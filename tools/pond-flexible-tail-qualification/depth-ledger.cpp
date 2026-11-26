#include "habitat_config.h"
#include "rl/flexible_tail_rate_chooser.h"
#include "rl/ppo_math.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace tail = cudarl::flexible_tail;
namespace rl = cudarl::rl;
using World = tail::rate_chooser::World;
constexpr int lanes = 1024, decisions = 8, batch = 1024, rows = lanes * decisions;
constexpr std::array<int,3> depth_mm{80,75,72};
struct Totals {
    int unique = 0, presentations = 0, lanes_started = 0, respawns = 0, ends = 0;
    uint64_t north_starts = 0, north_stops = 0, north_limits = 0, near_decisions = 0;
    int boundary_samples = 0;
    int responsive_fields = 0;
    float largest_history_delta = 0;
    double max_com_error_mm = 0;
};
enum class Practice { OpenApproach, NearParallel, Interior };
constexpr const char* practice_names[]={"open_approach","near_parallel","interior"};
Practice practice(int lane) {
    const auto group=(lane/2)%4;
    if(group<2) return Practice::OpenApproach;
    return group==2 ? Practice::NearParallel : Practice::Interior;
}
// Reconstruct the spawn stratum independently from the serialized lane index.
// Each 24-lane block covers both approach signs, all depths and all cohorts.
int reporting_group(int lane, bool bands) {
    const auto p=practice(lane);
    if(!bands) return static_cast<int>(p);
    return p==Practice::OpenApproach ? (lane/24)%3 : p==Practice::NearParallel ? 3 : 4;
}
constexpr const char* band_names[]={"short_moving","medium_moving","long_moving","tangent","interior"};
struct Cohort {
    int lanes=0, decisions=0, presentations=0;
    uint64_t starts=0, stops=0, limits=0, near=0;
};
void require(bool ok) { if (!ok) throw std::runtime_error("Inconsistent three-depth checkpoint record"); }

int main(int argc, char** argv) {
    try {
        const bool bands=argc==4 && std::string(argv[2])=="--moving-bands";
        const bool open_water = bands || (argc==4 && std::string(argv[2])=="--open-water");
        if (argc != 2 && !open_water) throw std::runtime_error("Expected checkpoint directory [--open-water|--moving-bands cohort-output.csv]");
        std::array<Totals,3> totals{};
        std::array<std::array<Cohort,5>,3> cohorts{};
        int previous_steps = 0, unexpected = 0;
        for (int u=0; u<=128; ++u) {
            const auto path = std::filesystem::path(argv[1]) / ("session-"+std::to_string(u)+".bin");
            const auto bytes = cudarl::checkpoint::read(path.u8string());
            cudarl::checkpoint::Reader in{bytes};
            require(in.value<uint32_t>()==0x52435446);
            const auto version=in.value<uint32_t>();
            require(version==2 || version==4);
            require(in.array<char>(64).size()==64);
            for(int field : {lanes,decisions,35,15,batch,4,40,128}) require(in.value<int>()==field);
            const auto lane_bytes=in.value<uint32_t>(), count_bytes=in.value<uint32_t>();
            require(lane_bytes>=sizeof(World) && count_bytes==136);
            for(int i=0;i<3;++i) in.value<uint64_t>();
            in.value<tail::steering::Kind>(); in.value<tail::steering::Config>();
            for(int i=0;i<8;++i) in.value<float>();
            in.value<cudarl::pond_geometry_v1::Parameters>();
            uint32_t episode_bytes=0;
            if(version==4) {
                require(in.value<int>()==1200);
                episode_bytes=in.value<uint32_t>();
                require(episode_bytes==32);
            }
            require(in.value<int>()==u);
            const int steps=in.value<int>();
            require(in.value<int>()==steps);
            if (u) {
                const int taken=steps-previous_steps;
                require(taken>=0 && taken<=32);
                std::array<bool,rows> seen{};
                for(int m=0;m<taken;++m) {
                    const auto perm=rl::make_permutation(rows,cudarl::rng::bits(20261124,u,m/8));
                    for(int i=0;i<batch;++i) {
                        const auto row=rl::permute(perm,(m%8)*batch+i);
                        const int lane=row%lanes, depth=(lane/2)%3;
                        auto& cohort=totals[depth];
                        auto& group=cohorts[depth][reporting_group(lane,bands)];
                        ++cohort.presentations;
                        ++group.presentations;
                        if(!seen[row]) { ++cohort.unique; ++group.decisions; seen[row]=true; }
                    }
                }
            }
            previous_steps=steps;
            // Episodic forks append the native per-lane clock/counter array.
            const size_t end_of_counts=episode_bytes ? bytes.size()-8-lanes*episode_bytes : bytes.size();
            if(episode_bytes){in.position=end_of_counts;require(in.value<uint64_t>()==lanes);}
            const size_t counts=end_of_counts-lanes*count_bytes;
            const size_t bodies=counts-8-lanes*lane_bytes;
            in.position=bodies-8; require(in.value<uint64_t>()==lanes);
            in.position=counts-8; require(in.value<uint64_t>()==lanes);
            for(int i=0;i<lanes;++i) {
                auto& total=totals[(i/2)%3];
                in.position=counts+i*count_bytes+4*sizeof(uint64_t);
                int actions=0;
                for(int a=0;a<15;++a) actions+=in.value<uint32_t>();
                require(actions==u*decisions);
                auto ends=in.value<uint32_t>(); in.value<uint32_t>(); in.value<uint32_t>();
                const auto starts=in.value<uint32_t>();
                if(!u) { require(!starts && !ends); continue; }
                in.position=bodies+i*lane_bytes;
                const auto world=in.value<World>();
                const int expected=depth_mm[(i/2)%3];
                const double origin_mm=-1000.*world.motor.navigation.motion.request.origin_m.y;
                if(!std::isfinite(origin_mm)||std::abs(origin_mm-expected)>.01) ++unexpected;
                const double com_error=std::abs(-1000.*world.pose.com_m.y-expected);
                require(std::isfinite(com_error));
                total.max_com_error_mm=std::max(total.max_com_error_mm,com_error);
                ++total.boundary_samples;
                if(u==128) {
                    require(starts>=1);
                    ++total.lanes_started; total.respawns+=starts-1; total.ends+=ends;
                    if(episode_bytes) {
                        in.position=end_of_counts+8+i*episode_bytes+8;
                        const auto limits=in.value<uint64_t>();
                        const auto near=in.value<uint64_t>();
                        total.near_decisions+=near;
                        require(in.value<uint64_t>()==decisions*u);
                        if(open_water) {
                            auto& group=cohorts[(i/2)%3][reporting_group(i,bands)];
                            ++group.lanes; group.starts+=starts; group.stops+=ends;
                            group.limits+=limits; group.near+=near;
                            if(practice(i)==Practice::Interior) require(limits==0);
                        } else {
                            if((i/2)%2==0) { total.north_starts+=starts; total.north_stops+=ends; total.north_limits+=limits; }
                            else require(limits==0);
                        }
                    }
                    auto fresh=world;
                    for(auto& word:fresh.visited) word=0;
                    for(auto& visit:fresh.recent) visit={};
                    float actual[35], cleared[35];
                    tail::rate_chooser::observe(world,larger_pond,actual,2);
                    tail::rate_chooser::observe(fresh,larger_pond,cleared,2);
                    float delta=0;
                    for(int k=32;k<35;++k) delta=std::max(delta,std::abs(cleared[k]-actual[k]));
                    total.responsive_fields+=delta>1.e-6f;
                    total.largest_history_delta=std::max(total.largest_history_delta,delta);
                }
            }
        }
        require(unexpected==0);
        if(open_water) {
            std::ofstream report;
            report.exceptions(std::ios::failbit|std::ios::badbit);
            report.open(argv[3]);
            report<<"depth_mm,practice,lanes,unique_PPO_decisions,minibatch_presentations,starts,clearance_stops,time_limits,near_decisions\n";
            for(int d=2;d>=0;--d) for(int g=0;g<(bands?5:3);++g) {
                const auto& v=cohorts[d][g];
                require(v.lanes>0 && v.decisions>0 && v.starts>=uint64_t(v.lanes));
                report<<depth_mm[d]<<','<<(bands?band_names[g]:practice_names[g])<<','<<v.lanes<<','<<v.decisions<<','<<v.presentations<<','
                      <<v.starts<<','<<v.stops<<','<<v.limits<<','<<v.near<<'\n';
            }
        }
        std::cout<<"depth_mm,initial_worlds,respawns,training_clearance_stops,unique_PPO_decisions,minibatch_presentations,boundary_COM_samples,max_boundary_COM_error_mm,final_fields_changed_when_history_cleared,largest_history_delta";
        if(!open_water) std::cout<<",north_training_starts,north_training_stops,north_time_limits";
        std::cout<<",near_decisions\n";
        for(int d=2;d>=0;--d) {
            const auto& t=totals[d];
            std::cout<<depth_mm[d]<<','<<t.lanes_started<<','<<t.respawns<<','<<t.ends<<','<<t.unique<<','<<t.presentations<<','
                <<t.boundary_samples<<','<<std::setprecision(10)<<t.max_com_error_mm<<','<<t.responsive_fields<<','<<t.largest_history_delta;
            if(!open_water) std::cout<<','<<t.north_starts<<','<<t.north_stops<<','<<t.north_limits;
            std::cout<<','<<t.near_decisions<<'\n';
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
