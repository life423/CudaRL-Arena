#include "rl/flexible_tail_fast_steering.h"
#include "nn/mlp_cpu.h"
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace ft = cudarl::flexible_tail;
namespace fast = ft::fast;
namespace nn = cudarl::nn;

struct Check {
    int samples = 0, differing = 0;
    float maximum = 0;
    void add(float a, float b) {
        ++samples;
        if (std::memcmp(&a, &b, sizeof(a)) != 0) ++differing;
        maximum = std::fmax(maximum, std::fabs(a-b));
    }
};

int main(int argc, char** argv) {
    try {
        const auto started = std::chrono::steady_clock::now();
        if (argc != 3) throw std::runtime_error("root and output required");
        const std::string root = argv[1], output = argv[2];
        const auto gait = ft::read_navigation_weights(root + "/flexible-tail-preview/live_data/fitted-gait.bin",ft::NavigationMode::Gait);
        const auto offset = ft::read_yaw_offset_weights(root + "/flexible-tail-preview/live_data/yaw-offset-update-8.bin");
        const auto starts = ft::read_navigation_worlds(root + "/build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin");
        const auto retention = ft::read_navigation_worlds(root + "/build/flexible-tail/yaw-beat-v1/capture/eval-straight-worlds.bin");
        const auto evaluation = ft::read_navigation_worlds(root + "/build/flexible-tail/yaw-beat-v1/capture/eval-turn-worlds.bin");
        const auto same_state=[](const ft::NavigationWorld& a,const ft::NavigationWorld& b){
            return std::memcmp(&a.motion.body,&b.motion.body,sizeof(a.motion.body))==0&&
                std::memcmp(a.motion.history,b.motion.history,sizeof(a.motion.history))==0&&
                std::memcmp(a.motion.last_targets,b.motion.last_targets,sizeof(a.motion.last_targets))==0;
        };
        int unique_physical_starts=0, heldout_overlaps=0;
        for(size_t i=0;i<starts.size();++i){bool prior=false;for(size_t j=0;j<i;++j)prior=prior||same_state(starts[i],starts[j]);if(!prior)++unique_physical_starts;}
        for(const auto& heldout:evaluation){bool overlap=false;for(const auto& training:starts)overlap=overlap||same_state(heldout,training);if(overlap)++heldout_overlaps;}
        fast::validate_frozen_gait(gait);
        fast::validate_actor(offset);
        nn::CpuMlp<float> full_gait(ft::navigation_motor_spec()), full_actor(ft::yaw_offset_spec());
        full_gait.params() = gait;
        full_actor.params() = offset;
        Check sparse, actor, observer, critic_steering, bypass;
        int unchanged_bodies = 0, unchanged_history = 0, finite_critic = 0, valid_worlds = 0;
        float largest_target_error = 0;
        std::array<float,fast::critic_inputs> minimum, maximum;
        minimum.fill(std::numeric_limits<float>::max());
        maximum.fill(-std::numeric_limits<float>::max());
        std::array<int,8> counts{};
        std::ofstream rows(output + "/requests.csv");
        rows << "lane,source,angle_deg,actual_request_deg,feature0,feature1,feature2,feature3,old_nn_latent,direct_latent,critic_finite\n";
        const int cases=int(starts.size())*8;
        for (int lane=0; lane<cases; ++lane) {
            const int band=lane%8, sign=(band%2)==0 ? -1:1;
            const float degrees=float(15+15*(band/2))*sign;
            const auto& source = starts[(lane/8)%starts.size()];
            fast::World world;
            fast::initialize(world,source,degrees*ft::pi/180.f);
            if(world.fault==0) ++valid_worlds;
            if(std::memcmp(&world.navigation.motion.body,&source.motion.body,sizeof(source.motion.body))==0) ++unchanged_bodies;
            if(std::memcmp(world.navigation.motion.history,source.motion.history,sizeof(source.motion.history))==0) ++unchanged_history;
            ++counts[band];
            const float actual=ft::yaw_offset_error(world.navigation.motion.entry_course,world.navigation.motion.request.heading)*180.f/ft::pi;
            largest_target_error=std::fmax(largest_target_error,std::fabs(actual-degrees));
            float gait_observation[ft::navigation_inputs], latent[ft::motor_outputs];
            ft::observe_gait(world.navigation,gait_observation);
            full_gait.forward(gait_observation,1);
            fast::frozen_gait_latent(world,gait.data(),latent);
            for(int axis=0;axis<ft::motor_outputs;++axis) sparse.add(nn::activate(nn::Activation::Tanh,latent[axis]),full_gait.output()[axis]);
            // Include positive, negative and zero carried trims, without physics.
            world.applied_offset=(lane%3-1)*.03125f;
            float original_observation[ft::navigation_inputs], observation[ft::navigation_inputs];
            ft::observe_yaw_offset(world.navigation,world.applied_offset,original_observation);
            fast::observe(world,observation);
            for(int i=0;i<ft::navigation_inputs;++i) observer.add(observation[i],original_observation[i]);
            full_actor.forward(observation,1);
            float direct=0;
            for(int i=0;i<4;++i) direct+=observation[i]*offset[i];
            actor.add(direct,full_actor.output()[0]);
            float critic[fast::critic_inputs];
            fast::observe_critic(world,critic);
            bool finite=true;
            for(int i=0;i<fast::critic_inputs;++i){
                finite=finite&&std::isfinite(critic[i]);
                minimum[i]=std::fmin(minimum[i],critic[i]);
                maximum[i]=std::fmax(maximum[i],critic[i]);
            }
            for(int i=0;i<4;++i) critic_steering.add(critic[ft::navigation_inputs+i],observation[i]);
            if(finite) ++finite_critic;
            rows << lane << ',' << (lane/8)%starts.size() << ',' << degrees << ',' << actual << ',' << observation[0]<<','<<observation[1]<<','<<observation[2]<<','<<observation[3]<<','<<full_actor.output()[0]<<','<<direct<<','<<finite<<'\n';
        }
        // Inactive direct bypass on every captured straight/from-rest pose and
        // each task kind, including a carried nonzero trim and extreme samples.
        for(const auto& source:retention) for(int task=0;task<2;++task) for(float sample:{-100.f,0.f,100.f}){
            fast::World world;
            world.navigation=source;
            world.navigation.active_leg=0;
            world.navigation.leg_count=1;
            world.navigation.motion.request.choice=task==0?ft::Choice::Swim:ft::Choice::Hold;
            float gait_observation[ft::navigation_inputs], latent[ft::motor_outputs];
            ft::observe_gait(world.navigation,gait_observation);
            full_gait.forward(gait_observation,1);
            fast::frozen_gait_latent(world,gait.data(),latent);
            float targets[ft::motor_outputs], applied=.2f;
            ft::decode_yaw_offset(world.navigation,latent,sample,applied,targets);
            bypass.add(applied,0.f);
            for(int axis=0;axis<ft::motor_outputs;++axis) bypass.add(targets[axis],full_gait.output()[axis]);
        }
        std::array<int,8> seeded_counts{};
        std::array<bool,256> seeded_sources{};
        int seeded_finite=0, seeded_equal_pairs=0;
        for(int pair=0;pair<512;++pair){
            const uint32_t paired=cudarl::rng::hash_u32(uint32_t(pair)^cudarl::rng::hash_u32(0u));
            const int source=int(cudarl::rng::bits(20261124u,paired,1)%uint32_t(starts.size()));
            const int band=int(cudarl::rng::bits(20261124u,paired,2)%4u);
            seeded_sources[source]=true;
            fast::World left,right;
            fast::initialize(left,starts[source],float(15+15*band)*ft::pi/180.f);
            fast::initialize(right,starts[source],float(-15-15*band)*ft::pi/180.f);
            if(std::memcmp(&left.navigation.motion.body,&right.navigation.motion.body,sizeof(left.navigation.motion.body))==0&&std::memcmp(left.navigation.motion.history,right.navigation.motion.history,sizeof(left.navigation.motion.history))==0) ++seeded_equal_pairs;
            for(const auto* world:{&left,&right}){
                float critic[fast::critic_inputs];
                fast::observe_critic(*world,critic);
                bool finite=true;
                for(float value:critic)finite=finite&&std::isfinite(value);
                if(finite)++seeded_finite;
            }
            ++seeded_counts[2*band];++seeded_counts[2*band+1];
        }
        int unique_sources=0;for(bool value:seeded_sources) if(value)++unique_sources;
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::ofstream report(output+"/report.json");
        report<<std::setprecision(10)<<"{\n\"physics_steps\":0,\"optimizer_calls\":0,\"exhaustive_request_cases\":"<<cases<<",\"source_count\":"<<starts.size()<<",\"retention_pose_count\":"<<retention.size()<<",\"valid_worlds\":"<<valid_worlds<<",\"unchanged_bodies\":"<<unchanged_bodies<<",\"unchanged_history\":"<<unchanged_history<<",\"finite_critic_rows\":"<<finite_critic<<",\"maximum_request_angle_error_deg\":"<<largest_target_error<<",\"angle_band_counts\":[";
        for(int i=0;i<8;++i) report<<(i?",":"")<<counts[i];
        report<<"],\"unique_physical_training_states\":"<<unique_physical_starts<<",\"heldout_exact_state_overlaps\":"<<heldout_overlaps<<",\"actual_seed_initial_worlds\":1024,\"actual_seed_unique_source_records\":"<<unique_sources<<",\"actual_seed_equal_state_pairs\":"<<seeded_equal_pairs<<",\"actual_seed_finite_critic_rows\":"<<seeded_finite<<",\"actual_seed_angle_counts\":[";
        for(int i=0;i<8;++i)report<<(i?",":"")<<seeded_counts[i];
        report<<"],\n";
        auto emit=[&](const char* name,const Check& c){report<<'"'<<name<<"\":{\"values\":"<<c.samples<<",\"bit_differences\":"<<c.differing<<",\"max_abs_difference\":"<<c.maximum<<"},\n";};
        emit("sparse_gait_vs_full_nn",sparse);emit("four_weight_mean_vs_full_nn",actor);emit("historical_observer",observer);emit("critic_steering",critic_steering);emit("inactive_bypass",bypass);
        report<<"\"critic_feature_ranges\":[";
        for(int i=0;i<fast::critic_inputs;++i) report<<(i?",":"")<<'['<<minimum[i]<<','<<maximum[i]<<']';
        report<<"],\"elapsed_s\":"<<seconds<<"}\n";
        const bool passed=sparse.differing==0&&actor.differing==0&&observer.differing==0&&critic_steering.differing==0&&bypass.differing==0&&valid_worlds==cases&&unchanged_bodies==cases&&unchanged_history==cases&&finite_critic==cases&&largest_target_error<.0001f&&seeded_equal_pairs==512&&seeded_finite==1024&&heldout_overlaps==0;
        std::cout<<"passed="<<passed<<" gaitmax="<<sparse.maximum<<" actormax="<<actor.maximum<<" bypassmax="<<bypass.maximum<<" elapsed="<<seconds<<'\n';
        return passed?0:2;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
