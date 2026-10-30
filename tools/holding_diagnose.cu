#include "rl/shelter_motor_run.h"
#include "rl/shelter_hold_cases.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>

// Frozen, matched controller diagnosis. No trainer is constructed, no PPO
// update occurs, and no source checkpoint is written or leased. Every variant
// uses the production shelter Batch/action-mean path with resets/practice OFF.
namespace {
using namespace cudarl;
namespace sm=cudarl::shelter_motor;
constexpr double gamma=.99;
constexpr uint32_t case_seed=0x481D2026u,noise_seed=0x481D9015u;
using Four=std::array<float,4>;
struct Case {std::string label,group;sm::State world;};
struct Geometry {
    float bed[3]{},arch[3]{},boulder[3]{},solid[3]{},surface[3]{};
    float clearance=1e6f,exposure=1,head=1,proximal=1,distal=1,distance=0;
    bool finite=true;
};
struct Reward {
    float progress=0,distance=0,exposure=0,alignment=0,holding=0,contact=0,penetration=0,effort=0;
    float gap=0,near=0,quiet=0,facing=0,quality=0;
    float total() const {return progress+distance+exposure+alignment+holding+contact+penetration+effort;}
};
struct Row {
    std::array<float,sm::kObs> observation{};
    Four raw{},mean{},noise{},sample{},applied{};
    deep::BodyState before{},after{};
    Geometry geometry{};
    Reward reward{};
    float actual=0,residual=0,done=0,truncated=0,cached_exposure=1,observation_residual=0;
    int clock=0,held=0,peak=0,contacts=0;
    double return_to_go=0,discounted_prefix=0;
};
struct Outcome {
    std::string controller,label,group,first_failure,first_penetration;
    Geometry initial{};
    int first_failure_step=-1,first_penetration_step=-1,longest_safe=0,longest_safe_near=0,cached_peak=0,contacts=0,cached_disagreements=0;
    bool exact_initial=true,finite=true;
    float minimum_bed=1e6f,minimum_solid=1e6f,minimum_surface=1e6f,maximum_exposure=0,max_residual=0,max_observation_residual=0;
    double total_return=0,undiscounted=0;
};
struct Variant {std::string name;int actor=0;bool noisy=false,saved_std=false;};

std::vector<char> raw_file(const std::string& path) {
    std::ifstream in(path,std::ios::binary);if(!in) throw std::runtime_error("Cannot read protected source: "+path);
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
uint64_t fingerprint(const std::vector<char>& bytes) {
    uint64_t h=14695981039346656037ull;
    for(char value:bytes) {h^=static_cast<unsigned char>(value);h*=1099511628211ull;}return h;
}
std::string hex(uint64_t value) {std::ostringstream out;out<<std::hex<<value;return out.str();}
std::string quoted(const std::string& value) {
    std::string result="\"";
    for(char c:value) {if(c=='\\' || c=='\"') result+='\\';if(c=='\n') result+="\\n";else result+=c;}
    return result+'\"';
}
void number(std::ostream& out,double value) {if(std::isfinite(value)) out<<value;else out<<"null";}
template<class T> void download(std::vector<T>& host,const T* device,cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(host.data(),device,host.size()*sizeof(T),cudaMemcpyDeviceToHost,stream));
}
bool finite_body(const deep::BodyState& b) {
    if(!std::isfinite(deep::norm(b.position)) || !std::isfinite(b.orientation.x) || !std::isfinite(b.orientation.y) ||
       !std::isfinite(b.orientation.z) || !std::isfinite(b.orientation.w)) return false;
    for(float value:b.velocity) if(!std::isfinite(value)) return false;
    for(float value:b.angle) if(!std::isfinite(value)) return false;return true;
}
Geometry geometry(const sm::State& world) {
    Geometry result;const auto& body=world.state.pond.body;const auto& habitat=world.state.habitat;
    result.finite=finite_body(body);result.distance=deep::norm(body.position-world.request.goal.position);
    const auto exposure=predator::exposure_detail(body,world.request.eye,habitat,{2});
    result.exposure=exposure.visible;result.head=exposure.head;result.proximal=exposure.proximal_tail;result.distal=exposure.distal_tail;
    const auto shape=deep::shape(body);
    for(int link=0;link<3;++link) {
        result.bed[link]=result.arch[link]=result.boulder[link]=result.solid[link]=result.surface[link]=1e6f;
        const auto& part=shape.link[link];const auto axis=part.q.rotate({1,0,0});
        for(int slice=0;slice<deep::arch_axial_samples;++slice) {
            const float f=(slice-4)*.25f;
            const auto point=(part.center+axis*(f*part.length*.5f))/deep::scale;
            const auto gradient=deep::habitat_gradient(point.x,point.z,habitat,{2});
            const float magnitude=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
            const auto bed_normal=deep::V{-gradient.x,1,-gradient.z}/magnitude;
            result.bed[link]=fminf(result.bed[link],(point.y-deep::habitat_floor(point.x,point.z,habitat,{2}))/magnitude-
                deep::arch_slice_support(part,link,f,bed_normal)/deep::scale);
            result.surface[link]=fminf(result.surface[link],-point.y-deep::arch_slice_support(part,link,f,{0,1,0})/deep::scale);
            deep::V normal;float distance=deep::arch_distance(point,habitat,&normal);
            const float arch_gap=distance-deep::arch_slice_support(part,link,f,normal)/deep::scale;
            distance=deep::boulder_distance(point,habitat,&normal);
            const float boulder_gap=distance-deep::arch_slice_support(part,link,f,normal)/deep::scale;
            result.arch[link]=fminf(result.arch[link],arch_gap);result.boulder[link]=fminf(result.boulder[link],boulder_gap);
            const float gap=fminf(arch_gap,boulder_gap);
            result.solid[link]=fminf(result.solid[link],gap);
        }
        result.clearance=fminf(result.clearance,fminf(result.bed[link],fminf(result.solid[link],result.surface[link])));
    }
    result.finite=result.finite && std::isfinite(result.clearance) && std::isfinite(result.distance) && std::isfinite(result.exposure);
    return result;
}
bool safe(const Geometry& g) {return g.finite && g.clearance>=.08f && g.exposure<=.05f;}
float minimum(const float (&values)[3]) {return std::min({values[0],values[1],values[2]});}
std::string failure(const Geometry& g) {
    if(!g.finite) return "nonfinite";
    if(minimum(g.bed)<.08f) return "bed_clearance_below_0.08";
    if(minimum(g.arch)<.08f) return "arch_clearance_below_0.08";
    if(minimum(g.boulder)<.08f) return "boulder_clearance_below_0.08";
    if(minimum(g.surface)<.08f) return "surface_clearance_below_0.08";
    if(g.exposure>.05f) return "exposure_above_0.05";return "none";
}
std::string penetration(const Geometry& g) {
    if(!g.finite) return "none"; // Invalid geometry is not evidence of negative penetration.
    if(minimum(g.bed)<0) return "bed_penetration";
    if(minimum(g.arch)<0) return "arch_penetration";
    if(minimum(g.boulder)<0) return "boulder_penetration";
    if(minimum(g.surface)<0) return "surface_penetration";return "none";
}
sm::State holding_world(const arch::State& state,int goal) {
    sm::State world;world.state=state;
    const auto cover=predator::cover(goal,world.request.eye,state.habitat,{2});
    if(!cover.valid) throw std::runtime_error("The source habitat has no legal nominal cover");
    world.request.cover=true;world.request.goal.position=cover.position;world.request.goal.cover=goal;world.request.goal.valid=true;
    world.request.heading=deep::axis_rotation({0,1,0},cover.yaw).rotate({1,0,0});
    world.state.pond.target=-1;world.state.pond.decision=false;world.state.pond.steps=0;world.state.pond.since_food=0;
    world.state.pond.anchor=world.state.pond.body.position;world.held_ticks=world.peak_hold=0;
    world.exposure=predator::exposure(world.state.pond.body,world.request.eye,state.habitat,{2});return world;
}
std::vector<Case> cases(const rl::BoulderSource& source,uint32_t near_seed) {
    std::vector<Case> result;std::vector<sm::State> nominal;
    for(int goal=0;goal<4;++goal) {
        // Fresh complete fixtures, not a copied production clock which may
        // already exceed the 400-step lesson timeout. These are diagnostic
        // starts only and never deployed relocation or rescue.
        auto state=arch::initial(case_seed,0,static_cast<uint32_t>(goal),false,{2});state.habitat=source.frame.habitat;
        const auto cover=predator::cover(goal,{-8,4,-1.25f},state.habitat,{2});state.pond.body=predator::cover_pose(cover);
        auto world=holding_world(state,goal);nominal.push_back(world);
        result.push_back({"nominal_"+std::to_string(goal),"nominal_rest",world});
    }
    const auto near=rl::shelter_hold_cases(source.frame,near_seed);
    for(size_t i=0;i<near.size();++i) {
        auto world=holding_world(near[i].state,near[i].goal.cover);
        result.push_back({"near_"+std::to_string(i),"near_rest",world});
    }
    for(int goal=0;goal<4;++goal) for(int perturbation=0;perturbation<4;++perturbation) {
        auto world=nominal[static_cast<size_t>(goal)];auto& body=world.state.pond.body;
        const float sign=perturbation%2==0?1.f:-1.f;
        if(perturbation<2) {
            const auto v=body.orientation.rotate({sign*.003f,0,0});
            body.velocity[0]=v.x;body.velocity[1]=v.y;body.velocity[2]=v.z;body.velocity[4]=sign*.3f;
        } else {
            body.velocity[1]=sign*.002f;const auto w=body.orientation.rotate({0,0,sign*.3f});
            body.velocity[3]=w.x;body.velocity[4]=w.y;body.velocity[5]=w.z;
        }
        const std::string label=perturbation<2?"forward_yaw":"vertical_pitch";
        result.push_back({"velocity_"+std::to_string(goal)+"_"+label+(sign>0?"_plus":"_minus"),"initial_velocity",world});
    }
    return result;
}
Four saved_log_std(const rl::ShelterMotorRun& run) {
    checkpoint::Reader in{run.training};
    (void)in.value<uint32_t>();(void)in.value<uint32_t>();(void)in.value<uint32_t>();
    (void)in.value<int>();(void)in.value<uint32_t>();(void)in.value<int>();(void)in.value<sm::ActuatorScale>();
    (void)in.array<float>(rl::swimmer_specification(4,sm::kObs).num_params());
    (void)in.array<float>(rl::swimmer_specification(1,sm::kObs).num_params());
    const auto values=in.array<float>(4);if(values.size()!=4) throw std::runtime_error("Missing saved physical log_std");
    Four result{};
    for(int j=0;j<4;++j) {result[j]=values[static_cast<size_t>(j)];if(!std::isfinite(expf(result[j])) || !(expf(result[j])>0))
        throw std::runtime_error("Saved exploration deviation is not finite and positive");}return result;
}
Reward reconstruct(const sm::State& before,const sm::State& after,const Four& applied) {
    Reward r;const auto& body=after.state.pond.body;
    const float distance=deep::norm(body.position-after.request.goal.position);
    r.gap=predator::body_clearance(body,after.state.habitat,{2});
    r.facing=deep::clamp(deep::dot(body.orientation.rotate({1,0,0}),after.request.heading),-1,1);
    r.near=expf(-distance*distance/(.45f*.45f));
    r.quiet=expf(-deep::norm({body.velocity[0],body.velocity[1],body.velocity[2]})/.015f);
    r.quality=(1.f-deep::clamp(after.exposure,0,1))*deep::clamp((r.gap+.02f)/.10f,0,1);
    r.progress=3.f*(deep::norm(before.state.pond.body.position-before.request.goal.position)-distance);
    r.distance=-.008f*fminf(distance,4.f);r.exposure=-.025f*after.exposure;
    r.alignment=.025f*fmaxf(0,r.facing);
    r.holding=.1f*r.near*(.35f+.65f*fmaxf(0,r.facing))*r.quiet*r.quality;
    r.contact=-.2f*static_cast<float>(after.state.contacts-before.state.contacts);
    r.penetration=-.02f*deep::clamp(-r.gap/.1f,0,1);
    for(float value:applied) r.effort-=.002f*value*value;return r;
}
void csv_header(std::ostream& out) {
    out<<"controller,case,group,tick,time_s,world_clock,done,truncated";
    for(int j=0;j<sm::kObs;++j) out<<",obs_"<<j;
    for(const auto* name:{"raw_mean","physical_mean","matched_standard_normal","sampled_action","applied_clamped_action"})
        for(int j=0;j<4;++j) out<<','<<name<<'_'<<j;
    for(const auto* phase:{"before","after"}) {
        for(const auto* axis:{"x","y","z"}) out<<','<<phase<<"_position_"<<axis;
        for(const auto* axis:{"x","y","z","w"}) out<<','<<phase<<"_quaternion_"<<axis;
        for(int j=0;j<deep::dofs;++j) out<<','<<phase<<"_velocity_"<<j;
        for(int j=0;j<4;++j) out<<','<<phase<<"_joint_angle_"<<j;
    }
    out<<",proof_exposure,proof_head_exposure,proof_proximal_exposure,proof_distal_exposure,cached_exposure,proof_clearance,goal_distance";
    for(const auto* name:{"bed","arch","boulder","solid","surface"}) for(int j=0;j<3;++j) out<<','<<name<<"_link_"<<j;
    out<<",geometry_safe,geometry_safe_near,cached_safe_near,production_held_ticks,production_peak_hold,obstacle_contacts_delta"
        ",reward_progress,reward_distance,reward_cached_exposure,reward_alignment,reward_dense_holding,reward_contact,reward_penetration,reward_effort"
        ",reward_gap,reward_near,reward_quiet,reward_alignment_cos,reward_cover_quality,reward_reconstructed,reward_actual,reconstruction_residual,obs_cpu_residual,discounted_return_to_go_gamma_0_99,discounted_prefix_gamma_0_99\n";
}
void body_csv(std::ostream& out,const deep::BodyState& b) {
    out<<','<<b.position.x<<','<<b.position.y<<','<<b.position.z<<','<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w;
    for(float value:b.velocity) out<<','<<value;for(float value:b.angle) out<<','<<value;
}
void row_csv(std::ostream& out,const Variant& variant,const Case& c,int tick,const Row& r) {
    out<<variant.name<<','<<c.label<<','<<c.group<<','<<tick<<','<<(tick+1)*deep::dt<<','<<r.clock<<','<<r.done<<','<<r.truncated;
    for(float value:r.observation) out<<','<<value;
    for(const auto* values:{&r.raw,&r.mean,&r.noise,&r.sample,&r.applied}) for(float value:*values) out<<','<<value;
    body_csv(out,r.before);body_csv(out,r.after);const auto& g=r.geometry;const auto& v=r.reward;
    const bool proof=safe(g),near=g.distance<=.6f,cached=r.cached_exposure<=.05f && v.gap>=.08f && near;
    out<<','<<g.exposure<<','<<g.head<<','<<g.proximal<<','<<g.distal<<','<<r.cached_exposure<<','<<g.clearance<<','<<g.distance;
    for(const auto* values:{g.bed,g.arch,g.boulder,g.solid,g.surface}) for(int j=0;j<3;++j) out<<','<<values[j];
    out<<','<<proof<<','<<(proof && near)<<','<<cached<<','<<r.held<<','<<r.peak<<','<<r.contacts
       <<','<<v.progress<<','<<v.distance<<','<<v.exposure<<','<<v.alignment<<','<<v.holding<<','<<v.contact<<','<<v.penetration<<','<<v.effort
       <<','<<v.gap<<','<<v.near<<','<<v.quiet<<','<<v.facing<<','<<v.quality<<','<<v.total()<<','<<r.actual<<','<<r.residual<<','<<r.observation_residual
       <<','<<r.return_to_go<<','<<r.discounted_prefix<<'\n';
}
void initial_json(std::ostream& out,const Geometry& g) {
    out<<"{\"exposure\":";number(out,g.exposure);out<<",\"head\":";number(out,g.head);out<<",\"proximal_tail\":";number(out,g.proximal);
    out<<",\"distal_tail\":";number(out,g.distal);out<<",\"clearance\":";number(out,g.clearance);out<<",\"distance\":";number(out,g.distance);
    const char* names[]={"bed_by_link","arch_by_link","boulder_by_link","solid_by_link","surface_by_link"};
    const float* arrays[]={g.bed,g.arch,g.boulder,g.solid,g.surface};
    for(int kind=0;kind<5;++kind) {
        out<<",\""<<names[kind]<<"\":[";const auto* values=arrays[kind];
        for(int j=0;j<3;++j) {if(j) out<<',';number(out,values[j]);}out<<']';
    }
    out<<",\"geometry_safe\":"<<(safe(g)?"true":"false")<<'}';
}
std::vector<Outcome> diagnose(const Variant& variant,const std::vector<Case>& test,const rl::ShelterMotorRun& run,
                            const std::vector<float>& inherited,const Four& learned_std,int ticks,std::ostream& csv) {
    const int n=static_cast<int>(test.size());rl::SwimmerStream stream;sm::Batch batch(n,case_seed);
    auto context=run.source.context;context.current={};context.basin={2};
    batch.configure(run.source.ancestry.policy.actor,context,stream.value,false);batch.set_action_scales(run.scales);
    std::vector<sm::State> starts;for(const auto& c:test) starts.push_back(c.world);
    batch.set_states(starts,stream.value);auto before=batch.host_states(stream.value);
    std::unique_ptr<nn::GpuMlp> actor;
    if(variant.actor) {
        actor=std::make_unique<nn::GpuMlp>(rl::swimmer_specification(4,sm::kObs),n,7);
        actor->set_params(variant.actor==1?inherited:run.actor);
        // Parameter upload is a default-stream helper. Establish an explicit
        // dependency before this diagnostic consumes it on a nonblocking stream.
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
    const size_t actions=static_cast<size_t>(n)*sm::kAct;
    device_ptr<float> means(actions),samples(actions),rewards(n),done(n),truncated(n);
    std::vector<float> host_obs(static_cast<size_t>(n)*sm::kObs),host_raw(actions),host_means(actions),host_samples(actions),host_rewards(n),host_done(n),host_truncated(n);
    std::vector<Row> rows(static_cast<size_t>(n)*ticks);std::vector<Outcome> outcomes(static_cast<size_t>(n));
    std::vector<int> consecutive(static_cast<size_t>(n)),near_consecutive(static_cast<size_t>(n));
    for(int i=0;i<n;++i) {
        auto& out=outcomes[static_cast<size_t>(i)];out.controller=variant.name;out.label=test[static_cast<size_t>(i)].label;out.group=test[static_cast<size_t>(i)].group;
        out.exact_initial=std::memcmp(&starts[static_cast<size_t>(i)],&before[static_cast<size_t>(i)],sizeof(sm::State))==0;
        if(!out.exact_initial) throw std::runtime_error("Production prepare changed a complete cover fixture");
        out.initial=geometry(before[static_cast<size_t>(i)]);
        if(!safe(out.initial)) {out.first_failure_step=0;out.first_failure=failure(out.initial);}else out.first_failure="none";
        out.first_penetration=penetration(out.initial);if(out.first_penetration!="none") out.first_penetration_step=0;
        out.minimum_bed=minimum(out.initial.bed);out.minimum_solid=minimum(out.initial.solid);out.minimum_surface=minimum(out.initial.surface);
        out.maximum_exposure=out.initial.exposure;out.finite=out.initial.finite;
    }
    for(int tick=0;tick<ticks;++tick) {
        download(host_obs,batch.obs(),stream.value);
        if(actor) {
            actor->forward(batch.obs(),n,stream.value);
            sm::policy_means(actor->output(),batch.obs(),means.get(),n,run.scales,stream.value);
            download(host_raw,actor->output(),stream.value);download(host_means,means.get(),stream.value);
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
        for(int i=0;i<n;++i) {
            auto& row=rows[static_cast<size_t>(i)*ticks+tick];row.before=before[static_cast<size_t>(i)].state.pond.body;
            for(int j=0;j<sm::kObs;++j) row.observation[static_cast<size_t>(j)]=host_obs[static_cast<size_t>(i)*sm::kObs+j];
            float cpu[sm::kObs];sm::observe(before[static_cast<size_t>(i)],cpu,{2});
            for(int j=0;j<sm::kObs;++j) row.observation_residual=fmaxf(row.observation_residual,fabsf(cpu[j]-row.observation[static_cast<size_t>(j)]));
            for(int j=0;j<4;++j) {
                const size_t offset=static_cast<size_t>(i)*4+j;
                row.raw[static_cast<size_t>(j)]=actor?host_raw[offset]:0;row.mean[static_cast<size_t>(j)]=actor?host_means[offset]:0;
                row.noise[static_cast<size_t>(j)]=rng::normal(noise_seed,static_cast<uint32_t>(i),static_cast<uint32_t>(tick*4+j));
                const float deviation=variant.noisy?expf(variant.saved_std?learned_std[static_cast<size_t>(j)]:-1.5f):0;
                row.sample[static_cast<size_t>(j)]=row.mean[static_cast<size_t>(j)]+deviation*row.noise[static_cast<size_t>(j)];
                row.applied[static_cast<size_t>(j)]=deep::clamp(row.sample[static_cast<size_t>(j)],-1,1);host_samples[offset]=row.sample[static_cast<size_t>(j)];
            }
        }
        CUDA_CHECK_THROW(cudaMemcpyAsync(samples.get(),host_samples.data(),samples.size_bytes(),cudaMemcpyHostToDevice,stream.value));
        batch.step(samples.get(),{rewards.get(),done.get(),truncated.get(),nullptr},stream.value,false,false);
        download(host_rewards,rewards.get(),stream.value);download(host_done,done.get(),stream.value);download(host_truncated,truncated.get(),stream.value);
        auto after=batch.host_states(stream.value);
        for(int i=0;i<n;++i) {
            const size_t index=static_cast<size_t>(i);auto& row=rows[index*ticks+tick];auto& out=outcomes[index];const auto& world=after[index];
            row.after=world.state.pond.body;row.geometry=geometry(world);row.reward=reconstruct(before[index],world,row.applied);
            row.actual=host_rewards[index];row.residual=row.actual-row.reward.total();row.done=host_done[index];row.truncated=host_truncated[index];
            row.clock=world.state.pond.steps;row.cached_exposure=world.exposure;row.held=world.held_ticks;row.peak=world.peak_hold;
            row.contacts=world.state.contacts-before[index].state.contacts;const auto& g=row.geometry;
            if(!safe(g) && out.first_failure_step<0) {out.first_failure_step=tick+1;out.first_failure=failure(g);}
            if(out.first_penetration_step<0 && penetration(g)!="none") {out.first_penetration_step=tick+1;out.first_penetration=penetration(g);}
            consecutive[index]=safe(g)?consecutive[index]+1:0;
            near_consecutive[index]=safe(g) && g.distance<=.6f?near_consecutive[index]+1:0;
            out.longest_safe=std::max(out.longest_safe,consecutive[index]);out.longest_safe_near=std::max(out.longest_safe_near,near_consecutive[index]);
            const bool cached_safe=world.exposure<=.05f && row.reward.gap>=.08f;
            out.cached_disagreements+=cached_safe!=safe(g)?1:0;out.cached_peak=std::max(out.cached_peak,world.peak_hold);
            out.minimum_bed=fminf(out.minimum_bed,minimum(g.bed));out.minimum_solid=fminf(out.minimum_solid,minimum(g.solid));out.minimum_surface=fminf(out.minimum_surface,minimum(g.surface));
            out.maximum_exposure=fmaxf(out.maximum_exposure,g.exposure);out.contacts+=row.contacts;out.max_residual=fmaxf(out.max_residual,fabsf(row.residual));
            out.max_observation_residual=fmaxf(out.max_observation_residual,row.observation_residual);out.finite=out.finite && g.finite && std::isfinite(row.actual);
            out.undiscounted+=row.actual;out.total_return+=std::pow(gamma,tick)*row.actual;row.discounted_prefix=out.total_return;
        }
        before=std::move(after);
    }
    for(int i=0;i<n;++i) {
        double future=0;
        for(int tick=ticks-1;tick>=0;--tick) {auto& row=rows[static_cast<size_t>(i)*ticks+tick];future=row.actual+gamma*future;row.return_to_go=future;}
        for(int tick=0;tick<ticks;++tick) row_csv(csv,variant,test[static_cast<size_t>(i)],tick,rows[static_cast<size_t>(i)*ticks+tick]);
    }
    if(actor && actor->get_params()!=(variant.actor==1?inherited:run.actor))
        throw std::runtime_error("A frozen diagnostic actor unexpectedly changed");
    return outcomes;
}
void summary_json(std::ostream& out,const std::vector<Outcome>& results,const std::vector<Case>& test,const Four& stds,
                  const rl::ShelterMotorRun& run,const std::string& source,uint64_t source_hash,int ticks,uint32_t near_seed,bool unchanged) {
    out<<std::setprecision(10)<<"{\n\"source\":"<<quoted(source)<<",\"source_raw_fnv1a64\":"<<quoted(hex(source_hash))
       <<",\"source_bytes_unchanged\":"<<(unchanged?"true":"false")<<",\"snapshot_protocol\":"<<rl::shelter_motor_protocol
       <<",\"saved_update\":"<<run.update<<",\"saved_training_worlds\":"<<run.worlds<<",\"case_count\":"<<test.size()<<",\"ticks\":"<<ticks
       <<",\"near_seed\":"<<near_seed<<",\"near_seed_hex\":"<<quoted("0x"+hex(near_seed))
       <<",\"gamma\":0.99,\"autoreset\":false,\"practice\":false,\"ppo_updates\":0,\"controller_weights_frozen\":true,\"current_m_s\":[0,0,0],\"basin_layout\":2,\n"
       <<"\"proof\":\"Every tick: all 216 visibility samples and nine axial supports per link, including distal fin; clearance >=0.08 and exposure <=0.05 continuously for five seconds. Geometry proof is independent of distance.\",\n"
       <<"\"production_bookkeeping\":\"Reward and held_ticks use cached exposure refreshed when pond.steps%4==0. held_ticks additionally requires goal distance<=0.6. Cached bookkeeping is logged, not substituted for every-tick proof. Cover reward overrides the preceding arch reward.\",\n"
       <<"\"contact_scope\":\"arch.contacts counts solid obstacle contacts, not bed/surface contacts. Proof samples nine axial supports and the full distal fin; production bed contact uses three axial points and a narrower tail radius. Their causal role is not assumed. First margin failure and first negative penetration are reported separately.\",\n"
       <<"\"noise\":\"Counter-based common standard normals by case/tick/joint. zero_mean_shared_noise is not zero torque. Shared log_std=-1.5 in physical actuator units; frozen_v5_saved_noise uses its separately saved physical log_std.\",\n"
       <<"\"saved_log_std\":[";
    for(int j=0;j<4;++j) {if(j) out<<',';out<<stds[static_cast<size_t>(j)];}out<<"],\"saved_action_scales\":[";
    for(int j=0;j<4;++j) {if(j) out<<',';out<<run.scales.value[j];}out<<"],\n\"cases\":[";
    for(size_t i=0;i<test.size();++i) {
        if(i) out<<',';const auto& c=test[i];const auto& b=c.world.state.pond.body;
        out<<"{\"label\":"<<quoted(c.label)<<",\"group\":"<<quoted(c.group)<<",\"initial_clock\":"<<c.world.state.pond.steps
           <<",\"initial_done\":"<<(c.world.state.pond.invalid || c.world.state.pond.steps>=sm::kCoverEpisode?"true":"false")
           <<",\"initial_truncated\":"<<(!c.world.state.pond.invalid && c.world.state.pond.steps>=sm::kCoverEpisode?"true":"false")<<",\"position\":["<<b.position.x<<','<<b.position.y<<','<<b.position.z<<"],\"velocity\":[";
        for(int j=0;j<deep::dofs;++j) {if(j) out<<',';out<<b.velocity[j];}out<<"],\"initial_geometry\":";initial_json(out,geometry(c.world));out<<'}';
    }
    out<<"],\n\"outcomes\":[\n";
    for(size_t i=0;i<results.size();++i) {
        if(i) out<<",\n";const auto& r=results[i];out<<"{\"controller\":"<<quoted(r.controller)<<",\"case\":"<<quoted(r.label)<<",\"group\":"<<quoted(r.group)
           <<",\"complete_initial_state_identical\":"<<(r.exact_initial?"true":"false")<<",\"initial\":";initial_json(out,r.initial);
        out<<",\"first_failure_step\":"<<r.first_failure_step<<",\"first_failure_condition\":"<<quoted(r.first_failure)
           <<",\"first_negative_penetration_step\":"<<r.first_penetration_step<<",\"first_negative_penetration_condition\":"<<quoted(r.first_penetration)
           <<",\"longest_geometry_safe_s\":"<<r.longest_safe*deep::dt<<",\"longest_geometry_safe_near_s\":"<<r.longest_safe_near*deep::dt
           <<",\"production_cached_peak_s\":"<<r.cached_peak*deep::dt<<",\"cached_vs_every_tick_disagreements\":"<<r.cached_disagreements
           <<",\"obstacle_contacts\":"<<r.contacts<<",\"finite\":"<<(r.finite?"true":"false")<<",\"minimum_bed\":";number(out,r.minimum_bed);
        out<<",\"minimum_solid\":";number(out,r.minimum_solid);out<<",\"minimum_surface\":";number(out,r.minimum_surface);
        out<<",\"maximum_exposure\":";number(out,r.maximum_exposure);out<<",\"max_reward_reconstruction_residual\":";number(out,r.max_residual);
        out<<",\"max_cpu_gpu_observation_residual\":";number(out,r.max_observation_residual);out<<",\"discounted_return\":";number(out,r.total_return);
        out<<",\"undiscounted_return\":";number(out,r.undiscounted);out<<'}';
    }
    out<<"],\n\"aggregates\":[";bool first=true;
    for(const auto* controller:{"zero_torque","calibrated_inherited","frozen_v5","zero_mean_shared_noise","inherited_shared_noise","frozen_v5_shared_noise","frozen_v5_saved_noise"})
        for(const auto* group:{"nominal_rest","near_rest","initial_velocity"}) {
            int count=0,holds=0,near=0;double returns=0,held=0,residual=0;
            for(const auto& r:results) if(r.controller==controller && r.group==group) {++count;holds+=r.longest_safe>=100?1:0;near+=r.longest_safe_near>=100?1:0;returns+=r.total_return;held+=r.longest_safe*deep::dt;residual=std::max(residual,static_cast<double>(r.max_residual));}
            if(!first) out<<',';first=false;out<<"{\"controller\":"<<quoted(controller)<<",\"group\":"<<quoted(group)<<",\"cases\":"<<count<<",\"geometry_5s\":"<<holds<<",\"geometry_near_5s\":"<<near<<",\"mean_longest_geometry_s\":"<<held/std::max(count,1)<<",\"mean_discounted_return\":"<<returns/std::max(count,1)<<",\"max_reward_residual\":"<<residual<<'}';
        }
    out<<"]\n}\n";
}
}
int main(int argc,char** argv) {
    try {
        std::string source,output;int ticks=400;uint32_t near_seed=rl::shelter_hold_evaluation_seed;
        const auto parse_seed=[](const std::string& value) {
            size_t used=0;const auto parsed=std::stoull(value,&used,0);
            if(used!=value.size() || parsed>std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("Near-case seed must be an unsigned 32-bit integer");
            return static_cast<uint32_t>(parsed);
        };
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg.rfind("--source=",0)==0) source=arg.substr(9);
            else if(arg.rfind("--output=",0)==0) output=arg.substr(9);
            else if(arg.rfind("--ticks=",0)==0) ticks=std::stoi(arg.substr(8));
            else if(arg.rfind("--near-seed=",0)==0) near_seed=parse_seed(arg.substr(12));
            else if(arg=="--source" && i+1<argc) source=argv[++i];
            else if(arg=="--output" && i+1<argc) output=argv[++i];
            else if(arg=="--ticks" && i+1<argc) ticks=std::stoi(argv[++i]);
            else if(arg=="--near-seed" && i+1<argc) near_seed=parse_seed(argv[++i]);
            else throw std::invalid_argument("Usage: holding_diagnose --source=<protected motor-v5-training.bin> --output=<new directory> [--ticks=400] [--near-seed=0x48D119A5]");
        }
        if(source.empty() || output.empty() || ticks<100 || ticks>800) throw std::invalid_argument("Source/output required; diagnostic ticks must be100..800");
        const auto protected_bytes=raw_file(source);const auto hash=fingerprint(protected_bytes);
        const auto run=rl::load_shelter_motor_run(source);const auto stds=saved_log_std(run);const auto test=cases(run.source,near_seed);
        const auto inherited=rl::normalized_shelter_actor(rl::transfer_shelter_motor(run.source.motor),run.scales);
        if(std::filesystem::exists(output) && (!std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)))
            throw std::invalid_argument("Diagnostic output must be a new or empty directory; existing reports/source files are preserved");
        std::filesystem::create_directories(output);
        std::ofstream trace(std::filesystem::path(output)/"trace.csv");if(!trace) throw std::runtime_error("Cannot create diagnostic trace");
        trace<<std::setprecision(10);csv_header(trace);std::vector<Outcome> outcomes;
        const Variant variants[]={{"zero_torque",0,false,false},{"calibrated_inherited",1,false,false},{"frozen_v5",2,false,false},
            {"zero_mean_shared_noise",0,true,false},{"inherited_shared_noise",1,true,false},{"frozen_v5_shared_noise",2,true,false},{"frozen_v5_saved_noise",2,true,true}};
        for(const auto& variant:variants) {
            const auto results=diagnose(variant,test,run,inherited,stds,ticks,trace);
            outcomes.insert(outcomes.end(),results.begin(),results.end());
            int holds=0;double max_residual=0;for(const auto& result:results) {holds+=result.longest_safe>=100?1:0;max_residual=std::max(max_residual,static_cast<double>(result.max_residual));}
            std::cout<<variant.name<<": geometry5s="<<holds<<'/'<<test.size()<<" max_reward_residual="<<max_residual<<'\n';
        }
        trace.close();if(!trace) throw std::runtime_error("Could not finish diagnostic trace");
        const bool unchanged=raw_file(source)==protected_bytes;
        std::ofstream summary(std::filesystem::path(output)/"summary.json");if(!summary) throw std::runtime_error("Cannot create diagnostic summary");
        summary_json(summary,outcomes,test,stds,run,source,hash,ticks,near_seed,unchanged);summary.close();
        if(!summary || !unchanged) throw std::runtime_error("Diagnostic report failed or protected source changed");
        std::cout<<"Holding diagnosis complete; source bytes unchanged; no training, autoreset or practice. "<<output<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"holding_diagnose: "<<e.what()<<'\n';return 1;}
}
