#if defined(CUDARL_WITH_CUDA) && CUDARL_WITH_CUDA
#error "The saved-failure inspector is CPU-only; link cudarl_runtime, not cudarl_core"
#endif
#ifndef CUDARL_WITH_CUDA
#define CUDARL_WITH_CUDA 0
#endif
#include "rl/paired_motor_policy.h"
#include "rl/boulder_source.h"
#include "nn/adam_config.h"
#include "motor_benchmark.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// Offline decoding and geometry only. No GPU, inference, integrator, replay,
// training, optimizer step or modification of the recorded study is performed.
namespace {
using namespace cudarl;namespace pm=paired_motor;namespace cm=command_motor;namespace fs=std::filesystem;
void need(bool okay,const std::string& message) {if(!okay) throw std::runtime_error(message);}
template<class T> bool exact(const std::vector<T>& a,const std::vector<T>& b) {return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
template<class T> uint64_t hash_array(const std::vector<T>& values) {checkpoint::Writer out;out.array(values);return pm::fingerprint(out.data);}
struct Adam {nn::AdamConfig config;int steps=0;std::vector<float> m,v;uint64_t hash=0;};
Adam read_adam(checkpoint::Reader& in,int parameters) {
    Adam out;out.config=in.value<nn::AdamConfig>();out.steps=in.value<int>();out.m=in.array<float>(parameters);out.v=in.array<float>(parameters);
    need(out.m.size()==static_cast<size_t>(parameters)&&out.v.size()==out.m.size(),"Saved Adam dimensions differ");
    checkpoint::Writer record;record.value(out.config);record.value(out.steps);record.array(out.m);record.array(out.v);out.hash=pm::fingerprint(record.data);return out;
}
struct Learner {
    pm::Interface mode{};pm::RegulationGate gate;int n=0,update=0;uint32_t seed=0;uint64_t environment=0;
    std::vector<float> actor,critic,sigma,observations;Adam actor_adam,critic_adam,std_adam;
    std::vector<pm::State> states,first;std::vector<uint32_t> invalid,safety;
};
Learner read_learner(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};Learner out;
    need(in.value<uint32_t>()==0x504D5452u&&in.value<uint32_t>()==rl::paired_motor_protocol_version&&in.value<uint32_t>()==sizeof(pm::State),"Unsupported saved paired learner layout");
    out.mode=in.value<pm::Interface>();out.environment=in.value<uint64_t>();out.gate=pm::RegulationGate::read(in.array<uint8_t>(1000000));out.n=in.value<int>();out.seed=in.value<uint32_t>();out.update=in.value<int>();
    need(out.n==512&&out.seed==rl::paired_motor_seed&&out.update>=0&&out.update<=64&&out.environment==rl::paired_motor_fingerprint(out.mode,out.gate.protocol)&&in.array<float>()==rl::paired_motor_protocol(),"Paired learner protocol mismatch");
    out.actor=in.array<float>(rl::paired_motor_specification().num_params());out.critic=in.array<float>(rl::paired_motor_specification(1).num_params());out.sigma=in.array<float>(4);
    need(out.actor.size()==static_cast<size_t>(rl::paired_motor_specification().num_params())&&out.critic.size()==static_cast<size_t>(rl::paired_motor_specification(1).num_params())&&out.sigma.size()==4,"Saved paired neural dimensions differ");
    out.actor_adam=read_adam(in,static_cast<int>(out.actor.size()));out.critic_adam=read_adam(in,static_cast<int>(out.critic.size()));out.std_adam=read_adam(in,4);
    need(in.value<uint32_t>()==0x504D4241u&&in.value<uint32_t>()==pm::kProtocol&&in.value<uint32_t>()==sizeof(pm::State)&&in.value<int>()==out.n&&in.value<uint32_t>()==out.seed&&in.value<pm::Interface>()==out.mode,"Saved paired environment layout mismatch");
    need(pm::same_gains(in.value<pm::Gains>(),out.gate.gains),"Saved environment gains mismatch");
    out.states=in.array<pm::State>(out.n);out.observations=in.array<float>(static_cast<size_t>(out.n)*59);out.invalid=in.array<uint32_t>(out.n);out.safety=in.array<uint32_t>(out.n);out.first=in.array<pm::State>(out.n);in.finish();
    need(out.states.size()==static_cast<size_t>(out.n)&&out.observations.size()==static_cast<size_t>(out.n)*59&&out.invalid.size()==out.states.size()&&out.safety.size()==out.states.size()&&out.first.size()==out.states.size(),"Saved paired world dimensions mismatch");
    need(out.actor_adam.steps==out.update*32&&out.critic_adam.steps==out.update*32&&out.std_adam.steps==out.update*32,"Saved completed-update Adam counters mismatch");return out;
}
struct Failure {
    pm::Interface mode{};int update=0,adam_before=0,actor_adam_after=0;uint64_t invalid=0,unsafe=0;
    std::vector<uint8_t> before;std::vector<pm::State> first,states;std::vector<float> obs,actions,logp,rewards;
};
Failure read_failure(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};Failure out;need(in.value<uint32_t>()==0x504D4641u&&in.value<uint32_t>()==1,"Unsupported rejected-rollout record");
    out.mode=in.value<pm::Interface>();out.update=in.value<int>();out.adam_before=in.value<int>();out.actor_adam_after=in.value<int>();out.before=in.array<uint8_t>(128000000);
    out.invalid=in.value<uint64_t>();out.unsafe=in.value<uint64_t>();out.first=in.array<pm::State>(512);out.states=in.array<pm::State>(512);
    out.obs=in.array<float>(512*64*59);out.actions=in.array<float>(512*64*4);out.logp=in.array<float>(512*64);out.rewards=in.array<float>(512*64);in.finish();return out;
}
uint64_t clock(const cm::State& state) {return static_cast<uint64_t>(state.episode)*600u+static_cast<uint64_t>(state.steps);}
const char* face_name(int face) {static const char* names[]={"x_min","x_max","bed","surface","z_min","z_max"};return face>=0&&face<6?names[face]:"unknown";}
const char* link_name(int link) {return link==0?"head":(link==1?"proximal_tail":"distal_tail_fin");}
struct Boundary {float envelope=1e6f,native_contact=1e6f;int face=-1,link=-1,native_face=-1,native_link=-1;deep::V point{};};
Boundary boundary(const deep::BodyState& body) {
    Boundary out;const auto geometry=deep::shape(body);const deep::V normals[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    const float offset[6]={4.5f*deep::scale,4.5f*deep::scale,4.f*deep::scale,0,3.5f*deep::scale,3.5f*deep::scale};
    auto consider=[&](int face,int link,float gap,deep::V point) {if(gap<out.envelope) {out.envelope=gap;out.face=face;out.link=link;out.point=point;}};
    for(int link=0;link<3;++link) {
        const auto& l=geometry.link[link];
        for(int face=0;face<6;++face) {
            const auto normal=normals[face],local=l.q.inverse(normal);
            if(link==0) {
                const float a=.010f,b=.0064f,c=.008f,support=sqrtf(a*a*local.x*local.x+b*b*local.y*local.y+c*c*local.z*local.z);
                const auto point=l.center+l.q.rotate({-a*a*local.x/support,-b*b*local.y/support,-c*c*local.z/support});
                consider(face,link,deep::dot(normal,l.center)+offset[face]-support,point);
            }
            for(int axial=0;axial<9;++axial) {
                const float fraction=(axial-4)*.25f,profile=link==0?sqrtf(fmaxf(0.f,1-fraction*fraction)):1.f;
                const auto center=l.center+l.q.rotate({fraction*l.length*.5f,0,0});const float ry=(link==0?.0064f:(link==2?.006f:.0015f))*profile,rz=(link==0?.008f:.0015f)*profile;
                const float support=deep::arch_slice_support(l,link,fraction,normal);deep::V point=center;
                if(support>0) point+=l.q.rotate({0,-ry*ry*local.y/support,-rz*rz*local.z/support});
                consider(face,link,deep::dot(normal,center)+offset[face]-support,point);
            }
            // Exactly the rectangular contact proxy used in deep::assemble:
            // three axial samples, head radius6.4mm and tail radius2mm.
            for(int sample=0;sample<3;++sample) {
                const auto point=l.center+l.q.rotate({(sample-1)*l.length/3.f,0,0});
                const float gap=deep::dot(normal,point)+offset[face]-(link==0?.0064f:.002f);
                if(gap<out.native_contact) {out.native_contact=gap;out.native_face=face;out.native_link=link;}
            }
        }
    }
    return out;
}
struct First {pm::State state;int world=0,rollout_tick=0;double rollout_seconds=0,training_seconds=0;Boundary shape;};
uint64_t capture_ticks(const fs::path& path,uint64_t source) {
    const auto bytes=checkpoint::read(path.string());checkpoint::Reader in{bytes};need(in.value<uint32_t>()==0x504D4649u&&in.value<uint32_t>()==1&&in.value<uint64_t>()==source,"Saved final fixture source differs");
    (void)in.value<uint64_t>();const auto fixtures=in.array<motor_benchmark::Fixture>(64);in.finish();need(fixtures.size()==64,"Expected64 preregistered final fixtures");
    uint64_t ticks=0;for(const auto& fixture:fixtures) {need(fixture.warmup_ticks>=0,"Negative saved fixture capture clock");ticks+=static_cast<uint64_t>(fixture.warmup_ticks);}return ticks;
}
void inspect(const fs::path& path,const fs::path& output,const fs::path& latest_path,const fs::path& policy_path) {
    const auto bytes=checkpoint::read(path.string());const auto failure=read_failure(bytes);const auto before=read_learner(failure.before);const int n=before.n;
    need(failure.mode==before.mode&&failure.update==before.update&&failure.adam_before==before.actor_adam.steps,"Rejected rollout and pre-update source disagree");
    need(failure.states.size()==static_cast<size_t>(n)&&failure.obs.size()==static_cast<size_t>(n)*64*59&&failure.actions.size()==static_cast<size_t>(n)*64*4&&failure.logp.size()==static_cast<size_t>(n)*64&&failure.rewards.size()==static_cast<size_t>(n)*64,"Rejected rollout arrays have unexpected dimensions");
    need(!failure.first.empty(),"This recorded failure contains no first-unsafe physical state");need(!fs::exists(output),"Choose a new offline inspection output directory");fs::create_directories(output);
    const fs::path study=path.parent_path().parent_path().parent_path();
    bool latest_exact=false,candidate_exact=false,source_matches=false;uint64_t latest_hash=0;int candidate_update=-1,source_motor_update=-1,source_learner_update=-1;
    if(fs::exists(latest_path)) {const auto latest=checkpoint::read(latest_path.string());latest_exact=latest==failure.before;latest_hash=pm::fingerprint(latest);}
    if(fs::exists(policy_path)) {const auto policy=rl::PairedMotorPolicy::load(policy_path.string());candidate_exact=exact(policy.actor,before.actor);candidate_update=policy.update;need(policy.source_fingerprint==before.gate.source_fingerprint,"Candidate has another original source");}
    const auto archived_source=study/"sources"/"source-boulder.bin";
    if(fs::exists(archived_source)) {const auto source=rl::read_boulder_source(checkpoint::read(archived_source.string()));source_matches=source.fingerprint==before.gate.source_fingerprint;source_motor_update=source.motor_update;source_learner_update=source.learner_update;}
    std::vector<First> failures;for(const auto& state:failure.first) {
        const int world=static_cast<int>(state.task.environment);need(world>=0&&world<n,"First unsafe world index is invalid");const auto start=clock(before.states[static_cast<size_t>(world)].task),end=clock(state.task);
        need(end>start&&end-start<=64&&state.first_unsafe_substep>=0&&state.first_unsafe_substep<=10,"Recorded unsafe clock/substep is invalid");First item;item.state=state;item.world=world;item.rollout_tick=static_cast<int>(end-start)-1;
        item.rollout_seconds=item.rollout_tick*.05+state.first_unsafe_substep*.005;item.training_seconds=start*.05+item.rollout_seconds;item.shape=boundary(state.first_unsafe_task.body);
        need(fabsf(item.shape.envelope-cm::physical_clearance(state.first_unsafe_task.body))<=1e-7f,"Offline boundary classification differs from native full-envelope query");failures.push_back(item);
    }
    std::sort(failures.begin(),failures.end(),[](const First& a,const First& b) {return a.rollout_seconds==b.rollout_seconds?a.world<b.world:a.rollout_seconds<b.rollout_seconds;});
    uint64_t integrated=0;std::ofstream worlds(output/"world-clocks.csv");worlds<<"world,before_episode,before_tick,after_episode,after_tick,integrated_control_ticks,frozen_rollout_slots\n";
    for(int i=0;i<n;++i) {const auto& a=before.states[i].task;const auto& b=failure.states[i].task;need(clock(b)>=clock(a)&&clock(b)-clock(a)<=64,"Recorded physical clock cannot represent this rollout");const auto ticks=clock(b)-clock(a);integrated+=ticks;
        worlds<<i<<','<<a.episode<<','<<a.steps<<','<<b.episode<<','<<b.steps<<','<<ticks<<','<<64-ticks<<'\n';}
    std::ofstream cases(output/"unsafe-worlds.csv");cases<<std::setprecision(17)<<"world,rollout_tick,global_rollout_row,substep,rollout_seconds,training_seconds,envelope_clearance_m,boundary,anatomy,native_proxy_min_gap_m,native_proxy_boundary,native_proxy_anatomy,unsafe_substep_samples\n";
    for(const auto& item:failures) cases<<item.world<<','<<item.rollout_tick<<','<<item.rollout_tick*n+item.world<<','<<item.state.first_unsafe_substep<<','<<item.rollout_seconds<<','<<item.training_seconds
        <<','<<item.shape.envelope<<','<<face_name(item.shape.face)<<','<<link_name(item.shape.link)<<','<<item.shape.native_contact<<','<<face_name(item.shape.native_face)<<','<<link_name(item.shape.native_link)<<','<<item.state.unsafe_substeps<<'\n';
    const auto& first=failures.front();std::ofstream window(output/"first-breach-window.csv");window<<std::setprecision(17);
    window<<"world,rollout_tick,global_row,physics_integrated,logp,reward,raw0,raw1,raw2,raw3,previous_actual0,previous_actual1,previous_actual2,previous_actual3,joint0_rad,joint1_rad,joint2_rad,joint3_rad,rate0_rad_s,rate1_rad_s,rate2_rad_s,rate3_rad_s,head_x_model,head_y_model,head_z_model";
    for(int j=0;j<59;++j) window<<",obs"<<j;window<<'\n';
    const auto actual_ticks=clock(failure.states[first.world].task)-clock(before.states[first.world].task);
    for(int tick=std::max(0,first.rollout_tick-3);tick<=std::min(63,first.rollout_tick+3);++tick) {
        const size_t row=static_cast<size_t>(tick)*n+first.world;const float* obs=failure.obs.data()+row*59;
        window<<first.world<<','<<tick<<','<<row<<','<<(static_cast<uint64_t>(tick)<actual_ticks)<<','<<failure.logp[row]<<','<<failure.rewards[row];
        for(int j=0;j<4;++j) window<<','<<failure.actions[row*4+j];for(int j=0;j<4;++j) window<<','<<obs[29+j];for(int j=0;j<4;++j) window<<','<<obs[18+j]*deep::limit;for(int j=0;j<4;++j) window<<','<<obs[22+j]*12.f;
        window<<','<<obs[26]*4.5f<<','<<obs[27]*2.f-2.f<<','<<obs[28]*3.5f;for(int j=0;j<59;++j) window<<','<<obs[j];window<<'\n';
    }
    checkpoint::Writer raw;raw.value(uint32_t{0x504D4953u});raw.value(uint32_t{1});raw.value(first.state);checkpoint::write_atomic((output/"first-unsafe-saved-state.bin").string(),raw.data);
    const auto warmups=capture_ticks(study/"sources"/"final-fixtures.bin",before.gate.source_fingerprint);
    const uint64_t completed=static_cast<uint64_t>(before.update)*n*64,slots=static_cast<uint64_t>(n)*64;
    std::ostringstream summary;summary<<std::setprecision(17)<<"{\n\"offline_only\":true,\n\"additional_physical_steps\":0,\n\"inference_or_learning_calls\":0,\n\"arm\":\""<<rl::paired_motor_arm(before.mode)<<"\",\n\"completed_update\":"<<before.update
        <<",\n\"attempted_update\":"<<before.update+1<<",\n\"actor_adam_before\":"<<failure.adam_before<<",\n\"actor_adam_after\":"<<failure.actor_adam_after<<",\n\"critic_adam_before\":"<<before.critic_adam.steps<<",\n\"std_adam_before\":"<<before.std_adam.steps
        <<",\n\"before_fullsave_hash\":\""<<pm::fingerprint(failure.before)<<"\",\n\"latest_fullsave_hash\":\""<<latest_hash<<"\",\n\"latest_fullsave_equals_before\":"<<(latest_exact?"true":"false")<<",\n\"exported_candidate_update\":"<<candidate_update<<",\n\"exported_actor_equals_before_bitwise\":"<<(candidate_exact?"true":"false")
        <<",\n\"abort_time_weights_and_moments_independently_recorded\":false,\n\"evidence_limit\":\"Recorded actor Adam counter plus exact last-good checkpoint/export; abort-time actor/critic/std parameters and moments were not separately serialized. Control-flow guard is before GAE/Adam.\",\n\"before_actor_hash\":\""<<hash_array(before.actor)<<"\",\n\"before_actor_adam_hash\":\""<<before.actor_adam.hash<<"\",\n\"before_critic_adam_hash\":\""<<before.critic_adam.hash<<"\",\n\"before_std_adam_hash\":\""<<before.std_adam.hash<<"\",\n"
        <<"\"bound_source_fingerprint\":\""<<before.gate.source_fingerprint<<"\",\n\"archived_source_matches\":"<<(source_matches?"true":"false")<<",\n\"source_motor_update\":"<<source_motor_update<<",\n\"source_learner_update\":"<<source_learner_update
        <<",\n\"invalid_world_events\":"<<failure.invalid<<",\n\"unsafe_geometric_samples\":"<<failure.unsafe<<",\n\"unsafe_worlds\":"<<failures.size()<<",\n\"rollout_slots\":"<<slots<<",\n\"actually_integrated_control_ticks\":"<<integrated<<",\n\"frozen_slots\":"<<slots-integrated<<",\n\"tick_count_exact_for_finite_failure\":"<<(failure.invalid==0?"true":"false")
        <<",\n\"completed_training_ticks\":"<<completed<<",\n\"total_training_ticks_including_rejected_collection\":"<<completed+integrated<<",\n\"regulation_physical_ticks\":"<<before.gate.cpu_steps+before.gate.cuda_steps<<",\n\"final_fixture_capture_ticks\":"<<warmups<<",\n\"final_policy_evaluation_ticks\":0,\n\"final_evaluation_budget_spent_on_capture\":"<<warmups
        <<",\n\"earliest\":{\"world\":"<<first.world<<",\"rollout_tick\":"<<first.rollout_tick<<",\"global_rollout_row\":"<<first.rollout_tick*n+first.world<<",\"substep\":"<<first.state.first_unsafe_substep<<",\"rollout_seconds\":"<<first.rollout_seconds<<",\"training_seconds\":"<<first.training_seconds<<",\"full_envelope_clearance_m\":"<<first.shape.envelope
        <<",\"boundary\":\""<<face_name(first.shape.face)<<"\",\"anatomy\":\""<<link_name(first.shape.link)<<"\",\"native_contact_proxy_min_gap_m\":"<<first.shape.native_contact<<",\"native_proxy_boundary\":\""<<face_name(first.shape.native_face)<<"\",\"native_proxy_anatomy\":\""<<link_name(first.shape.native_link)
        <<"\",\"full_envelope_breach_without_native_proxy_penetration\":"<<(first.shape.envelope<0&&first.shape.native_contact>=0?"true":"false")<<"},\n\"interpretation\":\"A saved full-envelope safety rejection; native contacts use three axial spheres with2mm tail proxy while the sampled distal fin is6mm vertically. Geometric disagreement or compliant contact must be distinguished from a global control-authority conclusion.\"\n}\n";
    std::ofstream report(output/"summary.json");need(bool(report),"Cannot create offline summary");report<<summary.str();report.flush();need(bool(report),"Cannot write offline summary");
    need(checkpoint::read(path.string())==bytes,"Input rejected rollout changed during offline inspection");std::cout<<"PAIRED_FAILURE_OFFLINE "<<summary.str();
}
}
int main(int argc,char** argv) {
    try {
        if(argc<3||argc>5) {std::cerr<<"Usage: paired_motor_failure REJECTED-ROLLOUT NEW-OUTPUT-DIR [LATEST-TRAINING] [CANDIDATE-POLICY]\n";return 2;}
        const fs::path input=fs::absolute(argv[1]);const auto latest=argc>=4?fs::path(argv[3]):input.parent_path()/"latest-training.bin";const auto candidate=argc>=5?fs::path(argv[4]):input.parent_path()/"candidate-policy.bin";
        inspect(input,fs::absolute(argv[2]),latest,candidate);return 0;
    } catch(const std::exception& error) {std::cerr<<"PAIRED_FAILURE_INSPECTION_STOPPED "<<error.what()<<'\n';return 1;}
}
