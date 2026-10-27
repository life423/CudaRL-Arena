#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "cuda/device.h"
#include "cuda/cuda_memory.h"
#include "rl/pond_selector_policy.h"
#include "rl/ppo_kernels.h"
#include "nn/mlp_gpu.h"
namespace current_diagnosis {
constexpr int population=64,subject=24;
constexpr uint32_t layout_seed=0xC071A201u;
// Preserve the exact recorded current. The original evaluator's kernels are
// also reused so this sensitive trajectory keeps its numerical execution path.
const d::V flow{-.007071067579f,0,-.007071067579f};
struct Frame {p::State state;int target=-1;float action[4]{},scores[3]{};};
__global__ void choose_override(const p::State* states,int* targets,int mode,int local_tick) {
    if(mode==0 || mode==1) return;
    const int i=threadIdx.x;if(i>=population) return;
    if(mode==2) {float o[33];p::observe(states[i],o);targets[i]=cudarl::demos::nearest_pond_food(o);}
    else if(mode>=3 && mode<=5) targets[i]=mode-3;
    else if(mode>=6 && mode<=8 && local_tick<20) targets[i]=mode-6;
}
struct World {
    cudarl::nn::GpuMlp motor,selector;
    device_ptr<p::State> states;device_ptr<int> targets;
    device_ptr<float> obs,features,motor_obs,logp;device_ptr<Case> metrics;
    const rl::SelectorPolicy& policy;
    explicit World(const rl::SelectorPolicy& weights):motor(rl::swimmer_specification(4,33),population,7),
        selector(rl::selector_spec(),population*3,7),states(population),targets(population),obs(population*36),features(population*3*32),
        motor_obs(population*33),logp(population),metrics(population),policy(weights) {
        motor.set_params(policy.motor);selector.set_params(policy.selector);
        CUDA_CHECK_THROW(cudaMemset(metrics.get(),0,metrics.size_bytes()));
        initialize<<<1,population>>>(states.get(),obs.get(),targets.get(),population,layout_seed);CUDA_CHECK_THROW(cudaGetLastError());
    }
    Frame capture() {
        Frame out;
        CUDA_CHECK_THROW(cudaMemcpy(&out.state,states.get()+subject,sizeof(p::State),cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy(&out.target,targets.get()+subject,sizeof(int),cudaMemcpyDeviceToHost));return out;
    }
    Frame step(int tick,d::V water,int mode=0) {
        if(tick%policy.decision_steps==0) {
            selector_inputs<<<1,population>>>(states.get(),targets.get(),obs.get(),population);CUDA_CHECK_THROW(cudaGetLastError());
            ps::make_features(obs.get(),features.get(),population,nullptr);selector.forward(features.get(),population*3,nullptr);
            rl::sample_actions(selector.output(),population,3,0,0,true,targets.get(),logp.get(),nullptr);
            choose_override<<<1,population>>>(states.get(),targets.get(),mode,tick);CUDA_CHECK_THROW(cudaGetLastError());
        }
        motor_inputs<<<1,population>>>(states.get(),targets.get(),motor_obs.get(),population);CUDA_CHECK_THROW(cudaGetLastError());motor.forward(motor_obs.get(),population,nullptr);
        advance_current<<<1,population>>>(states.get(),motor.output(),metrics.get(),population,water);CUDA_CHECK_THROW(cudaGetLastError());
        auto out=capture();
        CUDA_CHECK_THROW(cudaMemcpy(out.action,motor.output()+subject*4,4*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy(out.scores,selector.output()+subject*3,3*sizeof(float),cudaMemcpyDeviceToHost));return out;
    }
    void restore(const Frame& frame) {
        std::vector<p::State> copy(population,frame.state);std::vector<int> choice(population,frame.target);
        states.copy_from_host(copy.data(),copy.size());targets.copy_from_host(choice.data(),choice.size());
    }
    void verify() {if(motor.get_params()!=policy.motor || selector.get_params()!=policy.selector) throw std::runtime_error("Probe changed network weights");}
};
float min_gap(const p::State& s) {
    const auto geometry=d::shape(s.body);float minimum=1e6f;
    for(int b=0;b<3;++b) for(int sample=0;sample<3;++sample) {
        const auto& l=geometry.link[b];const auto point=l.center+l.q.rotate({1,0,0})*((sample-1)*l.length/3.f);
        const float x=point.x/d::scale,z=point.z/d::scale,r2=x*x/(4.6f*4.6f)+z*z/(3.5f*3.5f);
        const float gx=r2<1?27.f*x*r2*r2/(4.6f*4.6f):0,gz=r2<1?27.f*z*r2*r2/(3.5f*3.5f):0;
        minimum=std::min(minimum,(point.y-d::pond_floor(x,z)*d::scale)/std::sqrt(1+gx*gx+gz*gz)-(b==0?.0064f:.002f));
    }
    return minimum;
}
void vector_json(std::ostream& out,d::V v) {out<<'['<<v.x<<','<<v.y<<','<<v.z<<']';}
void quat_json(std::ostream& out,d::Q q) {out<<'['<<q.x<<','<<q.y<<','<<q.z<<','<<q.w<<']';}
void frame_json(std::ostream& out,const Frame& f) {
    const auto& s=f.state;const auto geometry=d::shape(s.body);
    out<<"{\"time\":"<<s.steps*.05<<",\"eaten\":"<<s.eaten<<",\"since_food\":"<<s.since_food*.05<<",\"target\":"<<f.target<<",\"position\":";
    vector_json(out,s.body.position);out<<",\"orientation\":";quat_json(out,s.body.orientation);
    out<<",\"speed_cm_s\":"<<d::norm({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]})*100<<",\"bank_gap_cm\":"<<min_gap(s)*100<<",\"foods\":[";
    for(int i=0;i<3;++i) {if(i) out<<',';vector_json(out,s.food[i]);}
    out<<"],\"links\":[";
    for(int i=0;i<3;++i) {if(i) out<<',';out<<"{\"position\":";vector_json(out,geometry.link[i].center/d::scale);out<<",\"orientation\":";quat_json(out,geometry.link[i].q);out<<'}';}
    out<<"],\"angles\":[";for(int i=0;i<4;++i) {if(i) out<<',';out<<s.body.angle[i];}
    out<<"],\"actions\":[";for(int i=0;i<4;++i) {if(i) out<<',';out<<f.action[i];}
    out<<"],\"scores\":[";for(int i=0;i<3;++i) {if(i) out<<',';out<<f.scores[i];}out<<"]}";
}
int run_current_diagnosis(const std::string& policy_path,const std::string& output_path) {
    try {
        const auto original=cudarl::checkpoint::read(policy_path);const auto policy=rl::SelectorPolicy::load(policy_path);
        const auto device=cudarl::probe_device();if(!device.available) throw std::runtime_error(device.error);
        const std::filesystem::path output(output_path);std::filesystem::create_directories(output);
        std::ofstream trace(output/"trace.csv"),summary(output/"branches.csv"),replay(output/"replay.json");
        if(!trace || !summary || !replay) throw std::runtime_error("Could not create diagnostic outputs");
        trace<<std::setprecision(9)<<"seconds,foods,since_food,target,x,y,z,speed_cm_s,bank_gap_cm,angle0,angle1,angle2,angle3,action0,action1,action2,action3,score0,score1,score2,food_distance0,food_distance1,food_distance2\n";
        replay<<std::setprecision(9)<<"{\"version\":1,\"seed\":"<<layout_seed<<",\"environment\":"<<subject<<",\"source\":\"Recorded CUDA physics and frozen neural inference; no training\",\"cases\":[{\"name\":\"Original current trial\",\"current_cm_s\":1,\"frames\":[";
        World world(policy);frame_json(replay,world.capture());Frame fixture,at120;int max_gap=0;
        for(int tick=0;tick<6000;++tick) {
            const auto frame=world.step(tick,flow);replay<<',';frame_json(replay,frame);
            const auto& s=frame.state;max_gap=std::max(max_gap,s.since_food);
            if(s.steps==1200) fixture=frame;
            if(s.steps==2400) at120=frame;
            if(s.steps%20==0) {
                trace<<s.steps*.05<<','<<s.eaten<<','<<s.since_food*.05<<','<<frame.target<<','<<s.body.position.x<<','<<s.body.position.y<<','<<s.body.position.z<<','
                    <<d::norm({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]})*100<<','<<min_gap(s)*100;
                for(float x:s.body.angle) trace<<','<<x;for(float x:frame.action) trace<<','<<x;for(float x:frame.scores) trace<<','<<x;
                for(auto food:s.food) trace<<','<<d::norm(food-s.body.position)*4;trace<<'\n';
            }
        }
        world.verify();const auto final=world.capture();
        std::printf("REPLAY_CHECK foods=%d max_gap_ticks=%d flow=%.12g\n",final.state.eaten,max_gap,flow.x);
        if(final.state.eaten!=8 || max_gap!=5312) throw std::runtime_error("The original five-minute stall did not reproduce");
        cudarl::checkpoint::Writer saved;saved.value(uint32_t{0x43535431});saved.value(fixture.state);saved.value(fixture.target);saved.value(flow);
        cudarl::checkpoint::write_atomic((output/"stall-at-60s.bin").string(),saved.data);
        std::printf("ORIGINAL foods=%d max_food_gap=%.2f fixture_target=%d bank_gap_cm=%.4f\n",final.state.eaten,max_gap*.05,fixture.target,min_gap(fixture.state)*100);
        replay<<"]}";summary<<"mode,new_foods,first_food_seconds,final_since_food_seconds,mean_speed_cm_s,mean_bank_penetration_cm,target_changes\n";
        const char* names[9]={"Original current from 60 s","Current removed at 60 s","Nearest-food choice (diagnostic)","Hold food 1 (diagnostic)","Hold food 2 (diagnostic)","Hold food 3 (diagnostic)","Food 1 for 1 s then original","Food 2 for 1 s then original","Food 3 for 1 s then original"};
        for(int mode=0;mode<9;++mode) {
            world.restore(fixture);Frame f=fixture;float first=-1;double speed=0,penetration=0;int changes=0,previous=fixture.target;
            replay<<",{\"name\":\""<<names[mode]<<"\",\"current_cm_s\":"<<(mode==1?0:1)<<",\"frames\":[";frame_json(replay,f);
            for(int tick=0;tick<1200;++tick) {
                f=world.step(tick,mode==1?d::V{}:flow,mode);replay<<',';frame_json(replay,f);
                if(f.state.eaten>fixture.state.eaten && first<0) first=(tick+1)*.05f;
                speed+=d::norm({f.state.body.velocity[0],f.state.body.velocity[1],f.state.body.velocity[2]})*100;
                penetration+=std::max(0.f,-min_gap(f.state))*100;
                if(f.target!=previous) ++changes;previous=f.target;
            }
            replay<<"]}";world.verify();const int foods=f.state.eaten-fixture.state.eaten;
            if(mode==0 && (d::norm(f.state.body.position-at120.state.body.position)!=0 || f.target!=at120.target || f.state.since_food!=at120.state.since_food))
                throw std::runtime_error("Restoring the stalled snapshot changed its original continuation");
            summary<<mode<<','<<foods<<','<<first<<','<<f.state.since_food*.05<<','<<speed/1200<<','<<penetration/1200<<','<<changes<<'\n';
            std::printf("BRANCH mode=%d foods=%d first=%.2f speed=%.4f penetration_cm=%.4f changes=%d\n",mode,foods,first,speed/1200,penetration/1200,changes);
        }
        replay<<"]}";
        if(original!=cudarl::checkpoint::read(policy_path)) throw std::runtime_error("Original policy changed");
        if(!trace || !summary || !replay) throw std::runtime_error("Diagnostic write failed");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Current diagnosis: %s\n",e.what());return 1;}
}

} // namespace current_diagnosis
