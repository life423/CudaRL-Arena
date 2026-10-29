#ifndef CUDARL_SHELTER_MOTOR_RUN_H
#define CUDARL_SHELTER_MOTOR_RUN_H
#include "boulder_source.h"
#include "shelter_motor_ppo.h"
#include "shelter_probe.h"

namespace cudarl::rl {
struct ShelterMotorRun {
    BoulderSource source;
    std::vector<uint8_t> training;
    std::vector<float> actor;
    shelter_motor::ActuatorScale scales{};
    int update=0,worlds=0;uint32_t seed=0;
};
inline ShelterMotorRun read_shelter_motor_run(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};
    if(in.value<uint32_t>()!=0x534D4631u || in.value<uint32_t>()!=1u) throw std::runtime_error("Unsupported full shelter motor run");
    const auto fingerprint=in.value<uint64_t>();ShelterMotorRun result;
    result.source=read_boulder_source(in.array<uint8_t>(128000000));result.training=in.array<uint8_t>(256000000);in.finish();
    if(result.source.fingerprint!=fingerprint) throw std::runtime_error("Shelter motor parent identity mismatch");
    checkpoint::Reader learner{result.training};
    if(learner.value<uint32_t>()!=0x534D5452u || learner.value<uint32_t>()!=shelter_motor_protocol || learner.value<uint32_t>()!=sizeof(shelter_motor::State))
        throw std::runtime_error("Invalid nested shelter motor snapshot");
    result.worlds=learner.value<int>();result.seed=learner.value<uint32_t>();result.update=learner.value<int>();
    result.scales=learner.value<shelter_motor::ActuatorScale>();
    for(float value:result.scales.value) if(!std::isfinite(value) || value<1 || value>1048576)
        throw std::runtime_error("Invalid saved shelter action calibration");
    result.actor=learner.array<float>(swimmer_specification(4,shelter_motor::kObs).num_params());
    if(result.worlds<16 || result.worlds>2048 || result.update<0 || result.actor.size()!=static_cast<size_t>(swimmer_specification(4,shelter_motor::kObs).num_params()))
        throw std::runtime_error("Invalid shelter motor run dimensions");
    for(float v:result.actor) if(!std::isfinite(v)) throw std::runtime_error("Non-finite shelter motor run actor");
    return result;
}
inline ShelterMotorRun load_shelter_motor_run(const std::string& path) {return read_shelter_motor_run(checkpoint::read(path));}
inline void save_shelter_motor_run(const std::string& path,const BoulderSource& source,const ShelterMotorTrainer& trainer) {
    checkpoint::Writer out;out.value(uint32_t{0x534D4631u});out.value(uint32_t{1u});out.value(source.fingerprint);
    out.array(source.parent);out.array(shelter_motor_snapshot(trainer));checkpoint::write_atomic(path,out.data);
}
inline std::vector<ShelterProbeCase> shelter_motor_probe_cases(const arch::State& reference,uint32_t seed=0x571E2026u) {
    std::vector<ShelterProbeCase> cases;const deep::V eye{-8,4,-1.25f};
    const float facing[4]={0,1.5707963f,3.1415927f,-1.5707963f};
    const deep::V currents[3]={{},{.005f,0,0},{.0075f,0,0}};
    for(int location=0;location<2;++location) for(int heading=0;heading<4;++heading) for(auto current:currents) {
        auto state=arch::initial(seed,0,static_cast<uint32_t>(location*4+heading),false,{2});state.habitat=reference.habitat;
        const float angle=facing[heading]+(rng::uniform(seed,static_cast<uint32_t>(location),static_cast<uint32_t>(heading))*2-1)*.12f;
        state.pond.body.orientation=deep::axis_rotation({0,1,0},angle);
        for(float& v:state.pond.body.velocity) v=0;for(float& a:state.pond.body.angle) a=0;
        const auto point=predator::cover(0,eye,state.habitat,{2});
        if(!point.valid || point.exposure>.05f) throw std::runtime_error("Probe needs a physically valid whole-body shelter pose");
        shelter::Goal goal;goal.position=point.position;goal.cover=0;goal.valid=true;
        bool found=false;const int offset=static_cast<int>(rng::bits(seed,static_cast<uint32_t>(location),static_cast<uint32_t>(heading)+9)%81u);
        for(int trial=0;trial<81 && !found;++trial) {
            const int cell=(trial+offset)%81;
            const float x=(location==0?3.4f:6.4f)+(cell%9-4)*.2f,z=(location==0?2.f:-.2f)+(cell/9-4)*.2f;
            state.pond.body.position={x,-1.7f,z};
            found=predator::body_clearance(state.pond.body,state.habitat,{2})>=.08f && arch::water(state.pond.body.position,state.habitat,.3f,{2}) &&
                deep::norm(state.pond.body.position-goal.position)>=1.3f;
        }
        if(!found) throw std::runtime_error("No legal full-body start for the held-out shelter probe");
        state.pond.target=-1;state.pond.anchor=state.pond.body.position;
        cases.push_back({state,goal,eye,current,{2}});
    }
    return cases;
}
inline double shelter_motor_food_rate(const std::vector<float>& actor,const BoulderSource& source,uint32_t seed,int count=16,int steps=600,deep::V current={},shelter_motor::ActuatorScale scales={}) {
    auto context=source.context;context.current=current;
    shelter_motor::Batch trials(count,seed);SwimmerStream stream;
    trials.configure(source.ancestry.policy.actor,context,stream.value,true);
    std::vector<shelter_motor::State> starts(static_cast<size_t>(count));
    for(int i=0;i<count;++i) starts[i].state=arch::initial(seed,static_cast<uint32_t>(i),0,false,context.basin);
    trials.set_states(starts,stream.value);
    nn::GpuMlp motor(swimmer_specification(4,shelter_motor::kObs),count,7);motor.set_params(actor);
    device_ptr<float> means(static_cast<size_t>(count)*4);
    for(int tick=0;tick<steps;++tick) {
        motor.forward(trials.obs(),count,stream.value);
        shelter_motor::policy_means(motor.output(),trials.obs(),means.get(),count,scales,stream.value);
        trials.step(means.get(),{},stream.value,false,false);
    }
    const auto results=trials.host_states(stream.value);int food=0;for(const auto& state:results) food+=state.state.pond.eaten;
    return double(food)*60/(count*steps*deep::dt);
}
}
#endif
