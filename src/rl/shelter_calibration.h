#ifndef CUDARL_SHELTER_CALIBRATION_H
#define CUDARL_SHELTER_CALIBRATION_H
#include "boulder_source.h"
#include "shelter_motor_ppo.h"
#include "nn/mlp_cpu.h"
#include <array>
#include <cstring>

namespace cudarl::rl {
// Development-only actuator-unit calibration. These starts are not the
// readiness or adoption cases, and no samples enter the PPO replay buffer.
constexpr uint32_t shelter_calibration_version=1;
constexpr uint32_t shelter_calibration_seed=0xC41B51A7u;
constexpr int shelter_calibration_per_cover=256;
struct ShelterActionCalibration {
    shelter_motor::ActuatorScale scale{};
    float median[4]{},p95[4]{},maximum[4]{},static_maximum[4]{};
    int samples=0;
    uint64_t protocol=0,observations=14695981039346656037ull,fingerprint=0;
};
inline void shelter_calibration_hash(uint64_t& hash,const void* data,size_t size) {
    const auto* bytes=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<size;++i) {hash^=bytes[i];hash*=1099511628211ull;}
}
inline uint64_t shelter_calibration_protocol_fingerprint() {
    checkpoint::Writer out;out.value(shelter_calibration_version);out.value(shelter_calibration_seed);
    out.value(shelter_calibration_per_cover);
    const std::vector<float> ranges={.45f,1.4f,.25f,.65f,2.4f,.04f,4.f,6.f,.65f,.35f,.25f,.08f,.95f};
    out.array(ranges);uint64_t hash=14695981039346656037ull;
    shelter_calibration_hash(hash,out.data.data(),out.data.size());return hash;
}
inline shelter_motor::ActuatorScale calibrate_shelter_actions(const BoulderSource& source,ShelterActionCalibration* report=nullptr) {
    const auto weights=transfer_shelter_motor(source.motor);
    for(float value:weights) if(!std::isfinite(value)) throw std::invalid_argument("Non-finite calibration source");
    nn::CpuMlp<float> actor(swimmer_specification(4,shelter_motor::kObs));actor.params()=weights;
    ShelterActionCalibration result;result.protocol=shelter_calibration_protocol_fingerprint();
    std::array<std::vector<float>,4> magnitudes;
    const auto record=[&](const shelter_motor::State& world,bool fixed_pose) {
        float input[shelter_motor::kObs];shelter_motor::observe(world,input,{2});
        for(float value:input) if(!std::isfinite(value)) throw std::runtime_error("Non-finite calibration observation");
        shelter_calibration_hash(result.observations,input,sizeof(input));actor.forward(input,1);
        for(int j=0;j<4;++j) {
            const float value=fabsf(actor.output()[j]);
            if(!std::isfinite(value)) throw std::runtime_error("Non-finite calibrated actuator output");
            magnitudes[j].push_back(value);result.maximum[j]=std::max(result.maximum[j],value);
            if(fixed_pose) result.static_maximum[j]=std::max(result.static_maximum[j],value);
        }
        ++result.samples;
    };
    for(int goal=0;goal<predator::kCoverCount;++goal) {
        shelter_motor::State fixed;fixed.state=source.frame;
        const auto cover=predator::cover(goal,fixed.request.eye,fixed.state.habitat,{2});
        if(!cover.valid) throw std::runtime_error("Calibration needs legal boulder cover geometry");
        fixed.state.pond.body=predator::cover_pose(cover);fixed.state.pond.target=-1;fixed.state.pond.decision=false;
        fixed.request.cover=true;fixed.request.goal={cover.position,{},-1,goal,true};
        fixed.request.heading=deep::axis_rotation({0,1,0},cover.yaw).rotate({1,0,0});record(fixed,true);
        int accepted=0;
        for(uint32_t attempt=0;attempt<16384u && accepted<shelter_calibration_per_cover;++attempt) {
            const uint32_t key=static_cast<uint32_t>(goal)*131071u+attempt*31u;
            const auto draw=[&](uint32_t offset) {return 2.f*rng::uniform(shelter_calibration_seed,static_cast<uint32_t>(goal),key+offset)-1.f;};
            shelter_motor::State world=fixed;
            // Food context is real, seeded geometry, not fabricated shelter food.
            world.state=arch::initial(shelter_calibration_seed,0,attempt+static_cast<uint32_t>(goal)*16384u,false,{2});
            world.state.habitat=fixed.state.habitat;world.state.pond.target=-1;world.state.pond.decision=false;
            auto& body=world.state.pond.body;body=predator::cover_pose(cover);
            const float distance=accepted%2?.45f:1.4f;
            body.position+=deep::V{draw(0)*distance,draw(1)*.25f,draw(2)*distance};
            const float yaw=cover.yaw+draw(3)*(accepted%2?.65f:2.4f);
            body.orientation=deep::axis_rotation({0,1,0},yaw)*deep::axis_rotation({0,0,1},draw(4)*.35f)*deep::axis_rotation({1,0,0},draw(5)*.25f);
            for(int j=0;j<4;++j) body.angle[j]=draw(static_cast<uint32_t>(j)+6u)*.65f;
            for(int j=0;j<deep::dofs;++j) body.velocity[j]=draw(static_cast<uint32_t>(j)+10u)*(j<3?.04f:(j<6?4.f:6.f));
            if(predator::body_clearance(body,world.state.habitat,{2})<.08f || !arch::water(body.position,world.state.habitat,.2f,{2})) continue;
            world.request=fixed.request;record(world,false);++accepted;
        }
        if(accepted!=shelter_calibration_per_cover) throw std::runtime_error("Insufficient legal calibration poses");
    }
    for(int j=0;j<4;++j) {
        auto& values=magnitudes[j];std::sort(values.begin(),values.end());
        result.median[j]=values[values.size()/2];
        result.p95[j]=values[(values.size()-1)*95/100];
        const float bound=std::max(1.f,std::max(result.p95[j],result.static_maximum[j]));
        float scale=1;
        while(scale<bound) {scale*=2;if(!std::isfinite(scale)) throw std::runtime_error("Calibration scale overflow");}
        result.scale.value[j]=scale;
    }
    result.fingerprint=result.protocol;
    shelter_calibration_hash(result.fingerprint,&source.fingerprint,sizeof(source.fingerprint));
    shelter_calibration_hash(result.fingerprint,&result.observations,sizeof(result.observations));
    shelter_calibration_hash(result.fingerprint,result.scale.value,sizeof(result.scale.value));
    if(report) *report=result;return result.scale;
}
}
#endif
