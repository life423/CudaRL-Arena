#ifndef CUDARL_SIMULATION_WORLD_H
#define CUDARL_SIMULATION_WORLD_H
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "physics/contact.h"

namespace cudarl::simulation {
enum class Actuation:uint32_t {Torque=0,JointTarget=1};
// Future motor contract; this diagnostic world does not act on requests.
struct MotorRequest {physics::V position{},terminal_velocity{},heading{1,0,0};};
struct Gains {
    float kp[4]{0.0000036832668683928205f,0.0000033448234262323240f,9.1073098928973195e-7f,0.0000014377696970768739f};
    float kd[4]{5.8621014886739431e-7f,5.3234515462463605e-7f,1.4494733591163822e-7f,2.2882814221247827e-7f};
};
struct ContactTelemetry {
    contact::Inspection inspection{};
    float minimum_floor_gap=1e6f,minimum_other_gap=1e6f;
    float floor_force_max=0,total_force_max=0;
    int contact_samples=0,executed_substeps=0,first_unsafe_substep=-1;
};
struct WorldState {
    physics::BodyState body{};
    physics::Parameters fluid{};
    physics::V current{};
    Actuation actuation=Actuation::JointTarget;
    Gains gains{};
    float held_targets[4]{}; // radians, two universal joints with two axes each
    float last_actual_torques[4]{}; // normalized [-1,1]; multiply by10 microNm
    float raw_action[4]{}; // requested normalized torque or normalized joint target
    float actual_torques[40]{}; // normalized applied torques: substep*4+axis
    float requested_torques[40]{}; // pre-saturation physical torque in N*m
    float mean_squared_torque=0; // mean square of actual normalized torque
    uint64_t control_steps=0,substeps=0;
    double time_seconds=0;
    bool paused=true,unsafe=false,invalid=false;
    ContactTelemetry telemetry{};
};
CUDARL_HD inline bool finite_state(const WorldState& s) {
    if(!physics::finite_body(s.body)||!physics::finite_vector(s.current)) return false;
    if(s.actuation!=Actuation::Torque&&s.actuation!=Actuation::JointTarget) return false;
    if(!physics::finite_number(s.fluid.density)||s.fluid.density<0||
       !physics::finite_number(s.fluid.viscosity)||s.fluid.viscosity<0||
       !physics::finite_number(s.fluid.gravity)||s.fluid.gravity<0) return false;
    for(int j=0;j<4;++j) if(!physics::finite_number(s.gains.kp[j])||s.gains.kp[j]<0||
        !physics::finite_number(s.gains.kd[j])||s.gains.kd[j]<0) return false;
    return true;
}
// One 50ms interval, at most ten 5ms physical substeps. Actuation only supplies
// internal torques. Safety freezes an unsafe world; it never projects or rescues it.
CUDARL_HD inline bool step_state(WorldState& s,const float* input) {
    if(s.paused||s.unsafe||s.invalid) return false;
    s.telemetry={};
    if(!finite_state(s)) {s.invalid=true;return false;}
    for(int j=0;j<4;++j) if(!physics::finite_number(input[j])) {s.invalid=true;return false;}
    for(int j=0;j<4;++j) {
        s.raw_action[j]=input[j];
        if(s.actuation==Actuation::JointTarget)
            s.held_targets[j]=physics::limit*physics::clamp(input[j],-1,1);
    }
    for(int j=0;j<40;++j) {s.actual_torques[j]=0;s.requested_torques[j]=0;}
    s.mean_squared_torque=0;
    auto initial=contact::inspect(s.body);
    s.telemetry.inspection=initial;
    s.telemetry.minimum_floor_gap=initial.floor_gap;
    s.telemetry.minimum_other_gap=initial.min_other_gap;
    if(initial.invalid) {s.invalid=true;return false;}
    if(initial.excessive||initial.escape) {s.unsafe=true;s.telemetry.first_unsafe_substep=0;return false;}
    for(int k=0;k<10;++k) {
        float actual[4]{};
        for(int j=0;j<4;++j) {
            const float requested=s.actuation==Actuation::Torque?physics::torque*physics::clamp(input[j],-1,1):
                s.gains.kp[j]*(s.held_targets[j]-s.body.angle[j])-s.gains.kd[j]*s.body.velocity[6+j];
            actual[j]=physics::clamp(requested/physics::torque,-1,1);
            s.requested_torques[k*4+j]=requested;s.actual_torques[k*4+j]=actual[j];
            s.last_actual_torques[j]=actual[j];s.mean_squared_torque+=actual[j]*actual[j];
        }
        contact::Telemetry force;
        contact::substep(s.body,actual,contact::kSubstep,s.fluid,s.current,&force);
        ++s.substeps;++s.telemetry.executed_substeps;
        s.time_seconds=static_cast<double>(s.substeps)*static_cast<double>(contact::kSubstep);
        s.telemetry.floor_force_max=fmaxf(s.telemetry.floor_force_max,force.floor_force);
        s.telemetry.total_force_max=fmaxf(s.telemetry.total_force_max,force.total_force);
        s.telemetry.contact_samples+=force.points;
        const auto inspection=contact::inspect(s.body);
        s.telemetry.inspection=inspection;
        s.telemetry.minimum_floor_gap=fminf(s.telemetry.minimum_floor_gap,inspection.floor_gap);
        s.telemetry.minimum_other_gap=fminf(s.telemetry.minimum_other_gap,inspection.min_other_gap);
        if(inspection.invalid||!finite_state(s)) {s.invalid=true;s.telemetry.first_unsafe_substep=k+1;break;}
        if(inspection.excessive||inspection.escape) {s.unsafe=true;s.telemetry.first_unsafe_substep=k+1;break;}
    }
    if(s.telemetry.executed_substeps>0) {
        s.mean_squared_torque/=static_cast<float>(4*s.telemetry.executed_substeps);
        ++s.control_steps;return true;
    }
    return false;
}
class PhysicalWorld {
public:
    PhysicalWorld();
    const WorldState& state() const {return state_;}
    void reset(const physics::BodyState& body={});
    void set_current(physics::V current);
    void set_actuation(Actuation mode);
    void set_paused(bool paused) {state_.paused=paused;}
    bool step(const std::array<float,4>& input) {return step_state(state_,input.data());}
    std::vector<uint8_t> serialize() const;
    void restore(const std::vector<uint8_t>& bytes);
    void save(const std::string& path) const;
    void load(const std::string& path);
private:
    WorldState state_{};
};
// Optional CUDA counterpart: one step per input state, no learner or implicit reset.
// Caller owns the CUDA stream (nullptr is the default stream). Results are returned
// only after that stream completes; no default/nonblocking stream dependency.
void cuda_step(std::vector<WorldState>& states,const std::vector<std::array<float,4>>& inputs,void* stream=nullptr);
}
#endif
