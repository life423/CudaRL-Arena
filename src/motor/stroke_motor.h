#ifndef CUDARL_STROKE_MOTOR_H
#define CUDARL_STROKE_MOTOR_H
#include "motor/forward_task.h"
#include "motor/movement_model.h"
namespace cudarl::stroke_motor {
namespace p=physics;namespace sim=simulation;namespace base=motor;namespace model=movement;
using ForwardConfig=base::ForwardConfig;using ForwardTask=base::ForwardTask;using Outcome=base::Outcome;using Transition=base::Transition;
constexpr int kTaskVersion=3,kObservations=42,kActions=2;
struct Action {float latent[2]{},command[4]{};};
struct World {
    sim::WorldState physical{};
    // First lesson controls amplitude[rad], frequency[Hz] only.
    // Straight gait: zero steering, fixed quarter-cycle lag. No steering learning.
    float parameter[2]{0,1},rate[2]{},phase=0;
    uint64_t attempted_substeps=0;
    double signed_work=0,positive_work=0,fluid_loss=0,numerical_loss=0,external_work=0,last_energy_error=0;
    int failure_code=0;
};
CUDARL_HD inline bool in_domain(const World& w,const ForwardConfig& c){
    if(!base::in_domain(w.physical,c)||model::clearance(w.physical.body)<c.clearance_m||!p::finite_number(w.phase))return false;
    for(int j=0;j<2;++j)if(!p::finite_number(w.parameter[j])||!p::finite_number(w.rate[j]))return false;return true;
}
CUDARL_HD inline bool begin_forward(const World& w,const ForwardConfig& c,ForwardTask& t){return in_domain(w,c)&&base::begin_forward(w.physical,c,t);}
CUDARL_HD inline bool action_from_latent(const float* z,Action& a){if(!z)return false;Action next;for(int j=0;j<2;++j){if(!p::finite_number(z[j]))return false;next.latent[j]=z[j];next.command[j]=tanhf(z[j]);}a=next;return true;}
CUDARL_HD inline bool observe(const ForwardTask& t,const World& w,const ForwardConfig& c,float* out){
    if(!in_domain(w,c)||!base::observe(t,w.physical,c,out))return false;
    out[36]=sinf(w.phase);out[37]=cosf(w.phase);
    const float scale[2]={.7f,1.5f};
    for(int j=0;j<2;++j){out[38+j]=w.parameter[j]/scale[j];out[40+j]=w.rate[j]/(scale[j]*2*model::pi);}
    return true;
}
CUDARL_HD inline base::Transition record_control(ForwardTask& t,const World& w,const ForwardConfig& c){
    if(!in_domain(w,c)){auto invalid=w.physical;invalid.invalid=true;return base::record_control(t,invalid,c);}return base::record_control(t,w.physical,c);
}
// A smooth internal reference generator, not a body force. Its phase/parameters
// and derivative state are observed and saved. Rate/acceleration references
// drive only the four torque-limited joints in movement model6.
CUDARL_HD inline bool step_state(World& w,const float* command){
    ForwardConfig config;if(w.physical.paused||!in_domain(w,config))return false;
    for(int j=0;j<2;++j)if(!p::finite_number(command[j])||fabsf(command[j])>1)return false;
    const float desired[2]={.35f*(command[0]+1),.25f+.625f*(command[1]+1)};
    auto& state=w.physical;state.telemetry={};state.mean_squared_torque=0;
    for(int j=0;j<4;++j)state.raw_action[j]=j<2?command[j]:0;
    for(int k=0;k<10;++k){const float h=.005f,omega=2*model::pi;float acceleration[2];
        for(int j=0;j<2;++j){acceleration[j]=omega*omega*(desired[j]-w.parameter[j])-2*omega*w.rate[j];w.rate[j]+=h*acceleration[j];w.parameter[j]+=h*w.rate[j];}
        model::Reference reference;
        for(int j=0;j<4;j+=2){const float phase=w.phase-(j==2?model::pi/2:0),phase_rate=2*model::pi*w.parameter[1],phase_acceleration=2*model::pi*w.rate[1];
            const float s=sinf(phase),c=cosf(phase),A=w.parameter[0];reference.angle[j]=A*s;reference.rate[j]=w.rate[0]*s+A*c*phase_rate;
            reference.acceleration[j]=acceleration[0]*s+2*w.rate[0]*c*phase_rate+A*c*phase_acceleration-A*s*phase_rate*phase_rate;}
        model::Telemetry telemetry;++w.attempted_substeps;
        if(!model::substep(state.body,reference,h,state.fluid,telemetry)){w.failure_code=telemetry.failure_code;state.invalid=true;return false;}
        w.signed_work+=telemetry.signed_work;w.positive_work+=telemetry.positive_work;w.fluid_loss+=telemetry.fluid_loss;w.numerical_loss+=telemetry.numerical_loss;w.external_work+=telemetry.external_work;
        w.last_energy_error=model::kinetic(state.body)-w.signed_work+w.fluid_loss+w.numerical_loss-w.external_work;
        if(fabs(w.last_energy_error)>.02*(w.positive_work+w.fluid_loss)+2e-10){w.failure_code=9;state.invalid=true;return false;}
        w.phase+=h*2*model::pi*w.parameter[1];if(w.phase>2*model::pi)w.phase-=2*model::pi;
        ++state.substeps;++state.telemetry.executed_substeps;state.time_seconds=static_cast<double>(state.substeps)*h;
        state.telemetry.minimum_floor_gap=fminf(state.telemetry.minimum_floor_gap,telemetry.minimum_gap);state.telemetry.minimum_other_gap=fminf(state.telemetry.minimum_other_gap,telemetry.minimum_gap);
        for(int j=0;j<4;++j){state.held_targets[j]=reference.angle[j];const float normalized=telemetry.torque[j]/p::torque;
            state.last_actual_torques[j]=normalized;state.actual_torques[k*4+j]=normalized;state.requested_torques[k*4+j]=telemetry.requested[j];state.mean_squared_torque+=normalized*normalized;}
    }
    state.mean_squared_torque/=40;++state.control_steps;return in_domain(w,config);
}
}
#endif
