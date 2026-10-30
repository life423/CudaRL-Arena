#ifndef CUDARL_MOTOR_H
#define CUDARL_MOTOR_H
#include "motor_types.h"
#include "motor_contact.h"

// Isolated motor experiment v2. Legacy pond/contact paths remain unchanged.
// The body has TWO universal joints, TWO controlled axes each: FOUR actuators.
namespace cudarl::motor {
namespace mc=motor_contact;
constexpr uint32_t kProtocol=2;
constexpr float kCurrent=.005f;
struct State {
    TaskState task{};
    deep::V water_velocity{};
    int lesson=0;
    uint64_t integrated_substeps=0;
    int executed_substeps=0;
    float held_target[4]{},actual_torques[40]{},unclamped_torques[40]{};
    float mean_squared_torque=0;
    int first_unsafe_substep=-1,unsafe_substeps=0;
    TaskState first_unsafe_task{};
    float substep_min_floor_gap=1e6f,substep_max_floor_gap=-1e6f,substep_min_other_gap=1e6f;
    int floor_contact_samples=0;
    float substep_max_floor_force=0,substep_max_head_speed=0,substep_max_com_speed=0;
    float substep_max_omega=0,substep_max_goal_error=0,substep_max_heading_error=0;
};
struct BodyMotion {float head_speed=0,com_speed=0,omega=0,goal_error=0,heading_error=0;};
CUDARL_HD inline BodyMotion body_motion(const State& state) {
    BodyMotion out;const auto& b=state.task.body;const auto shape=deep::shape(b);deep::V com{};float mass=0;
    for(int link=0;link<3;++link) {com+=deep::multiply(shape.link[link].linear,b.velocity)*shape.link[link].mass;mass+=shape.link[link].mass;}
    out.head_speed=deep::norm({b.velocity[0],b.velocity[1],b.velocity[2]});out.com_speed=deep::norm(com/fmaxf(mass,1e-12f));
    out.omega=deep::norm({b.velocity[3],b.velocity[4],b.velocity[5]});
    out.goal_error=deep::norm((b.position-state.task.request.position)*deep::scale);
    out.heading_error=acosf(deep::clamp(deep::dot(b.orientation.rotate({1,0,0}),state.task.request.heading),-1,1));return out;
}
CUDARL_HD inline bool finite(const State& s) {
    if(!motor::finite(s.task)||!motor::finite_vector(s.water_velocity)||s.lesson<0||s.lesson>6||s.executed_substeps<0||s.executed_substeps>10||s.first_unsafe_substep< -1||s.first_unsafe_substep>10||s.unsafe_substeps<0||s.unsafe_substeps>11||s.floor_contact_samples<0||s.floor_contact_samples>11) return false;
    for(int j=0;j<4;++j) if(!motor::finite_number(s.held_target[j])||fabsf(s.held_target[j])>deep::limit) return false;
    for(int j=0;j<40;++j) if(!motor::finite_number(s.actual_torques[j])||fabsf(s.actual_torques[j])>1||!motor::finite_number(s.unclamped_torques[j])) return false;
    const float values[]={s.mean_squared_torque,s.substep_min_floor_gap,s.substep_max_floor_gap,s.substep_min_other_gap,s.substep_max_floor_force,s.substep_max_head_speed,s.substep_max_com_speed,s.substep_max_omega,s.substep_max_goal_error,s.substep_max_heading_error};
    for(float value:values) if(!motor::finite_number(value)) return false;
    return s.mean_squared_torque>=0&&s.mean_squared_torque<=1.0001f&&(s.first_unsafe_substep<0||motor::finite(s.first_unsafe_task));
}
CUDARL_HD inline void observe(const State& s,float* out) {
    motor::observe(s.task,out);const auto flow=s.task.body.orientation.inverse(s.water_velocity)/motor::velocity_scale;
    out[4]=flow.x;out[5]=flow.y;out[6]=flow.z;
    const auto contact=mc::inspect(s.task.body);out[7]=contact.floor_gap/motor::body_length;out[8]=contact.floor_contact?1.f:0.f;
    // 36..53: future local obstacle geometry, deliberately unavailable here.
    out[54]=0; // no obstacle sensor is implemented in this experiment
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode=0,swimmer::Body body={}) {
    (void)body;State result;result.task.seed=seed;result.task.environment=environment;result.task.episode=episode;auto& s=result.task;
    const uint32_t base=episode*101u;result.lesson=static_cast<int>((environment+episode)%7u);
    const float yaw=6.28318530718f*rng::uniform(seed,environment,base+71);
    s.body.orientation=deep::axis_rotation({0,1,0},yaw)*deep::axis_rotation({0,0,1},(rng::uniform(seed,environment,base+72)*2-1)*.15f);
    s.body.position={(rng::uniform(seed,environment,base+73)*2-1)*.5f,-2.f,(rng::uniform(seed,environment,base+74)*2-1)*.4f};
    for(float& value:s.body.velocity) value=0;
    for(int j=0;j<4;++j) s.body.angle[j]=(rng::uniform(seed,environment,base+75+j)*2-1)*.15f;
    s.request.position=s.body.position;s.request.velocity={};s.request.heading=deep::axis_rotation({0,1,0},yaw).rotate({1,0,0});
    if(result.lesson==0||result.lesson==3) {
        deep::V direction{rng::uniform(seed,environment,base+80)*2-1,(rng::uniform(seed,environment,base+81)*2-1)*.5f,rng::uniform(seed,environment,base+82)*2-1};
        direction=direction/fmaxf(deep::norm(direction),1e-6f);
        s.request.position+=direction*((result.lesson==0?.75f:.5f)*motor::body_length/deep::scale);
    }
    if(result.lesson==1) s.request.heading=deep::axis_rotation({0,1,0},yaw+1.57079632679f).rotate({1,0,0});
    if(result.lesson==2||result.lesson==3) {
        const float speed=result.lesson==2?.01f+.02f*rng::uniform(seed,environment,base+83):.005f;
        const auto v=s.body.orientation.rotate({speed,0,0});s.body.velocity[0]=v.x;s.body.velocity[1]=v.y;s.body.velocity[2]=v.z;
        for(int j=3;j<10;++j) s.body.velocity[j]=(rng::uniform(seed,environment,base+84+j)*2-1)*(j<6?.1f:.25f);
    }
    if(result.lesson==5) {
        const float direction=6.28318530718f*rng::uniform(seed,environment,base+95);
        result.water_velocity={kCurrent*cosf(direction),0,kCurrent*sinf(direction)};
    }
    if(result.lesson==6) {
        s.body.orientation=deep::axis_rotation({0,1,0},yaw);
        for(float& q:s.body.angle) q=0;
        const float gap=mc::inspect(s.body).floor_gap;s.body.position.y-=gap/deep::scale;
        s.body.velocity[1]=-.001f;s.request.position=s.body.position;
    }
    return result;
}
CUDARL_HD inline void sample_telemetry(State& s,int substep,Result& result) {
    const auto contact=mc::inspect(s.task.body);const auto motion=body_motion(s);
    s.substep_min_floor_gap=fminf(s.substep_min_floor_gap,contact.floor_gap);s.substep_max_floor_gap=fmaxf(s.substep_max_floor_gap,contact.floor_gap);
    s.substep_min_other_gap=fminf(s.substep_min_other_gap,contact.min_other_gap);
    if(contact.floor_contact) ++s.floor_contact_samples;
    s.substep_max_head_speed=fmaxf(s.substep_max_head_speed,motion.head_speed);s.substep_max_com_speed=fmaxf(s.substep_max_com_speed,motion.com_speed);
    s.substep_max_omega=fmaxf(s.substep_max_omega,motion.omega);s.substep_max_goal_error=fmaxf(s.substep_max_goal_error,motion.goal_error);s.substep_max_heading_error=fmaxf(s.substep_max_heading_error,motion.heading_error);
    if(contact.invalid) result.invalid=true;
    if(contact.excessive||contact.escape) {result.unsafe=true;++s.unsafe_substeps;if(s.first_unsafe_substep<0) {s.first_unsafe_substep=substep;s.first_unsafe_task=s.task;}}
}
CUDARL_HD inline Result advance(State& s,const float* latent,Interface mode,const Gains& gains,deep::Physics physical={}) {
    Result result;if(!finite(s)||!motor::valid_gains(gains,mode)||physical.pond||physical.habitat.arch||physical.basin.layout!=0) {result.invalid=true;result.reward=-100;return result;}
    for(int j=0;j<4;++j) if(!motor::finite_number(latent[j])) {result.invalid=true;result.reward=-100;return result;}
    physical.water_velocity=s.water_velocity;
    s.first_unsafe_substep=-1;s.unsafe_substeps=0;s.floor_contact_samples=0;s.mean_squared_torque=0;s.executed_substeps=0;
    for(int j=0;j<40;++j) {s.actual_torques[j]=0;s.unclamped_torques[j]=0;}
    s.substep_min_floor_gap=s.substep_min_other_gap=1e6f;s.substep_max_floor_gap=-1e6f;
    s.substep_max_floor_force=s.substep_max_head_speed=s.substep_max_com_speed=s.substep_max_omega=s.substep_max_goal_error=s.substep_max_heading_error=0;
    for(int j=0;j<4;++j) s.held_target[j]=mode==Interface::JointTarget?deep::limit*deep::clamp(latent[j],-1,1):0;
    sample_telemetry(s,0,result);
    if(result.invalid||result.unsafe) {result.reward=-100;return result;}
    for(int step=0;step<10;++step) {
        float actual[4];for(int j=0;j<4;++j) {
            const float requested=mode==Interface::Torque?deep::torque*latent[j]:gains.kp[j]*(s.held_target[j]-s.task.body.angle[j])-gains.kd[j]*s.task.body.velocity[6+j];
            actual[j]=mode==Interface::Torque?deep::clamp(latent[j],-1,1):deep::clamp(requested/deep::torque,-1,1);
            s.unclamped_torques[step*4+j]=requested;s.actual_torques[step*4+j]=actual[j];s.mean_squared_torque+=actual[j]*actual[j]/40.f;
        }
        mc::Telemetry contact;mc::substep(s.task.body,actual,.005f,physical,&contact);
        ++s.executed_substeps;++s.integrated_substeps;
        s.substep_max_floor_force=fmaxf(s.substep_max_floor_force,contact.floor_force);
        for(int j=0;j<4;++j) s.task.previous_action[j]=actual[j];
        if(!motor::finite(s.task)) {result.invalid=true;result.reward=-100;return result;}
        sample_telemetry(s,step+1,result);
        if(result.invalid) {result.reward=-100;return result;}
        // Preserve the first failed control; no rescue/reset/remainder movement.
        if(result.unsafe) break;
    }
    ++s.task.steps;result.timeout=s.task.steps>=kEpisodeSteps;
    const auto& body=s.task.body;const auto e=(body.position-s.task.request.position)*(deep::scale/motor::body_length);
    const auto v=(deep::V{body.velocity[0],body.velocity[1],body.velocity[2]}-s.task.request.velocity)/motor::velocity_scale;
    const auto omega=deep::V{body.velocity[3],body.velocity[4],body.velocity[5]};
    const float heading=deep::clamp(deep::dot(body.orientation.rotate({1,0,0}),s.task.request.heading),-1,1);
    const float cost=deep::dot(e,e)+.25f*deep::dot(v,v)+.1f*(1-heading)+.02f*deep::dot(omega,omega)/.25f+.002f*s.mean_squared_torque;
    result.reward=-deep::dt*cost;result.invalid=!finite(s)||!motor::finite_number(result.reward);if(result.invalid) result.reward=-100;return result;
}
inline std::vector<float> protocol() {
    return {2,59,4,2,2,600,.05f,.005f,10,.04f,.072f,.02f,1e-5f,1,.25f,.1f,.02f,.002f,
        .005f,7,4,5,6,7,8,29,32,54,0,.0005f,.000005f,.002f,.0072f,.174532925f,.002f,.05f,10,
        .99f,.95f,.2f,.002f,.5f,64,4,8,.001f,.9f,.999f,.00001f,.5f,-.5f,512,64};
}
inline uint64_t physics_fingerprint() {checkpoint::Writer out;out.value(kProtocol);out.value(mc::physics_fingerprint());out.array(protocol());return motor::fingerprint(out.data);}
#if CUDARL_WITH_CUDA
class Batch {
    int n_;uint32_t seed_;Interface interface_=Interface::Torque;Gains gains_{};
    device_ptr<State> states_,first_unsafe_;device_ptr<float> observations_;device_ptr<uint32_t> invalid_,safety_;
public:
    Batch(int n,uint32_t seed,swimmer::Body body={});
    void configure(Interface,const Gains&,cudaStream_t);
    Interface mode() const {return interface_;} Gains gains() const {return gains_;}
    const float* obs() const {return observations_.get();} const State* states() const {return states_.get();}
    void step(const float*,swimmer::Outputs,cudaStream_t,bool autoreset=true);
    void step_physics(const float*,const deep::Physics*,const uint8_t*,cudaStream_t);
    std::vector<State> host_states(cudaStream_t) const;void set_states(const std::vector<State>&,cudaStream_t);
    uint64_t invalid_count(cudaStream_t) const;uint64_t unsafe_count(cudaStream_t) const;std::vector<State> unsafe_states(cudaStream_t) const;
    void save(checkpoint::Writer&) const;void load(checkpoint::Reader&);
};
#endif
}
#endif
