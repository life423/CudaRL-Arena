#ifndef CUDARL_COMMAND_MOTOR_H
#define CUDARL_COMMAND_MOTOR_H
#include "deep_physics.h"
#include "swimmer.h"
#include "core/checkpoint.h"

namespace cudarl::command_motor {
constexpr int kObs=59,kAct=4,kEpisodeSteps=600;
constexpr float body_length=.072f,velocity_scale=.02f;
constexpr uint32_t protocol=2,head_seed=0x4D072601u,training_seed=0x4D072602u;
// A persistent physical command: model-space position, SI terminal velocity,
// and a world-space unit forward vector. There is no food/motivation mode.
struct Request {deep::V position{},velocity{},heading{1,0,0};};
using Command=Request;
struct State {
    deep::BodyState body;
    Request request;
    float previous_action[kAct]{};
    uint32_t seed=1,environment=0,episode=0;
    int steps=0;
};
CUDARL_HD inline deep::Physics physics() {return {};}
CUDARL_HD inline bool finite_number(float x) {return x==x && fabsf(x)<1e30f;}
CUDARL_HD inline bool finite_vector(deep::V v) {return finite_number(v.x)&&finite_number(v.y)&&finite_number(v.z);}
CUDARL_HD inline bool finite(const State& s) {
    if(!finite_vector(s.body.position)||!finite_vector(s.request.position)||!finite_vector(s.request.velocity)||!finite_vector(s.request.heading)) return false;
    const auto q=s.body.orientation;const float q2=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if(!finite_number(q2)||fabsf(q2-1.f)>.01f||fabsf(deep::dot(s.request.heading,s.request.heading)-1.f)>.01f) return false;
    for(float v:s.body.velocity) if(!finite_number(v)) return false;
    for(int j=0;j<kAct;++j) if(!finite_number(s.body.angle[j])||!finite_number(s.previous_action[j])) return false;
    return finite_number(s.body.work)&&finite_number(s.body.travel)&&s.steps>=0;
}
// Body features9..32 deliberately retain the accepted55-input actor's exact
// semantics and scalers. Other columns are a new command contract; they must
// not inherit weights merely because their indices once described food/rays.
CUDARL_HD inline void observe(const State& s,float* out) {
    for(int i=0;i<kObs;++i) out[i]=0;
    const auto q=s.body.orientation;
    const deep::V p=q.inverse((s.request.position-s.body.position)*deep::scale)/body_length;
    const deep::V v=q.inverse({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]});
    const deep::V w=q.inverse({s.body.velocity[3],s.body.velocity[4],s.body.velocity[5]});
    const deep::V up=q.inverse({0,1,0}),h=q.inverse(s.request.heading);
    const deep::V velocity_error=q.inverse(s.request.velocity-deep::V{s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]})/velocity_scale;
    out[0]=p.x;out[1]=p.y;out[2]=p.z;
    out[3]=deep::clamp(1.f-static_cast<float>(s.steps)/kEpisodeSteps,0,1); // finite horizon, also visible to critic
    out[9]=v.x/.04f;out[10]=v.y/.04f;out[11]=v.z/.04f;
    out[12]=w.x/4;out[13]=w.y/4;out[14]=w.z/4;
    out[15]=up.x;out[16]=up.y;out[17]=up.z;
    for(int j=0;j<kAct;++j) {
        out[18+j]=s.body.angle[j]/deep::limit;out[22+j]=s.body.velocity[6+j]/12.f;out[29+j]=s.previous_action[j];
    }
    out[26]=s.body.position.x/4.5f;out[27]=(s.body.position.y+2)/2;out[28]=s.body.position.z/3.5f;
    out[33]=velocity_error.x;out[34]=velocity_error.y;out[35]=velocity_error.z;
    out[55]=1;out[56]=h.x;out[57]=h.y;out[58]=h.z;
}
// Full216-point surface sampling includes the head profile and the distal fin.
// It is a motion cost, not an exposure/contact approximation or a new force.
struct Motion {float mean_square_speed=0,clearance=1e6f;};
CUDARL_HD inline float physical_clearance(const deep::BodyState& body) {
    const auto geometry=deep::shape(body);float gap=1e6f;
    const deep::V normals[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for(int link=0;link<3;++link) {
        const auto& l=geometry.link[link];
        if(link==0) {
            const float planes[6]={l.center.x+4.5f*deep::scale,4.5f*deep::scale-l.center.x,l.center.y+4.f*deep::scale,
                -l.center.y,l.center.z+3.5f*deep::scale,3.5f*deep::scale-l.center.z};
            for(int face=0;face<6;++face) {
                const auto n=l.q.inverse(normals[face]);const float x=n.x*l.length*.5f,y=n.y*.0064f,z=n.z*.008f;
                gap=fminf(gap,planes[face]-sqrtf(x*x+y*y+z*z));
            }
        }
        for(int axial=0;axial<9;++axial) {
            const float f=(axial-4)*.25f;const auto center=l.center+l.q.rotate({f*l.length*.5f,0,0});
            const float planes[6]={center.x+4.5f*deep::scale,4.5f*deep::scale-center.x,center.y+4.f*deep::scale,
                -center.y,center.z+3.5f*deep::scale,3.5f*deep::scale-center.z};
            for(int face=0;face<6;++face) gap=fminf(gap,planes[face]-deep::arch_slice_support(l,link,f,normals[face]));
        }
    }
    return gap;
}
CUDARL_HD inline Motion motion(const deep::BodyState& body) {
    Motion result;result.clearance=physical_clearance(body);const auto geometry=deep::shape(body);
    for(int link=0;link<3;++link) {
        const auto& l=geometry.link[link];const auto center_velocity=deep::multiply(l.linear,body.velocity);
        for(int axial=0;axial<deep::arch_axial_samples;++axial) {
            const float fraction=-1.f+2.f*axial/(deep::arch_axial_samples-1);
            const float profile=link==0?sqrtf(fmaxf(0.f,1.f-fraction*fraction)):1.f;
            const float ry=(link==0?.0064f:(link==2?.006f:.0015f))*profile;
            const float rz=(link==0?.008f:.0015f)*profile;
            for(int radial=0;radial<8;++radial) {
                const float angle=radial*.7853981633974483f;
                const auto r=l.q.rotate({fraction*l.length*.5f,ry*cosf(angle),rz*sinf(angle)});
                const auto p=l.center+r,v=center_velocity+deep::cross(l.omega,r);
                result.mean_square_speed+=deep::dot(v,v)/216.f;
                result.clearance=fminf(result.clearance,fminf(fminf(p.x+4.5f*deep::scale,4.5f*deep::scale-p.x),
                    fminf(fminf(p.y+4.f*deep::scale,-p.y),fminf(p.z+3.5f*deep::scale,3.5f*deep::scale-p.z))));
            }
        }
    }
    return result;
}
struct Cost {
    float position=0,velocity=0,heading=0,whole_body=0,angular=0,joints=0,torque=0;
    CUDARL_HD float total() const {return position+velocity+heading+whole_body+angular+joints+torque;}
};
CUDARL_HD inline Cost tracking_cost(const State& s,const float* action,const Motion& measured) {
    Cost c;const auto e=(s.body.position-s.request.position)*(deep::scale/body_length);
    const auto v=(deep::V{s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]}-s.request.velocity)/velocity_scale;
    const auto w=deep::V{s.body.velocity[3],s.body.velocity[4],s.body.velocity[5]};
    c.position=deep::dot(e,e);c.velocity=.25f*deep::dot(v,v);
    c.heading=.1f*(1.f-deep::clamp(deep::dot(s.body.orientation.rotate({1,0,0}),s.request.heading),-1,1));
    c.whole_body=.05f*measured.mean_square_speed/(velocity_scale*velocity_scale);
    c.angular=.02f*deep::dot(w,w)/(.5f*.5f);
    for(int j=0;j<kAct;++j) {const float a=deep::clamp(action[j],-1,1),rate=s.body.velocity[6+j]/4.f;c.joints+=.02f*rate*rate/4.f;c.torque+=.002f*a*a/4.f;}
    return c;
}
struct Result {float reward=0;bool timeout=false,invalid=false,unsafe=false;};
CUDARL_HD inline Result advance(State& s,const float* action,deep::Physics p=physics()) {
    Result result;
    for(int j=0;j<kAct;++j) if(!finite_number(action[j])) result.invalid=true;
    if(!finite(s)||result.invalid) {result.invalid=true;result.reward=-100.f;return result;}
    deep::advance_physics(s.body,action,p);++s.steps;
    for(int j=0;j<kAct;++j) s.previous_action[j]=deep::clamp(action[j],-1,1);
    result.invalid=!finite(s);result.timeout=s.steps>=kEpisodeSteps;
    if(result.invalid) {result.reward=-100.f;return result;}
    const auto measured=motion(s.body);result.unsafe=measured.clearance<0;
    result.reward=-deep::dt*tracking_cost(s,action,measured).total();
    return result;
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t environment,uint32_t episode=0,swimmer::Body unused={}) {
    (void)unused;State s;s.seed=seed;s.environment=environment;s.episode=episode;
    const uint32_t base=episode*101u;
    const float yaw=6.28318530718f*rng::uniform(seed,environment,base+1);
    const float pitch=(rng::uniform(seed,environment,base+2)*2-1)*.35f;
    s.body.position={(rng::uniform(seed,environment,base+3)*2-1),-2.f+(rng::uniform(seed,environment,base+4)*2-1)*.5f,
        (rng::uniform(seed,environment,base+5)*2-1)*.6f};
    s.body.orientation=deep::axis_rotation({0,1,0},yaw)*deep::axis_rotation({0,0,1},pitch);
    const int kind=static_cast<int>(rng::bits(seed,environment,base+6)%6u);
    const float heading_yaw=yaw+(rng::uniform(seed,environment,base+7)*2-1)*1.57079632679f;
    // This first slice requests level terminal headings. Initial pitch remains
    // disturbed; depth commands remain3D. Quiet pitched equilibrium has not
    // been established for the existing buoyancy/restoring-torque model.
    s.request.heading=deep::axis_rotation({0,1,0},heading_yaw).rotate({1,0,0});
    s.request.position=s.body.position;
    if(kind<2) {
        deep::V direction{rng::uniform(seed,environment,base+9)*2-1,(rng::uniform(seed,environment,base+10)*2-1)*.5f,rng::uniform(seed,environment,base+11)*2-1};
        direction=direction/fmaxf(deep::norm(direction),1e-6f);
        s.request.position+=direction*((.5f+.5f*rng::uniform(seed,environment,base+12))*body_length/deep::scale);
        s.request.position.y=deep::clamp(s.request.position.y,-3.f,-1.f);
    }
    // Moving/braking/recovery starts carry real momentum and joint state. No
    // scripted phase later overrides the actor when it reaches the command.
    if(kind==1||kind==2||kind==5) {
        const float speed=kind==2?.01f+.02f*rng::uniform(seed,environment,base+13):.006f*rng::uniform(seed,environment,base+13);
        const auto travel=s.body.orientation.rotate({speed,0,0});s.body.velocity[0]=travel.x;s.body.velocity[1]=travel.y;s.body.velocity[2]=travel.z;
        for(int i=3;i<deep::dofs;++i) s.body.velocity[i]=(rng::uniform(seed,environment,base+14+i)*2-1)*(i<6?.15f:.5f);
    }
    for(int j=0;j<kAct;++j) s.body.angle[j]=(rng::uniform(seed,environment,base+30+j)*2-1)*.25f;
    if(kind==3) s.request.heading=deep::axis_rotation({0,1,0},yaw).rotate({1,0,0});
    return s;
}
#if CUDARL_WITH_CUDA
class Batch {
public:
    Batch(int n,uint32_t seed,swimmer::Body unused={});
    void step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true);
    // Frozen benchmark only: declared per-case force pulses and an active
    // mask stop shorter completed trials without resetting/moving their body.
    void step_physics(const float* actions,const deep::Physics* per_world,const uint8_t* active,cudaStream_t stream);
    const float* obs() const {return obs_.get();}
    const State* states() const {return states_.get();}
    void set_states(const std::vector<State>& states,cudaStream_t stream);
    std::vector<State> host_states(cudaStream_t stream) const;
    uint64_t invalid_count(cudaStream_t stream) const;
    uint64_t unsafe_count(cudaStream_t stream) const;
    std::vector<State> unsafe_states(cudaStream_t stream) const;
    void save(checkpoint::Writer&) const;
    void load(checkpoint::Reader&);
private:
    int n_;uint32_t seed_;
    device_ptr<State> states_,first_unsafe_;device_ptr<float> obs_;device_ptr<uint32_t> invalid_,safety_;
};
#endif
}
#endif
