#ifndef CUDARL_ENVS_SWIMMER_H
#define CUDARL_ENVS_SWIMMER_H

#include <cmath>
#include <cstdint>
#include "core/hd.h"
#include "core/rng.h"
#include "cuda/cuda_memory.h"
#include "inertial.h"

// Shared planar three-link swimmer. Historical bodies 0-7 retain the original
// overdamped model; body 8 uses torque-controlled rigid-body inertia and passive
// viscous/quadratic water forces. Neither is CFD or a land-contact simulator.
// Actions cannot apply external thrust or steering.
namespace cudarl::swimmer {
constexpr int kObs = 9, kAct = 2, kMaxSteps = 800;
constexpr float kDt = 0.05f, kLimit = 1.2f, kMotorSpeed = 4.f;
constexpr float kFoodRadius = 0.18f, kPi = 3.141592653589793f;
struct Body { float head=0.5f, proximal=0.65f, distal=0.65f, fin_normal=2.f, motor_speed=4.f; bool tapered=false,eel=false,inertial=false; };
CUDARL_HD inline Body body_preset(int kind) {
    if(kind==1) return {0.25f,0.6f,0.95f,2.f,4.f};
    if(kind==2) return {0.25f,0.6f,0.95f,4.f,4.f};
    if(kind==3) return {0.5f,0.65f,0.65f,2.f,8.f};
    if(kind==4) return {0.25f,0.6f,0.95f,4.f,8.f};
    if(kind==5) return {0.5f,0.65f,0.65f,2.f,16.f};
    if(kind==6) return {0.5f,0.65f,0.65f,4.f,16.f,true};
    // Equal-length, slender articulated segments; no broad tail membrane.
    // Same total length, motor limit and two independent joints as the tadpole.
    if(kind==7) return {0.6f,0.6f,0.6f,2.f,16.f,false,true};
    if(kind==8) return {0.5f,0.65f,0.65f,4.f,12.f,true,false,true};
    return {};
}
struct State {
    float x=0, y=0, heading=0, angle[2]{}, rate[2]{}, velocity[3]{};
    float food_x=0, food_y=0;
    int steps=0;
    Body body;
    float work=0,travel=0,strokes=0;
    float last_action[2]{};
};
CUDARL_HD inline float clamp(float x,float lo,float hi) { return fminf(hi,fmaxf(lo,x)); }
CUDARL_HD inline float head_width(float t) { return 0.06f+0.17f*sinf(kPi*t)*(0.6f+0.4f*t); }
CUDARL_HD inline float distance(const State& s) {
    const float x=s.food_x-s.x,y=s.food_y-s.y; return sqrtf(x*x+y*y);
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t env,uint32_t episode=0,Body body={}) {
    State s;
    s.body=body;
    const uint32_t i=episode*5;
    s.heading=(rng::uniform(seed,env,i)*2.f-1.f)*kPi;
    s.angle[0]=(rng::uniform(seed,env,i+1)*2.f-1.f)*0.25f;
    s.angle[1]=(rng::uniform(seed,env,i+2)*2.f-1.f)*0.25f;
    const float bearing=rng::uniform(seed,env,i+3)*2.f*kPi;
    const float radius=0.75f+0.75f*rng::uniform(seed,env,i+4);
    s.food_x=cosf(bearing)*radius; s.food_y=sinf(bearing)*radius;
    return s;
}
CUDARL_HD inline void observe(const State& s,float* o) {
    const float c=cosf(s.heading),n=sinf(s.heading),x=s.food_x-s.x,y=s.food_y-s.y;
    o[0]=(c*x+n*y)/2.f; o[1]=(-n*x+c*y)/2.f;
    o[2]=s.angle[0]/kLimit; o[3]=s.angle[1]/kLimit;
    o[4]=s.rate[0]/s.body.motor_speed; o[5]=s.rate[1]/s.body.motor_speed;
    o[6]=s.velocity[0]; o[7]=s.velocity[1]; o[8]=s.velocity[2];
}

// Add B^T D B and -B^T D v_joint for a quadrature point at p. Tangential
// resistance=1, normal resistance=2; weights integrate the link's length.
CUDARL_HD inline void resistance(float A[3][3],float b[3],float px,float py,
                                  float tx,float ty,float jx,float jy,float weight,float normal,float& joint_work) {
    const float d00=normal+(1.f-normal)*tx*tx,d01=(1.f-normal)*tx*ty,d11=normal+(1.f-normal)*ty*ty;
    const float bx[3]={1,0,-py},by[3]={0,1,px};
    joint_work+=weight*(jx*(d00*jx+d01*jy)+jy*(d01*jx+d11*jy));
    for(int i=0;i<3;++i) {
        b[i]-=weight*(bx[i]*(d00*jx+d01*jy)+by[i]*(d01*jx+d11*jy));
        for(int j=0;j<3;++j)
            A[i][j]+=weight*(bx[i]*(d00*bx[j]+d01*by[j])+by[i]*(d01*bx[j]+d11*by[j]));
    }
}
CUDARL_HD inline void integrate(State& s,const float* action,float dt=kDt,bool count_step=true) {
    float A[3][3]{},b[3]{},joint_work=0;
    for(int j=0;j<2;++j) {
        const float next=clamp(s.angle[j]+dt*s.body.motor_speed*clamp(action[j],-1,1),-kLimit,kLimit);
        s.rate[j]=(next-s.angle[j])/dt;
    }
    // Midpoint shape makes reciprocal strokes approximately reversible rather
    // than introducing a direction bias from evaluating only the old shape.
    const float a=s.angle[0]+0.5f*dt*s.rate[0];
    const float z=s.angle[1]+0.5f*dt*s.rate[1];
    const float c1=cosf(a),n1=sinf(a),c2=cosf(a+z),n2=sinf(a+z);
    const float body=s.body.head,tail=s.body.proximal,distal=s.body.distal;
    for(int q=0;q<3;++q) {
        const float t=(static_cast<float>(q)+0.5f)/3.f;
        const float width_factor=s.body.eel?0.5f:(s.body.tapered?head_width(t)/0.15f:1.f);
        resistance(A,b,(t-0.5f)*body,0,1,0,0,0,body*width_factor/3.f,2.f,joint_work);
        float px=-body*0.5f-tail*t*c1,py=-tail*t*n1;
        float jx=tail*t*n1*s.rate[0],jy=-tail*t*c1*s.rate[0];
        const float tail_width=s.body.eel?0.5f:1.f;
        resistance(A,b,px,py,c1,n1,jx,jy,tail*tail_width/3.f,2.f,joint_work);
        px=-body*0.5f-tail*c1-distal*t*c2; py=-tail*n1-distal*t*n2;
        jx=tail*n1*s.rate[0]+distal*t*n2*(s.rate[0]+s.rate[1]);
        jy=-tail*c1*s.rate[0]-distal*t*c2*(s.rate[0]+s.rate[1]);
        resistance(A,b,px,py,c2,n2,jx,jy,distal*tail_width/3.f,s.body.fin_normal,joint_work);
    }
    // SPD solve by Cholesky; all three links have positive drag.
    float L[3][3]{},v[3]{};
    for(int i=0;i<3;++i) for(int j=0;j<=i;++j) {
        float value=A[i][j]; for(int k=0;k<j;++k) value-=L[i][k]*L[j][k];
        L[i][j]=i==j ? sqrtf(fmaxf(value,1e-10f)) : value/L[j][j];
    }
    for(int i=0;i<3;++i) {
        float value=b[i]; for(int j=0;j<i;++j) value-=L[i][j]*v[j];
        v[i]=value/L[i][i];
    }
    for(int i=2;i>=0;--i) {
        float value=v[i]; for(int j=i+1;j<3;++j) value-=L[j][i]*s.velocity[j];
        s.velocity[i]=value/L[i][i];
    }
    float body_work=0;for(int i=0;i<3;++i) body_work+=s.velocity[i]*b[i];
    s.work+=dt*fmaxf(0.f,joint_work-body_work);
    s.travel+=dt*sqrtf(s.velocity[0]*s.velocity[0]+s.velocity[1]*s.velocity[1]);
    s.strokes+=dt*fabsf(s.rate[1])/(4.f*kLimit);
    const float mid=s.heading+0.5f*dt*s.velocity[2],c=cosf(mid),n=sinf(mid);
    s.x+=dt*(c*s.velocity[0]-n*s.velocity[1]);
    s.y+=dt*(n*s.velocity[0]+c*s.velocity[1]);
    s.heading+=dt*s.velocity[2];
    s.heading=atan2f(sinf(s.heading),cosf(s.heading));
    for(int j=0;j<2;++j) s.angle[j]+=dt*s.rate[j];
    if(count_step) ++s.steps;
}
struct Result { float reward; bool success,timeout; };
CUDARL_HD inline void advance_physics(State& s,const float* action) {
    if(s.body.inertial) {
        for(int i=0;i<10;++i) inertial::step(s,action,kDt/10.f,kLimit);
    } else {
        const int substeps=static_cast<int>(ceilf(s.body.motor_speed/4.f));
        for(int i=0;i<substeps;++i) integrate(s,action,kDt/static_cast<float>(substeps),false);
    }
    ++s.steps;
}
CUDARL_HD inline Result advance(State& s,const float* action) {
    const float before=distance(s);
    advance_physics(s,action);
    const float after=distance(s);
    const bool success=after<=kFoodRadius;
    const bool timeout=!success && s.steps>=kMaxSteps;
    const float effort=(s.rate[0]*s.rate[0]+s.rate[1]*s.rate[1])/(s.body.motor_speed*s.body.motor_speed);
    float reward=5.f*(before-after)-0.001f-0.0005f*effort+(success?3.f:0.f);
    if(s.body.inertial) {
        float change=0;
        for(int j=0;j<2;++j) {
            const float command=clamp(action[j],-1,1),difference=command-s.last_action[j];
            change+=difference*difference;s.last_action[j]=command;
        }
        reward-=0.01f*change+0.004f;
    }
    return {reward,success,timeout};
}
struct Outputs { float *reward,*done,*truncated,*final_obs; };
class Batch {
public:
    Batch(int n,uint32_t seed,Body body={});
    void step(const float* actions,Outputs out,cudaStream_t stream,bool autoreset=true);
    const float* obs() const { return obs_.get(); }
    const State* states() const { return states_.get(); }
private:
    int n_; uint32_t seed_; Body body_;
    device_ptr<State> states_; device_ptr<uint32_t> episodes_; device_ptr<float> obs_;
};
} // namespace cudarl::swimmer
#endif
