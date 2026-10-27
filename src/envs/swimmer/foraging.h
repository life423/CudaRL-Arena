#ifndef CUDARL_ENVS_FORAGING_H
#define CUDARL_ENVS_FORAGING_H
#include "swimmer.h"

namespace cudarl::foraging {
constexpr int kFoods=3,kObs=13,kEpisodeSteps=1200;
// Inset from the rendered water (4.6 x 3.5) to keep food off the bank.
constexpr float kFoodRadiusX=3.9f,kFoodRadiusY=2.8f,kFoodSeparation=2.2f,kRespawnDistance=2.6f;
struct State {
    swimmer::State body;
    float food_x[kFoods]{99,99,99},food_y[kFoods]{99,99,99};
    uint32_t seed=1,environment=0,episode=0,placements[kFoods]{};
    int eaten=0,since_food_steps=0,last_eaten=-1;
};
CUDARL_HD inline bool inside(float x,float y) {
    return x*x/(kFoodRadiusX*kFoodRadiusX)+y*y/(kFoodRadiusY*kFoodRadiusY)<1.f;
}
CUDARL_HD inline int region(float x,float y) {return (x>=0?1:0)+(y>=0?2:0);}
CUDARL_HD inline bool clear_of_body(const swimmer::State& s,float x,float y) {
    float px=s.x+0.25f*cosf(s.heading),py=s.y+0.25f*sinf(s.heading);
    float qx=s.x-0.25f*cosf(s.heading),qy=s.y-0.25f*sinf(s.heading);
    for(int segment=0;segment<3;++segment) {
        const float dx=qx-px,dy=qy-py;
        const float t=swimmer::clamp(((x-px)*dx+(y-py)*dy)/(dx*dx+dy*dy),0,1);
        const float ex=x-px-t*dx,ey=y-py-t*dy;
        if(ex*ex+ey*ey<0.32f*0.32f) return false;
        px=qx;py=qy;
        const float angle=s.heading+s.angle[0]+(segment==1?s.angle[1]:0.f);
        const float length=segment==0?s.body.proximal:s.body.distal;
        qx=px-length*cosf(angle);qy=py-length*sinf(angle);
    }
    return true;
}
CUDARL_HD inline bool valid_food(const State& s,int slot,float x,float y) {
    if(!inside(x,y) || !clear_of_body(s.body,x,y)) return false;
    for(int j=0;j<kFoods;++j) if(j!=slot) {
        const float dx=x-s.food_x[j],dy=y-s.food_y[j];
        if(dx*dx+dy*dy<kFoodSeparation*kFoodSeparation) return false;
    }
    return true;
}
CUDARL_HD inline bool place_food(State& s,int slot) {
    const uint32_t stream=s.environment*7u+static_cast<uint32_t>(slot)+71u;
    const uint32_t base=rng::bits(s.seed,s.episode,s.placements[slot]++);
    const float old_x=s.food_x[slot],old_y=s.food_y[slot];
    const bool replacing=inside(old_x,old_y);
    bool blocked[4]{};
    for(int j=0;j<kFoods;++j) if(inside(s.food_x[j],s.food_y[j]))
        blocked[region(s.food_x[j],s.food_y[j])]=true;
    for(uint32_t candidate=0;candidate<1024;++candidate) {
        const float bearing=rng::uniform(s.seed,stream,base+candidate*2u)*2.f*swimmer::kPi;
        const float draw=rng::uniform(s.seed,stream,base+candidate*2u+1u);
        // Use a different, currently empty quarter of the pond. The animal still
        // chooses among three stable foods; spawning cannot select its target.
        const float radius=sqrtf(draw);
        const float x=kFoodRadiusX*radius*cosf(bearing),y=kFoodRadiusY*radius*sinf(bearing);
        const float dx=x-old_x,dy=y-old_y;
        if(candidate<512 && blocked[region(x,y)]) continue;
        if(replacing && (region(x,y)==region(old_x,old_y) || dx*dx+dy*dy<kRespawnDistance*kRespawnDistance)) continue;
        if(valid_food(s,slot,x,y)) {s.food_x[slot]=x;s.food_y[slot]=y;return true;}
    }
    for(int row=0;row<19;++row) for(int column=0;column<25;++column) {
        const float x=kFoodRadiusX*(-0.99f+1.98f*column/24.f),y=kFoodRadiusY*(-0.99f+1.98f*row/18.f);
        const float dx=x-old_x,dy=y-old_y;
        if(replacing && (region(x,y)==region(old_x,old_y) || dx*dx+dy*dy<kRespawnDistance*kRespawnDistance)) continue;
        if(valid_food(s,slot,x,y)) {s.food_x[slot]=x;s.food_y[slot]=y;return true;}
    }
    return false;
}
CUDARL_HD inline State initial(uint32_t seed,uint32_t env,uint32_t episode=0,swimmer::Body body=swimmer::body_preset(8)) {
    State s;s.seed=seed;s.environment=env;s.episode=episode;
    s.body=swimmer::initial(seed,env,episode,body);
    // Learn recovery as well as forward swimming. Folded but valid resting
    // poses occur in training, rather than only almost-straight starts.
    for(int j=0;j<2;++j)
        s.body.angle[j]=(rng::uniform(seed,env,episode*7u+static_cast<uint32_t>(j)+9001u)*2.f-1.f)*swimmer::kLimit;
    for(int i=0;i<kFoods;++i) place_food(s,i);
    return s;
}
CUDARL_HD inline void observe(const State& s,float* o) {
    const float c=cosf(s.body.heading),n=sinf(s.body.heading);
    // Stable slots, never sorted by proximity: all three positions go into the
    // neural network. There is no selected target or target-following controller.
    for(int i=0;i<kFoods;++i) {
        const float x=s.food_x[i]-s.body.x,y=s.food_y[i]-s.body.y;
        o[i*2]=(c*x+n*y)/2.f;o[i*2+1]=(-n*x+c*y)/2.f;
    }
    o[6]=s.body.angle[0]/swimmer::kLimit;o[7]=s.body.angle[1]/swimmer::kLimit;
    o[8]=s.body.rate[0]/s.body.body.motor_speed;o[9]=s.body.rate[1]/s.body.body.motor_speed;
    o[10]=s.body.velocity[0];o[11]=s.body.velocity[1];o[12]=s.body.velocity[2];
}
CUDARL_HD inline float potential(const State& s) {
    // A smooth potential of ALL food distances supplies early learning signal.
    // It affects reward only; it cannot turn or propel the animal.
    constexpr float sharpness=4.f;
    float distances[kFoods],minimum=1e20f;
    for(int i=0;i<kFoods;++i) {
        const float dx=s.food_x[i]-s.body.x,dy=s.food_y[i]-s.body.y;
        distances[i]=sqrtf(dx*dx+dy*dy);minimum=fminf(minimum,distances[i]);
    }
    float total=0;
    for(int i=0;i<kFoods;++i) total+=expf(-sharpness*(distances[i]-minimum));
    return -minimum+logf(total)/sharpness;
}
struct Result {float reward;int eaten_slot;bool timeout;};
CUDARL_HD inline Result advance(State& s,const float* action) {
    const float before=potential(s);
    swimmer::advance_physics(s.body,action);
    ++s.since_food_steps;s.last_eaten=-1;
    // Compute progress before respawning, so a random replacement cannot award
    // progress. No reward is earned for simply idling far from food.
    float reward=3.f*(potential(s)-before)-0.005f;
    const float effort=(s.body.rate[0]*s.body.rate[0]+s.body.rate[1]*s.body.rate[1])/
        (s.body.body.motor_speed*s.body.body.motor_speed);
    reward-=0.0005f*effort;
    for(int j=0;j<2;++j) {
        const float command=swimmer::clamp(action[j],-1,1),difference=command-s.body.last_action[j];
        reward-=0.01f*difference*difference;s.body.last_action[j]=command;
        const float near_stop=swimmer::clamp((fabsf(s.body.angle[j])-0.8f*swimmer::kLimit)/(0.2f*swimmer::kLimit),0,1);
        const float outward=command*(s.body.angle[j]>=0?1.f:-1.f);
        reward-=0.05f*near_stop*fmaxf(0.f,outward);
    }
    for(int i=0;i<kFoods;++i) {
        const float dx=s.food_x[i]-s.body.x,dy=s.food_y[i]-s.body.y;
        if(dx*dx+dy*dy<=swimmer::kFoodRadius*swimmer::kFoodRadius) {
            reward+=3.f;++s.eaten;s.since_food_steps=0;s.last_eaten=i;
            place_food(s,i);break; // Other pieces remain exactly where they were.
        }
    }
    return {reward,s.last_eaten,s.body.steps>=kEpisodeSteps};
}
class Batch {
public:
    Batch(int n,uint32_t seed,swimmer::Body body=swimmer::body_preset(8));
    void step(const float* actions,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true);
    const float* obs() const {return obs_.get();}
    const State* states() const {return states_.get();}
private:
    int n_;uint32_t seed_;
    device_ptr<State> states_;device_ptr<uint32_t> episodes_;device_ptr<float> obs_;
};
}
#endif
