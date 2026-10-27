#ifndef CUDARL_DEMOS_TADPOLE_POND_H
#define CUDARL_DEMOS_TADPOLE_POND_H

#include <cmath>
#include <stdexcept>
#include <vector>
#include "envs/swimmer/swimmer.h"
#include "nn/mlp_cpu.h"

namespace cudarl::demos {
// A single continuously simulated pretrained tadpole. Food placement is a
// world rule; all movement and joint actions come from the saved neural policy.
// Horizontal mechanics are identical to the small CUDA swimming experiment.
class TadpolePond {
public:
    explicit TadpolePond(swimmer::Body body=swimmer::body_preset(6)):policy_(nn::MlpSpec({9,32,32,2},
        {nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity})),body_(body) {}
    void set_policy(const std::vector<float>& weights) {
        if(weights.size()!=policy_.params().size()) throw std::invalid_argument("Tadpole policy architecture does not match");
        for(float w:weights) if(!std::isfinite(w)) throw std::invalid_argument("Tadpole policy contains a non-finite weight");
        policy_.params()=weights;ready_=true;
    }
    void reset(uint32_t seed=1) {
        seed_=seed;placement_=0;eaten_=0;elapsed_=0;
        state_=swimmer::initial(seed_,0,0,body_);
        state_.heading=0;place_food();
    }
    static bool inside_food_area(float x,float y) { return x*x/(2.5f*2.5f)+y*y/(1.8f*1.8f)<1.f; }
    static bool clear_of_body(const swimmer::State& s,float x,float y) {
        float px=s.x+0.25f*cosf(s.heading),py=s.y+0.25f*sinf(s.heading);
        float qx=s.x-0.25f*cosf(s.heading),qy=s.y-0.25f*sinf(s.heading);
        for(int segment=0;segment<3;++segment) {
            const float dx=qx-px,dy=qy-py;
            const float t=swimmer::clamp(((x-px)*dx+(y-py)*dy)/(dx*dx+dy*dy),0,1);
            const float ex=x-px-t*dx,ey=y-py-t*dy;
            if(ex*ex+ey*ey<0.32f*0.32f) return false;
            px=qx;py=qy;
            const float angle=s.heading+s.angle[0]+(segment==1?s.angle[1]:0.f);
            qx=px-0.65f*cosf(angle);qy=py-0.65f*sinf(angle);
        }
        return true;
    }
    void place_food() {
        // Nearby, broadly forward-biased food makes the first presentation
        // legible. Reject bank/body intersections, never policy-test candidates.
        for(uint32_t candidate=0;candidate<256;++candidate) {
            const uint32_t index=placement_*512+candidate*2;
            const float radius=0.8f+0.6f*rng::uniform(seed_,71,index);
            const float offset=(rng::uniform(seed_,71,index+1)*2.f-1.f)*(candidate<128?1.7f:swimmer::kPi);
            const float angle=state_.heading+offset;
            const float x=state_.x+radius*cosf(angle),y=state_.y+radius*sinf(angle);
            if(inside_food_area(x,y) && clear_of_body(state_,x,y)) {
                state_.food_x=x;state_.food_y=y;state_.steps=0;++placement_;return;
            }
        }
        // This is a visible fallback food position, never a hidden body reset.
        state_.food_x=0;state_.food_y=0;state_.steps=0;++placement_;
    }
    bool step() {
        if(!ready_) return false;
        float observation[swimmer::kObs];swimmer::observe(state_,observation);
        policy_.forward(observation,1);
        const auto result=swimmer::advance(state_,policy_.output());
        elapsed_+=swimmer::kDt;
        if(result.success) {++eaten_;place_food();return true;}
        return false;
    }
    const swimmer::State& state() const { return state_; }
    int eaten() const { return eaten_; }
    float elapsed() const { return elapsed_; }
    bool ready() const { return ready_; }
private:
    nn::CpuMlp<float> policy_;
    swimmer::Body body_;
    swimmer::State state_;
    uint32_t seed_=1,placement_=0;
    int eaten_=0;
    float elapsed_=0;
    bool ready_=false;
};
}
#endif
