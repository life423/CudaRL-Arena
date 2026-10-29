#ifndef CUDARL_SHELTER_BATCH_H
#define CUDARL_SHELTER_BATCH_H
#include "shelter_motor.h"
#include "predator.h"
#include "rl/progress_selector_policy.h"

namespace cudarl::shelter {
// This bounded boulder experiment has two real foods, not a new player food cap.
// Import rejects a many-pellet world rather than silently omitting its food.
constexpr int kFoods=2,kChoices=kFoods+predator::kCoverCount,kFeatures=44;
constexpr int kObs=kChoices*(kFeatures+1),kPeriod=20;
struct Metrics {
    float exposure_seconds=0,threat_seconds=0;
    int safe_ticks=0,cover_choices=0,food_choices=0,post_threat_foods=0;
};
struct World {
    arch::State state{};
    predator::State predator{};
    Goal goal{};
    Metrics metrics{};
    float exposure=1,goal_distance=0,goal_progress=0;
    int chosen=-1;
    bool threats=true;
};
struct Options {
    bool training=true,threats=true,cues=true,cover=true;
};
CUDARL_HD inline predator::Event event(const World& world) {
    auto result=predator::sample(world.predator);
    if(!world.threats) {result.active=false;result.warning=false;result.strength=0;result.passed=false;}
    return result;
}
CUDARL_HD inline Goal option(const World& world,int choice,deep::Basin basin={2}) {
    Goal result;
    if(choice>=0 && choice<kFoods) {
        const auto& food=world.state.pond.pellets[choice];result.food=choice;result.position=food.position;
        result.velocity=food.velocity;result.valid=food.active;
    } else if(choice>=kFoods && choice<kChoices) {
        const auto location=predator::cover(choice-kFoods,event(world).eye,world.state.habitat,basin);
        result.cover=choice-kFoods;result.position=location.position;result.valid=location.valid;
    }
    return result;
}
// Features and validity are serialized with each on-policy decision, so masks
// and the sampled log probability are identical during PPO replay.
CUDARL_HD inline void observe(const World& world,float* out,deep::Basin basin={2},Options options={}) {
    auto threat=event(world);
    if(!options.cues) {threat.warning=false;threat.active=false;threat.strength=0;threat.phase=0;}
    const auto& pond=world.state.pond;
    int example=0;if(!pond.pellets[0].active && pond.pellets[1].active) example=1;
    for(int choice=0;choice<kChoices;++choice) {
        float* row=out+choice*kFeatures;
        for(int j=0;j<kFeatures;++j) row[j]=0;
        const auto goal=option(world,choice,basin);
        const bool valid=goal.valid && (choice<kFoods || options.cover);
        out[kChoices*kFeatures+choice]=valid?1.f:0.f;
        if(!valid) continue;
        feeding::features(pond,choice<kFoods?choice:example,row);
        float predicted=predator::occluded(goal.position,threat.eye,world.state.habitat,basin)?0.f:1.f,clearance=0;
        if(choice>=kFoods) {
            const auto point=pond.body.orientation.inverse(goal.position-pond.body.position)/2.f;
            row[0]=point.x;row[1]=point.y;row[2]=point.z;row[27]=world.chosen==choice?1.f:0.f;
            row[28]=deep::norm(goal.position-pond.body.position)/2.f;
            row[29]=row[30]=row[31]=0;
            row[33]=world.chosen==choice?world.goal_progress:0.f;
            const auto cover=predator::cover(choice-kFoods,threat.eye,world.state.habitat,basin);
            predicted=cover.exposure;clearance=cover.clearance;
        }
        const auto relative=pond.body.orientation.inverse(threat.eye-pond.body.position)/16.f;
        row[34]=choice>=kFoods?1.f:0.f;row[35]=threat.warning?1.f:0.f;
        row[36]=threat.strength;row[37]=threat.phase;
        row[38]=options.cues?world.exposure:0.f;row[39]=options.cues?predicted:0.f;
        row[40]=options.cues?relative.x:0.f;row[41]=options.cues?relative.y:0.f;row[42]=options.cues?relative.z:0.f;
        row[43]=clearance/2.f;
    }
}
#if CUDARL_WITH_CUDA
void make_features(const float* obs,float* rows,int n,cudaStream_t stream);
void mask_logits(const float* obs,float* logits,int n,cudaStream_t stream);
class Batch {
    int n_;uint32_t seed_;arch::Context context_;int period_;Options options_;
    device_ptr<World> worlds_;
    device_ptr<float> observations_,motor_inputs_;
    void prepare(cudaStream_t stream);
public:
    Batch(int n,uint32_t seed,arch::Context context,int period=kPeriod,Options options={});
    int count() const {return n_;}
    const float* obs() const {return observations_.get();}
    const World* worlds() const {return worlds_.get();}
    const float* motor_obs(cudaStream_t stream);
    void begin(const int* choices,swimmer::Outputs out,cudaStream_t stream);
    void step(const float* torques,swimmer::Outputs out,cudaStream_t stream,bool autoreset=true);
    World state(cudaStream_t stream) const;
    std::vector<World> states(cudaStream_t stream) const;
    void set_state(const World&,cudaStream_t stream);
    void configure(arch::Context context,cudaStream_t stream);
    void set_options(Options options,cudaStream_t stream);
    void save(checkpoint::Writer&) const;
    void load(checkpoint::Reader&);
};
#endif
}
#endif
