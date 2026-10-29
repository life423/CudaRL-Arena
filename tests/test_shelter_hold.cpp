#include "check.h"
#include "rl/shelter_hold_cases.h"
#include "rl/shelter_calibration.h"
#include "rl/pond_selector_policy.h"

using namespace cudarl;
static_assert(rl::shelter_hold_evaluation_seed!=rl::shelter_calibration_seed,"Holding evaluation must use a distinct seed from calibration");
TEST_CASE("shelter holding lesson evaluation uses fresh legal calm perturbations") {
    const auto reference=arch::initial(7,0,0,false,{2});const auto before=reference.pond.body.position;
    const auto cases=rl::shelter_hold_cases(reference);CHECK(cases.size()==24);int goals[4]{};
    for(const auto& item:cases) {
        CHECK(item.goal.cover>=0 && item.goal.cover<4);++goals[item.goal.cover];CHECK(item.goal.food==-1);CHECK(item.goal.valid);
        CHECK(deep::norm(item.current)==0);CHECK(item.state.pond.target==-1);CHECK(feeding::count(item.state.pond)==2);
        const float distance=deep::norm(item.state.pond.body.position-item.goal.position);
        CHECK(distance>=.025f && distance<.2f);CHECK(predator::body_clearance(item.state.pond.body,item.state.habitat,{2})>=.08f);
        for(float speed:item.state.pond.body.velocity) CHECK(speed==0);for(float angle:item.state.pond.body.angle) CHECK(angle==0);
    }
    for(int n:goals) CHECK(n==6);
    CHECK(reference.pond.body.position.x==before.x && reference.pond.body.position.y==before.y && reference.pond.body.position.z==before.z);
}
TEST_CASE("shelter holding lesson layouts are deterministic without implying escapes") {
    const auto reference=arch::initial(7,0,0,false,{2});const auto a=rl::shelter_hold_cases(reference),b=rl::shelter_hold_cases(reference);
    const auto c=rl::shelter_hold_cases(reference,rl::shelter_hold_evaluation_seed+1u);int changed=0,already_concealed=0;
    for(size_t i=0;i<a.size();++i) {
        const auto& x=a[i].state.pond.body;const auto& y=b[i].state.pond.body;
        CHECK(x.position.x==y.position.x && x.position.y==y.position.y && x.position.z==y.position.z);
        CHECK(x.orientation.x==y.orientation.x && x.orientation.y==y.orientation.y && x.orientation.z==y.orientation.z && x.orientation.w==y.orientation.w);
        changed+=deep::norm(x.position-c[i].state.pond.body.position)>1e-5f?1:0;
        already_concealed+=predator::exposure(x,a[i].predator_eye,a[i].state.habitat,{2})<=.05f?1:0;
    }
    CHECK(changed>0);CHECK(already_concealed>0);
    // Those already concealed starts are useful for holding. They cannot be
    // reported as a policy choosing cover or escaping from an exposed start.
}
GPU_TEST_CASE("shelter holding curriculum leaves food worlds intact and bounds new lesson starts") {
    constexpr int n=16;constexpr uint32_t seed=73;
    const auto parent=rl::initial_pond_policy();const auto shared=rl::transfer_progress_actor(parent.selector);
    const arch::Context context{{.0075f,0,0},1,{2}};rl::SwimmerStream stream;shelter_motor::Batch batch(n,seed);
    batch.configure(shared,context,stream.value,true);const auto states=batch.host_states(stream.value);
    int food=0,exact=0,perturbed=0;
    for(int i=0;i<n;++i) {
        const auto& world=states[static_cast<size_t>(i)];const auto& body=world.state.pond.body;
        CHECK(!world.state.pond.invalid);CHECK(feeding::count(world.state.pond)==2);
        if(i%2==0) {
            ++food;CHECK(!world.request.cover);const auto expected=arch::initial(seed,static_cast<uint32_t>(i),0,false,context.basin);
            CHECK(deep::norm(body.position-expected.pond.body.position)<1e-5f);
            CHECK(fabsf(body.orientation.x-expected.pond.body.orientation.x)<1e-6f && fabsf(body.orientation.y-expected.pond.body.orientation.y)<1e-6f && fabsf(body.orientation.z-expected.pond.body.orientation.z)<1e-6f && fabsf(body.orientation.w-expected.pond.body.orientation.w)<1e-6f);
            for(int j=0;j<2;++j) {
                CHECK(world.state.pond.pellets[j].active==expected.pond.pellets[j].active);
                CHECK(deep::norm(world.state.pond.pellets[j].position-expected.pond.pellets[j].position)<1e-6f);
                CHECK(world.state.pond.pellets[j].id==expected.pond.pellets[j].id);
            }
            continue;
        }
        CHECK(world.request.cover);CHECK(world.state.pond.target==-1);
        CHECK(predator::body_clearance(body,world.state.habitat,{2})>=.08f);
        for(float speed:body.velocity) CHECK(speed==0);for(float angle:body.angle) CHECK(angle==0);
        const float distance=deep::norm(body.position-world.request.goal.position);
        if(i%4==1) {++exact;CHECK(distance==0);}
        else {++perturbed;CHECK(distance<=.2601f);}
    }
    CHECK(food==8);CHECK(exact==4);CHECK(perturbed==4);
}
GPU_TEST_CASE("shelter holding practice is calm while evaluation keeps its physical current") {
    constexpr int n=16;const auto parent=rl::initial_pond_policy();const auto shared=rl::transfer_progress_actor(parent.selector);
    const arch::Context context{{.0075f,0,0},1,{2}};rl::SwimmerStream stream;
    shelter_motor::Batch practice(n,73),evaluation(n,73);practice.configure(shared,context,stream.value,true);evaluation.configure(shared,context,stream.value,true);
    const auto before=practice.host_states(stream.value);std::vector<float> zero(static_cast<size_t>(n)*4,0);device_ptr<float> actions(zero.size());actions.copy_from_host(zero.data(),zero.size());
    practice.step(actions.get(),{},stream.value,false,true);evaluation.step(actions.get(),{},stream.value,false,false);
    const auto trained=practice.host_states(stream.value),tested=evaluation.host_states(stream.value);int cover_flow_difference=0;
    for(int i=0;i<n;++i) {
        auto expected=before[static_cast<size_t>(i)].state;
        const auto water=before[static_cast<size_t>(i)].request.cover?deep::V{}:arch::practice_current(expected,context);
        const float no_action[4]={0,0,0,0};(void)arch::advance(expected,no_action,water,context.basin);arch::replenish(expected,context.basin);
        CHECK(deep::norm(expected.pond.body.position-trained[static_cast<size_t>(i)].state.pond.body.position)<2e-5f);
        auto expected_evaluation=before[static_cast<size_t>(i)].state;
        (void)arch::advance(expected_evaluation,no_action,context.current,context.basin);arch::replenish(expected_evaluation,context.basin);
        CHECK(deep::norm(expected_evaluation.pond.body.position-tested[static_cast<size_t>(i)].state.pond.body.position)<2e-5f);
        if(before[static_cast<size_t>(i)].request.cover)
            cover_flow_difference+=deep::norm(trained[static_cast<size_t>(i)].state.pond.body.position-tested[static_cast<size_t>(i)].state.pond.body.position)>1e-7f?1:0;
    }
    CHECK(cover_flow_difference>0);
}
