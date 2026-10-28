#include "check.h"
#include "demos/pond_reliability.h"
#include <limits>
namespace dm=cudarl::demos;
TEST_CASE("playback reliability: stationary, moving and short pauses remain distinct") {
    dm::PondFrame frame;frame.state=cudarl::deep_pond::initial(7,0);const auto origin=frame.state.body.position;
    dm::PondReliability stopped,moving,brief;
    for(int tick=0;tick<=200;++tick) {
        frame.state.steps=tick;frame.state.since_food=tick;frame.state.body.position=origin;stopped.observe(frame);
        frame.state.body.position=origin+cudarl::deep::V{tick*.02f,0,0};moving.observe(frame);
        const int position=tick<70?0:tick-70;frame.state.body.position=origin+cudarl::deep::V{position*.02f,0,0};brief.observe(frame);
    }
    stopped.finish();moving.finish();brief.finish();CHECK(!stopped.passes());CHECK(stopped.confirmed_events()==1);CHECK(stopped.events.size()==1);
    CHECK(stopped.max_immobility_seconds>=10);CHECK(moving.passes());CHECK(brief.passes());
}
TEST_CASE("playback reliability: invalid food, state and duplicate steps fail visibly") {
    dm::PondFrame frame;frame.state=cudarl::deep_pond::initial(7,0);dm::PondReliability monitor;monitor.observe(frame);monitor.observe(frame);
    frame.state.steps=1;frame.state.food[0].y=1;frame.state.body.position.x=std::numeric_limits<float>::quiet_NaN();monitor.observe(frame);
    CHECK(!monitor.passes());CHECK(monitor.invalid_states==1);CHECK(monitor.invalid_foods==1);CHECK(monitor.nonsequential_steps==1);
}
TEST_CASE("progress playback: transferred selector keeps the legacy CPU trajectory and tracks history") {
    auto original=cudarl::rl::initial_pond_policy();original.foods=0;
    cudarl::rl::ProgressPolicy progress{0,original,cudarl::rl::transfer_progress_actor(original.selector)};
    auto a=dm::create_pond_playback(original,"cpu"),b=dm::create_pond_playback(progress,"cpu");
    for(int tick=1;tick<=80;++tick) {
        const auto& x=a->step();const auto& y=b->step();CHECK(cudarl::deep::norm(x.state.body.position-y.state.body.position)<1e-6f);CHECK(x.selected==y.selected);
        if(tick%original.decision_steps==0) for(int f=0;f<3;++f) {
            CHECK(y.history.placements[f]==y.state.placements[f]);CHECK(std::fabs(y.history.distance[f]-cudarl::deep::norm(y.state.food[f]-y.state.body.position))<1e-6f);
        }
    }
    auto fallback=dm::create_pond_playback(progress,"auto","missing-module");CHECK(std::string(fallback->backend())=="CPU");
    auto bad=progress;bad.actor.resize(3);bool rejected=false;try {dm::create_pond_playback(bad,"cpu");} catch(const std::exception&) {rejected=true;}CHECK(rejected);
}
