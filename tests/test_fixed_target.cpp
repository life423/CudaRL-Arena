#include "check.h"
#include "../tools/pond_fixed_target.h"
namespace ft=cudarl::fixed_target;
TEST_CASE("fixed target: paired poses, bearing-relative currents and original motor inputs") {
    for(int i=0;i<32;++i) {
        const auto calm=ft::initial(0xF17ED101u,i,0,0,0),folded=ft::initial(0xF17ED101u,i,1,0,0);
        CHECK(ft::d::norm(calm.state.body.position-folded.state.body.position)==0);
        for(int f=0;f<3;++f) CHECK(ft::d::norm(calm.state.food[f]-folded.state.food[f])==0);
        for(float angle:folded.state.body.angle) CHECK(std::fabs(std::fabs(angle)-.98f)<1e-7f);
        const auto toward=ft::d::V{calm.goal.x-calm.state.body.position.x,0,calm.goal.z-calm.state.body.position.z};
        const auto up=ft::flow(calm.state.body.position,calm.goal,5),down=ft::flow(calm.state.body.position,calm.goal,6),cross=ft::flow(calm.state.body.position,calm.goal,7);
        CHECK(ft::d::dot(up,toward)<0);CHECK(ft::d::dot(down,toward)>0);CHECK(std::fabs(ft::d::dot(cross,toward))<1e-7f);
        CHECK(std::fabs(ft::d::norm(up)-.01f)<1e-8f);CHECK(ft::d::norm(up+down)==0);
        float actual[33],expected[33];ft::observe(calm,actual);ft::p::observe(calm.state,expected);cudarl::demos::primary_pond_food(expected,calm.target);
        for(int j=0;j<33;++j) CHECK(actual[j]==expected[j]);
    }
}
TEST_CASE("fixed target: shared physics, passive torque and selected-only stopping") {
    auto trial=ft::initial(0xF17ED101u,7,0,5,0);auto state=trial.state;auto m=ft::metrics(trial);const float action[4]={.2f,-.4f,.3f,.1f};
    ft::step(trial,m,action);ft::p::advance(state,action,trial.current);
    CHECK(ft::d::norm(trial.state.body.position-state.body.position)==0);for(int j=0;j<10;++j) CHECK(trial.state.body.velocity[j]==state.body.velocity[j]);
    auto passive=ft::initial(0xF17ED101u,7,0,5,1);state=passive.state;auto pm=ft::metrics(passive);const float zero[4]{};
    ft::step(passive,pm,action);ft::p::advance(state,zero,passive.current);CHECK(ft::d::norm(passive.state.body.position-state.body.position)==0);
    auto hit=ft::initial(4,0,0,0,0);hit.state.body=ft::d::BodyState{};hit.state.food[0]=hit.state.body.position;hit.goal=hit.state.food[0];auto hm=ft::metrics(hit);
    ft::step(hit,hm,zero);CHECK(hm.arrival==1);const auto stopped=hit.state.body.position;ft::step(hit,hm,action);CHECK(hm.ticks==1);CHECK(ft::d::norm(hit.state.body.position-stopped)==0);
    hit=ft::initial(4,0,0,0,0);hit.state.body=ft::d::BodyState{};hit.state.food[1]=hit.state.body.position;hm=ft::metrics(hit);
    ft::step(hit,hm,zero);CHECK(hm.arrival==0);CHECK(hm.incidental==1);CHECK(ft::d::norm(hit.goal-hit.state.food[0])==0);
}
