#include "check.h"
#include "envs/swimmer/predator.h"
#include <type_traits>
#include <cstring>

using namespace cudarl;

TEST_CASE("predator clock warns passes and preserves physical state") {
    static_assert(std::is_trivially_copyable<predator::State>::value,"standalone save descriptor");
    predator::State a;a.seed=73;CHECK(!predator::sample(a).active);
    predator::trigger(a);auto warning=predator::sample(a);
    CHECK(warning.warning);CHECK(!warning.active);CHECK(warning.strength==0);
    predator::advance(a,4);auto start=predator::sample(a);
    CHECK(!start.warning);CHECK(start.active);CHECK(start.phase==0);CHECK(start.strength==1);
    predator::advance(a,10);auto middle=predator::sample(a);
    CHECK(middle.active);CHECK(fabsf(middle.phase-.5f)<1e-6f);
    CHECK(fabsf(middle.eye.z+1.25f)<1e-6f);
    CHECK(middle.eye.x==start.eye.x+4);CHECK(middle.eye.y==start.eye.y);
    CHECK(middle.shadow.x==middle.eye.x+2*middle.eye.y);CHECK(middle.shadow.y==0);
    CHECK(middle.shadow.z==middle.eye.z);CHECK(fabsf(middle.velocity.x-.4f)<1e-6f);
    predator::advance(a,10);auto passed=predator::sample(a);
    CHECK(passed.passed);CHECK(!passed.active);CHECK(!passed.warning);CHECK(passed.strength==0);
    auto same=a;predator::advance(a,-1);CHECK(std::memcmp(&same,&a,sizeof(a))==0);
    predator::clear_manual(a);CHECK(a.manual_age<0);
}

TEST_CASE("predator automatic events are seeded deterministic and have calm gaps") {
    predator::State a,b;a.seed=b.seed=19;
    int active=0,warning=0,calm=0;
    for(int step=0;step<2400;++step) {
        const auto x=predator::sample(a),y=predator::sample(b);
        CHECK(x.eye.x==y.eye.x && x.eye.y==y.eye.y && x.eye.z==y.eye.z);
        CHECK(x.shadow.x==y.shadow.x && x.shadow.y==0 && x.shadow.z==y.shadow.z);
        CHECK(x.active==y.active && x.warning==y.warning);
        active+=x.active?1:0;warning+=x.warning?1:0;calm+=!x.active && !x.warning?1:0;
        predator::advance(a);predator::advance(b);
    }
    CHECK(active>400);CHECK(warning>100);CHECK(calm>400);
    predator::State other;other.seed=20;
    bool phase_difference=false;
    for(int i=0;i<200;++i) {other.elapsed=i*.05f;a.elapsed=other.elapsed;
        phase_difference=phase_difference || predator::sample(other).warning!=predator::sample(a).warning;}
    CHECK(phase_difference);
}

TEST_CASE("predator rays use actual sphere and open arch aperture") {
    auto h=arch::geometry();deep::Basin basin{2};
    CHECK(predator::sphere_segment({-3,0,0},{3,0,0},{0,0,0},1));
    CHECK(!predator::sphere_segment({-3,2,0},{3,2,0},{0,0,0},1));
    CHECK(!predator::sphere_segment({2,0,0},{3,0,0},{0,0,0},1));
    const deep::V eye{-8,4,-1.25f};
    CHECK(predator::occluded({5.5f,-2.5f,-1.25f},eye,h,basin));
    CHECK(!predator::occluded({-3,-2,-1.25f},eye,h,basin));
    CHECK(!predator::occluded({0,-2.25f,-2},{0,-2.25f,2},h,basin));
    CHECK(predator::occluded({1.35f,-2.25f,-2},{1.35f,-2.25f,2},h,basin));
    auto no_arch=h;no_arch.arch=false;
    CHECK(!predator::occluded({1.35f,-2.25f,-2},{1.35f,-2.25f,2},no_arch,{0}));
}

TEST_CASE("predator cover poses fit whole body and hide without solid goals") {
    const auto h=arch::geometry();const deep::Basin basin{2};const deep::V eye{-8,4,-1.25f};
    for(int i=0;i<predator::kCoverCount;++i) {
        const auto c=predator::cover(i,eye,h,basin);CHECK(c.valid);CHECK(c.clearance>.08f);
        CHECK(arch::water(c.position,h,.1f,basin));
        CHECK(c.exposure<.08f);
        const auto body=predator::cover_pose(c),before=body;
        const auto details=predator::exposure_detail(body,eye,h,basin);
        CHECK(details.samples==216);CHECK(details.visible==c.exposure);
        CHECK(std::memcmp(&body,&before,sizeof(body))==0);
        CHECK(deep::obstacle_body_clearance(body,h,basin)>.08f);
    }
    CHECK(!predator::cover(-1,eye,h,basin).valid);
    CHECK(!predator::cover(4,eye,h,basin).valid);
    CHECK(!predator::cover(0,eye,h,{0}).valid);
    deep::BodyState embedded;embedded.position=deep::boulder_center(h);
    CHECK(predator::body_clearance(embedded,h,basin)<0);
}

TEST_CASE("predator exposure sees protruding tail and changing sight line") {
    const auto h=arch::geometry();const deep::Basin basin{2};
    deep::BodyState body;body.position={0,-2.25f,0};
    body.orientation=deep::axis_rotation({0,1,0},-1.5707963267948966f);
    const auto overhead=predator::exposure_detail(body,{0,8,0},h,basin);
    CHECK(overhead.head<.05f);CHECK(overhead.distal_tail>.8f);CHECK(overhead.visible>.3f);
    const auto c=predator::cover(0,{-8,4,-1.25f},h,basin);body=predator::cover_pose(c);
    CHECK(predator::exposure(body,{-8,4,-1.25f},h,basin)<.05f);
    CHECK(predator::exposure(body,{8,4,-1.25f},h,basin)>.8f);
    CHECK(predator::exposure(body,{body.position.x,8,body.position.z},h,basin)>.8f);
    for(int i=0;i<=20;++i) {
        predator::State clock;clock.manual_age=4.f+i;
        const auto event=predator::sample(clock);float best=1;
        for(int goal=0;goal<predator::kCoverCount;++goal) {
            const auto candidate=predator::cover(goal,event.eye,h,basin);
            CHECK(candidate.valid);best=fminf(best,candidate.exposure);
        }
        CHECK(best<.10f);
    }
}
