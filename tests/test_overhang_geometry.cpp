#include "check.h"
#include "envs/swimmer/predator.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace cudarl::test {
// Direct native equations, without widening Batch checkpoint/layout validation.
int overhang_cpu_gpu_probe(const std::vector<arch::State>& states,
                           const std::vector<deep::V>& points,deep::V eye,
                           std::vector<float>& cpu,std::vector<float>& gpu);
}

namespace {
using namespace cudarl;
constexpr deep::Basin ledge_basin{3};
bool finite_ledge(deep::V value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
float contact_power(const deep::System& system,const deep::BodyState& body) {
    float power=0;
    for(int row=0;row<deep::dofs;++row) for(int column=0;column<deep::dofs;++column)
        power+=body.velocity[row]*system.R[row][column]*body.velocity[column];
    return power;
}
}

TEST_CASE("overhang geometry: rounded core dimensions normals and rotation share physical units") {
    using namespace cudarl;
    deep::OverhangPart part;part.center={.3f,-2.1f,.85f};part.half={1.f,.5f,.75f};
    part.rotation=deep::axis_rotation({0,1,0},.37f);part.bevel=.2f;
    for(const auto direction:{deep::V{1,0,0},deep::V{-1,0,0},deep::V{0,1,0},
                              deep::V{0,-1,0},deep::V{0,0,1},deep::V{0,0,-1}}) {
        const float support=fabsf(direction.x)*part.half.x+fabsf(direction.y)*part.half.y+fabsf(direction.z)*part.half.z;
        const auto point=part.center+part.rotation.rotate(direction*(support+part.bevel+.3f));
        deep::V normal;const float distance=deep::rounded_box_distance(point,part,&normal);
        CHECK(fabsf(distance-.3f)<1e-6f);
        CHECK(deep::norm(normal-part.rotation.rotate(direction))<2e-6f);
    }
    const auto corner=part.center+part.rotation.rotate({part.half.x+.3f,part.half.y+.4f,0});
    deep::V normal;CHECK(fabsf(deep::rounded_box_distance(corner,part,&normal)-.3f)<1e-6f);
    CHECK(deep::norm(normal-part.rotation.rotate({.6f,.8f,0}))<2e-6f);
    const float epsilon=.0005f;deep::V numerical;
    numerical.x=(deep::rounded_box_distance(corner+deep::V{epsilon,0,0},part)-deep::rounded_box_distance(corner-deep::V{epsilon,0,0},part))/(2*epsilon);
    numerical.y=(deep::rounded_box_distance(corner+deep::V{0,epsilon,0},part)-deep::rounded_box_distance(corner-deep::V{0,epsilon,0},part))/(2*epsilon);
    numerical.z=(deep::rounded_box_distance(corner+deep::V{0,0,epsilon},part)-deep::rounded_box_distance(corner-deep::V{0,0,epsilon},part))/(2*epsilon);
    CHECK(deep::norm(numerical-normal)<.001f);
    CHECK(deep::rounded_box_distance(part.center,part)<0);
    CHECK(finite_ledge(normal) && fabsf(deep::norm(normal)-1)<1e-6f);
    // A bounding box would incorrectly fill this rounded outer corner.
    const auto rounded_corner=part.center+part.rotation.rotate(part.half+deep::V{.18f,.18f,.18f});
    CHECK(deep::rounded_box_distance(rounded_corner,part)>.10f);
}

TEST_CASE("overhang geometry: isolated layout3 preserves every layout2 query and cove floor") {
    using namespace cudarl;
    const auto habitat=arch::geometry();
    static_assert(sizeof(deep::Basin)==sizeof(int),"external layout identity preserves raw world structs");
    for(int x=-12;x<=18;++x) for(int y=-8;y<=1;++y) for(int z=-10;z<=10;++z) {
        const deep::V point{x*.5f,y*.5f,z*.5f};deep::V arch_normal,boulder_normal,legacy_normal,ledge_normal;
        const float arch_distance=deep::arch_distance(point,habitat,&arch_normal);
        const float boulder_distance=deep::boulder_distance(point,habitat,&boulder_normal);
        const bool boulder_nearest=boulder_distance<arch_distance;
        const float legacy=deep::obstacle_distance(point,habitat,{2},&legacy_normal);
        CHECK(legacy==(boulder_nearest?boulder_distance:arch_distance));
        CHECK(deep::norm(legacy_normal-(boulder_nearest?boulder_normal:arch_normal))==0);
        const float ledge=deep::overhang_distance(point,habitat,&ledge_normal);
        CHECK(deep::obstacle_distance(point,habitat,ledge_basin)==fminf(legacy,ledge));
        CHECK(deep::habitat_floor(point.x,point.z,habitat,{2})==deep::habitat_floor(point.x,point.z,habitat,ledge_basin));
        CHECK(deep::norm(deep::habitat_gradient(point.x,point.z,habitat,{2})-
                         deep::habitat_gradient(point.x,point.z,habitat,ledge_basin))==0);
    }
    for(int i=0;i<deep::overhang_parts;++i) {
        const auto part=deep::overhang_part(i,habitat);
        CHECK(part.half.x>0 && part.half.y>0 && part.half.z>0 && part.bevel>0);
        CHECK(deep::overhang_distance(part.center,habitat)<0);
        CHECK(!arch::water(part.center,habitat,.1f,ledge_basin));
    }
}

TEST_CASE("overhang geometry: fixed whole-body cover stays clear throughout the real bird pass") {
    using namespace cudarl;
    const auto habitat=arch::geometry();predator::State clock;clock.manual_age=predator::kWarningSeconds;
    const auto first=predator::sample(clock);std::array<deep::BodyState,predator::kCoverCount> fixed;
    for(int i=0;i<predator::kCoverCount;++i) {
        const auto cover=predator::overhang_cover(i,first.eye,habitat,ledge_basin);
        CHECK(cover.valid && cover.clearance>=.08f);CHECK(cover.exposure<=.05f);
        CHECK(arch::water(cover.position,habitat,.1f,ledge_basin));
        fixed[i]=predator::cover_pose(cover);
        CHECK(deep::overhang_body_clearance(fixed[i],habitat)>=.08f);
    }
    // These are the same physical poses for the entire pass, not goal poses
    // recomputed or moved to follow a changing line of sight.
    for(int tick=0;tick<=400;++tick) {
        clock.manual_age=predator::kWarningSeconds+tick*deep::dt;const auto event=predator::sample(clock);
        for(const auto& body:fixed) {
            const auto before=body;const auto exposure=predator::exposure_detail(body,event.eye,habitat,ledge_basin);
            CHECK(exposure.samples==216 && exposure.visible<=.05f);
            CHECK(exposure.head<.08f && exposure.proximal_tail<.08f && exposure.distal_tail<.08f);
            CHECK(predator::body_clearance(body,habitat,ledge_basin)>=.08f);
            CHECK(std::memcmp(&before,&body,sizeof(body))==0);
        }
    }
    CHECK(!predator::overhang_cover(-1,first.eye,habitat,ledge_basin).valid);
    CHECK(!predator::overhang_cover(predator::kCoverCount,first.eye,habitat,ledge_basin).valid);
    CHECK(!predator::overhang_cover(0,first.eye,habitat,{2}).valid);
}

TEST_CASE("overhang geometry: open camera side stays visible and a protruding tail is not hidden") {
    using namespace cudarl;
    auto habitat=arch::geometry();habitat.arch=false;
    const auto cover=predator::overhang_cover(0,{-12,4,-2.25f},habitat,ledge_basin);
    CHECK(cover.valid);const auto body=predator::cover_pose(cover);
    const deep::V camera{body.position.x,body.position.y,3.f};
    const auto visible=predator::exposure_detail(body,camera,habitat,ledge_basin);
    CHECK(visible.samples==216 && visible.visible>.95f);
    const deep::V overview_direction{.22f,.67f,.74f};
    const auto overview_eye=body.position+overview_direction*(20.f/deep::norm(overview_direction));
    CHECK(predator::exposure_detail(body,overview_eye,habitat,ledge_basin).visible>.8f);
    for(int i=1;i<predator::kCoverCount;++i) {
        const auto fixed=predator::cover_pose(predator::overhang_cover(i,{-12,4,-2.25f},habitat,ledge_basin));
        const auto eye=fixed.position+overview_direction*(20.f/deep::norm(overview_direction));
        CHECK(predator::exposure_detail(fixed,eye,habitat,ledge_basin).visible>.8f);
    }
    CHECK(!predator::occluded(body.position,camera,habitat,ledge_basin));
    CHECK(predator::occluded(body.position,{body.position.x,8,body.position.z},habitat,ledge_basin));
    auto protruding=body;protruding.position={-2.4f,-2.1f,-2.4f};
    protruding.orientation=deep::axis_rotation({0,1,0},3.141592653589793f);
    CHECK(predator::body_clearance(protruding,habitat,ledge_basin)>.08f);
    const auto partial=predator::exposure_detail(protruding,{-2.4f,8,-2.4f},habitat,ledge_basin);
    CHECK(partial.head<.08f && partial.distal_tail>.8f && partial.visible>.3f);
    // Rounded corners must stay transparent where the native solid is absent.
    const auto roof=deep::overhang_part(0,habitat);
    const auto outside=roof.center+roof.rotation.rotate(roof.half+deep::V{roof.bevel*.9f,0,roof.bevel*.9f});
    CHECK(deep::rounded_box_distance(outside,roof)>0);
    CHECK(!predator::occluded(outside+deep::V{0,-1,0},outside+deep::V{0,1,0},habitat,ledge_basin));
}

TEST_CASE("overhang geometry: contacts include the fin and respond by forces without projecting the body") {
    using namespace cudarl;
    auto habitat=arch::geometry();habitat.arch=false;const auto roof=deep::overhang_part(0,habitat);
    const float underside=roof.center.y-roof.half.y-roof.bevel;
    deep::BodyState base;const auto initial=deep::shape(base);
    for(int link=0;link<3;++link) {
        auto body=base;const deep::V at{roof.center.x,underside-.02f,-2.4f};
        body.position=at-initial.link[link].center/deep::scale+base.position;
        body.velocity[1]=.01f;const auto before=body;const auto geometry=deep::shape(body);deep::System force;
        deep::arch_contact(force,geometry.link[link],link,body.velocity,habitat,ledge_basin);
        CHECK(force.arch_touched && force.force[1]<0 && force.R[1][1]>0);
        CHECK(contact_power(force,body)>=0);CHECK(std::memcmp(&body,&before,sizeof(body))==0);
        deep::System old;deep::arch_contact(old,geometry.link[link],link,body.velocity,habitat,{2});
        CHECK(!old.arch_touched && old.force[1]==0);
    }
    auto fin_body=base;fin_body.position=deep::V{roof.center.x,underside-.08f,-2.4f}-initial.link[2].center/deep::scale+base.position;
    const auto fin=deep::shape(fin_body).link[2];deep::System narrow,membrane;
    deep::arch_contact(narrow,fin,1,fin_body.velocity,habitat,ledge_basin);
    deep::arch_contact(membrane,fin,2,fin_body.velocity,habitat,ledge_basin);
    CHECK(narrow.force[1]==0 && membrane.force[1]<0);
    deep::BodyState body;body.position={-2.8f,underside-.08f,-2.4f};
    const auto before=body.position;const float gap=deep::overhang_body_clearance(body,habitat);CHECK(gap<0);
    deep::Physics physics;physics.pond=true;physics.habitat=habitat;physics.basin=ledge_basin;const float zero[4]{};
    deep::substep(body,zero,.0001f,physics);
    CHECK(deep::norm(body.position-before)<.001f); // No collision teleport or pose rescue.
    CHECK(deep::norm(body.position-(before+deep::V{body.velocity[0],body.velocity[1],body.velocity[2]}*(.0001f/deep::scale)))==0);
    float maximum_step=0,maximum_substep=0;
    for(int tick=0;tick<80;++tick) {
        const auto previous=body.position;auto explicit_body=body;
        for(int sub=0;sub<10;++sub) {
            const auto start=explicit_body.position;deep::substep(explicit_body,zero,.005f,physics);
            const auto integrated=start+deep::V{explicit_body.velocity[0],explicit_body.velocity[1],explicit_body.velocity[2]}*(.005f/deep::scale);
            CHECK(deep::norm(explicit_body.position-integrated)==0);
            maximum_substep=fmaxf(maximum_substep,deep::norm(explicit_body.position-start));
            CHECK(finite_ledge(explicit_body.position) && std::isfinite(deep::energy(explicit_body)));
            for(float velocity:explicit_body.velocity) CHECK(std::isfinite(velocity));
        }
        deep::advance_physics(body,zero,physics);
        CHECK(finite_ledge(body.position) && std::isfinite(deep::energy(body)));
        CHECK(deep::norm(body.position-explicit_body.position)==0);
        CHECK(body.orientation.x==explicit_body.orientation.x && body.orientation.y==explicit_body.orientation.y &&
              body.orientation.z==explicit_body.orientation.z && body.orientation.w==explicit_body.orientation.w);
        for(int i=0;i<4;++i) CHECK(body.angle[i]==explicit_body.angle[i]);
        for(int i=0;i<deep::dofs;++i) CHECK(body.velocity[i]==explicit_body.velocity[i]);
        CHECK(body.work==explicit_body.work && body.travel==explicit_body.travel);
        maximum_step=fmaxf(maximum_step,deep::norm(body.position-previous));
    }
    CHECK(deep::overhang_body_clearance(body,habitat)>gap);
    std::printf("    ledge contact trajectory: maximum50ms displacement %.7f model, maximum5ms displacement %.7f model; every substep equals integrated velocity\n",
                maximum_step,maximum_substep);
}

TEST_CASE("overhang geometry: old observations stay55 and food remains outside the new solid") {
    using namespace cudarl;
    const auto habitat=arch::geometry();const auto roof=deep::overhang_part(0,habitat);
    auto state=arch::initial(73,0,0,false,ledge_basin);
    const auto cover=predator::overhang_cover(0,{-12,4,-2.25f},habitat,ledge_basin);
    state.pond.body=predator::cover_pose(cover);
    float old[arch::kObs],added[arch::kObs];arch::observe(state,old,{2});arch::observe(state,added,ledge_basin);
    static_assert(arch::kObs==55 && arch::kAct==4,"ledge adds geometry without another policy output or observation");
    for(int i=0;i<feeding::kObs;++i) CHECK(old[i]==added[i]);
    CHECK(added[feeding::kObs+16]>old[feeding::kObs+16]+.5f); // Existing upward ray sees the roof.
    constexpr float radius=feeding::radius/deep::scale;
    const deep::V shallow_overlap{roof.center.x,roof.center.y-roof.half.y-roof.bevel+.01f,-2.4f};
    const auto corrected=arch::particle_contact(shallow_overlap,habitat,ledge_basin);
    CHECK(deep::overhang_distance(corrected,habitat)>=radius-3e-4f);
    CHECK(deep::obstacle_distance(corrected,habitat,ledge_basin)>=radius-3e-4f);
    int count=0;
    for(uint32_t seed=1;seed<=12;++seed) for(uint32_t environment=0;environment<8;++environment) {
        auto world=arch::initial(seed,environment,0,false,ledge_basin);
        for(int replacement=0;replacement<12;++replacement) {
            const int slot=replacement%2;world.pond.pellets[slot].active=false;arch::replenish(world,ledge_basin);
            CHECK(feeding::count(world.pond)==2);
            const auto position=world.pond.pellets[slot].position;
            CHECK(arch::water(position,world.habitat,.4f,ledge_basin));
            CHECK(deep::overhang_distance(position,world.habitat)>.4f);++count;
        }
    }
    CHECK(count==1152);
    auto tail_overlap=state.pond.body;tail_overlap.position={-3.55f,-2.1f,-3.3f};
    tail_overlap.orientation=deep::axis_rotation({0,1,0},-1.5707963267948966f);
    CHECK(arch::water(tail_overlap.position,habitat,.25f,ledge_basin)); // The head fits.
    CHECK(deep::overhang_body_clearance(tail_overlap,habitat)<0); // Its rear tail hits the root.
    auto invalid=state.pond.body;invalid.position=roof.center;
    CHECK(deep::overhang_body_clearance(invalid,habitat)<0);
    CHECK(predator::body_clearance(invalid,habitat,ledge_basin)<0);
}

TEST_CASE("overhang geometry: calm and disturbed zero-torque trajectories are diagnostics not learned hiding") {
    using namespace cudarl;
    const auto habitat=arch::geometry();const deep::V eye{-8,4,-1.25f};const float zero[4]{};
    deep::Physics physics;physics.pond=true;physics.habitat=habitat;physics.basin=ledge_basin;
    for(int goal=0;goal<predator::kCoverCount;++goal) for(int disturbed=0;disturbed<2;++disturbed) {
        const auto cover=predator::overhang_cover(goal,eye,habitat,ledge_basin);CHECK(cover.valid);
        auto body=predator::cover_pose(cover);const auto initial=body.position;
        if(disturbed) {body.velocity[0]=.001f;body.velocity[1]=.0005f;}
        float longest=0,consecutive=0,minimum=1e6f;
        for(int tick=0;tick<200;++tick) {
            const auto before=body.position;deep::advance_physics(body,zero,physics);
            const float clearance=predator::body_clearance(body,habitat,ledge_basin);
            const float exposure=predator::exposure(body,eye,habitat,ledge_basin);
            CHECK(finite_ledge(body.position) && std::isfinite(deep::energy(body)));
            CHECK(deep::norm(body.position-before)<.1f);
            minimum=fminf(minimum,clearance);
            consecutive=clearance>=.08f && exposure<=.05f?consecutive+deep::dt:0;
            longest=fmaxf(longest,consecutive);
        }
        std::printf("    ledge goal %d zero-torque %s: displacement %.5f model, minimum clearance %.5f, longest sampled hide %.2fs\n",
                    goal,disturbed?"initial-velocity":"calm",deep::norm(body.position-initial),minimum,longest);
    }
    // Finite, passive trajectories alone do not establish arrival, active
    // braking, PPO progress or that the existing neural policy can seek cover.
}

GPU_TEST_CASE("overhang geometry: direct CPU CUDA solids rays contacts observations and one step agree") {
    using namespace cudarl;
    const auto habitat=arch::geometry();const deep::V eye{-8,4,-1.25f};
    std::vector<arch::State> states;std::vector<deep::V> points;
    for(int goal=0;goal<predator::kCoverCount;++goal) {
        auto state=arch::initial(73,0,0,false,ledge_basin);
        state.pond.body=predator::cover_pose(predator::overhang_cover(goal,eye,habitat,ledge_basin));
        states.push_back(state);points.push_back(state.pond.body.position);
    }
    const auto roof=deep::overhang_part(0,habitat);const float underside=roof.center.y-roof.half.y-roof.bevel;
    for(const auto point:{deep::V{roof.center.x,underside-.02f,-2.4f},
                          roof.center+roof.rotation.rotate(roof.half+deep::V{.2f,.3f,.4f}),
                          roof.center,deep::V{3.8f,-1.35f,-1.25f}}) {
        auto state=arch::initial(73,0,0,false,ledge_basin);state.pond.body.position={-2.8f,underside-.08f,-2.4f};
        state.pond.body.velocity[1]=.001f;states.push_back(state);points.push_back(point);
    }
    std::vector<float> cpu,gpu;const int stride=test::overhang_cpu_gpu_probe(states,points,eye,cpu,gpu);
    CHECK(stride==200 && cpu.size()==states.size()*stride && gpu.size()==cpu.size());
    if(cpu.size()!=gpu.size() || stride!=200) return;
    for(size_t i=0;i<cpu.size();++i) {
        const int field=static_cast<int>(i%stride);
        // Solver integration has a wider tolerance than pure geometry/forces.
        const float absolute=field>=13 && field<30?.0015f:(field>=30 && field<140?1e-6f:3e-5f);
        CHECK(std::isfinite(cpu[i]) && std::isfinite(gpu[i]));
        CHECK(fabsf(cpu[i]-gpu[i])<=absolute+fabsf(cpu[i])*3e-4f);
    }
}
