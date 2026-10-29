#include "check.h"
#include "envs/swimmer/arch.h"
#include "envs/swimmer/arch_batch.h"
#include "rl/arch_cpu.h"
#include "rl/swimmer_ppo.h"
#include <array>

namespace br=cudarl::deep;
namespace ba=cudarl::arch;
namespace {
bool finite_boulder(br::V value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
float wet_body_gap(const br::BodyState& body,const br::Habitat& habitat) {
    const auto shape=br::shape(body);float gap=1e6f;
    for(int link=0;link<3;++link) for(int slice=0;slice<br::arch_axial_samples;++slice) {
        const auto& part=shape.link[link];const float fraction=(slice-4)*.25f;
        const auto p=(part.center+part.q.rotate({1,0,0})*(fraction*part.length*.5f))/br::scale;
        const auto gradient=br::habitat_gradient(p.x,p.z,habitat,br::Basin{2});
        const float length=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
        const br::V normal=br::V{-gradient.x,1,-gradient.z}/length;
        const float bed=(p.y-br::habitat_floor(p.x,p.z,habitat,br::Basin{2}))/length-
            br::arch_slice_support(part,link,fraction,normal)/br::scale;
        const float surface=-p.y-br::arch_slice_support(part,link,fraction,{0,1,0})/br::scale;
        gap=fminf(gap,fminf(bed,surface));
    }
    return gap;
}
}

TEST_CASE("boulder habitat: descriptor preserves the cove floor and legacy obstacle queries") {
    const auto h=ba::geometry();CHECK(br::norm(br::boulder_center(h)-br::V{3.8f,-2.65f,-1.25f})==0);
    CHECK(br::boulder_radius(h)==1.3f);
    const auto center=br::boulder_center(h);
    CHECK(center.y-br::boulder_radius(h)<br::habitat_floor(center.x,center.z,h,br::Basin{2}));
    CHECK(center.y+br::boulder_radius(h)<-.35f); // Bed-rooted and fully submerged.
    for(int x=-10;x<=18;++x) for(int z=-10;z<=10;++z) {
        const br::V p{x*.5f,-1.6f,z*.5f};
        CHECK(br::habitat_floor(p.x,p.z,h,br::Basin{1})==br::habitat_floor(p.x,p.z,h,br::Basin{2}));
        for(const auto basin:{br::Basin{},br::Basin{1}}) {
            br::V original,shared;
            CHECK(br::arch_distance(p,h,&original)==br::obstacle_distance(p,h,basin,&shared));
            CHECK(br::norm(original-shared)==0);
        }
    }
}

TEST_CASE("boulder habitat: one shared solid query supplies exact sphere normals and body rays") {
    const auto h=ba::geometry();const br::Basin basin{2};const auto center=br::boulder_center(h);const float radius=br::boulder_radius(h);
    for(const auto direction:{br::V{1,0,0},br::V{0,1,0},br::V{0,0,1},br::V{0,-1,0}}) {
        const auto p=center+direction*(radius+.2f);br::V normal;
        CHECK(fabsf(br::boulder_distance(p,h,&normal)-.2f)<4e-7f);CHECK(br::norm(normal-direction)<2e-7f);
    }
    br::V singular;CHECK(std::isfinite(br::boulder_distance(center,h,&singular)));CHECK(br::norm(singular-br::V{1,0,0})==0);
    for(int x=-8;x<=18;++x) for(int z=-8;z<=8;++z) {
        const br::V p{x*.5f,-1.6f,z*.5f};
        const float arch=br::arch_distance(p,h),sphere=br::boulder_distance(p,h);
        CHECK(br::obstacle_distance(p,h,basin)==fminf(arch,sphere));
    }
    const auto origin=center+br::V{radius+.8f,0,0};
    CHECK(fabsf(ba::proximity(origin,{-1,0,0},h,basin)-(1.f-.8f/ba::sensing_range))<.01f);
    CHECK(ba::proximity(origin,{-1,0,0},h,br::Basin{1})==0);
    auto s=ba::initial(7,0,0,false,basin);s.pond.body.position=origin;s.pond.body.orientation=br::axis_rotation({0,1,0},3.14159265f);
    float legacy[ba::kObs],active[ba::kObs];ba::observe(s,legacy,br::Basin{1});ba::observe(s,active,basin);
    for(int i=0;i<cudarl::feeding::kObs;++i) CHECK(legacy[i]==active[i]);
    CHECK(active[cudarl::feeding::kObs]>.7f && legacy[cudarl::feeding::kObs]==0);
}

TEST_CASE("boulder habitat: a blocked direct line has two wet whole-body detours") {
    const auto h=ba::geometry();const br::Basin basin{2};
    CHECK(!ba::water({3.8f,-1.35f,-1.25f},h,.15f,basin));
    float minimum_solid=1e6f,minimum_water=1e6f;int poses=0;
    const auto pose=[&](br::V position,float yaw) {
        br::BodyState body;body.position=position;body.orientation=br::axis_rotation({0,1,0},yaw);
        const float solid=br::obstacle_body_clearance(body,h,basin),water=wet_body_gap(body,h);
        minimum_solid=fminf(minimum_solid,solid);minimum_water=fminf(minimum_water,water);++poses;
        CHECK(solid>.12f);CHECK(water>.08f);
    };
    const auto line=[&](br::V start,br::V end) {
        const auto direction=end-start;
        for(int sample=0;sample<=40;++sample) {
            pose(start+direction*(sample/40.f),-atan2f(direction.z,direction.x));
        }
    };
    const br::V start{2.4f,-1.35f,-1.25f},end{6.9f,-1.35f,-1.25f};
    // The north approach bends before the bank: an abrupt 90-degree pose
    // change would put a straight 1.55-unit tail outside the water.
    line(start,{2.4f,-1.35f,-2.25f});line({2.4f,-1.35f,-2.25f},{4.6f,-1.35f,-2.25f});
    for(int sample=0;sample<=40;++sample) {
        const float theta=-1.57079632679f+sample/40.f*1.57079632679f;
        pose({4.6f+1.6f*cosf(theta),-1.35f,-.65f+1.6f*sinf(theta)},-atan2f(cosf(theta),-sinf(theta)));
    }
    for(int sample=0;sample<=20;++sample) {
        pose({6.2f,-1.35f,-.65f},-1.57079632679f+sample/20.f*(1.57079632679f+atan2f(.6f,.7f)));
    }
    line({6.2f,-1.35f,-.65f},end);
    const std::array<br::V,4> south{{start,{2.4f,-1.35f,1.4f},{4.7f,-1.35f,1.4f},end}};
    for(size_t leg=1;leg<south.size();++leg) line(south[leg-1],south[leg]);
    for(int sample=0;sample<=20;++sample) {
        pose({2.4f,-1.35f,-2.25f},sample/20.f*1.57079632679f);
    }
    std::printf("    boulder route poses %d, minimum solid %.5f, minimum water %.5f model units\n",poses,minimum_solid,minimum_water);
    // These geometric routes are feasibility checks, not scripted AI actions.
}

TEST_CASE("boulder habitat: compliant contact covers head links and fin without projecting the body") {
    auto h=ba::geometry();h.arch=false;const br::Basin basin{2};const auto center=br::boulder_center(h);const float radius=br::boulder_radius(h);
    br::BodyState base;const auto initial=br::shape(base);
    for(int link=0;link<3;++link) {
        auto body=base;body.position=base.position+center+br::V{radius+.03f,0,0}-initial.link[link].center/br::scale;
        body.velocity[0]=-.01f;const auto geometry=br::shape(body);br::System force;
        br::arch_contact(force,geometry.link[link],link,body.velocity,h,basin);
        CHECK(force.arch_touched && force.force[0]>0 && force.R[0][0]>0);
        float power=0;for(int i=0;i<br::dofs;++i) for(int j=0;j<br::dofs;++j) power+=body.velocity[i]*force.R[i][j]*body.velocity[j];
        CHECK(power>=0);
        br::System disabled;br::arch_contact(disabled,geometry.link[link],link,body.velocity,h,br::Basin{1});CHECK(!disabled.arch_touched && disabled.force[0]==0);
    }
    auto fin_body=base;fin_body.position=base.position+center+br::V{0,radius+.08f,0}-initial.link[2].center/br::scale;
    const auto fin=br::shape(fin_body).link[2];br::System core,membrane;
    br::arch_contact(core,fin,1,fin_body.velocity,h,basin);br::arch_contact(membrane,fin,2,fin_body.velocity,h,basin);
    CHECK(core.force[1]==0 && membrane.force[1]>0);
    br::BodyState body;body.position=center+br::V{radius+.1f,0,0};body.orientation=br::axis_rotation({0,1,0},3.14159265f);
    const auto before=body.position;const float old_gap=br::boulder_body_clearance(body,h);CHECK(old_gap<0);
    br::Physics physics;physics.pond=true;physics.habitat=h;physics.basin=basin;const float action[4]{};
    br::substep(body,action,.0001f,physics);CHECK(br::norm(body.position-before)<.001f);
    for(int tick=0;tick<80;++tick) {
        br::advance_physics(body,action,physics);CHECK(finite_boulder(body.position));
        CHECK(std::isfinite(br::energy(body)));CHECK(br::boulder_body_clearance(body,h)>-.4f);
    }
    CHECK(br::boulder_body_clearance(body,h)>old_gap);
    body={};body.position=center+br::V{radius-.2f,0,0};body.orientation=br::axis_rotation({0,1,0},3.14159265f);
    for(int tick=0;tick<40;++tick) {
        br::advance_physics(body,action,physics);CHECK(finite_boulder(body.position));
        for(float velocity:body.velocity) CHECK(std::isfinite(velocity));
    } // A newly changed practice layout may overlap; the live import is preflighted separately.
}

TEST_CASE("boulder habitat: food particles flow and targets stay clear of both solids") {
    const auto h=ba::geometry();const br::Basin basin{2};const auto center=br::boulder_center(h);const br::V current{.0075f,0,0};
    CHECK(br::norm(br::habitat_flow(center,h,current,basin))==0);
    int cove_foods=0;
    for(uint32_t seed=1;seed<=10;++seed) for(uint32_t environment=0;environment<8;++environment) {
        auto state=ba::initial(seed,environment,0,false,basin);
        for(int replacement=0;replacement<12;++replacement) {
            const int slot=replacement%2;state.pond.pellets[slot].active=false;ba::replenish(state,basin);
            CHECK(cudarl::feeding::count(state.pond)==2);const auto point=state.pond.pellets[slot].position;
            CHECK(ba::water(point,state.habitat,.4f,basin));
            if(br::habitat_floor(point.x,point.z,state.habitat)>=0) ++cove_foods;
        }
    }
    CHECK(cove_foods>150);
    // Regression from a five-minute live-path probe: a surviving cove pellet
    // must not crowd every subsequent intended cove arrival back into main.
    for(uint32_t seed=1;seed<=128;++seed) for(uint32_t environment=0;environment<4;++environment) {
        auto state=ba::initial(seed,environment,0,false,basin);
        state.pond.body.position={-2.5f,-2.1f,2.2f};state.pond.body.orientation={};
        for(auto& pellet:state.pond.pellets) pellet={};
        const br::V existing{7.4f,-1.65f,-1.5f};
        state.pond.pellets[0].active=true;state.pond.pellets[0].id=1;state.pond.pellets[0].position=existing;
        state.pond.next_id=3;ba::place(state,1,1,basin);
        const auto added=state.pond.pellets[1].position;
        CHECK(cudarl::feeding::count(state.pond)==2 && state.pond.pellets[1].id==3);
        CHECK(added.x>6.1f && br::norm(added-existing)>=1.2f);
        CHECK(ba::water(added,state.habitat,.4f,basin));
        CHECK(state.pond.pellets[0].active && state.pond.pellets[0].id==1 && br::norm(state.pond.pellets[0].position-existing)==0);
    }
    auto particle=ba::particle_contact(center,h,basin);
    constexpr float pellet_radius=cudarl::feeding::radius/br::scale;
    CHECK(br::obstacle_distance(particle,h,basin)>=pellet_radius-2e-4f);
    particle=center+br::V{-2.f,.4f,0};
    for(int tick=0;tick<600;++tick) {
        const auto flow=br::habitat_flow(particle,h,current,basin);CHECK(finite_boulder(flow));CHECK(br::norm(flow)<=br::norm(current)+1e-8f);
        const auto previous=particle;particle=ba::advect(particle,current,h,br::dt,basin);
        CHECK(finite_boulder(particle));CHECK(br::obstacle_distance(particle,h,basin)>=pellet_radius-3e-4f);
        CHECK(br::norm(particle-previous)<.011f);
    }
}

TEST_CASE("boulder habitat: bounded directional wakes shelter real drag and drifting particles") {
    auto h=ba::geometry();h.arch=false;const br::Basin basin{2};const auto center=br::boulder_center(h);const float radius=br::boulder_radius(h);
    const br::V current{.005f,0,0},reverse{-.005f,0,0},turned{0,0,.005f};
    const auto downstream=center+br::V{2*radius,0,0},upstream=center-br::V{2*radius,0,0};
    const auto sideways=center+br::V{2*radius,1.05f,-2.05f};
    const float sheltered=br::obstacle_wake_factor(downstream,h,current,basin);
    CHECK(sheltered>.5f && sheltered<.6f);
    CHECK(br::obstacle_wake_factor(upstream,h,current,basin)==1);
    CHECK(br::obstacle_wake_factor(sideways,h,current,basin)==1);
    CHECK(br::obstacle_wake_factor(downstream,h,reverse,basin)==1);
    CHECK(fabsf(br::obstacle_wake_factor(upstream,h,reverse,basin)-sheltered)<1e-6f);
    CHECK(fabsf(br::obstacle_wake_factor(center+br::V{0,0,2*radius},h,turned,basin)-sheltered)<1e-6f);
    CHECK(br::norm(br::habitat_flow(downstream,h,current,basin))<.61f*br::norm(br::habitat_flow(upstream,h,current,basin)));
    CHECK(br::norm(br::habitat_flow(sideways,h,current,basin))>.0049f);
    CHECK(br::obstacle_wake_factor(center+br::V{100*radius,0,0},h,current,basin)>.998f);
    CHECK(br::obstacle_wake_factor(downstream,h,current,br::Basin{1})==1);
    CHECK(br::norm(br::habitat_flow(downstream,h,{},basin))==0);
    CHECK(br::norm(br::habitat_flow(center,h,current,basin))==0);
    for(int x=-15;x<=25;++x) for(int z=-15;z<=15;++z) {
        const br::V point{x*.4f,-1.6f,z*.4f};const auto velocity=br::habitat_flow(point,h,current,basin);
        const float factor=br::obstacle_wake_factor(point,h,current,basin);
        CHECK(finite_boulder(velocity) && factor>=.279999f && factor<=1);
        CHECK(br::norm(velocity)<=br::norm(current)+1e-8f);
    }
    auto arch=ba::geometry();const br::V axial{0,0,.005f};
    CHECK(br::obstacle_wake_factor(arch.center+br::V{0,0,1},arch,axial,basin)==1);
    CHECK(br::obstacle_wake_factor(arch.center+br::V{arch.major,0,1},arch,axial,basin)<.6f);
    arch.yaw=.7f;const auto direction=br::arch_to_world({0,0,1},arch)-arch.center;
    CHECK(br::obstacle_wake_factor(br::arch_to_world({0,0,1},arch),arch,direction*.005f,basin)==1);
    CHECK(br::obstacle_wake_factor(br::arch_to_world({arch.major,0,1},arch),arch,direction*.005f,basin)<.6f);
    br::BodyState body;body.position=center+br::V{2*radius,.35f,0};body.orientation=br::axis_rotation({0,1,0},3.14159265f);
    CHECK(br::boulder_body_clearance(body,h)>.7f);
    br::Physics physical;physical.pond=true;physical.habitat=h;physical.water_velocity=current;physical.basin=basin;
    auto exposed=physical;exposed.basin={1};
    const auto wake_force=br::assemble(body,physical),reference_force=br::assemble(body,exposed);
    CHECK(reference_force.force[0]>0 && wake_force.force[0]>0 && wake_force.force[0]<reference_force.force[0]*.8f);
    const auto sheltered_food=ba::advect(body.position,current,h,br::dt,basin);
    const auto exposed_food=ba::advect(body.position,current,h,br::dt,br::Basin{1});
    CHECK(sheltered_food.x>body.position.x && sheltered_food.x-body.position.x<(exposed_food.x-body.position.x)*.8f);
    const auto sampled=br::habitat_flow(body.position,h,current,basin);
    CHECK(br::norm((sheltered_food-body.position)*(br::scale/br::dt)-sampled)<2e-5f);
}

GPU_TEST_CASE("boulder habitat: CUDA CPU mixed retention observations live contact and resume agree") {
    auto selector=cudarl::nn::init_params(cudarl::rl::progress_selector_spec(),7);std::fill(selector.begin(),selector.end(),0.f);
    ba::Context context;context.current={.005f,0,0};context.version=4;context.basin.layout=2;
    cudarl::rl::SwimmerStream stream;ba::Batch mixed(8,7);mixed.configure(selector,context,stream.value,true);
    ba::CpuBatch reference(8,7,selector,context);std::vector<float> observations(8*ba::kObs);
    for(uint32_t environment=0;environment<8;++environment) {
        auto state=ba::initial(7,environment,0,environment%4u==0u,context.basin);state.pond.steps=0;
        const auto initial=ba::practice_current(state,context);state.pond.steps=100;
        const auto later=ba::practice_current(state,context);
        CHECK(br::norm(initial)<=.0050001f && br::norm(later)<=.0050001f);
        if(environment%4u<2u) CHECK(br::norm(initial)==0 && br::norm(later)==0);
        else if(environment<4u) CHECK(br::norm(initial-later)==0);
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));CUDA_CHECK_THROW(cudaMemcpy(observations.data(),mixed.obs(),observations.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<observations.size();++i) CHECK(fabsf(observations[i]-reference.observations[i])<3e-5f);
    ba::Batch gpu(1,7);gpu.configure(selector,context,stream.value);ba::CpuBatch cpu(1,7,selector,context);
    auto state=ba::initial(7,0,0,false,context.basin);state.pond.body.position=br::boulder_center(state.habitat)+br::V{br::boulder_radius(state.habitat)+.1f,0,0};
    state.pond.body.orientation=br::axis_rotation({0,1,0},3.14159265f);gpu.set_state(state,stream.value);cpu.states[0]=state;
    const float action[4]={.1f,-.1f,.1f,-.1f};device_ptr<float> packed(4);packed.copy_from_host(action,4);
    for(int tick=0;tick<40;++tick) {
        gpu.step(packed.get(),{},stream.value,false,false);cpu.step(action,nullptr,nullptr,nullptr,false,false);
        const auto actual=gpu.state(stream.value);const auto& expected=cpu.states[0];
        CHECK(br::norm(actual.pond.body.position-expected.pond.body.position)<.0015f);CHECK(actual.contacts==expected.contacts);
        CHECK(actual.pond.target==expected.pond.target && actual.pond.eaten==expected.pond.eaten);
        for(int food=0;food<2;++food) CHECK(br::norm(actual.pond.pellets[food].position-expected.pond.pellets[food].position)<.0003f);
    }
    state=ba::initial(7,0,0,false,context.basin);state.pond.body.position=br::boulder_center(state.habitat)+br::V{2*br::boulder_radius(state.habitat),.35f,0};
    state.pond.body.orientation=br::axis_rotation({0,1,0},3.14159265f);gpu.set_state(state,stream.value);cpu.states[0]=state;cpu.prepare();
    for(int tick=0;tick<20;++tick) {
        gpu.step(packed.get(),{},stream.value,false,false);cpu.step(action,nullptr,nullptr,nullptr,false,false);
        const auto actual=gpu.state(stream.value);const auto& expected=cpu.states[0];
        CHECK(br::norm(actual.pond.body.position-expected.pond.body.position)<.0015f);
        for(int food=0;food<2;++food) CHECK(br::norm(actual.pond.pellets[food].position-expected.pond.pellets[food].position)<.0003f);
    }
    cudarl::checkpoint::Writer saved;gpu.save(saved);ba::Batch resumed(1,7);resumed.configure(selector,stream.value);
    cudarl::checkpoint::Reader input{saved.data};resumed.load(input);input.finish();CHECK(resumed.context().basin.layout==2);
    gpu.step(packed.get(),{},stream.value,false,false);resumed.step(packed.get(),{},stream.value,false,false);
    const auto a=gpu.state(stream.value),b=resumed.state(stream.value);CHECK(br::norm(a.pond.body.position-b.pond.body.position)==0);
    CHECK(a.pond.eaten==b.pond.eaten && a.contacts==b.contacts);
}
