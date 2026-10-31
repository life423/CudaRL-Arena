#include "check.h"
#include "motor/straight_task.h"
#include <limits>
namespace sm=cudarl::straight_motor;
TEST_CASE("stroke_motor: viewer arena extends horizontal clearance without changing benchmark or floor") {
    namespace contact=cudarl::contact;namespace p=cudarl::physics;namespace model=cudarl::movement;
    sm::World original;CHECK(std::fabs(original.arena.half_width_m-.18f)<1e-7f&&std::fabs(original.arena.half_depth_m-.14f)<1e-7f);
    auto viewer=original;viewer.arena={.6f,.4f};
    original.physical.body.position.x=9;viewer.physical.body=original.physical.body;
    CHECK(!cudarl::stroke_motor::in_domain(original));CHECK(cudarl::stroke_motor::in_domain(viewer));
    CHECK(model::clearance(viewer.physical.body,true,viewer.arena)>.01f);
    CHECK(contact::inspect(original.physical.body).floor_gap==contact::inspect(viewer.physical.body,viewer.arena).floor_gap);
    for(int face:{2,3})CHECK(contact::offset(face)==contact::offset(face,viewer.arena));
    CHECK(!contact::valid_arena({.1f,.4f}));CHECK(!contact::valid_arena({std::numeric_limits<float>::infinity(),.4f}));
    CHECK(original.attempted_substeps==0&&viewer.attempted_substeps==0);static_assert(p::torque==.00001f,"Actuator torque changed");
}
TEST_CASE("stroke_motor: invalid explicit references reject without physical work") {
    sm::World world;const float command[4]{};
    CHECK(!cudarl::stroke_motor::step_reference(world,-.1f,1,command));
    CHECK(!cudarl::stroke_motor::step_reference(world,.35f,-2,command));
    CHECK(!cudarl::stroke_motor::step_reference(world,.35f,std::numeric_limits<float>::quiet_NaN(),command));
    CHECK(world.attempted_substeps==0&&world.physical.control_steps==0);
}
TEST_CASE("stroke_motor: oscillator phase and filter rates are observed without stepping") {
    sm::World world;sm::ForwardConfig config;sm::ForwardTask task;CHECK(sm::begin_forward(world,config,task));
    float before[sm::kObservations]{},after[sm::kObservations]{};CHECK(sm::observe(task,world,config,before));
    world.phase=cudarl::movement::pi/2;world.rate[0]=.7f;CHECK(sm::observe(task,world,config,after));
    for(int j=0;j<36;++j)CHECK(before[j]==after[j]);CHECK(std::fabs(after[36]-1)<1e-6f);CHECK(std::fabs(after[37])<1e-6f);
    CHECK(after[40]>0&&before[40]==0);CHECK(world.attempted_substeps==0&&world.physical.substeps==0);
}
TEST_CASE("stroke_motor: corrupt hidden state rejects before observation or learning") {
    sm::World world;sm::ForwardConfig config;sm::ForwardTask task;CHECK(sm::begin_forward(world,config,task));float observation[sm::kObservations]{};
    world.phase=std::numeric_limits<float>::quiet_NaN();CHECK(!sm::observe(task,world,config,observation));world.phase=0;
    world.rate[1]=std::numeric_limits<float>::infinity();CHECK(!sm::observe(task,world,config,observation));CHECK(world.attempted_substeps==0);
}
