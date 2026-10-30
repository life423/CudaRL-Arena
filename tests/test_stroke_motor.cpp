#include "check.h"
#include "motor/stroke_motor.h"
#include <limits>
namespace sm=cudarl::stroke_motor;
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
