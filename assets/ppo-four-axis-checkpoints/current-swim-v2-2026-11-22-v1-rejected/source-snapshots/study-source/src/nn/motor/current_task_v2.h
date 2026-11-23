#ifndef CUDARL_CURRENT_TASK_V2_H
#define CUDARL_CURRENT_TASK_V2_H
#include "motor/current_motor_v2.h"
#include "motor/straight_task.h"
namespace cudarl::current_task_v2 {
namespace p=physics;namespace old=straight_motor;namespace motor=current_motor_v2;
using World=motor::World;using Task=old::ForwardTask;using Config=old::ForwardConfig;
constexpr uint32_t task_version=2,observation_version=5;
constexpr int observations=44,backbone_observations=41,actions=4;
constexpr float flow_scale_m_s=.002f;
CUDARL_HD inline Config config(int controls=160) {auto c=old::config_for_speed(.008f);c.observation_schema=4;c.maximum_controls=controls;return c;}
CUDARL_HD inline bool in_domain(const World& world) {return motor::in_domain(world);}
// Read-only domain adaptation for historical pure task functions. Actual flow
// has already passed the new current domain. This temporary view is never
// integrated or copied back into the physical world; all physics sees real flow.
CUDARL_HD inline World calm_task_view(const World& world) {auto view=world;view.physical.current={};return view;}
CUDARL_HD inline bool initialize(int pose,p::V flow,World& world,Task& task,int controls=160) {
    auto next=old::initial_world(pose);next.arena=old::long_swim_arena();next.physical.current=flow;
    if(!motor::in_domain(next))return false;
    const auto view=calm_task_view(next);Task initialized;
    if(!old::begin_forward(view,config(controls),initialized))return false;
    world=next;task=initialized;return true;
}
CUDARL_HD inline bool observe(const Task& task,const World& world,float* backbone,float* body_flow,int controls=160) {
    if(!backbone||!body_flow||!motor::in_domain(world))return false;
    const auto view=calm_task_view(world);
    if(!old::observe_appended(task,view,config(controls),backbone))return false;
    const auto local=world.physical.body.orientation.inverse(world.physical.current)/flow_scale_m_s;
    body_flow[0]=local.x;body_flow[1]=local.y;body_flow[2]=local.z;
    for(int j=0;j<3;++j)if(!p::finite_number(body_flow[j]))return false;
    return true;
}
CUDARL_HD inline old::Transition record(Task& task,const World& world,int controls=160) {
    if(!motor::in_domain(world))return {};
    const auto view=calm_task_view(world);return old::record_control(task,view,config(controls));
}
}
#endif
