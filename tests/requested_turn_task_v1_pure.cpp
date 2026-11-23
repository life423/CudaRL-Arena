// Synthetic geometry, observation and measurement fixtures only. No call to a
// physical stepper, neural network, CUDA kernel or optimizer occurs here.
#include "motor/requested_turn_task_v1.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace task=cudarl::requested_turn_task_v1;
namespace old=cudarl::straight_motor;
namespace p=cudarl::physics;
int main(){int checks=0;try{
    const auto check=[&](bool value,const char* why){if(!value)throw std::runtime_error(why);++checks;};
    check(task::curriculum_limit(0)==90&&task::curriculum_limit(1)==15&&task::curriculum_limit(8)==15&&
        task::curriculum_limit(9)==30&&task::curriculum_limit(16)==30&&task::curriculum_limit(17)==60&&
        task::curriculum_limit(24)==60&&task::curriculum_limit(25)==90&&task::curriculum_limit(32)==90&&task::curriculum_limit(33)==0,"Curriculum contract differs");
    for(bool withheld:{false,true})for(float cap:{15.f,30.f,60.f,90.f})for(int id=0;id<24;++id){
        task::World world;task::Task state;check(task::initialize(id,world,state,cap,withheld),"Initial case rejected");
        const auto fixture=task::case_for(id,cap,withheld);const auto center=cudarl::motor::motion(world.physical.body).com_model;
        check(std::abs(center.x*p::scale)<1e-8&&std::abs(center.z*p::scale)<1e-8,"Initial COM was not centered horizontally");
        check(fixture.pose==(id<12?(withheld?1:0):(withheld?2:3))&&std::abs(fixture.effective_offset_deg)<=cap,"Case pose/angle cap differs");
        check(world.attempted_substeps==0&&world.physical.substeps==0&&world.physical.control_steps==0&&state.controls==0,"Initialization advanced physics");
        float observation[42],legacy[41];check(task::observe(state,world,observation)&&old::observe_appended(state.forward,world,task::config(state.holding),legacy),"Initial observation rejected");
        check(std::equal(observation,observation+41,legacy)&&observation[41]==0,"Flag-zero inputs changed the old41 meanings");
        check(state.holding==(id%12==0)&&!state.heading_only&&!state.request_latched,"Initial intent flags differ");
        // Artificial counters for a pure request-latch fixture, not a rollout.
        state.controls=state.last_world_control=80;state.forward.controls=80;state.forward.last_world_control=80;
        world.physical.control_steps=80;world.physical.substeps=world.attempted_substeps=800;world.physical.time_seconds=800*.005f;
        const auto before=world;const auto old_request=state.forward;
        check(task::update_request(state,world),"Persistent request latch rejected");
        check(std::memcmp(&world,&before,sizeof(world))==0,"Request changed a physical state");
        check(state.controls==80&&state.forward.controls==80&&state.forward.last_world_control==80,"Request reset task clocks");
        if(state.kind!=task::Turn){check(std::memcmp(&old_request,&state.forward,sizeof(old_request))==0&&!state.heading_only,"Rest/Swim zero-offset intent changed");}
        else{
            check(state.heading_only&&state.request_latched&&state.forward.start_com_model.x==old_request.start_com_model.x&&
                state.forward.start_com_model.y==old_request.start_com_model.y&&state.forward.start_com_model.z==old_request.start_com_model.z,"Turn rebased the original lane/depth anchor");
            const auto heading=p::axis_rotation({0,1,0},fixture.effective_offset_deg*cudarl::movement::pi/180.f).rotate(state.original_heading);
            check(p::norm(state.forward.request.heading-heading)<1e-7f,"Requested +Y course sign differs");
            check(task::observe(state,world,observation)&&observation[41]==1,"Heading-only observation missing");
            auto view=state.forward;view.start_com_model.x=center.x;view.start_com_model.z=center.z;
            check(old::observe_appended(view,world,task::config(false),legacy)&&std::equal(observation,observation+41,legacy),"Heading-only did not mask WORLD XZ lane error");
        }
    }
    // A tilted synthetic input proves vertical goal feedback can occupy the
    // local X channel; clearing local X/Z instead of world X/Z is incorrect.
    task::World world;task::Task state;check(task::initialize(2,world,state),"Tilted sensor fixture init rejected");
    state.controls=state.last_world_control=80;state.forward.controls=80;state.forward.last_world_control=80;world.physical.control_steps=80;
    check(task::update_request(state,world),"Tilted sensor latch rejected");
    world.physical.body.orientation=p::axis_rotation({0,0,1},.12f)*world.physical.body.orientation;
    world.physical.body.position.y+=.0004f/p::scale;
    float observation[42];check(task::observe(state,world,observation)&&std::abs(observation[0])>1e-5f,"Heading-only removed depth projected onto local X");

    check(task::initialize(1,world,state),"Quality fixture init rejected");
    world.physical.body.velocity[0]=state.forward.request.terminal_velocity.x;
    world.physical.body.velocity[1]=state.forward.request.terminal_velocity.y;
    world.physical.body.velocity[2]=state.forward.request.terminal_velocity.z;
    const double ideal=task::quality(state,world,0);check(ideal>.99999&&ideal<=1,"Ideal moving quality differs");
    const double effort=task::quality(state,world,.00001);check(effort<ideal&&effort>=0,"Positive-power quality cost differs");
    world.physical.body.position.z+=.001f/p::scale;
    const double tracked=task::quality(state,world,0);state.heading_only=true;
    const double heading_only=task::quality(state,world,0);check(tracked<heading_only&&heading_only<=ideal+1e-6,"Flag-zero lane quality cost was not retained");
    check(task::quality(state,world,-1)<0,"Negative positive-power input accepted");
    task::bank::Bounds boundary;boundary.inset_radius_squared=1;
    check(task::behavior_reason(.006f,25,boundary)==task::Boundary,"Bank guard priority differs");
    boundary.inset_radius_squared=0;
    check(task::behavior_reason(.006f,25,boundary)==task::Depth&&task::behavior_reason(.005f,21,boundary)==task::Tilt&&
        task::behavior_reason(.005f,20,boundary)==task::Running,"Depth/tilt strict limits differ");

    // These records contain manually supplied counters and measurement values.
    // They test task algebra only; none is asserted to be a physical trajectory.
    task::StepInfo info;info.valid=true;info.accepted_substeps=info.attempted_substeps=info.substeps=10;info.seconds=info.dt=10*.005f;
    check(task::initialize(0,world,state),"Rest record fixture init rejected");
    world.physical.control_steps=1;
    const auto quiet=task::record(state,world,info);
    check(quiet.accepted&&!quiet.done&&quiet.quality>.99999&&std::abs(quiet.reward-info.seconds/12)<1e-7,"Survivor reward differs");
    check(task::initialize(1,world,state),"Stall record fixture init rejected");
    bool premature=false;task::Transition transition;
    for(int control=1;control<=140;++control){world.physical.control_steps=uint64_t(control);transition=task::record(state,world,info);if(control<140)premature=premature||!transition.accepted||transition.done;}
    check(!premature&&transition.accepted&&transition.done&&transition.reason==task::Stalled&&state.controls==140,"Moving stall did not begin at nominal seven seconds");
    const double expected_failure=-(task::remaining_seconds(state)+4)/12;
    check(std::abs(transition.reward-expected_failure)<1e-7,"Remaining-time cost did not apply exactly once after the terminal transition");
    const auto sum=state.reward_sum;check(!task::record(state,world,info).accepted&&state.reward_sum==sum,"Terminal episode accepted more reward");
    check(task::initialize(0,world,state),"Rest deadline fixture init rejected");
    bool early=false;
    for(int control=1;control<=240;++control){world.physical.control_steps=uint64_t(control);transition=task::record(state,world,info);if(control<240)early=early||!transition.accepted||transition.done;}
    check(!early&&transition.accepted&&transition.done&&!transition.truncated&&transition.reason==task::Deadline&&!transition.behavior_terminal,"Rest stalled or deadline was bootstrapped");
    check(std::abs(state.reward_sum-1)<1e-6,"Bounded quiet-Rest episode return differs");
    check(task::initialize(1,world,state),"Partial record fixture init rejected");
    info.accepted_substeps=info.attempted_substeps=info.substeps=3;info.seconds=info.dt=3*.005f;info.behavior_terminal=true;info.terminal_reason=task::Depth;
    info.maximum_depth_m=.0051f;info.maximum_tilt_deg=7;info.minimum_clearance_m=.02f;world.physical.control_steps=1;
    transition=task::record(state,world,info);
    check(transition.accepted&&transition.done&&transition.reason==task::Depth&&transition.substeps==3&&state.elapsed_seconds==double(info.seconds),"Partial terminal time/accounting differs");
    check(state.maximum_depth_m==info.maximum_depth_m&&state.maximum_tilt_deg==info.maximum_tilt_deg&&state.minimum_clearance_m<=info.minimum_clearance_m,"Substep extrema were not propagated");
    std::cout<<"{\"passed\":true,\"checks\":"<<checks<<",\"physical_controls_executed\":0,\"physical_substeps_executed\":0,\"GPU_kernels\":0,\"optimizer_steps\":0,\"synthetic_records_only\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 3;}}
