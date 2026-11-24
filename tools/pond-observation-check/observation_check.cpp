// Zero-integration migration check: construct states, compare inputs/metrics.
#include "motor/pond_motor_observation.h"
#include "motor/straight_task.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace fresh=cudarl::pond_motor_observation;
namespace old=cudarl::straight_motor;
namespace p=cudarl::physics;
namespace {
int assertions=0,fixtures=0,observation_arrays=0;
void require(bool condition,const char* message) {++assertions;if(!condition)throw std::runtime_error(message);}
void same(float a,float b,const char* message) {require(std::memcmp(&a,&b,sizeof(a))==0,message);}
void same(p::V a,p::V b,const char* message) {same(a.x,b.x,message);same(a.y,b.y,message);same(a.z,b.z,message);}
void compare_metrics(const fresh::Metrics& a,const old::Metrics& b) {
    same(a.forward_m,b.forward_m,"forward metric differs");same(a.lateral_m,b.lateral_m,"lane metric differs");
    same(a.depth_m,b.depth_m,"depth metric differs");same(a.speed_error_m_s,b.speed_error_m_s,"speed metric differs");
    same(a.head_yaw_deg,b.head_yaw_deg,"head metric differs");same(a.tilt_deg,b.tilt_deg,"tilt metric differs");
    same(a.signed_depth_m,b.signed_depth_m,"signed depth metric differs");
    same(a.vertical_velocity_m_s,b.vertical_velocity_m_s,"vertical velocity metric differs");
    same(a.horizontal_speed_error_m_s,b.horizontal_speed_error_m_s,"horizontal speed metric differs");
    same(a.roll_deg,b.roll_deg,"roll metric differs");same(a.pitch_deg,b.pitch_deg,"pitch metric differs");
    same(a.hold_error_m,b.hold_error_m,"anchor metric differs");same(a.velocity_m_s,b.velocity_m_s,"velocity metric differs");
}
void fixture(int id,bool holding) {
    ++fixtures;
    auto world=old::initial_world(holding?(id%2?3:0):id);world.arena=old::long_swim_arena();
    auto config=old::config_for_speed(.008f);config.observation_schema=4;config.maximum_controls=holding?1280:160;
    config.target_speed_m_s=holding?0.f:.008f;config.stationary_request=holding;config.position_hold=holding;
    old::ForwardTask task;
    require(old::begin_forward(world,config,task),"reference fixture rejected");
    if(holding) {
        require(old::offset_episode_start(world,task,config,id/2==1?-.0005f:id/2==2?.0005f:0.f),"reference depth fixture rejected");
        require(old::update_velocity_request(world,config,task),"reference hold request rejected");
    }
    p::BodyState body;float targets[4]{},rates[4]{},torques[4]{};fresh::RequestState state;
    require(fresh::initialize(id,holding,body,targets,rates,torques,state),"new fixture rejected");
    same(body.position,world.physical.body.position,"initial body position differs");
    same(body.orientation.x,world.physical.body.orientation.x,"initial quaternion differs");
    same(body.orientation.y,world.physical.body.orientation.y,"initial quaternion differs");
    same(body.orientation.z,world.physical.body.orientation.z,"initial quaternion differs");
    same(body.orientation.w,world.physical.body.orientation.w,"initial quaternion differs");
    same(state.start_com_model,task.start_com_model,"request origin differs");
    same(state.request.position,task.request.position,"request position differs");
    same(state.request.heading,task.request.heading,"request heading differs");
    same(state.request.terminal_velocity,task.request.terminal_velocity,"request velocity differs");
    same(state.hold_anchor_model,task.hold_anchor_model,"hold anchor differs");
    require(state.hold_anchor_set==task.hold_anchor_set,"hold anchor presence differs");
    for(int j=0;j<4;++j) {
        same(body.angle[j],world.physical.body.angle[j],"initial joint differs");
        same(targets[j],world.physical.held_targets[j],"initial filter target differs");
        same(rates[j],world.reference_rate[j],"initial filter rate differs");
        same(torques[j],world.physical.last_actual_torques[j],"initial torque differs");
    }
    // These are assigned finite fixtures, never integrated trajectories.
    const std::array<int,7> counters{0,1,39,40,41,160,1280};
    for(int variant=0;variant<4;++variant) {
        auto varied=world;auto varied_body=body;
        if(variant>0) {
            varied_body.position.x+=float(variant)*.007f;varied_body.position.y-=float(variant)*.0008f;
            varied_body.position.z-=float(variant)*.0011f;
            for(int j=0;j<10;++j)varied_body.velocity[j]=float(j-5)*float(variant)*.0001f;
            for(int j=0;j<4;++j) {
                varied_body.angle[j]+=float(j-2)*float(variant)*.002f;
                varied.physical.held_targets[j]=targets[j]+float(j-1)*float(variant)*.001f;
                varied.reference_rate[j]=float(j-1)*float(variant)*.003f;
                varied.physical.last_actual_torques[j]=float(j-2)*float(variant)*.04f;
            }
            varied_body.orientation=varied_body.orientation*p::axis_rotation({0,0,1},float(variant)*.004f);
        }
        varied.physical.body=varied_body;
        for(int counter:counters) {
            auto varied_task=task;auto varied_request=state;
            varied_task.controls=counter;varied_request.controls=static_cast<uint64_t>(counter);
            float a[41]{},b[41]{};
            require(old::observe_appended(varied_task,varied,config,a),"reference input fixture rejected");
            require(fresh::observe(varied_request,varied_body,varied.physical.held_targets,varied.reference_rate,
                varied.physical.last_actual_torques,b),"new input fixture rejected");
            ++observation_arrays;
            for(int j=0;j<41;++j)same(a[j],b[j],"saved actor input differs bitwise");
            compare_metrics(fresh::measure(varied_request,varied_body),old::measure(varied_task,varied));
        }
    }
}
}
int main() {
    try {
        for(int id=0;id<6;++id)fixture(id,true);
        for(int id=0;id<4;++id)fixture(id,false);
        p::BodyState body;float targets[4]{},rates[4]{},torques[4]{},output[41]{};fresh::RequestState state;
        require(!fresh::initialize(-1,false,body,targets,rates,torques,state),"negative fixture accepted");
        require(!fresh::initialize(4,false,body,targets,rates,torques,state),"unavailable swim fixture accepted");
        require(!fresh::initialize(6,true,body,targets,rates,torques,state),"unavailable hold fixture accepted");
        require(fresh::initialize(0,false,body,targets,rates,torques,state),"valid fixture rejected");
        body.position.x=std::numeric_limits<float>::infinity();
        require(!fresh::observe(state,body,targets,rates,torques,output),"nonfinite body accepted");
        // Translation beyond the old guard must not be a hidden adapter domain.
        require(fresh::initialize(0,false,body,targets,rates,torques,state),"valid fixture rejected");
        body.position.x+=20;state.start_com_model.x+=20;state.request.position.x+=20;
        require(fresh::observe(state,body,targets,rates,torques,output),"observation retained old spatial guard");
        std::cout<<"{\"passed\":true,\"assertions\":"<<assertions<<",\"initial_fixtures\":"<<fixtures
            <<",\"bitwise_observation_arrays\":"<<observation_arrays
            <<",\"physical_controls\":0,\"physical_substeps\":0,\"CUDA_kernels\":0,\"optimizer_steps\":0}\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
