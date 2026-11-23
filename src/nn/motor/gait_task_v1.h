#ifndef CUDARL_GAIT_TASK_V1_H
#define CUDARL_GAIT_TASK_V1_H
#include "motor/straight_task.h"
namespace cudarl::gait_task_v1 {
namespace p=physics;namespace old=straight_motor;
using World=old::World;using Task=old::ForwardTask;
constexpr unsigned task_version=1,observation_version=4;
constexpr int observations=41,actions=4,controls=160,gait_measure_start=80;
constexpr float pitch_limit_degrees=10.f,gait_cost_weight=.05f,movement_failure_cost=1.f;
constexpr float minimum_centered_joint_rms_degrees=5.f;
CUDARL_HD inline old::ForwardConfig config(int maximum_controls=controls) {
    auto result=old::config_for_speed(.008f);result.observation_schema=4;
    result.maximum_controls=maximum_controls;return result;
}
CUDARL_HD inline bool initialize(int pose,World& world,Task& task,int maximum_controls=controls) {
    world=old::initial_world(pose);world.arena=old::long_swim_arena();
    return old::begin_forward(world,config(maximum_controls),task);
}
CUDARL_HD inline float tail_vertical_bend_degrees(const p::BodyState& state) {
    const auto links=p::shape(state);
    const auto tail_in_head=state.orientation.inverse(links.link[2].q.rotate({1,0,0}));
    return atan2f(tail_in_head.y,sqrtf(tail_in_head.x*tail_in_head.x+
                                     tail_in_head.z*tail_in_head.z))*180.f/movement::pi;
}
struct Measurements {
    double pitch_sum[2]{},tail_bend_sum=0;
    double pitch_square_sum[2]{},tail_bend_square_sum=0;
    double angle_sum[4]{},angle_square_sum[4]{};
    float pitch_peak[2]{},tail_bend_peak=0;
    float yaw_min[2]{},yaw_max[2]{};
    int samples=0;
    CUDARL_HD bool record(int control,const p::BodyState& body) {
        if(control<=gait_measure_start)return true;
        const float pitches[2]={body.angle[1]*180.f/movement::pi,body.angle[3]*180.f/movement::pi};
        const float bend=tail_vertical_bend_degrees(body);
        if(!p::finite_number(bend))return false;
        for(int j=0;j<2;++j){
            if(!p::finite_number(pitches[j]))return false;
            pitch_sum[j]+=pitches[j];pitch_square_sum[j]+=double(pitches[j])*pitches[j];
            pitch_peak[j]=fmaxf(pitch_peak[j],fabsf(pitches[j]));
            const float yaw=body.angle[j*2];
            if(samples==0)yaw_min[j]=yaw_max[j]=yaw;
            else {yaw_min[j]=fminf(yaw_min[j],yaw);yaw_max[j]=fmaxf(yaw_max[j],yaw);}
        }
        tail_bend_sum+=bend;tail_bend_square_sum+=double(bend)*bend;
        for(int j=0;j<4;++j){angle_sum[j]+=body.angle[j];angle_square_sum[j]+=double(body.angle[j])*body.angle[j];}
        tail_bend_peak=fmaxf(tail_bend_peak,fabsf(bend));++samples;return true;
    }
    CUDARL_HD double mean_pitch(int joint) const {return samples?pitch_sum[joint]/samples:0;}
    CUDARL_HD double mean_bend() const {return samples?tail_bend_sum/samples:0;}
    CUDARL_HD double centered_joint_rms_degrees() const {
        if(!samples)return 0;
        double sum=0;for(int j=0;j<4;++j){const double mean=angle_sum[j]/samples;
            sum+=fmax(0.,angle_square_sum[j]/samples-mean*mean);}
        return sqrt(sum/4)*180./movement::pi;
    }
    CUDARL_HD double cost() const {
        const double a=mean_pitch(0)/pitch_limit_degrees,b=mean_pitch(1)/pitch_limit_degrees,
                     c=mean_bend()/pitch_limit_degrees;
        return gait_cost_weight*(a*a+b*b+c*c);
    }
    CUDARL_HD bool passed() const {
        return samples>0&&fabs(mean_pitch(0))<=pitch_limit_degrees&&
               fabs(mean_pitch(1))<=pitch_limit_degrees&&fabs(mean_bend())<=pitch_limit_degrees&&
               centered_joint_rms_degrees()>=minimum_centered_joint_rms_degrees;
    }
};
}
#endif
