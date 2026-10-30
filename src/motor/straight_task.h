#ifndef CUDARL_STRAIGHT_TASK_H
#define CUDARL_STRAIGHT_TASK_H
#include "motor/stroke_motor.h"
// Task4 changes the lesson, not the mechanics/action generator. It deliberately
// reports head yaw separately from COM path direction. Earlier task3 remains 0/4.
namespace cudarl::straight_motor {
namespace p=physics;namespace stroke=stroke_motor;namespace old=motor;
using World=stroke::World;using Action=stroke::Action;using Outcome=old::Outcome;
using stroke::action_from_latent;using stroke::step_state;
constexpr int kTaskVersion=4,kObservations=42,kActions=2;
constexpr int kRewardVersion=2;
struct ForwardConfig {
    float target_speed_m_s=.008f,clearance_m=.01f;
    float minimum_speed_m_s=.0072f,maximum_speed_m_s=.0088f,speed_rmse_limit=.0025f;
    float lateral_limit_m=.002f,depth_limit_m=.001f,path_angle_limit_deg=5,tilt_limit_deg=10;
    float work_per_distance_limit=0.001f; // J/m = 1 microJ/mm
    int maximum_controls=80,measure_start=40;
};
struct Metrics {
    float forward_m=0,lateral_m=0,depth_m=0,speed_error_m_s=0,head_yaw_deg=0,tilt_deg=0;
};
struct ForwardTask {
    simulation::MotorRequest request{};
    p::V start_com_model{},steady_start_com_model{};
    uint64_t last_world_control=0;
    int controls=0,steady_samples=0;
    double velocity_error_sq=0,head_yaw_sq=0;
    float steady_speed_m_s=0,speed_rmse=0,path_angle_deg=0,maximum_lateral=0,maximum_depth=0,maximum_tilt=0,maximum_head_yaw=0;
    Outcome outcome=Outcome::Running;
};
struct Transition {Metrics metrics{};float reward=0;bool accepted=false,done=false,truncated=false;Outcome outcome=Outcome::Rejected;};
CUDARL_HD inline bool valid(const ForwardConfig& c){return c.target_speed_m_s>0&&p::finite_number(c.target_speed_m_s)&&c.minimum_speed_m_s>0&&c.maximum_speed_m_s>=c.minimum_speed_m_s&&
    c.speed_rmse_limit>0&&c.clearance_m>=.01f&&c.lateral_limit_m>0&&c.depth_limit_m>0&&c.path_angle_limit_deg>0&&c.tilt_limit_deg>0&&c.work_per_distance_limit>0&&c.measure_start>0&&c.maximum_controls>c.measure_start;}
CUDARL_HD inline bool in_domain(const World& w,const ForwardConfig& c){old::ForwardConfig domain;domain.clearance_m=c.clearance_m;return valid(c)&&stroke::in_domain(w,domain);}
CUDARL_HD inline bool begin_forward(const World& w,const ForwardConfig& c,ForwardTask& output){
    if(!in_domain(w,c))return false;const auto motion=old::motion(w.physical.body);auto heading=w.physical.body.orientation.rotate({1,0,0});heading=heading/p::norm(heading);
    ForwardTask next;next.start_com_model=motion.com_model;next.steady_start_com_model=motion.com_model;next.last_world_control=w.physical.control_steps;
    next.request={motion.com_model+heading*(.072f/p::scale),heading*c.target_speed_m_s,heading};output=next;return true;
}
CUDARL_HD inline bool observe(const ForwardTask& t,const World& w,const ForwardConfig& c,float* out){
    if(!out||t.outcome!=Outcome::Running||!in_domain(w,c))return false;old::ForwardTask view;view.request=t.request;view.controls=t.controls;old::ForwardConfig domain;domain.maximum_controls=c.maximum_controls;domain.clearance_m=c.clearance_m;
    if(!stroke::observe(view,w,domain,out))return false;
    const auto motion=old::motion(w.physical.body);const auto displacement=(motion.com_model-t.start_com_model)*p::scale;
    const auto lane_error=t.request.heading*p::dot(displacement,t.request.heading)-displacement;
    old::vector_features(out,0,w.physical.body.orientation.inverse(lane_error)/old::kBodyLength);return true;
}
CUDARL_HD inline Metrics measure(const ForwardTask& t,const World& w){const auto motion=old::motion(w.physical.body);const auto displacement=(motion.com_model-t.start_com_model)*p::scale;Metrics result;
    result.forward_m=p::dot(displacement,t.request.heading);const auto perpendicular=displacement-t.request.heading*result.forward_m;
    result.lateral_m=p::norm({perpendicular.x,0,perpendicular.z});result.depth_m=fabsf(displacement.y);result.speed_error_m_s=p::norm(motion.velocity_m_s-t.request.terminal_velocity);
    const auto forward=w.physical.body.orientation.rotate({1,0,0});const auto up=w.physical.body.orientation.rotate({0,1,0});
    result.head_yaw_deg=fabsf(atan2f(p::dot(p::cross(t.request.heading,forward),{0,1,0}),p::dot(t.request.heading,forward)))*180/movement::pi;
    result.tilt_deg=acosf(p::clamp(p::dot(up,{0,1,0}),-1,1))*180/movement::pi;return result;
}
CUDARL_HD inline Transition record_control(ForwardTask& t,const World& w,const ForwardConfig& c){Transition out;
    if(t.outcome!=Outcome::Running){out.done=true;out.outcome=t.outcome;return out;}
    if(!in_domain(w,c)||w.physical.control_steps!=t.last_world_control+1){t.outcome=out.outcome=Outcome::Rejected;out.done=true;return out;}
    out.metrics=measure(t,w);const float velocity_error=out.metrics.speed_error_m_s/c.target_speed_m_s;
    const float lane_error=out.metrics.lateral_m/c.lateral_limit_m;
    // Interval40 ends at2s; intervals41..80 are the scored2..4s window.
    // Warmup state/actions remain in the full return, with effort charged over
    // every interval. Physical work, safety and acceptance bookkeeping never reset.
    const bool scored=t.controls>=c.measure_start;
    const float tracking=velocity_error*velocity_error+.05f*lane_error*lane_error;
    out.reward=-p::dt*((scored?tracking:0.f)+.002f*w.physical.mean_squared_torque);
    if(!p::finite_number(out.reward)){t.outcome=out.outcome=Outcome::Rejected;out.done=true;return out;}
    ++t.controls;t.last_world_control=w.physical.control_steps;t.maximum_lateral=fmaxf(t.maximum_lateral,out.metrics.lateral_m);t.maximum_depth=fmaxf(t.maximum_depth,out.metrics.depth_m);t.maximum_tilt=fmaxf(t.maximum_tilt,out.metrics.tilt_deg);
    t.maximum_head_yaw=fmaxf(t.maximum_head_yaw,out.metrics.head_yaw_deg);t.head_yaw_sq+=out.metrics.head_yaw_deg*out.metrics.head_yaw_deg;
    const auto motion=old::motion(w.physical.body);
    if(t.controls==c.measure_start)t.steady_start_com_model=motion.com_model;
    if(t.controls>c.measure_start){++t.steady_samples;t.velocity_error_sq+=out.metrics.speed_error_m_s*out.metrics.speed_error_m_s;
        const auto delta=(motion.com_model-t.steady_start_com_model)*p::scale;const float forward=p::dot(delta,t.request.heading);const auto perpendicular=delta-t.request.heading*forward;
        t.steady_speed_m_s=forward/(float(t.steady_samples)*p::dt);t.speed_rmse=sqrtf(static_cast<float>(t.velocity_error_sq/t.steady_samples));
        t.path_angle_deg=atan2f(p::norm({perpendicular.x,0,perpendicular.z}),forward)*180/movement::pi;}
    if(t.controls>=c.maximum_controls){const float work_per_m=w.physical.body.work/fmaxf(out.metrics.forward_m,1e-9f);
        const bool success=t.steady_speed_m_s>=c.minimum_speed_m_s&&t.steady_speed_m_s<=c.maximum_speed_m_s&&t.speed_rmse<=c.speed_rmse_limit&&
            t.maximum_lateral<=c.lateral_limit_m&&t.maximum_depth<=c.depth_limit_m&&t.path_angle_deg<=c.path_angle_limit_deg&&t.maximum_tilt<=c.tilt_limit_deg&&work_per_m<=c.work_per_distance_limit;
        t.outcome=success?Outcome::Success:Outcome::Timeout;}
    out.accepted=true;out.outcome=t.outcome;out.done=t.outcome!=Outcome::Running;return out;
}
}
#endif
