#include "frozen_swimmer.h"
#include "motor/frozen_controller.h"
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector3.hpp>
namespace godot {
namespace p=cudarl::physics;namespace m=cudarl::movement;
struct FrozenSwimmer::Impl {cudarl::motor::FrozenController controller;};
FrozenSwimmer::FrozenSwimmer():impl_(std::make_unique<Impl>()){}FrozenSwimmer::~FrozenSwimmer()=default;
void FrozenSwimmer::_bind_methods(){ClassDB::bind_method(D_METHOD("set_speed","speed_m_s"),&FrozenSwimmer::set_speed);ClassDB::bind_method(D_METHOD("load_controller"),&FrozenSwimmer::load_controller);ClassDB::bind_method(D_METHOD("reset","case_id"),&FrozenSwimmer::reset,DEFVAL(0));
    ClassDB::bind_method(D_METHOD("set_continuous","enabled"),&FrozenSwimmer::set_continuous);
    ClassDB::bind_method(D_METHOD("start"),&FrozenSwimmer::start);ClassDB::bind_method(D_METHOD("pause"),&FrozenSwimmer::pause);ClassDB::bind_method(D_METHOD("step"),&FrozenSwimmer::step);
    ClassDB::bind_method(D_METHOD("get_state"),&FrozenSwimmer::get_state);ClassDB::bind_method(D_METHOD("get_geometry"),&FrozenSwimmer::get_geometry);ClassDB::bind_method(D_METHOD("get_error"),&FrozenSwimmer::get_error);}
bool FrozenSwimmer::load_controller(){return impl_->controller.load(ProjectSettings::get_singleton()->globalize_path("res://assets/frozen-controller.bin").utf8().get_data())&&impl_->controller.configure_arena({.6f,.4f});}
bool FrozenSwimmer::set_speed(double speed){return impl_->controller.set_speed(static_cast<float>(speed));}
void FrozenSwimmer::set_continuous(bool enabled){impl_->controller.set_continuous(enabled);}
bool FrozenSwimmer::reset(int id){return impl_->controller.reset(id);}bool FrozenSwimmer::start(){return impl_->controller.start();}void FrozenSwimmer::pause(){impl_->controller.pause();}bool FrozenSwimmer::step(){return impl_->controller.step();}
String FrozenSwimmer::get_error()const{return String(impl_->controller.error().c_str());}
Dictionary FrozenSwimmer::get_geometry()const{Dictionary out;out["model"]=6;out["scale_m"]=p::scale;out["rod_radius_m"]=m::Geometry::rod_radius;out["fin_extension_m"]=m::Geometry::fin_extension;out["fin_thickness_m"]=m::Geometry::fin_thickness;
    out["arena_width_m"]=2*impl_->controller.arena().half_width_m;out["arena_depth_m"]=2*impl_->controller.arena().half_depth_m;out["floor_y_m"]=cudarl::contact::bounds_min_model().y*p::scale;
    Array links;for(int b=0;b<3;++b){const auto g=m::body_geometry(b);Dictionary d;d["length"]=g.length;d["half_height"]=g.half_height;d["half_width"]=g.half_width;links.append(d);}out["links"]=links;return out;}
Dictionary FrozenSwimmer::get_state()const{const auto& c=impl_->controller;const auto& w=c.world();const auto& b=w.physical.body;const auto metrics=c.metrics();Dictionary out;
    out["loaded"]=c.loaded();out["paused"]=c.paused();out["finished"]=c.finished();out["error"]=get_error();out["model"]=6;out["training"]=false;
    out["segment_time_s"]=c.task().controls*p::dt;out["controls"]=static_cast<int64_t>(c.viewing_controls());out["substeps"]=static_cast<int64_t>(c.viewing_substeps());out["time_s"]=c.viewing_controls()*p::dt;
    out["continuous"]=c.continuous();out["loops"]=static_cast<int64_t>(c.loops());out["benchmark_complete"]=c.benchmark_complete();out["episode_time_s"]=w.physical.time_seconds;
    out["target_speed_m_s"]=c.target_speed();out["minimum_speed_m_s"]=c.minimum_speed();out["maximum_speed_m_s"]=c.maximum_speed();
    out["action_mapping"]=c.action_mapping();out["speed_rmse_m_s"]=c.task().speed_rmse;out["tilt_deg"]=metrics.tilt_deg;
    out["segment_work_J"]=b.work-c.task().start_work_J;
    out["arena_width_m"]=2*w.arena.half_width_m;out["arena_depth_m"]=2*w.arena.half_depth_m;
    const auto gates=cudarl::straight_motor::config_for_speed(c.target_speed());const auto& task=c.task();Dictionary score;
    score["task_controls"]=task.controls;score["maximum_controls"]=c.benchmark_controls();score["complete"]=task.controls>=c.benchmark_controls();
    out["substeps_per_control"]=c.substeps_per_control();out["training_update"]=c.training_update();out["observation_schema"]=c.observation_schema();
    score["outcome"]=static_cast<int>(task.outcome);score["steady_speed_m_s"]=task.steady_speed_m_s;score["minimum_speed_m_s"]=gates.minimum_speed_m_s;score["maximum_speed_m_s"]=gates.maximum_speed_m_s;
    score["speed_rmse_m_s"]=task.speed_rmse;score["speed_rmse_limit"]=gates.speed_rmse_limit;score["maximum_lateral_m"]=task.maximum_lateral;score["lateral_limit_m"]=gates.lateral_limit_m;
    score["maximum_depth_m"]=task.maximum_depth;score["depth_limit_m"]=gates.depth_limit_m;score["path_angle_deg"]=task.path_angle_deg;score["path_angle_limit_deg"]=gates.path_angle_limit_deg;
    score["maximum_tilt_deg"]=task.maximum_tilt;score["tilt_limit_deg"]=gates.tilt_limit_deg;score["current_work_J_per_m"]=(b.work-task.start_work_J)/fmaxf(metrics.forward_m,1e-9f);score["work_J_per_m_limit"]=gates.work_per_distance_limit;
    score["current_clearance_m"]=m::clearance(b,true,w.arena);out["score"]=score;
    out["case"]=c.case_id();out["passed"]=c.task().outcome==cudarl::straight_motor::Outcome::Success;out["forward_m"]=metrics.forward_m;out["depth_m"]=metrics.depth_m;out["lateral_m"]=metrics.lateral_m;
    out["speed_error_m_s"]=metrics.speed_error_m_s;out["speed_m_s"]=p::norm(cudarl::motor::motion(b).velocity_m_s);out["steady_speed_m_s"]=c.task().steady_speed_m_s;out["head_yaw_deg"]=metrics.head_yaw_deg;
    out["head_x"]=b.position.x;out["head_y"]=b.position.y;out["head_z"]=b.position.z;out["qx"]=b.orientation.x;out["qy"]=b.orientation.y;out["qz"]=b.orientation.z;out["qw"]=b.orientation.w;
    for(int j=0;j<4;++j)out[String("j")+String::num_int64(j)]=b.angle[j];
    out["positive_work_J"]=b.work;for(int j=0;j<4;++j){out[String("target")+String::num_int64(j)]=w.physical.held_targets[j];out[String("target_rate")+String::num_int64(j)]=w.reference_rate[j];}
    Array links;for(const auto& link:m::shape(b).link){Dictionary d;d["position"]=Vector3(link.center.x/p::scale,link.center.y/p::scale,link.center.z/p::scale);d["orientation"]=Quaternion(link.q.x,link.q.y,link.q.z,link.q.w);links.append(d);}out["links"]=links;return out;}
}
