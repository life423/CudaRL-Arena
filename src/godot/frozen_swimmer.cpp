#include "frozen_swimmer.h"
#include "motor/frozen_controller.h"
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector3.hpp>
namespace godot {
namespace p=cudarl::physics;namespace m=cudarl::movement;
struct FrozenSwimmer::Impl {cudarl::motor::FrozenController controller;};
FrozenSwimmer::FrozenSwimmer():impl_(std::make_unique<Impl>()){}FrozenSwimmer::~FrozenSwimmer()=default;
void FrozenSwimmer::_bind_methods(){ClassDB::bind_method(D_METHOD("load_controller"),&FrozenSwimmer::load_controller);ClassDB::bind_method(D_METHOD("reset","case_id"),&FrozenSwimmer::reset,DEFVAL(0));
    ClassDB::bind_method(D_METHOD("start"),&FrozenSwimmer::start);ClassDB::bind_method(D_METHOD("pause"),&FrozenSwimmer::pause);ClassDB::bind_method(D_METHOD("step"),&FrozenSwimmer::step);
    ClassDB::bind_method(D_METHOD("get_state"),&FrozenSwimmer::get_state);ClassDB::bind_method(D_METHOD("get_geometry"),&FrozenSwimmer::get_geometry);ClassDB::bind_method(D_METHOD("get_error"),&FrozenSwimmer::get_error);}
bool FrozenSwimmer::load_controller(){return impl_->controller.load(ProjectSettings::get_singleton()->globalize_path("res://assets/frozen-controller.bin").utf8().get_data());}
bool FrozenSwimmer::reset(int id){return impl_->controller.reset(id);}bool FrozenSwimmer::start(){return impl_->controller.start();}void FrozenSwimmer::pause(){impl_->controller.pause();}bool FrozenSwimmer::step(){return impl_->controller.step();}
String FrozenSwimmer::get_error()const{return String(impl_->controller.error().c_str());}
Dictionary FrozenSwimmer::get_geometry()const{Dictionary out;out["model"]=6;out["scale_m"]=p::scale;out["rod_radius_m"]=m::Geometry::rod_radius;out["fin_extension_m"]=m::Geometry::fin_extension;out["fin_thickness_m"]=m::Geometry::fin_thickness;
    Array links;for(int b=0;b<3;++b){const auto g=m::body_geometry(b);Dictionary d;d["length"]=g.length;d["half_height"]=g.half_height;d["half_width"]=g.half_width;links.append(d);}out["links"]=links;return out;}
Dictionary FrozenSwimmer::get_state()const{const auto& c=impl_->controller;const auto& w=c.world();const auto& b=w.physical.body;const auto metrics=c.metrics();Dictionary out;
    out["loaded"]=c.loaded();out["paused"]=c.paused();out["finished"]=c.finished();out["error"]=get_error();out["model"]=6;out["training"]=false;
    out["controls"]=static_cast<int64_t>(w.physical.control_steps);out["substeps"]=static_cast<int64_t>(w.attempted_substeps);out["time_s"]=w.physical.time_seconds;
    out["case"]=c.case_id();out["passed"]=c.task().outcome==cudarl::straight_motor::Outcome::Success;out["forward_m"]=metrics.forward_m;out["depth_m"]=metrics.depth_m;out["lateral_m"]=metrics.lateral_m;
    out["speed_error_m_s"]=metrics.speed_error_m_s;out["speed_m_s"]=p::norm(cudarl::motor::motion(b).velocity_m_s);out["steady_speed_m_s"]=c.task().steady_speed_m_s;out["head_yaw_deg"]=metrics.head_yaw_deg;
    out["head_x"]=b.position.x;out["head_y"]=b.position.y;out["head_z"]=b.position.z;out["qx"]=b.orientation.x;out["qy"]=b.orientation.y;out["qz"]=b.orientation.z;out["qw"]=b.orientation.w;
    for(int j=0;j<4;++j)out[String("j")+String::num_int64(j)]=b.angle[j];
    out["phase"]=w.phase;out["amplitude"]=w.parameter[0];out["frequency"]=w.parameter[1];out["positive_work_J"]=b.work;
    Array links;for(const auto& link:m::shape(b).link){Dictionary d;d["position"]=Vector3(link.center.x/p::scale,link.center.y/p::scale,link.center.z/p::scale);d["orientation"]=Quaternion(link.q.x,link.q.y,link.q.z,link.q.w);links.append(d);}out["links"]=links;return out;}
}
