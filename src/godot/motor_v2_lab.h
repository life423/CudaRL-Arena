#ifndef CUDARL_GODOT_MOTOR_V2_LAB_H
#define CUDARL_GODOT_MOTOR_V2_LAB_H
#include <memory>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
namespace godot {
class MotorV2Lab:public RefCounted {
    GDCLASS(MotorV2Lab,RefCounted);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    String error_;
protected:
    static void _bind_methods();
public:
    MotorV2Lab();
    ~MotorV2Lab() override;
    bool configure(String study_root);
    Dictionary get_controller_info(int arm) const;
    Array get_cases() const;
    bool start_trial(int case_id,int arm);
    Dictionary step();
    Dictionary get_state() const;
    String get_error() const {return error_;}
};
}
#endif
