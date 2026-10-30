#ifndef CUDARL_GODOT_MOTOR_CONTROL_LAB_H
#define CUDARL_GODOT_MOTOR_CONTROL_LAB_H
#include <memory>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
namespace godot {
class MotorControlLab:public RefCounted {
    GDCLASS(MotorControlLab,RefCounted);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    String error_;
protected:
    static void _bind_methods();
public:
    MotorControlLab();
    ~MotorControlLab() override;
    bool configure(String original,String calibrated,String candidate="",String paired_root="");
    bool start_trial(int suite,int index,int controller);
    Dictionary step();
    Dictionary get_state() const;
    Array get_cases(int suite) const;
    bool has_candidate() const;
    Dictionary get_controller_info(int controller) const;
    String get_error() const {return error_;}
};
}
#endif
