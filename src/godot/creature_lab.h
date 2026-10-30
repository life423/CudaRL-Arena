#ifndef CUDARL_GODOT_CREATURE_LAB_H
#define CUDARL_GODOT_CREATURE_LAB_H

#include <memory>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

// A physical-world inspection adapter. No policy, learner or automatic commands.
class CreatureLab final : public RefCounted {
    GDCLASS(CreatureLab, RefCounted);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    String error_;
protected:
    static void _bind_methods();
public:
    CreatureLab();
    ~CreatureLab() override;
    Dictionary get_state() const;
    bool set_commands(PackedFloat32Array commands);
    bool set_mode(int mode);
    bool set_current(double speed);
    void set_paused(bool paused);
    Dictionary step();
    Dictionary single_step();
    void reset_world();
    bool save_state();
    bool load_state();
    bool has_saved_state() const;
    String get_error() const { return error_; }
};

} // namespace godot
#endif
