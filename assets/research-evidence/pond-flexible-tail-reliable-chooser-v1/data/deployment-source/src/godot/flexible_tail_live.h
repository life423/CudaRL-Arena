#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include <memory>

namespace godot {
class FlexibleTailLive final : public RefCounted {
    GDCLASS(FlexibleTailLive, RefCounted)
    struct Impl;
    std::unique_ptr<Impl> impl_;

protected:
    static void _bind_methods();

public:
    FlexibleTailLive();
    ~FlexibleTailLive() override;
    bool load_motor(const String& gait_path, const String& offset_path, const String& starts_path);
    bool load_autonomous(const String& gait_path, const String& motor_path, const String& chooser_path, const String& starts_path);
    Dictionary snapshot() const;
    bool step_control();
    bool request_turn(int direction);
    bool reset();
    String get_error() const;
};
} // namespace godot

