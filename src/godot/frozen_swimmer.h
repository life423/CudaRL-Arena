#ifndef CUDARL_GODOT_FROZEN_SWIMMER_H
#define CUDARL_GODOT_FROZEN_SWIMMER_H
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <memory>
namespace godot {
class FrozenSwimmer final:public RefCounted {
    GDCLASS(FrozenSwimmer,RefCounted);
    struct Impl;std::unique_ptr<Impl> impl_;
protected:static void _bind_methods();
public:
    FrozenSwimmer();~FrozenSwimmer()override;
    bool load_controller();bool reset(int case_id=0);bool start();
    bool set_speed(double speed);void pause();bool step();
    Dictionary get_state()const;Dictionary get_geometry()const;String get_error()const;
};
}
#endif
