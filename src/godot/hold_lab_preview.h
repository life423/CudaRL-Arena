#ifndef CUDARL_GODOT_HOLD_LAB_PREVIEW_H
#define CUDARL_GODOT_HOLD_LAB_PREVIEW_H
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <memory>
namespace godot {
// Saved-motor inference in the shared native pond. Never owns an optimizer.
class HoldLabPreview final : public RefCounted {
    GDCLASS(HoldLabPreview, RefCounted);
    struct Impl;
    std::unique_ptr<Impl> impl_;
protected:
    static void _bind_methods();
public:
    HoldLabPreview();
    ~HoldLabPreview() override;
    bool load_controller(const String& path);
    bool reset(int case_id, bool holding);
    bool reset_untimed_swim(int case_id);
    bool start();
    void pause();
    bool step();
    Dictionary get_state() const;
    Dictionary get_geometry() const;
    Dictionary get_pond_geometry_descriptor() const;
    Dictionary sample_pond_surface(const PackedVector2Array& positions_m) const;
    Dictionary sample_pond_outline(const PackedVector2Array& angle_radius) const;
    String get_error() const;
};
}
#endif
