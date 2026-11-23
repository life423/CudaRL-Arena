#ifndef CUDARL_GODOT_HOLD_LAB_PREVIEW_H
#define CUDARL_GODOT_HOLD_LAB_PREVIEW_H
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <memory>
namespace godot {
// Explicit research inspection. Frozen inference only; never owns an optimizer.
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
    bool reset_free_swim(int case_id);
    bool start();
    void pause();
    bool step();
    Dictionary get_state() const;
    Dictionary get_geometry() const;
    String get_error() const;
};
}
#endif
