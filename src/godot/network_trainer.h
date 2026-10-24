#ifndef CUDARL_GODOT_NETWORK_TRAINER_H
#define CUDARL_GODOT_NETWORK_TRAINER_H

#include <memory>

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include "demos/function_fit.h"

namespace godot {

// GDScript wrapper around cudarl::demos::FunctionFit: train a small MLP on the
// GPU a few steps per frame and read its predictions back for drawing.
// Nothing throws into Godot: failures (e.g. no CUDA device) are reported by
// configure() returning false and get_error().
class NetworkTrainer : public RefCounted {
    GDCLASS(NetworkTrainer, RefCounted);

protected:
    static void _bind_methods();

public:
    // target: index into get_target_names(); activation: 0 identity, 1 tanh, 2 relu.
    bool       configure(int target, int width, int depth, int activation, double learning_rate);
    bool       is_ready() const { return fit_ != nullptr; }
    String     get_error() const { return error_; }
    int        get_num_params() const;

    // Runs `steps` training steps; returns { step, loss, ms_per_step }.
    Dictionary train(int steps);
    PackedFloat32Array predict(const PackedFloat32Array& xs);
    PackedFloat32Array target_values(const PackedFloat32Array& xs) const;

    PackedStringArray get_target_names() const;
    double            get_x_min() const { return cudarl::demos::FunctionFit::kXMin; }
    double            get_x_max() const { return cudarl::demos::FunctionFit::kXMax; }

private:
    std::unique_ptr<cudarl::demos::FunctionFit> fit_;
    String                                      error_;
};

} // namespace godot

#endif // CUDARL_GODOT_NETWORK_TRAINER_H
