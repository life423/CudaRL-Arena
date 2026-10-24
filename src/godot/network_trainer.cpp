#include "network_trainer.h"

#include <exception>
#include <vector>

#include <godot_cpp/core/class_db.hpp>

#include "cuda/device.h"

namespace godot {

using cudarl::demos::FitConfig;
using cudarl::demos::FitTarget;
using cudarl::demos::FunctionFit;

void NetworkTrainer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("configure", "target", "width", "depth", "activation", "learning_rate"),
                         &NetworkTrainer::configure);
    ClassDB::bind_method(D_METHOD("is_ready"), &NetworkTrainer::is_ready);
    ClassDB::bind_method(D_METHOD("get_error"), &NetworkTrainer::get_error);
    ClassDB::bind_method(D_METHOD("get_num_params"), &NetworkTrainer::get_num_params);
    ClassDB::bind_method(D_METHOD("train", "steps"), &NetworkTrainer::train);
    ClassDB::bind_method(D_METHOD("predict", "xs"), &NetworkTrainer::predict);
    ClassDB::bind_method(D_METHOD("target_values", "xs"), &NetworkTrainer::target_values);
    ClassDB::bind_method(D_METHOD("get_target_names"), &NetworkTrainer::get_target_names);
    ClassDB::bind_method(D_METHOD("get_x_min"), &NetworkTrainer::get_x_min);
    ClassDB::bind_method(D_METHOD("get_x_max"), &NetworkTrainer::get_x_max);
}

bool NetworkTrainer::configure(int target, int width, int depth, int activation, double learning_rate) {
    fit_.reset();
    error_ = String();
    if (target < 0 || target >= cudarl::demos::kNumFitTargets || activation < 0 || activation > 2) {
        error_ = "invalid target or activation";
        return false;
    }
    const cudarl::DeviceInfo device = cudarl::probe_device();
    if (!device.available) {
        error_ = String("CUDA unavailable: ") + device.error.c_str();
        return false;
    }
    FitConfig cfg;
    cfg.target     = static_cast<FitTarget>(target);
    cfg.width      = width;
    cfg.depth      = depth;
    cfg.activation = static_cast<cudarl::nn::Activation>(activation);
    cfg.lr         = static_cast<float>(learning_rate);
    try {
        fit_ = std::make_unique<FunctionFit>(cfg);
    } catch (const std::exception& e) {
        error_ = e.what();
        return false;
    }
    return true;
}

int NetworkTrainer::get_num_params() const {
    return fit_ ? fit_->num_params() : 0;
}

Dictionary NetworkTrainer::train(int steps) {
    Dictionary d;
    if (!fit_) return d;
    try {
        const auto s = fit_->train(steps);
        d["step"]        = static_cast<int64_t>(s.step);
        d["loss"]        = s.loss;
        d["ms_per_step"] = s.ms_per_step;
    } catch (const std::exception& e) {
        error_ = e.what();
        fit_.reset();
    }
    return d;
}

PackedFloat32Array NetworkTrainer::predict(const PackedFloat32Array& xs) {
    PackedFloat32Array out;
    if (!fit_) return out;
    try {
        const auto ys = fit_->predict(std::vector<float>(xs.ptr(), xs.ptr() + xs.size()));
        out.resize(static_cast<int64_t>(ys.size()));
        for (size_t i = 0; i < ys.size(); ++i) out.set(static_cast<int64_t>(i), ys[i]);
    } catch (const std::exception& e) {
        error_ = e.what();
        fit_.reset();
    }
    return out;
}

PackedFloat32Array NetworkTrainer::target_values(const PackedFloat32Array& xs) const {
    PackedFloat32Array out;
    out.resize(xs.size());
    const FitTarget target = fit_ ? fit_->config().target : FitTarget::Sine;
    for (int64_t i = 0; i < xs.size(); ++i) out.set(i, FunctionFit::target_value(target, xs[i]));
    return out;
}

PackedStringArray NetworkTrainer::get_target_names() const {
    PackedStringArray names;
    for (int t = 0; t < cudarl::demos::kNumFitTargets; ++t) {
        names.push_back(FunctionFit::target_name(static_cast<FitTarget>(t)));
    }
    return names;
}

} // namespace godot
