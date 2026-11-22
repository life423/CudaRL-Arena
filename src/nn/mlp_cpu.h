#ifndef CUDARL_NN_MLP_CPU_H
#define CUDARL_NN_MLP_CPU_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "adam_config.h"
#include "mlp_spec.h"
#include "core/checkpoint.h"

// Plain-loop CPU reference for the MLP and Adam. It is written for obvious
// correctness, not speed: it is the ground truth the GPU kernels are tested
// against, and it runs in double precision for finite-difference gradient checks.
namespace cudarl::nn {

template <typename T>
class CpuMlp {
public:
    explicit CpuMlp(MlpSpec spec)
        : spec_(std::move(spec)),
          params_(static_cast<size_t>(spec_.num_params()), T(0)),
          grads_(params_.size(), T(0)),
          acts_(static_cast<size_t>(spec_.num_layers())) {}

    const MlpSpec& spec() const { return spec_; }
    std::vector<T>& params() { return params_; }
    const std::vector<T>& params() const { return params_; }
    const std::vector<T>& grads() const { return grads_; }

    // x: [batch x input_size], row-major. Keeps a copy of x for backward().
    void forward(const T* x, int batch) {
        if (batch <= 0 || x == nullptr)
            throw std::invalid_argument("CpuMlp::forward: positive batch and input required");
        batch_ = batch;
        input_.assign(x, x + static_cast<size_t>(batch) * spec_.input_size());
        const T* in = input_.data();
        for (int l = 0; l < spec_.num_layers(); ++l) {
            const int n_in = spec_.in(l), n_out = spec_.out(l);
            const T*  W    = &params_[spec_.weight_offset(l)];
            const T*  b    = &params_[spec_.bias_offset(l)];
            std::vector<T>& y = acts_[l];
            y.assign(static_cast<size_t>(batch) * n_out, T(0));
            for (int m = 0; m < batch; ++m) {
                for (int o = 0; o < n_out; ++o) {
                    T z = b[o];
                    for (int i = 0; i < n_in; ++i) z += in[m * n_in + i] * W[o * n_in + i];
                    y[m * n_out + o] = activate(spec_.activations[l], z);
                }
            }
            in = y.data();
        }
    }

    // [batch x output_size] from the last forward().
    const T* output() const { return acts_.back().data(); }

    void zero_grad() { std::fill(grads_.begin(), grads_.end(), T(0)); }

    // dy = dL/d(output) for the last forward(). Accumulates parameter gradients;
    // writes dL/d(input) to dx when it is non-null.
    void backward(const T* dy, int batch, T* dx = nullptr) {
        if (batch_ <= 0 || batch != batch_)
            throw std::logic_error("CpuMlp::backward: batch must match the preceding forward()");
        if (dy == nullptr) throw std::invalid_argument("CpuMlp::backward: output gradient required");
        std::vector<T> d_out(dy, dy + static_cast<size_t>(batch) * spec_.output_size());
        for (int l = spec_.num_layers() - 1; l >= 0; --l) {
            const int n_in = spec_.in(l), n_out = spec_.out(l);
            const T*  W    = &params_[spec_.weight_offset(l)];
            T*        dW   = &grads_[spec_.weight_offset(l)];
            T*        db   = &grads_[spec_.bias_offset(l)];
            const T*  x    = l == 0 ? input_.data() : acts_[l - 1].data();
            const std::vector<T>& y = acts_[l];

            std::vector<T> dz(d_out.size());
            for (size_t i = 0; i < dz.size(); ++i) {
                dz[i] = d_out[i] * activate_grad_from_output(spec_.activations[l], y[i]);
            }
            for (int m = 0; m < batch; ++m) {
                for (int o = 0; o < n_out; ++o) {
                    const T g = dz[m * n_out + o];
                    db[o] += g;
                    for (int i = 0; i < n_in; ++i) dW[o * n_in + i] += g * x[m * n_in + i];
                }
            }
            if (l == 0 && dx == nullptr) break;
            std::vector<T> d_in(static_cast<size_t>(batch) * n_in, T(0));
            for (int m = 0; m < batch; ++m) {
                for (int i = 0; i < n_in; ++i) {
                    T s = T(0);
                    for (int o = 0; o < n_out; ++o) s += dz[m * n_out + o] * W[o * n_in + i];
                    d_in[m * n_in + i] = s;
                }
            }
            if (l == 0) {
                std::copy(d_in.begin(), d_in.end(), dx);
            } else {
                d_out = std::move(d_in);
            }
        }
    }

private:
    MlpSpec                     spec_;
    std::vector<T>              params_;
    std::vector<T>              grads_;
    std::vector<T>              input_;
    std::vector<std::vector<T>> acts_;   // acts_[l] = output of layer l
    int                         batch_ = 0;
};

template <typename T>
class CpuAdam {
public:
    CpuAdam(int num_params, AdamConfig cfg)
        : cfg_(cfg), m_(static_cast<size_t>(num_params), T(0)), v_(m_.size(), T(0)) {}

    void step(std::vector<T>& params, const std::vector<T>& grads) {
        ++t_;
        T scale = T(1);
        if (cfg_.max_grad_norm > 0.f) {
            T sq = T(0);
            for (T g : grads) sq += g * g;
            const T norm = std::sqrt(sq);
            scale = std::min(T(1), T(cfg_.max_grad_norm) / (norm + T(1e-6)));
        }
        const T b1 = T(cfg_.beta1), b2 = T(cfg_.beta2);
        const T bc1 = T(1) - std::pow(b1, T(t_));
        const T bc2 = T(1) - std::pow(b2, T(t_));
        for (size_t i = 0; i < params.size(); ++i) {
            const T g = grads[i] * scale;
            m_[i] = b1 * m_[i] + (T(1) - b1) * g;
            v_[i] = b2 * v_[i] + (T(1) - b2) * g * g;
            params[i] -= T(cfg_.lr) * (m_[i] / bc1) / (std::sqrt(v_[i] / bc2) + T(cfg_.eps));
        }
    }

    int steps() const { return t_; }

    void save(checkpoint::Writer& out) const {
        out.value(cfg_); out.value(t_); out.array(m_); out.array(v_);
    }
    void load(checkpoint::Reader& in) {
        const auto cfg=in.value<AdamConfig>(); const int steps=in.value<int>();
        auto m=in.array<T>(m_.size()), v=in.array<T>(v_.size());
        if(m.size()!=m_.size() || v.size()!=v_.size() || steps<0 ||
           !std::isfinite(cfg.lr) || cfg.lr<=0 || cfg.beta1!=cfg_.beta1 || cfg.beta2!=cfg_.beta2 ||
           cfg.eps!=cfg_.eps || cfg.max_grad_norm!=cfg_.max_grad_norm)
            throw std::runtime_error("Incompatible CPU Adam checkpoint");
        for(size_t i=0;i<m.size();++i)
            if(!std::isfinite(m[i]) || !std::isfinite(v[i]) || v[i]<0)
                throw std::runtime_error("Invalid CPU Adam moments");
        cfg_=cfg; t_=steps; m_=std::move(m); v_=std::move(v);
    }

private:
    AdamConfig     cfg_;
    std::vector<T> m_, v_;
    int            t_ = 0;
};

} // namespace cudarl::nn

#endif // CUDARL_NN_MLP_CPU_H
