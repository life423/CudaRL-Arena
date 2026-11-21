#ifndef CUDARL_NN_ADAM_CONFIG_H
#define CUDARL_NN_ADAM_CONFIG_H

namespace cudarl::nn {

// Adam with optional global-norm gradient clipping: gradients are first scaled
// by min(1, max_grad_norm / (||g|| + 1e-6)), then
// p -= lr * m_hat / (sqrt(v_hat) + eps) with bias-corrected moments.
struct AdamConfig {
    float lr            = 3e-4f;
    float beta1         = 0.9f;
    float beta2         = 0.999f;
    float eps           = 1e-8f;
    float max_grad_norm = 0.f;   // <= 0 disables clipping
};

} // namespace cudarl::nn

#endif // CUDARL_NN_ADAM_CONFIG_H
