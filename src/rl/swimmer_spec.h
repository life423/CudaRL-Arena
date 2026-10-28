#ifndef CUDARL_RL_SWIMMER_SPEC_H
#define CUDARL_RL_SWIMMER_SPEC_H
#include "nn/mlp_spec.h"
#include "envs/swimmer/swimmer.h"
namespace cudarl::rl {
constexpr int hidden=32;
inline nn::MlpSpec swimmer_specification(int outputs,int inputs=swimmer::kObs) {
    return nn::MlpSpec({inputs,hidden,hidden,outputs},
        {nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity});
}
}
#endif
