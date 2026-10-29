#ifndef CUDARL_FEEDING_POLICY_H
#define CUDARL_FEEDING_POLICY_H
#include "swimmer_spec.h"
#include "envs/swimmer/feeding.h"
namespace cudarl::rl {
inline std::vector<float> feeding_motor_parameters(const std::vector<float>& original) {
    const auto old=swimmer_specification(4,33),next=swimmer_specification(4,feeding::kObs);
    if(original.size()!=static_cast<size_t>(old.num_params())) throw std::runtime_error("Invalid feeding source motor");
    std::vector<float> weights(static_cast<size_t>(next.num_params()),0);
    for(int layer=0;layer<old.num_layers();++layer) {
        for(int row=0;row<old.out(layer);++row) for(int col=0;col<old.in(layer);++col)
            weights[next.weight_offset(layer)+row*next.in(layer)+col]=original[old.weight_offset(layer)+row*old.in(layer)+col];
        for(int row=0;row<old.out(layer);++row) weights[next.bias_offset(layer)+row]=original[old.bias_offset(layer)+row];
    }
    return weights;
}
}
#endif
