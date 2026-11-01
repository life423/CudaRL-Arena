#ifndef CUDARL_OBSERVATION_CONTRACT_H
#define CUDARL_OBSERVATION_CONTRACT_H
#include "core/hd.h"
#include <cstdint>
namespace cudarl::motor_observation {
// Only body-frame path-error channels 0..2 differ. This scales lane AND depth.
constexpr uint32_t legacy = 1, task_scale = 2;
CUDARL_HD inline bool valid(uint32_t schema) { return schema == legacy || schema == task_scale; }
CUDARL_HD inline float path_scale_m(uint32_t schema) { return schema == task_scale ? .002f : .072f; }
} // namespace cudarl::motor_observation
#endif
