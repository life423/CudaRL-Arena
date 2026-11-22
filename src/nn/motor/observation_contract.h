#ifndef CUDARL_OBSERVATION_CONTRACT_H
#define CUDARL_OBSERVATION_CONTRACT_H
#include "core/hd.h"
#include <cstdint>
namespace cudarl::motor_observation {
// Only body-frame path-error channels 0..2 differ. This scales lane AND depth.
constexpr uint32_t legacy = 1, task_scale = 2, position_hold = 3, appended_hold = 4;
CUDARL_HD inline bool valid(uint32_t schema) { return schema == legacy || schema == task_scale || schema == position_hold || schema == appended_hold; }
CUDARL_HD inline float path_scale_m(uint32_t schema) { return schema == legacy ? .072f : .002f; }
} // namespace cudarl::motor_observation
#endif
