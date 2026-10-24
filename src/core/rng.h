#ifndef CUDARL_CORE_RNG_H
#define CUDARL_CORE_RNG_H

#include <cmath>
#include <cstdint>

#include "hd.h"

// Counter-based random numbers: every value is a pure function of
// (seed, stream, index), so CPU and GPU code produce identical sequences with
// no shared state, and any element can be regenerated on demand.
namespace cudarl::rng {

// "lowbias32" integer hash (Chris Wellons), a strong 32-bit bijective mixer.
CUDARL_HD inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

CUDARL_HD inline uint32_t bits(uint32_t seed, uint32_t stream, uint32_t index) {
    return hash_u32(seed ^ hash_u32(stream ^ hash_u32(index)));
}

// Uniform in the open interval (0, 1), using the top 24 bits.
CUDARL_HD inline float uniform(uint32_t seed, uint32_t stream, uint32_t index) {
    return (static_cast<float>(bits(seed, stream, index) >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

// Standard normal via Box-Muller on two independent uniforms.
CUDARL_HD inline float normal(uint32_t seed, uint32_t stream, uint32_t index) {
    const float u1 = uniform(seed, stream, 2u * index);
    const float u2 = uniform(seed, stream, 2u * index + 1u);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

} // namespace cudarl::rng

#endif // CUDARL_CORE_RNG_H
