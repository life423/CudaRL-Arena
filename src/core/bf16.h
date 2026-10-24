#ifndef CUDARL_CORE_BF16_H
#define CUDARL_CORE_BF16_H

#include <cstdint>
#include <cstring>

#include "hd.h"

// bfloat16 as raw 16-bit patterns: the top half of an IEEE fp32 (1 sign,
// 8 exponent, 7 mantissa bits). Same range as fp32, ~3 significant digits.
// Host and device share this code, so conversions agree bit for bit.
namespace cudarl::bf16 {

CUDARL_HD inline uint32_t float_bits(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
#endif
}

CUDARL_HD inline float bits_float(uint32_t u) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
#endif
}

// Round to nearest, ties to even. NaN stays NaN (quiet); infinities and
// overflow to infinity behave as in IEEE rounding.
CUDARL_HD inline uint16_t from_float(float f) {
    const uint32_t u = float_bits(f);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);   // keep sign, force a quiet-NaN bit
    }
    const uint32_t rounding_bias = 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

CUDARL_HD inline float to_float(uint16_t b) {
    return bits_float(static_cast<uint32_t>(b) << 16);
}

// The fp32 value the bf16 rounding of f represents.
CUDARL_HD inline float round_trip(float f) {
    return to_float(from_float(f));
}

} // namespace cudarl::bf16

#endif // CUDARL_CORE_BF16_H
