#ifndef CUDARL_CUDA_POND_MATH_PROFILE_H
#define CUDARL_CUDA_POND_MATH_PROFILE_H
#if defined(CUDARL_MOVEMENT_MODEL_H)
#error "Include pond_math_profile.h before movement_model.h in CUDA physics translation units"
#endif
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "physics/dynamics.h"

// Isolated shared-physics arithmetic compatibility profile. This preserves the
// original CPU cbrtf operation and the exact results of its unchanged three
// geometry arguments. It changes no volume, force equation or radius parameter.
// CUDA physics targets using this header must also compile with --fmad=false;
// optimized neural/PPO targets are not assigned that flag by this header.
namespace cudarl::pond_math_profile {
constexpr uint32_t version=1;
constexpr uint32_t argument_bits[3]={889811040u,859599123u,861293837u};
constexpr uint32_t original_cpu_result_bits[3]={1006834286u,996615495u,997278936u};
CUDARL_HD inline float radius(float argument) {
#if defined(__CUDA_ARCH__)
    switch(__float_as_uint(argument)) {
        case 889811040u:return __uint_as_float(1006834286u);
        case 859599123u:return __uint_as_float(996615495u);
        case 861293837u:return __uint_as_float(997278936u);
        default:return __uint_as_float(0x7fc00000u);
    }
#else
    return ::cbrtf(argument);
#endif
}
inline uint32_t bits(float value) {uint32_t result=0;std::memcpy(&result,&value,sizeof(result));return result;}
inline void validate_host_geometry() {
    const auto links=physics::shape(physics::BodyState{});constexpr float pi=3.14159265358979323846f;
    for(int body=0;body<3;++body) {
        const float argument=3*links.link[body].volume/(4*pi);
        if(bits(argument)!=argument_bits[body]||bits(::cbrtf(argument))!=original_cpu_result_bits[body])
            throw std::runtime_error("Pond CUDA radius profile differs from original CPU geometry/arithmetic");
    }
}
}
#define cbrtf(argument) cudarl::pond_math_profile::radius(argument)
#if defined(CUDARL_DEFINE_POND_INSPECT) || defined(CUDARL_DEFINE_POND_INSPECT_LINK)
#include "environment/pond_body_geometry.h"
#else
#include "simulation/pond_runtime.h"
#endif
#undef cbrtf
#endif
