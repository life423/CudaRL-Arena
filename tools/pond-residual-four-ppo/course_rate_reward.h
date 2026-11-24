#pragma once
#include "core/hd.h"
#include <cmath>

namespace cudarl::residual_four_ppo {

// The vectors describe whole-body horizontal COM movement before and after
// one 50 ms command. This changes reward only, never physical motion.
CUDARL_HD inline float course_rate_quality(float pre_x,float pre_z,float post_x,float post_z,
                                          float heading_x,float heading_z) {
    constexpr float neutral=.5f,pi=3.14159265358979323846f;
    constexpr float minimum_speed=.003f,control_seconds=.05f;
    constexpr float settling_seconds=2.f,rate_cap=10.f*pi/180.f,rate_scale=7.5f*pi/180.f;
    const float values[6]={pre_x,pre_z,post_x,post_z,heading_x,heading_z};
    for(float value:values)if(!(value==value&&fabsf(value)<=3.402823466e38f))return neutral;
    const float pre_squared=pre_x*pre_x+pre_z*pre_z,post_squared=post_x*post_x+post_z*post_z;
    const float heading_squared=heading_x*heading_x+heading_z*heading_z;
    if(!(pre_squared>=minimum_speed*minimum_speed&&post_squared>=minimum_speed*minimum_speed&&heading_squared>0)||
       !((pre_squared+post_squared+heading_squared)<=3.402823466e38f))return neutral;
    const float pre_length=sqrtf(pre_squared),post_length=sqrtf(post_squared),heading_length=sqrtf(heading_squared);
    pre_x/=pre_length;pre_z/=pre_length;post_x/=post_length;post_z/=post_length;
    heading_x/=heading_length;heading_z/=heading_length;
    // Positive pond yaw rotates +X toward -Z. atan2 wraps the course delta.
    const float error=atan2f(pre_z*heading_x-pre_x*heading_z,pre_x*heading_x+pre_z*heading_z);
    const float measured_rate=atan2f(pre_z*post_x-pre_x*post_z,pre_x*post_x+pre_z*post_z)/control_seconds;
    const float requested_rate=fminf(rate_cap,fmaxf(-rate_cap,error/settling_seconds));
    const float normalized=(measured_rate-requested_rate)/rate_scale;
    return neutral+neutral*expf(-normalized*normalized);
}

} // namespace cudarl::residual_four_ppo
