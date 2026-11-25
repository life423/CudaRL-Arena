#ifndef CUDARL_POND_SWEPT_CLEARANCE_H
#define CUDARL_POND_SWEPT_CLEARANCE_H
#include "environment/pond_body_geometry.h"

namespace cudarl::pond_swept_clearance {
namespace d=pond_body_geometry::detail;
namespace t=pond_geometry_v1;
namespace p=physics;

// A lower bound only. No closest point, force direction or upper-gap witness.
struct Certificate {
    double lower_m=0;
    bool separated=false;
};

template<class T>
CUDARL_HD inline double translation_speed_bound(const T* velocity) {
    return fabs(double(velocity[0]))+fabs(double(velocity[1]))+fabs(double(velocity[2]));
}
template<class T>
CUDARL_HD inline double angular_speed_bound(const T* velocity,int link) {
    double speed=fabs(double(velocity[3]))+fabs(double(velocity[4]))+fabs(double(velocity[5]));
    for(int joint=0;joint<2*link;++joint)speed+=fabs(double(velocity[6+joint]));
    return speed;
}

// The integrator translates the root at constant v, rotates it through h*omega
// and increments each incident joint angle by h*qdot. A rotation's arc length
// bounds every intermediate chord. The sum of link box diagonals bounds the
// distance from ANY upstream pivot to ANY downstream material point. The same
// lever lengths bound the old Jacobian used by the implicit contact predictor.
CUDARL_HD inline Certificate certify(const p::BodyState& body,const p::Shape& links,
        const t::Parameters& terrain,const double* free_velocity,float h,double threshold_m) {
    if(!(h>0&&h<=.005001f)||!t::finite(threshold_m)||threshold_m<0||!t::valid_parameters(terrain))return {};
    for(int i=0;i<p::dofs;++i)if(!t::finite(free_velocity[i])||!p::finite_number(body.velocity[i]))return {};
    const auto q=body.orientation;
    const double norm2=double(q.x)*q.x+double(q.y)*q.y+double(q.z)*q.z+double(q.w)*q.w;
    if(!t::finite(norm2)||fabs(norm2-1)>2e-6)return {};
    const auto bounds=d::wet_bounds(terrain);
    if(!t::finite(bounds.gradient)||!t::finite(bounds.hessian))return {};
    const double gap_gradient_length=hypot(1.,bounds.gradient);
    const double translation_old=translation_speed_bound(body.velocity);
    const double translation_free=translation_speed_bound(free_velocity);
    double lower=1e30,reach=0;
    for(int b=0;b<3;++b) {
        auto shape=d::shape(links.link[b],b);
        // This material box encloses each curved hull and includes both corners
        // of the tail membrane (its half thickness is smaller than shape.c).
        shape.box=true;
        const double diagonal=hypot(hypot(shape.a,shape.b),shape.c);
        reach+=2*diagonal;
        const double radius=diagonal*shape.gain;
        const double old_speed=translation_old+2*reach*angular_speed_bound(body.velocity,b);
        const double free_speed=translation_free+2*reach*angular_speed_bound(free_velocity,b);
        // Factor two on each angular lever covers the near-unit rotate matrices
        // and the float Jacobian arithmetic. Up to four elementary joint
        // rotations and root normalization are used. Pad the paths for their
        // endpoint errors and for float coordinate/angle conversion. Very large
        // displacements fail the wet/surface tests and use the exact path.
        const double roundoff=256*1.1920928955078125e-7*
            (1+fabs(shape.center.x)+fabs(shape.center.y)+fabs(shape.center.z)+reach);
        const double move=double(h)*fmax(old_speed,free_speed)*(1.+1e-5)+roundoff;
        if(!t::finite(move)||!d::wet_ball(terrain,shape.center,radius+move)||
           d::support(shape,{0,-1,0}).y+move>=0)return {};
        const auto bed=t::sample(terrain,shape.center.x,shape.center.z);
        if(!bed.valid||bed.region==t::DryLand||bed.shoreline_coordinate>=1)return {};
        const auto gradient=d::gap_gradient(bed);
        const double linear=d::dot(gradient,d::support(shape,gradient)-shape.center);
        const double initial_lower=shape.center.y-bed.height_m+linear-.5*bounds.hessian*radius*radius;
        // On this entirely wet disk, Taylor bounds bed curvature. The global
        // wet gradient bounds both the actual swept gap and the contact law's
        // linear old-witness predictor, for BOTH old and free-solved velocities.
        const double swept=initial_lower-gap_gradient_length*move-d::arithmetic_margin(terrain,shape.center);
        if(!t::finite(swept)||swept<=threshold_m)return {};
        lower=fmin(lower,swept);
    }
    return {lower,true};
}
}
#endif
