#ifndef CUDARL_POND_VIEW_DOMAIN_H
#define CUDARL_POND_VIEW_DOMAIN_H
#include "motor/movement_model.h"

namespace cudarl::pond_view {
constexpr float flat_radius_x_m=.22f,flat_radius_z_m=.14f;
constexpr float domain_radius_x_m=.20f,domain_radius_z_m=.12f;
constexpr float water_radius_x_m=.52f,water_radius_z_m=.34f;

// Presentation-domain validation only. No force, contact or state correction.
// An ellipse is convex: containing every corner of each conservative link box
// contains the whole head/rod/fin geometry, with a buffer before the visual slope.
inline bool contains(const physics::BodyState& body) {
    if(!physics::finite_body(body)) return false;
    const auto shape=movement::shape(body);
    for(int b=0;b<3;++b) {
        const auto dimensions=movement::body_geometry(b);
        const auto& link=shape.link[b];
        for(int x=-1;x<=1;x+=2) for(int y=-1;y<=1;y+=2) for(int z=-1;z<=1;z+=2) {
            const auto point=link.center+link.q.rotate({x*dimensions.length*.5f,y*dimensions.half_height,z*dimensions.half_width});
            const double nx=double(point.x)/domain_radius_x_m,nz=double(point.z)/domain_radius_z_m;
            if(!physics::finite_vector(point)||nx*nx+nz*nz>1.) return false;
        }
    }
    return true;
}
}
#endif
