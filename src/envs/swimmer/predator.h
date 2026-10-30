#ifndef CUDARL_PREDATOR_H
#define CUDARL_PREDATOR_H
#include <cstdint>
#include "arch.h"

// A bounded geometric threat experiment, not predator dynamics or a vision
// simulator. This descriptor never owns, resets, or moves the swimmer/food.
// All positions are model units (one unit is four centimetres).
namespace cudarl::predator {
using deep::V;
constexpr int kCoverCount=4;
constexpr float kWarningSeconds=4.f,kPassingSeconds=20.f,kCycleSeconds=48.f;
constexpr float kRayTolerance=.002f;
struct State {
    float elapsed=0;
    uint32_t seed=1;
    float manual_age=-1; // >=0 overrides the automatic clock until cleared
};
struct Event {
    V eye{},shadow{},velocity{};
    float strength=0,phase=0;
    bool warning=false,active=false,passed=false;
};
CUDARL_HD inline uint32_t event_key(uint32_t seed) {
    seed^=seed>>16;seed*=0x7FEB352Du;seed^=seed>>15;seed*=0x846CA68Bu;return seed^(seed>>16);
}
CUDARL_HD inline Event sample(const State& s) {
    Event e;
    const float first=6.f+static_cast<float>(event_key(s.seed)%81u)*.05f;
    float age=s.manual_age>=0?s.manual_age:s.elapsed-first;
    if(age<0) {e.eye={-12,4,-2.25f};e.shadow={-4,0,-2.25f};return e;}
    if(s.manual_age<0) age=fmodf(age,kCycleSeconds);
    e.warning=age<kWarningSeconds;
    e.active=age>=kWarningSeconds && age<kWarningSeconds+kPassingSeconds;
    e.passed=age>=kWarningSeconds+kPassingSeconds;
    e.phase=deep::clamp((age-kWarningSeconds)/kPassingSeconds,0,1);
    // An oblique pass is deliberate. A bird directly above the bed-rooted
    // sphere would offer no physically accessible hiding volume beneath it.
    e.eye={-12.f+8.f*e.phase,4.f,-2.25f+2.f*e.phase};
    // A fixed oblique sunlight direction (2,-1,0) projects the bird onto the
    // water plane. The readable shadow sweeps x=-4..4 while actual detection
    // uses the distinct eye above. A shadow is a warning cue, not a cover test.
    e.shadow={e.eye.x+2.f*e.eye.y,0,e.eye.z};
    e.velocity=e.active?V{8.f/kPassingSeconds,0,2.f/kPassingSeconds}:V{};
    e.strength=e.active?1.f:0.f;
    return e;
}
CUDARL_HD inline void advance(State& s,float seconds=deep::dt) {
    if(!(seconds>0)) return;
    s.elapsed+=seconds;
    if(s.manual_age>=0) s.manual_age+=seconds;
}
CUDARL_HD inline void trigger(State& s) {s.manual_age=0;}
CUDARL_HD inline void clear_manual(State& s) {s.manual_age=-1;}

CUDARL_HD inline bool sphere_segment(V start,V end,V center,float radius) {
    const V direction=end-start,relative=start-center;
    const float length2=deep::dot(direction,direction);
    const float t=length2>1e-12f?deep::clamp(-deep::dot(relative,direction)/length2,0,1):0;
    const V nearest=relative+direction*t;
    return deep::dot(nearest,nearest)<=radius*radius;
}
// Rays use the same exact sphere/torus/ledge solids as contact. Water and decorative
// plants are transparent in this simplified detection model. Terrain supplies
// physical clearance, but is not an additional opaque cover primitive here.
CUDARL_HD inline bool occluded(V point,V eye,const deep::Habitat& h,deep::Basin basin={2}) {
    if((basin.layout==2 || basin.layout==3) && sphere_segment(point,eye,deep::boulder_center(h),deep::boulder_radius(h))) return true;
    if(!h.arch && basin.layout!=3) return false;
    const V delta=eye-point;const float length=deep::norm(delta);
    if(!(length>1e-6f)) return (basin.layout==3?fminf(deep::arch_distance(point,h),deep::overhang_distance(point,h)):
                                               deep::arch_distance(point,h))<=kRayTolerance;
    const V direction=delta/length;float travel=0;
    // The exact torus SDF is 1-Lipschitz. A 0.9 safety factor cannot step
    // through a tube; the finite budget reports unresolved tangencies visible.
    for(int step=0;step<128 && travel<=length;++step) {
        const V at=point+direction*travel;
        const float arch=deep::arch_distance(at,h);
        const float gap=basin.layout==3?fminf(arch,deep::overhang_distance(at,h)):arch;
        if(gap<=kRayTolerance) return true;
        travel+=fmaxf(.0005f,gap*.9f);
    }
    return false;
}
struct Exposure {
    float visible=1,head=1,proximal_tail=1,distal_tail=1;
    int visible_samples=0,samples=0;
};
// Nine axial slices, eight elliptical rim points per slice, on all three
// articulated links, including the distal fin. This is sampled whole-body
// visibility, not exact continuous silhouette coverage or a head-only test.
CUDARL_HD inline Exposure exposure_detail(const deep::BodyState& body,V eye,const deep::Habitat& h,deep::Basin basin={2}) {
    const auto shape=deep::shape(body);Exposure e;int counts[3]{};
    constexpr int rim=8;
    for(int b=0;b<3;++b) {
        const auto& link=shape.link[b];
        for(int axial=0;axial<deep::arch_axial_samples;++axial) {
            const float f=(axial-4)*.25f;
            const float profile=b==0?sqrtf(fmaxf(0.f,1.f-f*f)):1.f;
            const float ry=(b==0?.0064f:(b==2?.006f:.0015f))*profile;
            const float rz=(b==0?.008f:.0015f)*profile;
            for(int j=0;j<rim;++j) {
                const float angle=j*.7853981633974483f;
                const V local{f*link.length*.5f,ry*cosf(angle),rz*sinf(angle)};
                const V point=(link.center+link.q.rotate(local))/deep::scale;
                const bool visible=!occluded(point,eye,h,basin);
                counts[b]+=visible?1:0;e.visible_samples+=visible?1:0;++e.samples;
            }
        }
    }
    constexpr float per_link=static_cast<float>(deep::arch_axial_samples*rim);
    e.head=counts[0]/per_link;e.proximal_tail=counts[1]/per_link;e.distal_tail=counts[2]/per_link;
    e.visible=static_cast<float>(e.visible_samples)/static_cast<float>(e.samples);return e;
}
CUDARL_HD inline float exposure(const deep::BodyState& body,V eye,const deep::Habitat& h,deep::Basin basin={2}) {
    return exposure_detail(body,eye,h,basin).visible;
}
// The same sampled supports as contact, including bed slope and the surface.
// A legal static pose is necessary but does not prove motor reachability/hold.
CUDARL_HD inline float body_clearance(const deep::BodyState& body,const deep::Habitat& h,deep::Basin basin={2}) {
    float gap=deep::obstacle_body_clearance(body,h,basin);const auto shape=deep::shape(body);
    for(int b=0;b<3;++b) {
        const auto& link=shape.link[b];const V axis=link.q.rotate({1,0,0});
        for(int axial=0;axial<deep::arch_axial_samples;++axial) {
            const float f=(axial-4)*.25f;
            const V point=(link.center+axis*(f*link.length*.5f))/deep::scale;
            const V gradient=deep::habitat_gradient(point.x,point.z,h,basin);
            const float magnitude=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
            const V normal=V{-gradient.x,1,-gradient.z}/magnitude;
            gap=fminf(gap,(point.y-deep::habitat_floor(point.x,point.z,h,basin))/magnitude-
                deep::arch_slice_support(link,b,f,normal)/deep::scale);
            gap=fminf(gap,-point.y-deep::arch_slice_support(link,b,f,{0,1,0})/deep::scale);
        }
    }
    return gap;
}
struct Cover {
    V position{};
    float yaw=0;
    bool valid=false;
    float exposure=1,clearance=0;
};
CUDARL_HD inline deep::BodyState cover_pose(const Cover& c) {
    deep::BodyState body;body.position=c.position;body.orientation=deep::axis_rotation({0,1,0},c.yaw);return body;
}
// Four finite, geometrically checked goal poses around the boulder's accessible
// far side. They are navigation metadata, never pellets or scripted routes.
// valid means physically clear; exposure must also be checked for the eye.
CUDARL_HD inline Cover cover(int index,V eye,const deep::Habitat& h,deep::Basin basin={2}) {
    Cover c;if(index<0 || index>=kCoverCount || basin.layout!=2) return c;
    const float factor=h.scale/1.5f;
    if(index==0) {c.position={5.5f*factor,-2.5f,-.5f*factor};c.yaw=-1.5707963267948966f;}
    if(index==1) {c.position={5.5f*factor,-2.5f,-.8f*factor};c.yaw=-1.5707963267948966f;}
    if(index==2) {c.position={5.4f*factor,-2.65f,-.7f*factor};c.yaw=-1.5707963267948966f;}
    if(index==3) {c.position={5.5f*factor,-2.5f,-1.8f*factor};c.yaw=1.5707963267948966f;}
    const auto body=cover_pose(c);c.clearance=body_clearance(body,h,basin);
    c.exposure=predator::exposure(body,eye,h,basin);c.valid=c.clearance>.08f;return c;
}
// Static front-opening poses for geometry validation, not a movement policy.
// They do not track the bird, teleport the creature, or claim learned shelter.
CUDARL_HD inline Cover overhang_cover(int index,V eye,const deep::Habitat& h,deep::Basin basin={3}) {
    Cover c;if(index<0 || index>=kCoverCount || basin.layout!=3) return c;
    const float factor=h.scale/1.5f;
    if(index==0) c.position={-2.25f*factor,-2.10f,-2.40f*factor};
    if(index==1) c.position={-2.10f*factor,-2.15f,-2.35f*factor};
    if(index==2) c.position={-2.40f*factor,-2.15f,-2.45f*factor};
    if(index==3) c.position={-2.25f*factor,-2.20f,-2.50f*factor};
    const auto body=cover_pose(c);c.clearance=body_clearance(body,h,basin);
    c.exposure=predator::exposure(body,eye,h,basin);c.valid=c.clearance>.08f;return c;
}
}
#endif
