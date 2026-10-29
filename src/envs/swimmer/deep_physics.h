#ifndef CUDARL_DEEP_PHYSICS_H
#define CUDARL_DEEP_PHYSICS_H
#include <cmath>
#include "core/hd.h"

// SI articulated dynamics, 6 floating-base + 4 universal-joint DOFs.
// Head/body orientation is free in 3D. Actuators apply ONLY internal torques.
// Reduced-order distributed drag, not CFD; training and display share this code.
namespace cudarl::deep {
constexpr float scale=0.04f,dt=0.05f,limit=1.0f,torque=1e-5f;
constexpr int dofs=10;
struct V {
    float x=0,y=0,z=0;
    CUDARL_HD V operator+(V b) const {return {x+b.x,y+b.y,z+b.z};}
    CUDARL_HD V operator-(V b) const {return {x-b.x,y-b.y,z-b.z};}
    CUDARL_HD V operator*(float a) const {return {x*a,y*a,z*a};}
    CUDARL_HD V operator/(float a) const {return *this*(1.f/a);}
    CUDARL_HD V& operator+=(V b) {x+=b.x;y+=b.y;z+=b.z;return *this;}
};
CUDARL_HD inline float dot(V a,V b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
CUDARL_HD inline V cross(V a,V b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
CUDARL_HD inline float norm(V a) {return sqrtf(dot(a,a));}
CUDARL_HD inline float clamp(float a,float lo,float hi) {return fminf(hi,fmaxf(lo,a));}
struct Q {
    float x=0,y=0,z=0,w=1;
    CUDARL_HD Q operator*(Q b) const {
        return {w*b.x+x*b.w+y*b.z-z*b.y,w*b.y-x*b.z+y*b.w+z*b.x,
                w*b.z+x*b.y-y*b.x+z*b.w,w*b.w-x*b.x-y*b.y-z*b.z};
    }
    CUDARL_HD V rotate(V v) const {V q{x,y,z};return v+cross(q,cross(q,v)+v*w)*2.f;}
    CUDARL_HD V inverse(V v) const {return Q{-x,-y,-z,w}.rotate(v);}
    CUDARL_HD void normalize() {const float r=1.f/sqrtf(x*x+y*y+z*z+w*w);x*=r;y*=r;z*=r;w*=r;}
};
CUDARL_HD inline Q axis_rotation(V a,float angle) {const float n=sinf(angle*.5f);return {a.x*n,a.y*n,a.z*n,cosf(angle*.5f)};}
struct BodyState {
    V position{0,-2,0}; // model units, Godot axes: Y up
    Q orientation;
    float angle[4]{},velocity[dofs]{}; // world m/s, world rad/s, relative rad/s
    float work=0,travel=0;
};
// Geometry is expressed in the same model units as BodyState/Godot. The arch
// is a solid torus about local Z, with its lower half buried in the basin.
struct Habitat {
    float scale=1;
    bool arch=false;
    V center{0,-2.25f,0};
    float major=1.35f,minor=.5f,yaw=0;
};
// Layout identity is external to Habitat/BodyState: old raw world snapshots
// remain readable. Layout 1 is the fixed, versioned outward-cove definition.
struct Basin {int layout=0;};
struct Physics {
    float density=1000,viscosity=.001f,gravity=9.81f;
    bool boundaries=true,pond=false;
    V water_velocity{}; // imposed far-field fluid velocity in world m/s
    Habitat habitat{};
    Basin basin{};
};
// Shared analytic basin, also sampled by Godot for its visible terrain.
CUDARL_HD inline float pond_floor(float x,float z) {
    const float r2=fminf(1.f,x*x/(4.6f*4.6f)+z*z/(3.5f*3.5f));
    return -4.f+4.5f*r2*r2*r2;
}
CUDARL_HD inline float habitat_floor(float x,float z,const Habitat& h,Basin basin={});
CUDARL_HD inline V habitat_gradient(float x,float z,const Habitat& h,Basin basin={});
CUDARL_HD inline V cove_center(const Habitat& h) {
    const float factor=h.scale/1.5f;return {5.8f*factor,0,-1.5f*factor};
}
CUDARL_HD inline V cove_radii(const Habitat& h) {
    const float factor=h.scale/1.5f;return {3.2f*factor,0,2.8f*factor};
}
CUDARL_HD inline float main_basin_floor(float x,float z,const Habitat& h) {
    if(h.scale==1.f) return pond_floor(x,z);
    const float rx=4.6f*h.scale,rz=3.5f*h.scale;
    const float r2=fminf(1.f,x*x/(rx*rx)+z*z/(rz*rz));
    return -4.f+4.5f*r2*r2*r2;
}
CUDARL_HD inline V main_basin_gradient(float x,float z,const Habitat& h) {
    const float rx=4.6f*h.scale,rz=3.5f*h.scale;
    const float r2=x*x/(rx*rx)+z*z/(rz*rz);
    return r2<1.f?V{27.f*x*r2*r2/(rx*rx),0,27.f*z*r2*r2/(rz*rz)}:V{};
}
CUDARL_HD inline float cove_floor(float x,float z,const Habitat& h) {
    const V center=cove_center(h),radii=cove_radii(h);x-=center.x;z-=center.z;
    const float r2=fminf(1.f,x*x/(radii.x*radii.x)+z*z/(radii.z*radii.z));
    return -3.f+3.5f*r2*r2*r2;
}
CUDARL_HD inline V cove_gradient(float x,float z,const Habitat& h) {
    const V center=cove_center(h),radii=cove_radii(h);x-=center.x;z-=center.z;
    const float r2=x*x/(radii.x*radii.x)+z*z/(radii.z*radii.z);
    return r2<1.f?V{21.f*x*r2*r2/(radii.x*radii.x),0,21.f*z*r2*r2/(radii.z*radii.z)}:V{};
}
CUDARL_HD inline float habitat_floor(float x,float z,const Habitat& h,Basin basin) {
    const float main=main_basin_floor(x,z,h);
    if(basin.layout==0) return main;
    constexpr float blend=.35f;const float cove=cove_floor(x,z,h);
    const float weight=clamp(.5f+.5f*(cove-main)/blend,0,1);
    if(weight>=1) return main;if(weight<=0) return cove;
    return cove+(main-cove)*weight-blend*weight*(1-weight);
}
CUDARL_HD inline V habitat_gradient(float x,float z,const Habitat& h,Basin basin) {
    const auto main=main_basin_gradient(x,z,h);if(basin.layout==0) return main;
    const float weight=clamp(.5f+.5f*(cove_floor(x,z,h)-main_basin_floor(x,z,h))/.35f,0,1);
    if(weight>=1) return main;
    const auto cove=cove_gradient(x,z,h);if(weight<=0) return cove;
    return main*weight+cove*(1-weight);
}
// Rare render/diagnostic contour query. Particle/drag integration uses the
// analytic floor and gradient above, without a root search in its hot path.
CUDARL_HD inline V basin_boundary(float angle,const Habitat& h,Basin basin={},float level=0) {
    const float x=cosf(angle),z=sinf(angle);float low=0,high=16.f*h.scale;
    for(int step=0;step<32;++step) {
        const float mid=(low+high)*.5f;
        if(habitat_floor(x*mid,z*mid,h,basin)<level) low=mid;else high=mid;
    }
    return {x*(low+high)*.5f,level,z*(low+high)*.5f};
}
CUDARL_HD inline float basin_radius(V p,const Habitat& h,Basin basin={}) {
    const float radius=sqrtf(p.x*p.x+p.z*p.z);if(radius==0) return 0;
    const V boundary=basin_boundary(atan2f(p.z,p.x),h,basin);
    return radius/sqrtf(boundary.x*boundary.x+boundary.z*boundary.z);
}
CUDARL_HD inline V arch_to_local(V point,const Habitat& h) {
    const V p=point-h.center;const float c=cosf(h.yaw),s=sinf(h.yaw);
    return {c*p.x-s*p.z,p.y,s*p.x+c*p.z};
}
CUDARL_HD inline V arch_to_world(V point,const Habitat& h) {
    const float c=cosf(h.yaw),s=sinf(h.yaw);
    return h.center+V{c*point.x+s*point.z,point.y,-s*point.x+c*point.z};
}
// Exact point-to-torus signed distance in model units. Outward unit normal
// uses finite deterministic fallbacks at the two nondifferentiable loci.
CUDARL_HD inline float arch_distance(V point,const Habitat& h,V* normal=nullptr) {
    if(!h.arch) {if(normal) *normal=V{};return 1e6f;}
    const V p=arch_to_local(point,h);const float radial=sqrtf(p.x*p.x+p.y*p.y);
    const float ring=radial-h.major,tube=sqrtf(ring*ring+p.z*p.z);
    if(normal) {
        V n{0,0,1};
        if(tube>1e-8f) {
            const float x=radial>1e-8f?p.x/radial:1.f,y=radial>1e-8f?p.y/radial:0.f;
            n={x*ring/tube,y*ring/tube,p.z/tube};
        }
        const float c=cosf(h.yaw),s=sinf(h.yaw);
        *normal={c*n.x+s*n.z,n.y,-s*n.x+c*n.z};
    }
    return tube-h.minor;
}
// A prescribed, bounded boundary-aware current, not a pressure/CFD solver.
// Model-unit distances define a smooth free-slip projection near the arch and
// basin, with a thin zero-velocity layer at every solid/surface intersection.
// Disabled habitats retain the exact legacy uniform flow.
CUDARL_HD inline float flow_smooth(float x) {
    const float t=clamp(x,0,1);return t*t*(3.f-2.f*t);
}
CUDARL_HD inline V habitat_flow(V point,const Habitat& h,V imposed,Basin basin={}) {
    if(!h.arch && basin.layout==0) return imposed;
    if(imposed.x==0 && imposed.y==0 && imposed.z==0) return {};
    V rock_normal;const float rock=arch_distance(point,h,&rock_normal);
    const V gradient=habitat_gradient(point.x,point.z,h,basin);
    const float magnitude=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
    const V bed_normal=V{-gradient.x,1,-gradient.z}/magnitude;
    const float bed=(point.y-habitat_floor(point.x,point.z,h,basin))/magnitude,surface=-point.y;
    if(rock<=0 || bed<=0 || surface<=0) return {};
    V velocity=imposed-rock_normal*(dot(imposed,rock_normal)*(1.f-flow_smooth(rock/.75f)));
    velocity=velocity-bed_normal*(dot(velocity,bed_normal)*(1.f-flow_smooth(bed/.75f)));
    velocity.y*=flow_smooth(surface/.3f);
    // Each orthogonal projection is a contraction, so the field never exceeds
    // the selected speed. This final layer also prevents leakage where the
    // individual projected boundary normals are not orthogonal.
    return velocity*(flow_smooth(rock/.05f)*flow_smooth(bed/.05f)*flow_smooth(surface/.05f));
}
struct Link {
    Q q;
    V center,omega,acceleration,alpha,inertia;
    V linear[dofs]{},angular[dofs]{};
    float length=0,mass=0,volume=0,vertical_radius=0;
};
struct Shape {Link link[3];};
constexpr int arch_axial_samples=9;
// Each axial slice has an elliptical cross section; support is projected on
// the contact normal in that link's frame, including the distal tail fin.
CUDARL_HD inline float arch_slice_support(const Link& l,int body,float fraction,V normal) {
    const float profile=body==0?sqrtf(fmaxf(0.f,1.f-fraction*fraction)):1.f;
    const float ry=(body==0?.0064f:(body==2?.006f:.0015f))*profile;
    const float rz=(body==0?.008f:.0015f)*profile;const V n=l.q.inverse(normal);
    return sqrtf(n.y*n.y*ry*ry+n.z*n.z*rz*rz);
}
CUDARL_HD inline V inertial(const Link& l,V v) {
    const V u=l.q.inverse(v);return l.q.rotate({u.x*l.inertia.x,u.y*l.inertia.y,u.z*l.inertia.z});
}
CUDARL_HD inline V multiply(const V* J,const float* velocity) {V v;for(int i=0;i<dofs;++i) v+=J[i]*velocity[i];return v;}
CUDARL_HD inline Shape shape(const BodyState& s) {
    Shape out;
    const V basis[3]={{1,0,0},{0,1,0},{0,0,1}};
    const float lengths[3]={.020f,.026f,.026f};
    V anchors[2],yaw_axes[2],pitch_axes[2];
    for(int b=0;b<3;++b) {
        auto& l=out.link[b];l.length=lengths[b];
        if(b==0) {l.q=s.orientation;l.center=s.position*scale;l.omega={s.velocity[3],s.velocity[4],s.velocity[5]};}
        else {
            const auto& p=out.link[b-1];const int j=(b-1)*2;
            const V r=p.q.rotate({-p.length*.5f,0,0});
            anchors[b-1]=p.center+r;yaw_axes[b-1]=p.q.rotate({0,1,0});
            const Q yaw=p.q*axis_rotation({0,1,0},s.angle[j]);
            pitch_axes[b-1]=yaw.rotate({0,0,1});l.q=yaw*axis_rotation({0,0,1},s.angle[j+1]);
            const V r2=l.q.rotate({-l.length*.5f,0,0});l.center=anchors[b-1]+r2;
            const V wy=yaw_axes[b-1]*s.velocity[6+j],wp=pitch_axes[b-1]*s.velocity[7+j];
            l.omega=p.omega+wy+wp;
            l.alpha=p.alpha+cross(p.omega,wy)+cross(p.omega+wy,wp);
            const V anchor_acc=p.acceleration+cross(p.alpha,r)+cross(p.omega,cross(p.omega,r));
            l.acceleration=anchor_acc+cross(l.alpha,r2)+cross(l.omega,cross(l.omega,r2));
        }
        for(int k=0;k<3;++k) {
            l.linear[k]=basis[k];l.angular[3+k]=basis[k];
            l.linear[3+k]=cross(basis[k],l.center-out.link[0].center);
        }
        for(int a=0;a<b;++a) {
            l.angular[6+2*a]=yaw_axes[a];l.angular[7+2*a]=pitch_axes[a];
            l.linear[6+2*a]=cross(yaw_axes[a],l.center-anchors[a]);
            l.linear[7+2*a]=cross(pitch_axes[a],l.center-anchors[a]);
        }
        constexpr float pi=3.141592653589793f;
        if(b==0) {
            const float a=.01f,h=.0064f,w=.008f;l.volume=4.f*pi*a*h*w/3.f;l.mass=l.volume*1000.f;
            l.inertia={l.mass*(h*h+w*w)/5.f,l.mass*(a*a+w*w)/5.f,l.mass*(a*a+h*h)/5.f};
            l.vertical_radius=h;
        } else {
            constexpr float r=.0015f,membrane=.0097f;
            l.volume=pi*r*r*l.length+(b==2?membrane*l.length*.0001f:0.f);l.mass=l.volume*1000.f;
            const float axial=l.mass*(r*r+(b==2?membrane*membrane/12.f:0.f))*.5f;
            const float transverse=l.mass*(l.length*l.length+3*r*r)/12.f;
            l.inertia={axial,transverse,transverse};l.vertical_radius=b==2?.006f:r;
        }
    }
    return out;
}
CUDARL_HD inline float arch_body_clearance(const BodyState& s,const Habitat& h) {
    if(!h.arch) return 1e6f;
    const auto geometry=shape(s);float gap=1e6f;
    for(int b=0;b<3;++b) {
        const auto& l=geometry.link[b];const V axis=l.q.rotate({1,0,0});
        for(int sample=0;sample<arch_axial_samples;++sample) {
            const float f=(sample-4)*.25f;const V point=l.center+axis*(f*l.length*.5f);V normal;
            const float distance=arch_distance(point/scale,h,&normal);
            gap=fminf(gap,distance-arch_slice_support(l,b,f,normal)/scale);
        }
    }
    return gap;
}
struct System {float M[dofs][dofs]{},R[dofs][dofs]{},C[dofs]{},force[dofs]{};bool arch_touched=false;};
CUDARL_HD inline void resist(System& a,const V* J,V axis,float coefficient) {
    float j[dofs];for(int i=0;i<dofs;++i) j[i]=dot(J[i],axis);
    for(int i=0;i<dofs;++i) for(int k=0;k<dofs;++k) a.R[i][k]+=coefficient*j[i]*j[k];
}
CUDARL_HD inline void load(System& a,const V* J,V force) {for(int i=0;i<dofs;++i) a.force[i]+=dot(J[i],force);}
CUDARL_HD inline void contact(System& a,const V* J,V point,V velocity,float radius,bool pond=false) {
    if(pond) {
        const float x=point.x/scale,z=point.z/scale,r2=x*x/(4.6f*4.6f)+z*z/(3.5f*3.5f);
        const float gx=r2<1?27.f*x*r2*r2/(4.6f*4.6f):0,gz=r2<1?27.f*z*r2*r2/(3.5f*3.5f):0;
        const float magnitude=sqrtf(1+gx*gx+gz*gz);const V n=V{-gx,1,-gz}/magnitude;
        const float gap=(point.y-pond_floor(x,z)*scale)/magnitude-radius;
        if(gap<0) {load(a,J,n*(-.8f*gap));if(dot(velocity,n)<0) resist(a,J,n,.006f);}
        // Swimming-only surface containment; shore locomotion is a later task.
        const float surface=-point.y-radius;
        if(surface<0) {load(a,J,{0,.8f*surface,0});if(velocity.y>0) resist(a,J,{0,-1,0},.006f);}
        return;
    }
    const V normal[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    const float gap[6]={point.x+4.5f*scale-radius,4.5f*scale-point.x-radius,point.y+4.f*scale-radius,
                       -point.y-radius,point.z+3.5f*scale-radius,3.5f*scale-point.z-radius};
    for(int wall=0;wall<6;++wall) if(gap[wall]<0) {
        load(a,J,normal[wall]*(-.8f*gap[wall]));
        if(dot(velocity,normal[wall])<0) resist(a,J,normal[wall],.006f);
    }
}
// Opt-in basin branch. The original contact routine above remains unchanged
// so existing environments/checkpoints retain their exact arithmetic path.
CUDARL_HD inline void habitat_contact(System& a,const V* J,V point,V velocity,float radius,const Habitat& h,Basin basin={}) {
    const float x=point.x/scale,z=point.z/scale;const V gradient=habitat_gradient(x,z,h,basin);
    const float magnitude=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
    const V n=V{-gradient.x,1,-gradient.z}/magnitude;
    const float gap=(point.y-habitat_floor(x,z,h,basin)*scale)/magnitude-radius;
    if(gap<0) {load(a,J,n*(-.8f*gap));if(dot(velocity,n)<0) resist(a,J,n,.006f);}
    const float surface=-point.y-radius;
    if(surface<0) {load(a,J,{0,.8f*surface,0});if(velocity.y>0) resist(a,J,{0,-1,0},.006f);}
}
CUDARL_HD inline void arch_contact(System& a,const Link& l,int body,const float* velocity,const Habitat& h) {
    if(!h.arch) return;
    const V axis=l.q.rotate({1,0,0});
    for(int sample=0;sample<arch_axial_samples;++sample) {
        const float f=(sample-4)*.25f;const V r=axis*(f*l.length*.5f);V normal;
        // Populate the normal before evaluating cross-section support. The
        // operands of subtraction may be evaluated in either order on CPU/CUDA.
        const float distance=arch_distance((l.center+r)/scale,h,&normal);
        const float gap=distance*scale-arch_slice_support(l,body,f,normal);
        if(gap<0) {
            a.arch_touched=true;
            V J[dofs];for(int i=0;i<dofs;++i) J[i]=l.linear[i]+cross(l.angular[i],r);
            load(a,J,normal*(-.8f*gap));
            if(dot(multiply(J,velocity),normal)<0) resist(a,J,normal,.006f);
        }
    }
}
CUDARL_HD inline System assemble(const BodyState& s,Physics physics={}) {
    const auto geometry=shape(s);System a;
    for(int b=0;b<3;++b) {
        const auto& l=geometry.link[b];
        const V angular_bias=inertial(l,l.alpha)+cross(l.omega,inertial(l,l.omega));
        for(int i=0;i<dofs;++i) {
            a.C[i]+=l.mass*dot(l.linear[i],l.acceleration)+dot(l.angular[i],angular_bias);
            for(int j=0;j<dofs;++j)
                a.M[i][j]+=l.mass*dot(l.linear[i],l.linear[j])+dot(l.angular[i],inertial(l,l.angular[j]));
        }
        const float submersion=clamp((l.vertical_radius-l.center.y)/(2.f*l.vertical_radius),0,1);
        load(a,l.linear,{0,-l.mass*physics.gravity,0});
        load(a,l.linear,{0,physics.density*l.volume*submersion*physics.gravity,0});
        // Slightly higher centre of buoyancy gives physical roll/pitch stability.
        const V buoyancy_offset=l.q.rotate({0,b==0?.0004f:0.f,0});
        load(a,l.angular,cross(buoyancy_offset,{0,physics.density*l.volume*submersion*physics.gravity,0}));
        const V axis[3]={l.q.rotate({1,0,0}),l.q.rotate({0,1,0}),l.q.rotate({0,0,1})};
        const float area[3]={b==0?.3f*3.14159265f*.0064f*.008f:1e-6f,
                            b==0?.8f*3.14159265f*.01f*.008f:l.length*.003f,
                            b==0?.8f*3.14159265f*.01f*.0064f:l.length*(b==2?.0127f:.003f)};
        for(int sample=0;sample<3;++sample) {
            const V r=axis[0]*((sample-1)*l.length/3.f);V J[dofs];
            for(int i=0;i<dofs;++i) J[i]=l.linear[i]+cross(l.angular[i],r);
            const V v=multiply(J,s.velocity);
            const bool moving_water=physics.water_velocity.x!=0 || physics.water_velocity.y!=0 || physics.water_velocity.z!=0;
            const V water=(physics.habitat.arch || physics.basin.layout!=0) && moving_water?
                habitat_flow((l.center+r)/scale,physics.habitat,physics.water_velocity,physics.basin):physics.water_velocity;
            const V relative=v-water;
            for(int direction=0;direction<3;++direction) {
                const float viscous=6.f*3.14159265f*physics.viscosity*(b==0?.008f:.004f);
                const float coefficient=submersion*(viscous+.5f*physics.density*area[direction]*fabsf(dot(relative,axis[direction])))/3.f;
                resist(a,J,axis[direction],coefficient);
                // Semi-implicit drag is -R*(v_new-u). The coefficient is
                // evaluated from the previous water-relative point velocity.
                // Keep the zero-current force path unchanged.
                if(moving_water)
                    load(a,J,axis[direction]*(coefficient*dot(water,axis[direction])));
            }
            if(physics.boundaries) {
                if(physics.pond && (physics.habitat.scale!=1.f || physics.basin.layout!=0))
                    habitat_contact(a,J,l.center+r,v,b==0?.0064f:.002f,physics.habitat,physics.basin);
                else contact(a,J,l.center+r,v,b==0?.0064f:.002f,physics.pond);
            }
        }
        if(physics.boundaries && physics.habitat.arch) arch_contact(a,l,b,s.velocity,physics.habitat);
        for(int k=0;k<3;++k) resist(a,l.angular,axis[k],submersion*8.f*3.14159265f*physics.viscosity*powf(b==0?.008f:.002f,3.f));
    }
    return a;
}
CUDARL_HD inline float energy(const BodyState& s) {
    Physics p;p.density=0;p.viscosity=0;p.gravity=0;p.boundaries=false;const auto a=assemble(s,p);float e=0;
    for(int i=0;i<dofs;++i) for(int j=0;j<dofs;++j) e+=.5f*s.velocity[i]*a.M[i][j]*s.velocity[j];return e;
}
CUDARL_HD inline void substep(BodyState& s,const float* action,float h,Physics physics={},bool* arch_touched=nullptr) {
    auto a=assemble(s,physics);
    if(arch_touched && a.arch_touched) *arch_touched=true;
    for(int j=0;j<4;++j) {
        const float displacement=s.angle[j]-clamp(s.angle[j],-limit,limit);
        a.force[6+j]+=torque*clamp(action[j],-1,1)-1e-4f*displacement;
        a.R[6+j][6+j]+=torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    // Each lower R entry is consumed once to form M+hR, then becomes the
    // corresponding Cholesky factor entry. Later rows only need that factor;
    // no subsequent operation needs the overwritten resistance coefficient.
    // Reuse removes a separate 100-float per-thread factor workspace.
    auto& L=a.R;float rhs[dofs]{},v[dofs]{};
    for(int i=0;i<dofs;++i) {
        rhs[i]=h*(a.force[i]-a.C[i]);for(int j=0;j<dofs;++j) rhs[i]+=a.M[i][j]*s.velocity[j];
        for(int j=0;j<=i;++j) {
            float x=a.M[i][j]+h*a.R[i][j];for(int k=0;k<j;++k) x-=L[i][k]*L[j][k];
            L[i][j]=i==j?sqrtf(fmaxf(x,1e-18f)):x/L[j][j];
        }
    }
    for(int i=0;i<dofs;++i) {float x=rhs[i];for(int j=0;j<i;++j) x-=L[i][j]*v[j];v[i]=x/L[i][i];}
    for(int i=dofs-1;i>=0;--i) {float x=v[i];for(int j=i+1;j<dofs;++j) x-=L[j][i]*v[j];v[i]=x/L[i][i];}
    s.position+=V{v[0],v[1],v[2]}*(h/scale);
    const V omega{v[3],v[4],v[5]};const float speed=norm(omega);
    if(speed>1e-10f) s.orientation=axis_rotation(omega/speed,h*speed)*s.orientation;
    s.orientation.normalize();
    for(int j=0;j<4;++j) {s.angle[j]+=h*v[6+j];s.work+=h*fmaxf(0,torque*clamp(action[j],-1,1)*v[6+j]);}
    for(int i=0;i<dofs;++i) s.velocity[i]=v[i];
    s.travel+=h*norm({v[0],v[1],v[2]})/scale;
}
CUDARL_HD inline void advance_physics(BodyState& s,const float* action,Physics p={},bool* arch_touched=nullptr) {
    if(arch_touched) *arch_touched=false;
    for(int i=0;i<10;++i) substep(s,action,.005f,p,arch_touched);
}
}
#endif
