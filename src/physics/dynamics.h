#ifndef CUDARL_PHYSICS_DYNAMICS_H
#define CUDARL_PHYSICS_DYNAMICS_H
#include <cmath>
#include "core/hd.h"

// Articulated SI dynamics extracted from baseline e5e9bd1 deep_physics.h.
// Body, inertia, buoyancy, distributed drag and reduced solver algebra are retained.
// Uniform water only. No environment task, goal, food or learning dependency.
namespace cudarl::physics {
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
    float work=0,travel=0; // positive actuator work in joules; travel in model units
};
struct Parameters {float density=1000,viscosity=.001f,gravity=9.81f;};
CUDARL_HD inline bool finite_number(float v) {return v==v&&fabsf(v)<1e30f;}
CUDARL_HD inline bool finite_vector(V v) {return finite_number(v.x)&&finite_number(v.y)&&finite_number(v.z);}
CUDARL_HD inline bool finite_body(const BodyState& b) {
    if(!finite_vector(b.position)) return false;
    const auto q=b.orientation;const float n=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if(!finite_number(n)||fabsf(n-1.f)>.01f) return false;
    for(float v:b.velocity) if(!finite_number(v)) return false;
    for(float v:b.angle) if(!finite_number(v)) return false;
    return finite_number(b.work)&&finite_number(b.travel);
}
struct Link {
    Q q;
    V center,omega,acceleration,alpha,inertia;
    V linear[dofs]{},angular[dofs]{};
    float length=0,mass=0,volume=0,vertical_radius=0;
};
struct Shape {Link link[3];};
constexpr int surface_axial_samples=9;
// Each axial slice has an elliptical cross section; support is projected on
// the contact normal in that link's frame, including the distal tail fin.
CUDARL_HD inline float slice_support(const Link& l,int body,float fraction,V normal) {
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
struct System {float M[dofs][dofs]{},R[dofs][dofs]{},C[dofs]{},force[dofs]{};};
CUDARL_HD inline void resist(System& a,const V* J,V axis,float coefficient) {
    float j[dofs];for(int i=0;i<dofs;++i) j[i]=dot(J[i],axis);
    for(int i=0;i<dofs;++i) for(int k=0;k<dofs;++k) a.R[i][k]+=coefficient*j[i]*j[k];
}
CUDARL_HD inline void load(System& a,const V* J,V force) {for(int i=0;i<dofs;++i) a.force[i]+=dot(J[i],force);}
CUDARL_HD inline System assemble(const BodyState& s,Parameters physics={},V current={}) {
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
            const bool moving_water=current.x!=0 || current.y!=0 || current.z!=0;
            const V water=current;
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
        }
        for(int k=0;k<3;++k) resist(a,l.angular,axis[k],submersion*8.f*3.14159265f*physics.viscosity*powf(b==0?.008f:.002f,3.f));
    }
    return a;
}
CUDARL_HD inline float energy(const BodyState& s) {
    Parameters p;p.density=0;p.viscosity=0;p.gravity=0;const auto a=assemble(s,p);float e=0;
    for(int i=0;i<dofs;++i) for(int j=0;j<dofs;++j) e+=.5f*s.velocity[i]*a.M[i][j]*s.velocity[j];return e;
}
}
#endif
