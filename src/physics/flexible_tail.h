#pragma once
#include "motor/movement_model.h"

// Isolated fully submerged model. Coordinates 0/1 drive the root universal
// joint. Coordinates 2/3 bend the entire tapered tail and have no actuators.
namespace cudarl::flexible_tail {
namespace p = cudarl::physics;
namespace movement_model = cudarl::movement;
constexpr int stations = 6;
constexpr float length_m = .052f, ds = length_m / stations;
constexpr float pi = 3.14159265358979323846f;
constexpr float body_density = 1000.f, young_modulus_Pa = 70000.f, relaxation_s = .02f;

// Explicit experimental material variant. Shape, density, actuator authority
// and integration are unchanged. The default preserves the original material.
enum class Material : uint32_t { Baseline = 0, YawHalfStiffness = 1 };
CUDARL_HD constexpr bool valid_material(Material material) {
    return material == Material::Baseline || material == Material::YawHalfStiffness;
}
CUDARL_HD constexpr double yaw_stiffness_multiplier(Material material) {
    return material == Material::YawHalfStiffness ? .5 : 1.;
}

struct Geometry {
    p::Link body[1 + stations];
    p::V spine[1 + stations];
};
struct Snapshot {
    p::V com_m, com_velocity_m_s, spine_m[stations + 1];
    float tilt_deg = 0, depth_m = 0;
    p::V linear_momentum_kg_m_s, angular_momentum_about_com_kg_m2_s;
};
struct StepInfo {
    double root_work_J = 0, elastic_energy_J = 0, drag_loss_J = 0, bend_loss_J = 0, kinetic_energy_J = 0;
    float torque_Nm[2]{};
};

CUDARL_HD constexpr float rib_radius(float t) { return .0015f * (1.f - .8f*t); }
CUDARL_HD constexpr float membrane_span(float t) { return .0097f * (1.f - t*t); }
CUDARL_HD constexpr float membrane_thickness(float t) { return .0001f * (1.f - .6f*t); }
CUDARL_HD constexpr float bend_stiffness() {
    // Uniform curvature q/L costs integral(E I (q/L)^2 ds)/2.
    float sum = 0;
    for (int i=0; i<stations; ++i) {
        const float r = rib_radius((i+.5f)/stations);
        sum += .25f*pi*r*r*r*r;
    }
    return young_modulus_Pa*sum/(stations*length_m);
}
CUDARL_HD inline double elastic_energy(const p::BodyState& state, Material material = Material::Baseline) {
    if (material == Material::Baseline)
        return .5*double(bend_stiffness())*(double(state.angle[2])*state.angle[2]+double(state.angle[3])*state.angle[3]);
    return .5*double(bend_stiffness())*(yaw_stiffness_multiplier(material)*double(state.angle[2])*state.angle[2]
        +double(state.angle[3])*state.angle[3]);
}

// Keep the large device functions separate to avoid expanding all sections and
// both solves inside every CUDA caller. CPU builds retain the inline definitions.
CUDARL_HD CUDARL_DEVICE_NOINLINE void shape(const p::BodyState& s, Geometry& geom) {
    const p::V basis[3]={{1,0,0},{0,1,0},{0,0,1}};
    auto& head=geom.body[0];
    const auto g=p::body_geometry(0);
    const float a=.5f*g.length, b=g.half_height, c=g.half_width;
    head.q=s.orientation; head.center=s.position*p::scale;
    head.length=g.length; head.vertical_radius=b;
    head.volume=4.f*pi*a*b*c/3.f; head.mass=body_density*head.volume;
    head.inertia={head.mass*(b*b+c*c)/5,head.mass*(a*a+c*c)/5,head.mass*(a*a+b*b)/5};
    head.omega={s.velocity[3],s.velocity[4],s.velocity[5]};
    for (int k=0;k<3;++k) {head.linear[k]=basis[k]; head.angular[k+3]=basis[k];}
    const p::V offset=head.q.rotate({-a,0,0});
    geom.spine[0]=head.center+offset;
    p::V end_j[p::dofs], end_bias=p::cross(head.omega,p::cross(head.omega,offset));
    for (int k=0;k<p::dofs;++k) end_j[k]=head.linear[k]+p::cross(head.angular[k],offset);
    const p::Q yaw=head.q*p::axis_rotation(basis[1],s.angle[0]);
    const p::Q root=yaw*p::axis_rotation(basis[2],s.angle[1]);
    for (int i=0;i<stations;++i) {
        const float t=(i+.5f)/stations, r=rib_radius(t), span=membrane_span(t), thick=membrane_thickness(t);
        auto& l=geom.body[1+i]; l.length=ds; l.vertical_radius=r+span*.5f;
        const p::Q curved_yaw=root*p::axis_rotation(basis[1],t*s.angle[2]);
        l.q=curved_yaw*p::axis_rotation(basis[2],t*s.angle[3]);
        const p::V axes[4]={head.q.rotate(basis[1]),yaw.rotate(basis[2]),root.rotate(basis[1])*t,curved_yaw.rotate(basis[2])*t};
        l.omega=head.omega;
        for(int k=0;k<3;++k) l.angular[3+k]=basis[k];
        for(int j=0;j<4;++j) {
            l.angular[6+j]=axes[j];
            const p::V dw=axes[j]*s.velocity[6+j];
            l.alpha+=p::cross(l.omega,dw); l.omega+=dw;
        }
        const p::V edge=l.q.rotate({-ds,0,0});
        const p::V edge_bias=p::cross(l.alpha,edge)+p::cross(l.omega,p::cross(l.omega,edge));
        l.center=geom.spine[i]+edge*.5f; geom.spine[1+i]=geom.spine[i]+edge;
        l.acceleration=end_bias+edge_bias*.5f; end_bias+=edge_bias;
        for(int k=0;k<p::dofs;++k) {
            const p::V edge_j=p::cross(l.angular[k],edge);
            l.linear[k]=end_j[k]+edge_j*.5f; end_j[k]+=edge_j;
        }
        const float rib_volume=pi*r*r*ds, fin_volume=span*thick*ds;
        const float mr=rib_volume*body_density, mf=fin_volume*body_density;
        l.volume=rib_volume+fin_volume; l.mass=l.volume*body_density;
        const float hh=.5f*span, d=r+.5f*hh;
        const float trans=mr*(ds*ds+3*r*r)/12;
        l.inertia={mr*r*r*.5f+mf*((hh*hh+thick*thick)/12+d*d),
                   trans+mf*(ds*ds+thick*thick)/12,
                   trans+mf*((ds*ds+hh*hh)/12+d*d)};
    }
}

CUDARL_HD CUDARL_DEVICE_NOINLINE Snapshot snapshot(const p::BodyState& state) {
    Geometry geom{}; shape(state,geom); Snapshot out{}; float mass=0;
    for(int i=0;i<=stations;++i) {
        const auto& l=geom.body[i];
        out.spine_m[i]=geom.spine[i]; mass+=l.mass; out.com_m+=l.center*l.mass;
        out.linear_momentum_kg_m_s+=p::multiply(l.linear,state.velocity)*l.mass;
    }
    out.com_m=out.com_m/mass; out.com_velocity_m_s=out.linear_momentum_kg_m_s/mass;
    for(int i=0;i<=stations;++i) {
        const auto& l=geom.body[i];
        const p::V momentum=p::multiply(l.linear,state.velocity)*l.mass;
        out.angular_momentum_about_com_kg_m2_s+=p::cross(l.center-out.com_m,momentum)+p::inertial(l,l.omega);
    }
    out.tilt_deg=acosf(p::clamp(state.orientation.rotate({0,1,0}).y,-1.f,1.f))*(180.f/pi);
    out.depth_m=-out.com_m.y; return out;
}

CUDARL_HD CUDARL_DEVICE_NOINLINE double kinetic_energy(const p::BodyState& state) {
    Geometry geom{}; shape(state,geom); double energy=0;
    for (int i=0;i<=stations;++i) {
        const auto& l=geom.body[i]; const auto v=p::multiply(l.linear,state.velocity);
        energy+=.5*(double(l.mass)*p::dot(v,v)+p::dot(l.omega,p::inertial(l,l.omega)));
    }
    return energy;
}

CUDARL_HD CUDARL_DEVICE_NOINLINE void assemble(const p::BodyState& s,p::Parameters fluid,p::V water,p::System& sys) {
    Geometry geom{}; shape(s,geom);
    for(int i=0;i<=stations;++i) {
        const auto& l=geom.body[i];
        const auto bias=p::inertial(l,l.alpha)+p::cross(l.omega,p::inertial(l,l.omega));
        for(int j=0;j<p::dofs;++j) {
            sys.C[j]+=l.mass*p::dot(l.linear[j],l.acceleration)+p::dot(l.angular[j],bias);
            const auto inertia_axis=p::inertial(l,l.angular[j]);
            for(int k=0;k<=j;++k) {
                const float mk=l.mass*p::dot(l.linear[k],l.linear[j])+p::dot(l.angular[k],inertia_axis);
                sys.M[k][j]+=mk; if(k!=j) sys.M[j][k]+=mk;
            }
        }
        p::load(sys,l.linear,{0,(fluid.density*l.volume-l.mass)*fluid.gravity,0});
        const auto g=p::body_geometry(0);
        const float t=i?(i-.5f)/stations:0, r=rib_radius(t), span=membrane_span(t);
        const float area[3]={i?(pi*r*r+span*membrane_thickness(t))/stations:pi*g.half_height*g.half_width,
                             i?ds*2*r:pi*g.length*.5f*g.half_width,
                             i?ds*(2*r+span):pi*g.length*.5f*g.half_height};
        const float cd[3]={i?.15f:.3f,i?1.f:.8f,i?1.f:.8f};
        const p::V axis[3]={l.q.rotate({1,0,0}),l.q.rotate({0,1,0}),l.q.rotate({0,0,1})};
        const float radius=i?r:cbrtf(3*l.volume/(4*pi));
        const float visc=6*pi*fluid.viscosity*radius/(i?stations:1);
        const float nodes[3]={-.3872983346f,0,.3872983346f},weights[3]={5.f/18,4.f/9,5.f/18};
        for(int k=0;k<(i?1:3);++k) {
            const auto offset=axis[0]*(i?0:nodes[k]*l.length);
            p::V jac[p::dofs];
            for(int j=0;j<p::dofs;++j)jac[j]=l.linear[j]+p::cross(l.angular[j],offset);
            const auto relative=p::multiply(jac,s.velocity)-water;
            for(int d=0;d<3;++d) {
                const float coeff=(i?1.f:weights[k])*(visc+.5f*fluid.density*cd[d]*area[d]*fabsf(p::dot(relative,axis[d])));
                p::resist(sys,jac,axis[d],coeff);
                p::load(sys,jac,axis[d]*(coeff*p::dot(water,axis[d])));
            }
        }
        const float rot=8*pi*fluid.viscosity*radius*radius*radius/(i?stations:1);
        for(int d=0;d<3;++d)p::resist(sys,l.angular,axis[d],rot);
    }
}

// A null target disconnects both root motors. Only passive material response
// and the external fluid remain. No downstream joint target is constructed.
CUDARL_HD CUDARL_DEVICE_NOINLINE bool step(p::BodyState& body,const float* root_targets_rad,float h,StepInfo& info,
                            p::Parameters fluid={},p::V current_m_s={},Material material=Material::Baseline) {
    info={};
    if(!p::finite_body(body)||!(h>0&&h<=.005001f)||!p::finite_vector(current_m_s)
       ||!p::finite_number(fluid.density)||!p::finite_number(fluid.viscosity)||!p::finite_number(fluid.gravity)
       ||fluid.density<0||fluid.viscosity<0||fluid.gravity<0||!valid_material(material)) return false;
    for(int j=0;j<2;++j) {
        if(fabsf(body.angle[j])>p::limit) return false;
        if(root_targets_rad&&(!p::finite_number(root_targets_rad[j])||fabsf(root_targets_rad[j])>p::limit))return false;
    }
    p::System sys{}; assemble(body,fluid,current_m_s,sys);
    double A[10][10]{},rhs[10]{},next[10]{};
    movement_model::discrete_system(sys,body,h,A,rhs);
    constexpr double stiffness=bend_stiffness(), damping=stiffness*relaxation_s;
    for(int j=0;j<2;++j) {
        const double material_scale=j==0?yaw_stiffness_multiplier(material):1.;
        const double axis_stiffness=stiffness*material_scale, axis_damping=damping*material_scale;
        const int k=8+j; A[k][k]+=h*(axis_damping+h*axis_stiffness); rhs[k]-=h*axis_stiffness*body.angle[2+j];
    }
    if(root_targets_rad) {
        constexpr double freq=2*pi*3,zeta=.9;
        double desired[2]{},free_matrix[10][10]{},free_rhs[10]{},free_v[10]{};
        const int free_index[8]={0,1,2,3,4,5,8,9};
        for(int j=0;j<2;++j) desired[j]=body.velocity[6+j]+h*(freq*freq*(root_targets_rad[j]-body.angle[j])-2*zeta*freq*body.velocity[6+j]);
        for(int i=0;i<8;++i) {
            const int k=free_index[i];
            free_rhs[i]=rhs[k]-A[k][6]*desired[0]-A[k][7]*desired[1];
            for(int j=0;j<8;++j) free_matrix[i][j]=A[k][free_index[j]];
        }
        if(!movement_model::solve(free_matrix,free_rhs,free_v,8))return false;
        for(int j=0;j<2;++j) {
            const int k=6+j;
            double impulse=-rhs[k]+A[k][6]*desired[0]+A[k][7]*desired[1];
            for(int i=0;i<8;++i)impulse+=A[k][free_index[i]]*free_v[i];
            const float requested=float(impulse/h); if(!p::finite_number(requested))return false;
            info.torque_Nm[j]=p::clamp(requested,-p::torque,p::torque);
            rhs[k]+=h*info.torque_Nm[j];
        }
    }
    if(!movement_model::solve(A,rhs,next,10))return false;
    const double water[3]={current_m_s.x,current_m_s.y,current_m_s.z};
    for(int i=0;i<10;++i) {
        double residual=-rhs[i],scale=fabs(rhs[i])+1e-15;
        const double vi=next[i]-(i<3?water[i]:0);
        for(int j=0;j<10;++j) {
            residual+=A[i][j]*next[j]; scale+=fabs(A[i][j]*next[j]);
            info.drag_loss_J+=h*vi*sys.R[i][j]*(next[j]-(j<3?water[j]:0));
        }
        if(fabs(residual)>1e-8*scale)return false;
    }
    if(!(info.drag_loss_J>=-1e-14))return false;
    auto candidate=body;
    for(int i=0;i<10;++i)candidate.velocity[i]=float(next[i]);
    const p::V v={candidate.velocity[0],candidate.velocity[1],candidate.velocity[2]};
    const p::V w={candidate.velocity[3],candidate.velocity[4],candidate.velocity[5]};
    candidate.position+=v*(h/p::scale); candidate.travel+=p::norm(v)*(h/p::scale);
    const float speed=p::norm(w);
    if(speed>1e-10f)candidate.orientation=p::axis_rotation(w/speed,h*speed)*candidate.orientation;
    candidate.orientation.normalize();
    for(int j=0;j<4;++j)candidate.angle[j]+=h*candidate.velocity[6+j];
    for(int j=0;j<2;++j) {
        if(fabsf(candidate.angle[j])>p::limit)return false;
        const double work=h*info.torque_Nm[j]*next[6+j];
        info.root_work_J+=work; candidate.work+=float(fmax(0.,work));
        const double material_scale=j==0?yaw_stiffness_multiplier(material):1.;
        info.bend_loss_J+=h*(damping*material_scale)*next[8+j]*next[8+j];
    }
    if(!p::finite_body(candidate))return false;
    info.elastic_energy_J=elastic_energy(candidate,material); info.kinetic_energy_J=kinetic_energy(candidate);
    if(!::isfinite(info.elastic_energy_J)||!::isfinite(info.kinetic_energy_J))return false;
    body=candidate; return true;
}
}
