#ifndef CUDARL_MOVEMENT_MODEL_H
#define CUDARL_MOVEMENT_MODEL_H
#include "physics/contact.h"

// Opt-in submerged movement model, not the live physics2/save contract.
// Functions operate on caller-owned BodyState; no new world/session owner.
namespace cudarl::movement {
namespace p=physics;
constexpr uint32_t kVersion=6;
constexpr float pi=3.14159265358979323846f;
struct Geometry {
    static constexpr float rod_radius=.0015f,fin_extension=.0097f,fin_thickness=.0001f;
    static constexpr float tail_height=2*rod_radius+fin_extension;
};
CUDARL_HD inline p::Shape shape(const p::BodyState& state) {
    auto result=p::shape(state);auto& tail=result.link[2];
    const float r=Geometry::rod_radius,L=tail.length,h=Geometry::fin_extension*.5f;
    const float rod_mass=1000*pi*r*r*L,fin_mass=1000*Geometry::fin_extension*L*Geometry::fin_thickness;
    const float offset=r+h*.5f,t=Geometry::fin_thickness;
    tail.inertia={rod_mass*r*r*.5f+fin_mass*((h*h+t*t)/12+offset*offset),
        rod_mass*(L*L+3*r*r)/12+fin_mass*(L*L+t*t)/12,
        rod_mass*(L*L+3*r*r)/12+fin_mass*((L*L+h*h)/12+offset*offset)};
    tail.vertical_radius=Geometry::tail_height*.5f;
    return result;
}
// Rod plus two thin rectangular membrane wings. The enclosing cylinder is
// conservative contact geometry; it is not used as a solid mass/volume.
CUDARL_HD inline p::BodyGeometry body_geometry(int body) {
    auto g=p::body_geometry(body);if(body==2)g.half_height=Geometry::tail_height*.5f;return g;
}
CUDARL_HD inline float clearance(const p::BodyState& body,bool composite=true) {
    if(!p::finite_body(body))return -1e6f;
    const auto links=composite?cudarl::movement::shape(body):p::shape(body);float gap=1e6f;
    for(int b=0;b<3;++b)for(int face=0;face<6;++face){const auto& link=links.link[b];
        const auto n=contact::normal(face),u=link.q.inverse(n);const auto g=composite?body_geometry(b):p::body_geometry(b);
        float support=0;
        if(b==0)support=sqrtf(g.length*g.length*.25f*u.x*u.x+g.half_height*g.half_height*u.y*u.y+g.half_width*g.half_width*u.z*u.z);
        else support=fabsf(u.x)*g.length*.5f+sqrtf(g.half_height*g.half_height*u.y*u.y+g.half_width*g.half_width*u.z*u.z);
        gap=fminf(gap,p::dot(n,link.center)+contact::offset(face)-support);
    }return gap;
}
CUDARL_HD inline p::System assemble(const p::BodyState& state,p::Parameters fluid={},bool composite=true) {
    if(!composite)return p::assemble(state,fluid);
    const auto links=cudarl::movement::shape(state);p::System out;
    for(int b=0;b<3;++b){const auto& l=links.link[b];const auto g=body_geometry(b);
        const auto bias=p::inertial(l,l.alpha)+p::cross(l.omega,p::inertial(l,l.omega));
        for(int i=0;i<10;++i){out.C[i]+=l.mass*p::dot(l.linear[i],l.acceleration)+p::dot(l.angular[i],bias);
            for(int j=0;j<10;++j)out.M[i][j]+=l.mass*p::dot(l.linear[i],l.linear[j])+p::dot(l.angular[i],p::inertial(l,l.angular[j]));}
        // Uniform density, common mass/volume centre, fully submerged scope.
        p::load(out,l.linear,{0,(fluid.density*l.volume-l.mass)*fluid.gravity,0});
        const p::V axis[3]={l.q.rotate({1,0,0}),l.q.rotate({0,1,0}),l.q.rotate({0,0,1})};
        const float projected[3]={b==0?pi*g.half_height*g.half_width:
            pi*g.half_width*g.half_width+(b==2?Geometry::fin_extension*Geometry::fin_thickness:0),
            b==0?pi*g.length*.5f*g.half_width:l.length*2*g.half_width,
            b==0?pi*g.length*.5f*g.half_height:l.length*2*g.half_height};
        // Explicit pressure-drag coefficients, no forward-thrust term. Linear
        // viscous term is an equal-volume sphere approximation, not calibrated CFD.
        const float Cd[3]={b==0?.3f:.15f,b==0?.8f:1.f,b==0?.8f:1.f};
        const float radius=cbrtf(3*l.volume/(4*pi));
        const float node[3]={-.3872983346207417f,0,.3872983346207417f};
        const float weight[3]={5.f/18,4.f/9,5.f/18};
        for(int sample=0;sample<3;++sample){const auto r=axis[0]*(node[sample]*l.length);p::V J[10];
            for(int i=0;i<10;++i)J[i]=l.linear[i]+p::cross(l.angular[i],r);
            const auto velocity=p::multiply(J,state.velocity);
            for(int direction=0;direction<3;++direction){const float coefficient=weight[sample]*(6*pi*fluid.viscosity*radius+
                    .5f*fluid.density*Cd[direction]*projected[direction]*fabsf(p::dot(velocity,axis[direction])));
                p::resist(out,J,axis[direction],coefficient);}
        }
        for(int k=0;k<3;++k)p::resist(out,l.angular,axis[k],8*pi*fluid.viscosity*radius*radius*radius);
    }return out;
}
// Positive-definite solve without pivot flooring. Double arithmetic prevents
// translation/rotation scale differences from hiding a failed factorization.
CUDARL_HD inline bool solve(const double A[10][10],const double* rhs,double* x,int n) {
    double L[10][10]{};
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j){double value=A[i][j];for(int k=0;k<j;++k)value-=L[i][k]*L[j][k];
        if(i==j){if(!(value>0)||!::isfinite(value))return false;L[i][j]=sqrt(value);}else L[i][j]=value/L[j][j];}
    for(int i=0;i<n;++i){double value=rhs[i];for(int j=0;j<i;++j)value-=L[i][j]*x[j];x[i]=value/L[i][i];}
    for(int i=n-1;i>=0;--i){double value=x[i];for(int j=i+1;j<n;++j)value-=L[j][i]*x[j];x[i]=value/L[i][i];if(!::isfinite(x[i]))return false;}
    return true;
}
struct Reference {float angle[4]{},rate[4]{},acceleration[4]{};};
struct Telemetry {
    float torque[4]{},requested[4]{},minimum_gap=1e6f,solve_residual=0;
    int failure_code=0;
    double signed_work=0,positive_work=0,fluid_loss=0,external_work=0,numerical_loss=0;
};
CUDARL_HD inline void discrete_system(const p::System& system,const p::BodyState& body,float h,double A[10][10],double rhs[10]) {
    for(int i=0;i<10;++i){rhs[i]=h*(system.force[i]-system.C[i]);
        for(int j=0;j<10;++j){
            // Both tensors are symmetric by construction. Float dot-product
            // ordering can leave their stored triangles a few ULPs apart.
            // Use one symmetric double operator for tracking, solve and audit.
            const double mass=.5*(static_cast<double>(system.M[i][j])+system.M[j][i]);
            const double resistance=.5*(static_cast<double>(system.R[i][j])+system.R[j][i]);
            A[i][j]=mass+static_cast<double>(h)*resistance;rhs[i]+=mass*body.velocity[j];
        }
    }
}
CUDARL_HD inline float kinetic(const p::BodyState& body,bool composite=true) {
    const auto a=assemble(body,{0,0,0},composite);double value=0;
    for(int i=0;i<10;++i)for(int j=0;j<10;++j)value+=.5*body.velocity[i]*a.M[i][j]*body.velocity[j];return static_cast<float>(value);
}
CUDARL_HD inline bool substep(p::BodyState& body,const Reference& reference,float h,p::Parameters fluid,Telemetry& telemetry,bool composite=true) {
    telemetry={};telemetry.failure_code=1;telemetry.minimum_gap=clearance(body,composite);
    if(!(h>0)||h>.005001f||telemetry.minimum_gap<.01f)return false;
    telemetry.failure_code=2;
    for(int j=0;j<4;++j)if(!p::finite_number(reference.angle[j])||fabsf(reference.angle[j])>p::limit||
        !p::finite_number(reference.rate[j])||fabsf(reference.rate[j])>12||
        !p::finite_number(reference.acceleration[j])||fabsf(reference.acceleration[j])>200)return false;
    const auto system=assemble(body,fluid,composite);double A[10][10]{},rhs[10]{},root_rhs[10]{},root_v[10]{},desired_v[4]{};
    const double omega=2*static_cast<double>(pi)*3,zeta=.9;
    discrete_system(system,body,h,A,rhs);
    for(int j=0;j<4;++j){const double acceleration=reference.acceleration[j]+omega*omega*(reference.angle[j]-body.angle[j])+2*zeta*omega*(reference.rate[j]-body.velocity[6+j]);
        desired_v[j]=body.velocity[6+j]+h*acceleration;}
    // Floating base remains unactuated. Solve its momentum equations, then
    // obtain the four INTERNAL torques needed for desired joint velocities.
    for(int i=0;i<6;++i){root_rhs[i]=rhs[i];for(int j=0;j<4;++j)root_rhs[i]-=A[i][6+j]*desired_v[j];}
    telemetry.failure_code=3;if(!solve(A,root_rhs,root_v,6))return false;
    telemetry.failure_code=4;
    for(int j=0;j<4;++j){double impulse=-rhs[6+j];for(int i=0;i<6;++i)impulse+=A[6+j][i]*root_v[i];for(int i=0;i<4;++i)impulse+=A[6+j][6+i]*desired_v[i];
        telemetry.requested[j]=static_cast<float>(impulse/h);telemetry.torque[j]=p::clamp(telemetry.requested[j],-p::torque,p::torque);
        if(!p::finite_number(telemetry.requested[j]))return false;rhs[6+j]+=h*telemetry.torque[j];}
    telemetry.failure_code=5;double next[10]{};if(!solve(A,rhs,next,10))return false;
    for(int i=0;i<10;++i){double product=0,scale=fabs(rhs[i])+1e-15;for(int j=0;j<10;++j){product+=A[i][j]*next[j];scale+=fabs(A[i][j]*next[j]);}
        telemetry.solve_residual=fmaxf(telemetry.solve_residual,static_cast<float>(fabs(product-rhs[i])/scale));
        telemetry.external_work+=h*system.force[i]*next[i];for(int j=0;j<10;++j){telemetry.fluid_loss+=h*next[i]*system.R[i][j]*next[j];
            telemetry.numerical_loss+=.5*(next[i]-body.velocity[i])*system.M[i][j]*(next[j]-body.velocity[j]);}}
    telemetry.failure_code=6;if(telemetry.solve_residual>1e-8f||telemetry.fluid_loss< -1e-14)return false;
    auto candidate=body;candidate.position+=p::V{static_cast<float>(next[0]),static_cast<float>(next[1]),static_cast<float>(next[2])}*(h/p::scale);
    const p::V angular{static_cast<float>(next[3]),static_cast<float>(next[4]),static_cast<float>(next[5])};const float speed=p::norm(angular);
    if(speed>1e-10f)candidate.orientation=p::axis_rotation(angular/speed,h*speed)*candidate.orientation;candidate.orientation.normalize();
    telemetry.failure_code=7;for(int j=0;j<4;++j){candidate.angle[j]+=h*static_cast<float>(next[6+j]);telemetry.signed_work+=h*telemetry.torque[j]*next[6+j];telemetry.positive_work+=h*fmax(0.,telemetry.torque[j]*next[6+j]);if(fabsf(candidate.angle[j])>p::limit)return false;}
    candidate.work+=static_cast<float>(telemetry.positive_work);for(int i=0;i<10;++i)candidate.velocity[i]=static_cast<float>(next[i]);
    candidate.travel+=h*p::norm({candidate.velocity[0],candidate.velocity[1],candidate.velocity[2]})/p::scale;
    telemetry.failure_code=8;telemetry.minimum_gap=fminf(telemetry.minimum_gap,clearance(candidate,composite));if(telemetry.minimum_gap<.01f)return false;
    body=candidate;telemetry.failure_code=0;return true;
}
// Smooth analytic reference used only by the prescribed mechanical diagnostic.
CUDARL_HD inline Reference prescribed(float time) {
    const float frequency=1,amplitude=.6f,w=2*pi*frequency;float envelope=1,rate=0,acceleration=0;
    if(time<2){envelope=.5f*(1-cosf(pi*time/2));rate=pi*.25f*sinf(pi*time/2);acceleration=pi*pi*.125f*cosf(pi*time/2);}
    else if(time>6){const float t=8-time;envelope=.5f*(1-cosf(pi*t/2));rate=-pi*.25f*sinf(pi*t/2);acceleration=pi*pi*.125f*cosf(pi*t/2);}
    Reference out;for(int j=0;j<4;j+=2){const float phase=w*time-(j==2?pi/2:0),s=sinf(phase),c=cosf(phase);
        out.angle[j]=amplitude*envelope*s;out.rate[j]=amplitude*(rate*s+envelope*w*c);
        out.acceleration[j]=amplitude*(acceleration*s+2*rate*w*c-envelope*w*w*s);}
    return out;
}
}
#endif
