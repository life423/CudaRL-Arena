#ifndef CUDARL_CONTACT_RESPONSE_EXPERIMENT_H
#define CUDARL_CONTACT_RESPONSE_EXPERIMENT_H
#include "contact.h"

// Opt-in experiment only. Live contact.h/version2 and diagnostic save contracts
// remain unchanged. This supplies normal contact impulses, never a policy,
// position projection, tangential force or whole-body stabilizer.
namespace cudarl::contact_response {
constexpr uint32_t kPhysicsVersion=3;
constexpr int kRows=30,kMaxSweeps=64,kMaxActivationPasses=4;
constexpr float kResidualAbsolute=1e-6f,kResidualRelative=1e-5f;
struct SolverStats {
    int sweeps=0;
    float residual=0,tolerance=0;
    bool converged=false,finite=true;
};
struct Row {
    int body=0,face=0,end=0;
    float gap=0,k=0,c=0,impulse=0,velocity_free=0,velocity_after=0,residual=0,effective_mass=0;
    bool active=false;
    float jacobian[physics::dofs]{};
};
struct Telemetry {
    Row rows[kRows]{};
    int active_rows=0,sweeps=0,activation_passes=0;
    bool finite=true,converged=false,missed_activation=false;
    float projected_residual=0,tolerance=0,floor_impulse=0,total_impulse=0;
    physics::V linear_impulse{};
    // Discrete contact-only work in the frozen A metric (free -> constrained
    // velocity), not the complete body's gravity/fluid energy balance.
    double contact_work=0,elastic_before=0,elastic_linear_after=0,passivity_residual=0;
};

// Matrix is SPD on active rows. Projected Gauss-Seidel solves p>=0,
// r=matrix*p+q>=0,p*r=0. Residual is in normal-velocity units, not force units.
// This API performs algebra only; it never advances any physical pose.
CUDARL_HD inline bool solve_lcp(int count,const float matrix[kRows][kRows],const float* q,
    const bool* active,float* impulses,SolverStats& stats,float tolerance=1e-6f) {
    stats={};stats.tolerance=tolerance;
    if(count<1||count>kRows||!physics::finite_number(tolerance)||tolerance<=0) {stats.finite=false;return false;}
    int enabled=0;
    for(int i=0;i<count;++i) {
        if(!active[i]) {impulses[i]=0;continue;}
        ++enabled;
        if(!physics::finite_number(q[i])||!physics::finite_number(impulses[i])||impulses[i]<0||
           !physics::finite_number(matrix[i][i])||matrix[i][i]<=0) {stats.finite=false;return false;}
        for(int j=0;j<count;++j) if(active[j]&&!physics::finite_number(matrix[i][j])) {stats.finite=false;return false;}
    }
    if(enabled==0) {stats.converged=true;return true;}
    for(int sweep=0;sweep<kMaxSweeps;++sweep) {
        for(int i=0;i<count;++i) if(active[i]) {
            float residual=q[i];for(int j=0;j<count;++j) if(active[j]) residual+=matrix[i][j]*impulses[j];
            if(!physics::finite_number(residual)) {stats.finite=false;return false;}
            impulses[i]=fmaxf(0,impulses[i]-residual/matrix[i][i]);
        }
        stats.sweeps=sweep+1;stats.residual=0;
        for(int i=0;i<count;++i) if(active[i]) {
            float residual=q[i];for(int j=0;j<count;++j) if(active[j]) residual+=matrix[i][j]*impulses[j];
            if(!physics::finite_number(residual)||!physics::finite_number(impulses[i])) {stats.finite=false;return false;}
            const float projected=fabsf(fminf(impulses[i]*matrix[i][i],residual));
            stats.residual=fmaxf(stats.residual,projected);
        }
        if(stats.residual<=tolerance) {stats.converged=true;return true;}
    }
    return false;
}

namespace detail {
using namespace physics;
CUDARL_HD inline float product(const float* a,const float* b) {
    float value=0;for(int i=0;i<dofs;++i) value+=a[i]*b[i];return value;
}
CUDARL_HD inline bool factor(const System& system,float h,float L[dofs][dofs]) {
    for(int i=0;i<dofs;++i) for(int j=0;j<=i;++j) {
        float value=system.M[i][j]+h*system.R[i][j];
        for(int k=0;k<j;++k) value-=L[i][k]*L[j][k];
        if(!finite_number(value)||(i==j&&value<=0)) return false;
        L[i][j]=i==j?sqrtf(value):value/L[j][j];
        if(!finite_number(L[i][j])) return false;
    }
    return true;
}
CUDARL_HD inline bool linear_solve(const float L[dofs][dofs],const float* rhs,float* velocity) {
    for(int i=0;i<dofs;++i) {
        float value=rhs[i];for(int j=0;j<i;++j) value-=L[i][j]*velocity[j];velocity[i]=value/L[i][i];
    }
    for(int i=dofs-1;i>=0;--i) {
        float value=velocity[i];for(int j=i+1;j<dofs;++j) value-=L[j][i]*velocity[j];velocity[i]=value/L[i][i];
        if(!finite_number(velocity[i])) return false;
    }
    return true;
}
// Kinematic preview only: no force evaluation, work/travel accumulation or
// committed world advance. Nonlinear support can rotate substantially in5ms.
CUDARL_HD inline BodyState preview(const BodyState& initial,const float* velocity,float h) {
    auto next=initial;next.position+=V{velocity[0],velocity[1],velocity[2]}*(h/scale);
    const V omega{velocity[3],velocity[4],velocity[5]};const float speed=norm(omega);
    if(speed>1e-10f) next.orientation=axis_rotation(omega/speed,h*speed)*next.orientation;
    next.orientation.normalize();for(int j=0;j<4;++j) next.angle[j]+=h*velocity[6+j];
    for(int j=0;j<dofs;++j) next.velocity[j]=velocity[j];return next;
}
CUDARL_HD inline float gap(const Shape& geometry,const Row& row) {
    const auto& link=geometry.link[row.body];const auto n=contact::normal(row.face);
    return dot(n,link.center+contact::support_offset(link,row.body,n,row.end))+contact::offset(row.face);
}
CUDARL_HD inline void integrate(BodyState& state,const float* velocity,const float* action,float h) {
    state=preview(state,velocity,h);
    for(int j=0;j<4;++j) state.work+=h*fmaxf(0,torque*clamp(action[j],-1,1)*velocity[6+j]);
    state.travel+=h*norm({velocity[0],velocity[1],velocity[2]})/scale;
}
}

// Exactly one attempted physical substep. Failure leaves the input body intact;
// the harness must still count that attempt and retain this diagnostic record.
// Eligible rows: existing contact OR linear/nonlinear predicted crossing. Up to
// four monotone activation passes close contacts induced by coupled impulses.
CUDARL_HD inline bool substep(physics::BodyState& state,const float* action,float h,
    physics::Parameters physical={},physics::V current={},Telemetry* output=nullptr) {
    using namespace physics;
    Telemetry local;auto& out=output?*output:local;out={};
    if(!finite_body(state)||!finite_vector(current)||!finite_number(h)||h<=0||
       !finite_number(physical.density)||physical.density<0||!finite_number(physical.viscosity)||physical.viscosity<0||
       !finite_number(physical.gravity)||physical.gravity<0) {out.finite=false;return false;}
    for(int j=0;j<4;++j) if(!finite_number(action[j])) {out.finite=false;return false;}
    auto system=physics::assemble(state,physical,current);
    for(int j=0;j<4;++j) {
        const float displacement=state.angle[j]-clamp(state.angle[j],-limit,limit);
        system.force[6+j]+=torque*clamp(action[j],-1,1)-1e-4f*displacement;
        system.R[6+j][6+j]+=torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    float L[dofs][dofs]{},rhs[dofs]{},free_velocity[dofs]{};
    for(int i=0;i<dofs;++i) {rhs[i]=h*(system.force[i]-system.C[i]);for(int j=0;j<dofs;++j) rhs[i]+=system.M[i][j]*state.velocity[j];}
    if(!detail::factor(system,h,L)||!detail::linear_solve(L,rhs,free_velocity)) {out.finite=false;return false;}
    const auto free_trial=detail::preview(state,free_velocity,h);
    if(!finite_body(free_trial)) {out.finite=false;return false;}
    const auto geometry=shape(state),predicted=shape(free_trial);
    bool active[kRows]{};int row_count=0;
    for(int body=0;body<3;++body) {
        const auto& link=geometry.link[body];const int endpoints=body==0?1:2;const float weight=1.f/endpoints;
        for(int face=0;face<6;++face) for(int end=0;end<endpoints;++end) {
            auto& row=out.rows[row_count];row.body=body;row.face=face;row.end=end;
            const auto n=contact::normal(face),r=contact::support_offset(link,body,n,end);
            for(int j=0;j<dofs;++j) row.jacobian[j]=dot(n,link.linear[j]+cross(link.angular[j],r));
            row.gap=detail::gap(geometry,row);row.k=contact::kContactStiffness*weight;
            row.c=2.f*contact::kContactDampingRatio*sqrtf(contact::kContactStiffness*link.mass)*weight;
            row.velocity_free=detail::product(row.jacobian,free_velocity);
            active[row_count]=row.gap<=0||row.gap+h*row.velocity_free<=0||detail::gap(predicted,row)<=0;
            row.active=active[row_count];if(row.active) ++out.active_rows;++row_count;
        }
    }
    out.tolerance=kResidualAbsolute;
    if(out.active_rows==0) {
        // Both paths use precisely the unchanged free-water integrator, not
        // merely a numerically similar replacement with reordered arithmetic.
        for(int i=0;i<kRows;++i) out.rows[i].velocity_after=out.rows[i].velocity_free;
        auto next=state;contact::substep(next,action,h,physical,current);
        if(!finite_body(next)) {out.finite=false;return false;}
        state=next;out.converged=true;return true;
    }
    float response[kRows][dofs]{},matrix[kRows][kRows]{},q[kRows]{},impulses[kRows]{};
    for(int i=0;i<kRows;++i) if(!detail::linear_solve(L,out.rows[i].jacobian,response[i])) {out.finite=false;return false;}
    for(int i=0;i<kRows;++i) {
        auto& row=out.rows[i];
        for(int j=i;j<kRows;++j) {const float value=detail::product(row.jacobian,response[j]);matrix[i][j]=matrix[j][i]=value;}
        if(!finite_number(matrix[i][i])||matrix[i][i]<=0) {out.finite=false;return false;}
        row.effective_mass=1.f/matrix[i][i];
        matrix[i][i]+=1.f/(h*(row.c+h*row.k));
        q[i]=row.velocity_free+row.k*row.gap/(row.c+h*row.k);
    }
    float velocity[dofs]{};
    for(int pass=0;pass<kMaxActivationPasses;++pass) {
        float velocity_scale=0;
        for(int i=0;i<kRows;++i) if(active[i]) {
            const auto& row=out.rows[i];const float bias=row.k*row.gap/(row.c+h*row.k);
            velocity_scale=fmaxf(velocity_scale,fmaxf(fabsf(row.velocity_free),fabsf(bias)));
        }
        out.tolerance=kResidualAbsolute+kResidualRelative*velocity_scale;
        SolverStats stats;out.activation_passes=pass+1;
        const bool solved=solve_lcp(kRows,matrix,q,active,impulses,stats,out.tolerance);
        out.sweeps+=stats.sweeps;out.projected_residual=stats.residual;out.finite=stats.finite;
        for(int j=0;j<dofs;++j) {velocity[j]=free_velocity[j];for(int i=0;i<kRows;++i) if(active[i]) velocity[j]+=response[i][j]*impulses[i];}
        out.active_rows=0;
        for(int i=0;i<kRows;++i) {
            auto& row=out.rows[i];row.active=active[i];row.impulse=impulses[i];
            row.velocity_after=detail::product(row.jacobian,velocity);
            row.residual=q[i];for(int j=0;j<kRows;++j) if(active[j]) row.residual+=matrix[i][j]*impulses[j];
            if(row.active) ++out.active_rows;
        }
        if(!solved) return false;
        const auto trial=detail::preview(state,velocity,h);
        if(!finite_body(trial)) {out.finite=false;return false;}
        const auto trial_shape=shape(trial);bool added=false;
        for(int i=0;i<kRows;++i) if(!active[i]) {
            const auto& row=out.rows[i];
            if(row.gap+h*detail::product(row.jacobian,velocity)<=0||detail::gap(trial_shape,row)<=0) {
                active[i]=true;added=true;
            }
        }
        if(added) {
            if(pass+1==kMaxActivationPasses) {out.missed_activation=true;return false;}
            continue;
        }
        for(int i=0;i<kRows;++i) {
            const auto& row=out.rows[i];
            if(row.active) {
                out.total_impulse+=row.impulse;
                out.linear_impulse+=contact::normal(row.face)*row.impulse;
                if(row.face==2) out.floor_impulse+=row.impulse;
                out.contact_work+=double(row.impulse)*.5*(double(row.velocity_free)+row.velocity_after);
                const double before=fminf(0,row.gap),after=fminf(0,row.gap+h*row.velocity_after);
                out.elastic_before+=.5*row.k*before*before;out.elastic_linear_after+=.5*row.k*after*after;
            }
        }
        out.passivity_residual=out.contact_work+out.elastic_linear_after-out.elastic_before;
        auto next=state;detail::integrate(next,velocity,action,h);
        if(!finite_body(next)) {out.finite=false;return false;}
        state=next;out.converged=true;return true;
    }
    return false;
}
}
#endif
