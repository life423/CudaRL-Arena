#ifndef CUDARL_CONTACT_PILOT_H
#define CUDARL_CONTACT_PILOT_H
#include "contact_response.h"

// Opt-in physical-law comparison only. Neither law changes the live runtime,
// body, actuator limits, water forces or saved-world contact version.
namespace cudarl::contact_pilot {
namespace p=physics;
constexpr uint32_t kPhysicsVersion=4;
constexpr int kRows=30,kComplianceSweeps=128,kBasisIterations=64,kActivationPasses=4;
constexpr float kStiffness=320.f,kDamping=1.9595918f;
constexpr double kEnergyCap=25e-6,kLoadCap=.0003,kMassCap=.003;
constexpr double kDesignCompression=.0004,kImpactSpeedCap=.05;
constexpr double kHardGapTolerance=2e-6,kResidualAbsolute=1e-6,kResidualRelative=1e-5;
// Rank detection, not a compliance ridge. No term is added to hard W.
constexpr double kRankRelative=1e-10,kImpulseRoundoff=1e-14;
enum class Law:uint32_t {Compliant=0,Hard=1};
using Row=contact_response::Row;
struct SolverStats {
    int iterations=0,basis_rank=0;
    double residual=0,tolerance=0;
    bool finite=true,converged=false,rank_failure=false;
};
struct Telemetry {
    Row rows[kRows]{};
    float free_velocity[p::dofs]{},after_velocity[p::dofs]{};
    int active_rows=0,iterations=0,basis_rank=0,activation_passes=0;
    bool finite=true,converged=false,missed_activation=false,geometry_valid=true,pose_ready=false;
    double residual=0,tolerance=0,contact_work=0;
    // Before/after refer ONLY to the physical-M contact substage: before is
    // the already-computed free velocity, not the beginning of the full step.
    double kinetic_before=0,kinetic_after=0,kinetic_delta=0;
    double elastic_before=0,elastic_linear_after=0,elastic_actual_after=0,passivity_residual=0;
    double scalar_gap=0,scalar_velocity=0,scalar_free_velocity=0,external_work=0;
    float floor_impulse=0,total_impulse=0;
    p::BodyState trial{};
};
namespace detail {
CUDARL_HD inline bool finite(double x) {return x==x&&fabs(x)<1e300;}
CUDARL_HD inline double product(const float* a,const float* b) {
    double x=0;for(int i=0;i<p::dofs;++i) x+=double(a[i])*b[i];return x;
}
CUDARL_HD inline double residual(int count,const double H[kRows][kRows],const double* q,
    const bool* active,const double* impulse) {
    double worst=0;
    for(int i=0;i<count;++i) if(active[i]) {
        double value=q[i];for(int j=0;j<count;++j) if(active[j]) value+=H[i][j]*impulse[j];
        if(!finite(value)||!finite(impulse[i])) return 1e300;
        worst=fmax(worst,fabs(fmin(impulse[i]*H[i][i],value)));
    }
    return worst;
}
// Solve an independent active Gram block. A dependent row is handled by the
// dual basis pivot, never by replacing a pivot or adding diagonal compliance.
CUDARL_HD inline bool basis_solve(int n,const int* basis,const double W[kRows][kRows],
    const double* rhs,double* solution) {
    double L[p::dofs][p::dofs]{};
    for(int i=0;i<n;++i) for(int j=0;j<=i;++j) {
        double value=W[basis[i]][basis[j]];
        for(int k=0;k<j;++k) value-=L[i][k]*L[j][k];
        if(!finite(value)||(i==j&&value<=kRankRelative*W[basis[i]][basis[i]])) return false;
        L[i][j]=i==j?sqrt(value):value/L[j][j];
    }
    for(int i=0;i<n;++i) {double x=rhs[i];for(int j=0;j<i;++j) x-=L[i][j]*solution[j];solution[i]=x/L[i][i];}
    for(int i=n-1;i>=0;--i) {double x=solution[i];for(int j=i+1;j<n;++j) x-=L[j][i]*solution[j];solution[i]=x/L[i][i];if(!finite(solution[i])) return false;}
    return true;
}
CUDARL_HD inline bool hard_solve(int count,const double W[kRows][kRows],const double* q,
    const bool* active,double* impulse,SolverStats& stats,double tolerance) {
    int basis[p::dofs]{},rank=0;
    for(int i=0;i<count;++i) impulse[i]=0;
    while(stats.iterations<kBasisIterations) {
        int entering=-1;double minimum=-tolerance;
        for(int i=0;i<count;++i) if(active[i]) {
            double value=q[i];for(int j=0;j<count;++j) value+=W[i][j]*impulse[j];
            if(value<minimum) {minimum=value;entering=i;}
        }
        if(entering<0) {
            stats.residual=residual(count,W,q,active,impulse);stats.basis_rank=rank;
            stats.finite=finite(stats.residual);stats.converged=stats.finite&&stats.residual<=tolerance;return stats.converged;
        }
        bool inserted=false;
        while(!inserted&&stats.iterations<kBasisIterations) {
            ++stats.iterations;
            double rhs[p::dofs]{},alpha[p::dofs]{};
            for(int j=0;j<rank;++j) rhs[j]=W[basis[j]][entering];
            if(!basis_solve(rank,basis,W,rhs,alpha)) {stats.rank_failure=true;return false;}
            double schur=W[entering][entering],value=q[entering];
            for(int j=0;j<rank;++j) schur-=W[entering][basis[j]]*alpha[j];
            for(int j=0;j<count;++j) value+=W[entering][j]*impulse[j];
            const double rank_limit=kRankRelative*W[entering][entering];
            if(!finite(schur)||!finite(value)||schur< -rank_limit) {stats.finite=false;return false;}
            double blocking=1e300;int leaving=-1;
            for(int j=0;j<rank;++j) if(alpha[j]>0) {
                const double step=impulse[basis[j]]/alpha[j];
                if(step<blocking) {blocking=step;leaving=j;}
            }
            const bool independent=schur>rank_limit;
            const double complete=independent?fmax(0.,-value/schur):1e300;
            const bool reaches=independent&&complete<=blocking;
            const double step=reaches?complete:blocking;
            if(!finite(step)||step>=1e299) {stats.rank_failure=true;return false;}
            impulse[entering]+=step;
            for(int j=0;j<rank;++j) {
                auto& amount=impulse[basis[j]];amount-=step*alpha[j];
                if(amount< -kImpulseRoundoff||!finite(amount)) {stats.finite=false;return false;}
                if(amount<0) amount=0;
            }
            if(reaches) {
                if(rank>=p::dofs) {stats.rank_failure=true;return false;}
                basis[rank++]=entering;inserted=true;
            } else {
                if(leaving<0) {stats.rank_failure=true;return false;}
                impulse[basis[leaving]]=0;
                for(int j=leaving+1;j<rank;++j) basis[j-1]=basis[j];--rank;
            }
            stats.basis_rank=rank;
        }
    }
    stats.residual=residual(count,W,q,active,impulse);stats.basis_rank=rank;
    stats.finite=finite(stats.residual);stats.converged=stats.finite&&stats.residual<=tolerance;return stats.converged;
}
CUDARL_HD inline bool mass_factor(const p::System& system,double L[p::dofs][p::dofs]) {
    for(int i=0;i<p::dofs;++i) for(int j=0;j<=i;++j) {
        double x=system.M[i][j];for(int k=0;k<j;++k) x-=L[i][k]*L[j][k];
        if(!finite(x)||(i==j&&x<=0)) return false;
        L[i][j]=i==j?sqrt(x):x/L[j][j];
    }
    return true;
}
CUDARL_HD inline bool mass_solve(const double L[p::dofs][p::dofs],const float* rhs,double* x) {
    for(int i=0;i<p::dofs;++i) {double v=rhs[i];for(int j=0;j<i;++j) v-=L[i][j]*x[j];x[i]=v/L[i][i];}
    for(int i=p::dofs-1;i>=0;--i) {double v=x[i];for(int j=i+1;j<p::dofs;++j) v-=L[j][i]*x[j];x[i]=v/L[i][i];if(!finite(x[i])) return false;}
    return true;
}
CUDARL_HD inline double kinetic(const p::System& system,const float* velocity) {
    double value=0;for(int i=0;i<p::dofs;++i) for(int j=0;j<p::dofs;++j) value+=.5*double(velocity[i])*system.M[i][j]*velocity[j];return value;
}
}

// Zero-integration algebra. H is W+D for A and exactly W for B. All impulses
// are reset; no hidden warm start, diagonal ridge or tolerance adaptation.
CUDARL_HD inline bool solve_contact(int count,const double H[kRows][kRows],const double* q,
    const bool* active,Law law,double* impulse,SolverStats& stats,double tolerance=kResidualAbsolute) {
    stats={};stats.tolerance=tolerance;
    if(count<1||count>kRows||!detail::finite(tolerance)||tolerance<=0||
       (law!=Law::Compliant&&law!=Law::Hard)) {stats.finite=false;return false;}
    for(int i=0;i<count;++i) {
        impulse[i]=0;
        if(!active[i]) continue;
        if(!detail::finite(q[i])||!detail::finite(H[i][i])||H[i][i]<=0) {stats.finite=false;return false;}
        for(int j=0;j<count;++j) if(active[j]&&!detail::finite(H[i][j])) {stats.finite=false;return false;}
    }
    if(law==Law::Hard) return detail::hard_solve(count,H,q,active,impulse,stats,tolerance);
    for(int sweep=0;sweep<kComplianceSweeps;++sweep) {
        for(int i=0;i<count;++i) if(active[i]) {
            double value=q[i];for(int j=0;j<count;++j) if(active[j]) value+=H[i][j]*impulse[j];
            impulse[i]=fmax(0.,impulse[i]-value/H[i][i]);
        }
        stats.iterations=sweep+1;stats.residual=detail::residual(count,H,q,active,impulse);
        if(!detail::finite(stats.residual)||stats.residual>=1e299) {stats.finite=false;return false;}
        if(stats.residual<=tolerance) {stats.converged=true;return true;}
    }
    return false;
}

struct ScalarState {double gap=0,velocity=0;};
CUDARL_HD inline bool scalar_step(ScalarState& state,double mass,double inward_load,double h,Law law,
    Telemetry* output=nullptr) {
    Telemetry local;auto& out=output?*output:local;out={};
    if(!detail::finite(state.gap)||!detail::finite(state.velocity)||!detail::finite(mass)||mass<=0||
       !detail::finite(inward_load)||!detail::finite(h)||h<=0||
       (law!=Law::Compliant&&law!=Law::Hard)) {out.finite=false;return false;}
    const double allowed=law==Law::Hard?kHardGapTolerance:contact::kOverlapTolerance;
    if(state.gap< -allowed) {out.geometry_valid=false;return false;}
    const double free=state.velocity-h*inward_load/mass;
    out.scalar_free_velocity=free;out.kinetic_before=.5*mass*free*free;
    out.external_work=.5*mass*(free*free-state.velocity*state.velocity);
    double H[kRows][kRows]{},q[kRows]{},impulse[kRows]{};bool active[kRows]{};
    H[0][0]=1./mass;
    if(law==Law::Compliant) {
        active[0]=state.gap<=0||state.gap+h*free<=0;
        H[0][0]+=1./(h*(kDamping+h*kStiffness));
        q[0]=free+kStiffness*state.gap/(kDamping+h*kStiffness);
    } else {active[0]=true;q[0]=free+fmax(0.,state.gap)/h;}
    out.tolerance=kResidualAbsolute+kResidualRelative*fabs(free);
    SolverStats stats;const bool solved=solve_contact(1,H,q,active,law,impulse,stats,out.tolerance);
    out.finite=stats.finite;out.converged=solved;out.iterations=stats.iterations;out.basis_rank=stats.basis_rank;out.residual=stats.residual;
    if(!solved) return false;
    const double velocity=free+impulse[0]/mass,gap=state.gap+h*velocity;
    out.scalar_gap=gap;out.scalar_velocity=velocity;out.pose_ready=true;
    out.kinetic_after=.5*mass*velocity*velocity;out.kinetic_delta=out.kinetic_after-out.kinetic_before;
    out.contact_work=impulse[0]*.5*(free+velocity);
    if(law==Law::Compliant) {
        const double x0=fmin(0.,state.gap),x1=fmin(0.,gap);
        out.elastic_before=.5*kStiffness*x0*x0;out.elastic_linear_after=out.elastic_actual_after=.5*kStiffness*x1*x1;
    }
    out.passivity_residual=out.kinetic_delta+out.elastic_actual_after-out.elastic_before;
    auto& row=out.rows[0];row.body=0;row.face=2;row.gap=static_cast<float>(state.gap);row.active=active[0];
    row.k=law==Law::Compliant?kStiffness:0;row.c=law==Law::Compliant?kDamping:0;row.impulse=static_cast<float>(impulse[0]);
    row.velocity_free=static_cast<float>(free);row.velocity_after=static_cast<float>(velocity);row.effective_mass=static_cast<float>(mass);
    row.residual=static_cast<float>(H[0][0]*impulse[0]+q[0]);out.active_rows=active[0]?1:0;
    out.floor_impulse=out.total_impulse=row.impulse;
    out.finite=detail::finite(gap)&&detail::finite(velocity)&&detail::finite(out.passivity_residual);
    out.geometry_valid=out.finite&&gap>=-allowed;
    if(!out.finite||!out.geometry_valid) return false;
    state={gap,velocity};return true;
}

// One attempted articulated substep. First solve the unchanged free water /
// gravity / actuator equations, THEN apply normal impulses through M^{-1}.
// This explicit operator split is shared by A and B. Each endpoint is a full
// calibrated patch (k320/c1.95959), unlike the prior half-weight endpoint law.
// False leaves the caller unchanged. pose_ready/trial retain a solved but
// geometrically rejected pose, so failure cannot disappear from the record.
CUDARL_HD inline bool substep(p::BodyState& state,const float* action,float h,Law law,
    p::Parameters physical={},p::V current={},Telemetry* output=nullptr) {
    Telemetry local;auto& out=output?*output:local;out={};out.trial=state;
    if(!p::finite_body(state)||!p::finite_vector(current)||!p::finite_number(h)||h<=0||
       !p::finite_number(physical.density)||physical.density<0||!p::finite_number(physical.viscosity)||physical.viscosity<0||
       !p::finite_number(physical.gravity)||physical.gravity<0||
       (law!=Law::Compliant&&law!=Law::Hard)) {out.finite=false;return false;}
    for(int j=0;j<4;++j) if(!p::finite_number(action[j])) {out.finite=false;return false;}
    const auto initial=contact::inspect(state);
    if(initial.invalid||initial.escape||initial.excessive||(law==Law::Hard&&initial.min_gap< -kHardGapTolerance)) {out.geometry_valid=false;return false;}
    auto system=p::assemble(state,physical,current);
    for(int j=0;j<4;++j) {
        const float displacement=state.angle[j]-p::clamp(state.angle[j],-p::limit,p::limit);
        system.force[6+j]+=p::torque*p::clamp(action[j],-1,1)-1e-4f*displacement;
        system.R[6+j][6+j]+=p::torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    float Lfree[p::dofs][p::dofs]{},rhs[p::dofs]{};
    for(int i=0;i<p::dofs;++i) {rhs[i]=h*(system.force[i]-system.C[i]);for(int j=0;j<p::dofs;++j) rhs[i]+=system.M[i][j]*state.velocity[j];}
    if(!contact_response::detail::factor(system,h,Lfree)||!contact_response::detail::linear_solve(Lfree,rhs,out.free_velocity)) {out.finite=false;return false;}
    const auto free_trial=contact_response::detail::preview(state,out.free_velocity,h);
    if(!p::finite_body(free_trial)) {out.finite=false;return false;}
    const auto geometry=p::shape(state),predicted=p::shape(free_trial);
    bool active[kRows]{};int index=0;double speed=0;
    for(int body=0;body<3;++body) {
        const auto& link=geometry.link[body];const int endpoints=body==0?1:2;
        for(int face=0;face<6;++face) for(int end=0;end<endpoints;++end) {
            auto& row=out.rows[index];row.body=body;row.face=face;row.end=end;
            const auto n=contact::normal(face),offset=contact::support_offset(link,body,n,end);
            for(int j=0;j<p::dofs;++j) row.jacobian[j]=p::dot(n,link.linear[j]+p::cross(link.angular[j],offset));
            row.gap=contact_response::detail::gap(geometry,row);row.k=law==Law::Compliant?kStiffness:0;row.c=law==Law::Compliant?kDamping:0;
            row.velocity_free=static_cast<float>(detail::product(row.jacobian,out.free_velocity));speed=fmax(speed,fabs(double(row.velocity_free)));
            active[index]=law==Law::Hard||row.gap<=0||row.gap+h*row.velocity_free<=0||contact_response::detail::gap(predicted,row)<=0;
            row.active=active[index];++index;
        }
    }
    out.tolerance=kResidualAbsolute+kResidualRelative*speed;
    double Lmass[p::dofs][p::dofs]{},response[kRows][p::dofs]{},H[kRows][kRows]{},q[kRows]{},impulse[kRows]{};
    if(!detail::mass_factor(system,Lmass)) {out.finite=false;return false;}
    for(int i=0;i<kRows;++i) if(!detail::mass_solve(Lmass,out.rows[i].jacobian,response[i])) {out.finite=false;return false;}
    for(int i=0;i<kRows;++i) {
        auto& row=out.rows[i];
        for(int j=i;j<kRows;++j) {double x=0;for(int k=0;k<p::dofs;++k) x+=double(row.jacobian[k])*response[j][k];H[i][j]=H[j][i]=x;}
        if(!detail::finite(H[i][i])||H[i][i]<=0) {out.finite=false;return false;}
        row.effective_mass=static_cast<float>(1./H[i][i]);
        if(law==Law::Compliant) {H[i][i]+=1./(double(h)*(kDamping+double(h)*kStiffness));q[i]=row.velocity_free+kStiffness*double(row.gap)/(kDamping+double(h)*kStiffness);}
        else q[i]=row.velocity_free+fmax(0.,double(row.gap))/h;
    }
    for(int pass=0;pass<(law==Law::Compliant?kActivationPasses:1);++pass) {
        SolverStats stats;out.activation_passes=pass+1;
        const bool solved=solve_contact(kRows,H,q,active,law,impulse,stats,out.tolerance);
        out.iterations+=stats.iterations;out.basis_rank=stats.basis_rank;out.residual=stats.residual;out.finite=stats.finite;out.converged=solved;
        out.active_rows=0;
        for(int j=0;j<p::dofs;++j) {double x=out.free_velocity[j];for(int i=0;i<kRows;++i) x+=response[i][j]*impulse[i];out.after_velocity[j]=static_cast<float>(x);}
        for(int i=0;i<kRows;++i) {
            auto& row=out.rows[i];row.active=active[i];row.impulse=static_cast<float>(impulse[i]);row.velocity_after=static_cast<float>(detail::product(row.jacobian,out.after_velocity));
            double value=q[i];for(int j=0;j<kRows;++j) value+=H[i][j]*impulse[j];row.residual=static_cast<float>(value);if(row.active) ++out.active_rows;
        }
        if(!solved) return false;
        auto next=state;contact_response::detail::integrate(next,out.after_velocity,action,h);
        out.trial=next;out.pose_ready=true;
        if(!p::finite_body(next)) {out.finite=false;return false;}
        const auto final_geometry=p::shape(next);bool added=false;
        if(law==Law::Compliant) for(int i=0;i<kRows;++i) if(!active[i]) {
            const auto& row=out.rows[i];
            if(row.gap+h*row.velocity_after<=0||contact_response::detail::gap(final_geometry,row)<=0) {active[i]=true;added=true;}
        }
        if(added) {if(pass+1==kActivationPasses) {out.missed_activation=true;return false;}continue;}
        out.kinetic_before=detail::kinetic(system,out.free_velocity);out.kinetic_after=detail::kinetic(system,out.after_velocity);out.kinetic_delta=out.kinetic_after-out.kinetic_before;
        for(int i=0;i<kRows;++i) {
            const auto& row=out.rows[i];out.total_impulse+=row.impulse;if(row.face==2) out.floor_impulse+=row.impulse;
            out.contact_work+=impulse[i]*.5*(double(row.velocity_free)+detail::product(row.jacobian,out.after_velocity));
            if(law==Law::Compliant) {
                const double g0=fmin(0.,double(row.gap)),g1=fmin(0.,double(row.gap)+h*detail::product(row.jacobian,out.after_velocity));
                const double actual=fmin(0.,double(contact_response::detail::gap(final_geometry,row)));
                out.elastic_before+=.5*kStiffness*g0*g0;out.elastic_linear_after+=.5*kStiffness*g1*g1;out.elastic_actual_after+=.5*kStiffness*actual*actual;
            }
        }
        out.passivity_residual=out.kinetic_delta+out.elastic_actual_after-out.elastic_before;
        const auto proof=contact::inspect(next);
        out.geometry_valid=!proof.invalid&&!proof.escape&&!proof.excessive&&(law!=Law::Hard||proof.min_gap>=-kHardGapTolerance);
        if(!out.geometry_valid) return false;
        state=next;return true;
    }
    return false;
}
}
#endif
