#ifndef CUDARL_CONTACT_MOMENTUM_AUDIT_H
#define CUDARL_CONTACT_MOMENTUM_AUDIT_H
#include "physics/contact_pilot.h"
#include <cfloat>

// Diagnostic-only instrumentation of the frozen compliant-A step. No caller of
// the production world includes this header. The two authorized replay calls
// count as physical advances; certify() below performs algebra only.
namespace cudarl::contact_momentum_audit {
namespace a=contact_pilot;
namespace p=physics;
constexpr int kRows=a::kRows;
constexpr double kDoubleUnitRoundoff=0x1p-53;
constexpr double kOldAbsolute=1e-13,kOldRelative=2e-5;

struct Coordinate {
    // Coordinates 0..2 are linear impulse (N s), 3..9 generalized angular
    // impulse (N m s). Scales/normalizations reproduce the frozen checker.
    double raw_delta=0,raw_impulse=0,raw_residual=0,raw_scale=0;
    double stored_delta=0,stored_impulse=0,stored_residual=0,stored_scale=0;
    double raw_allowance=0,old_stored_allowance=0;
    double old_stored_normalized=0,raw_normalized=0;
    double velocity_rounding_signed=0,impulse_rounding_signed=0;
    double predicted_stored_residual=0,rounding_identity_error=0;
    double velocity_half_ulp_bound=0,impulse_half_ulp_bound=0;
    double arithmetic_bound=0,bounded_stored_allowance=0,bounded_normalized=0;
    bool finite=true,raw_passed=false,identity_passed=false;
    bool quantization_explained=false,bounded_passed=false;
};
struct Audit {
    a::Telemetry telemetry{};
    // The original float M/J entries are promoted exactly, not reassembled by
    // an independent call. free_velocity is the ACTUAL float free solve result
    // promoted to double; this experiment does not invent a double free solve.
    double mass[p::dofs][p::dofs]{},jacobian[kRows][p::dofs]{};
    double response[kRows][p::dofs]{};
    double free_velocity[p::dofs]{},raw_velocity[p::dofs]{},raw_impulse[kRows]{};
    float stored_velocity[p::dofs]{},stored_impulse[kRows]{};
    double velocity_conversion_error[p::dofs]{},impulse_conversion_error[kRows]{};
    double velocity_half_ulp[p::dofs]{},impulse_half_ulp[kRows]{};
    Coordinate coordinate[p::dofs]{};
    bool captured=false,rounding_consistent=false,finite=false;
    bool raw_passed=false,bounded_passed=false;
    double maximum_old_normalized=0,maximum_raw_normalized=0;
    double maximum_bounded_normalized=0;
};

namespace detail {
CUDARL_HD inline double gamma(int n) {
    return (n*kDoubleUnitRoundoff)/(1-n*kDoubleUnitRoundoff);
}
// Conservative RN error radius around the actual float conversion. Taking
// the larger adjacent spacing handles powers of two and subnormal values.
// Conversion consistency is checked separately so corrupt output cannot
// increase its own allowance by presenting a large "conversion error".
CUDARL_HD inline double half_ulp(double raw) {
    const float rounded=static_cast<float>(raw);
    if(!p::finite_number(rounded)) return 1e300;
    const float up=nextafterf(rounded,FLT_MAX),down=nextafterf(rounded,-FLT_MAX);
    const double spacing=fmax(fabs(double(up)-rounded),fabs(double(rounded)-down));
    return .5*spacing;
}
}

// r_stored = r_raw + M(v_float-v_double) - J^T(p_float-p_double).
// The conversion allowance is sum |M|*halfULP(v) + |J|*halfULP(p).
// The free velocity is identical in both residuals, hence contributes no
// conversion uncertainty. gamma_n bounds double arithmetic evaluating this
// identity, with a conservative path count of 2*dofs+3*rows+16 operations.
// Crucially, the original consistency gate is still required on r_raw. This
// only avoids treating representational rounding as a physical impulse error.
CUDARL_HD inline bool certify(Audit& audit) {
    audit.rounding_consistent=audit.captured;audit.finite=audit.captured;
    audit.raw_passed=audit.captured;audit.bounded_passed=audit.captured;
    audit.maximum_old_normalized=audit.maximum_raw_normalized=audit.maximum_bounded_normalized=0;
    for(int j=0;j<p::dofs;++j) {
        audit.velocity_conversion_error[j]=double(audit.stored_velocity[j])-audit.raw_velocity[j];
        audit.velocity_half_ulp[j]=detail::half_ulp(audit.raw_velocity[j]);
        const bool finite=a::detail::finite(audit.raw_velocity[j])&&p::finite_number(audit.stored_velocity[j]);
        const bool rounded=audit.stored_velocity[j]==static_cast<float>(audit.raw_velocity[j])&&
            fabs(audit.velocity_conversion_error[j])<=audit.velocity_half_ulp[j];
        audit.finite=audit.finite&&finite;audit.rounding_consistent=audit.rounding_consistent&&finite&&rounded;
    }
    for(int row=0;row<kRows;++row) {
        audit.impulse_conversion_error[row]=double(audit.stored_impulse[row])-audit.raw_impulse[row];
        audit.impulse_half_ulp[row]=detail::half_ulp(audit.raw_impulse[row]);
        const bool finite=a::detail::finite(audit.raw_impulse[row])&&p::finite_number(audit.stored_impulse[row]);
        const bool rounded=audit.stored_impulse[row]==static_cast<float>(audit.raw_impulse[row])&&
            fabs(audit.impulse_conversion_error[row])<=audit.impulse_half_ulp[row];
        audit.finite=audit.finite&&finite;audit.rounding_consistent=audit.rounding_consistent&&finite&&rounded;
    }
    for(int i=0;i<p::dofs;++i) {
        auto& c=audit.coordinate[i];c={};double arithmetic_scale=0;
        for(int j=0;j<p::dofs;++j) {
            const double m=audit.mass[i][j],free=audit.free_velocity[j];
            const double raw_term=m*(audit.raw_velocity[j]-free);
            const double stored_term=m*(double(audit.stored_velocity[j])-free);
            c.raw_delta+=raw_term;c.raw_scale+=fabs(raw_term);
            c.stored_delta+=stored_term;c.stored_scale+=fabs(stored_term);
            c.velocity_rounding_signed+=m*audit.velocity_conversion_error[j];
            c.velocity_half_ulp_bound+=fabs(m)*audit.velocity_half_ulp[j];
            arithmetic_scale+=fabs(m)*(fabs(audit.raw_velocity[j])+fabs(double(audit.stored_velocity[j]))+2*fabs(free));
        }
        for(int row=0;row<kRows;++row) {
            const double j=audit.jacobian[row][i];
            c.raw_impulse+=j*audit.raw_impulse[row];c.stored_impulse+=j*audit.stored_impulse[row];
            c.impulse_rounding_signed-=j*audit.impulse_conversion_error[row];
            c.impulse_half_ulp_bound+=fabs(j)*audit.impulse_half_ulp[row];
            arithmetic_scale+=fabs(j)*(fabs(audit.raw_impulse[row])+fabs(double(audit.stored_impulse[row])));
        }
        c.raw_residual=c.raw_delta-c.raw_impulse;c.stored_residual=c.stored_delta-c.stored_impulse;
        c.raw_allowance=kOldAbsolute+kOldRelative*fmax(c.raw_scale,fabs(c.raw_impulse));
        c.old_stored_allowance=kOldAbsolute+kOldRelative*fmax(c.stored_scale,fabs(c.stored_impulse));
        c.arithmetic_bound=detail::gamma(2*p::dofs+3*kRows+16)*arithmetic_scale+DBL_MIN*(2*p::dofs+3*kRows+16);
        c.predicted_stored_residual=c.raw_residual+c.velocity_rounding_signed+c.impulse_rounding_signed;
        c.rounding_identity_error=c.stored_residual-c.predicted_stored_residual;
        c.bounded_stored_allowance=c.raw_allowance+c.velocity_half_ulp_bound+c.impulse_half_ulp_bound+c.arithmetic_bound;
        c.old_stored_normalized=fabs(c.stored_residual)/c.old_stored_allowance;
        c.raw_normalized=fabs(c.raw_residual)/c.raw_allowance;
        c.bounded_normalized=fabs(c.stored_residual)/c.bounded_stored_allowance;
        c.finite=audit.finite&&a::detail::finite(c.raw_residual)&&a::detail::finite(c.stored_residual)&&
            a::detail::finite(c.bounded_stored_allowance)&&a::detail::finite(c.arithmetic_bound);
        c.raw_passed=c.finite&&fabs(c.raw_residual)<=c.raw_allowance+c.arithmetic_bound;
        c.identity_passed=c.finite&&fabs(c.rounding_identity_error)<=c.arithmetic_bound;
        c.quantization_explained=c.finite&&fabs(c.stored_residual-c.raw_residual)<=
            c.velocity_half_ulp_bound+c.impulse_half_ulp_bound+c.arithmetic_bound;
        c.bounded_passed=c.raw_passed&&audit.rounding_consistent&&c.identity_passed&&c.quantization_explained&&
            fabs(c.stored_residual)<=c.bounded_stored_allowance;
        audit.finite=audit.finite&&c.finite;audit.raw_passed=audit.raw_passed&&c.raw_passed;
        audit.bounded_passed=audit.bounded_passed&&c.bounded_passed;
        audit.maximum_old_normalized=fmax(audit.maximum_old_normalized,c.old_stored_normalized);
        audit.maximum_raw_normalized=fmax(audit.maximum_raw_normalized,c.raw_normalized);
        audit.maximum_bounded_normalized=fmax(audit.maximum_bounded_normalized,c.bounded_normalized);
    }
    return audit.bounded_passed;
}

// Literal instrumented clone of contact_pilot::substep with law fixed to A.
// Captures are assignments only. Original operation order, factorization,
// solve, float conversions, integration and rejection rules are unchanged.
CUDARL_HD inline bool substep(p::BodyState& state,const float* action,float h,
    p::Parameters physical,p::V current,Audit& audit) {
    audit={};auto& out=audit.telemetry;out.trial=state;
    if(!p::finite_body(state)||!p::finite_vector(current)||!p::finite_number(h)||h<=0||
       !p::finite_number(physical.density)||physical.density<0||!p::finite_number(physical.viscosity)||physical.viscosity<0||
       !p::finite_number(physical.gravity)||physical.gravity<0) {out.finite=false;return false;}
    for(int j=0;j<4;++j) if(!p::finite_number(action[j])) {out.finite=false;return false;}
    const auto initial=contact::inspect(state);
    if(initial.invalid||initial.escape||initial.excessive) {out.geometry_valid=false;return false;}
    auto system=p::assemble(state,physical,current);
    for(int i=0;i<p::dofs;++i) for(int j=0;j<p::dofs;++j) audit.mass[i][j]=system.M[i][j];
    for(int j=0;j<4;++j) {
        const float displacement=state.angle[j]-p::clamp(state.angle[j],-p::limit,p::limit);
        system.force[6+j]+=p::torque*p::clamp(action[j],-1,1)-1e-4f*displacement;
        system.R[6+j][6+j]+=p::torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    float Lfree[p::dofs][p::dofs]{},rhs[p::dofs]{};
    for(int i=0;i<p::dofs;++i) {rhs[i]=h*(system.force[i]-system.C[i]);for(int j=0;j<p::dofs;++j) rhs[i]+=system.M[i][j]*state.velocity[j];}
    if(!contact_response::detail::factor(system,h,Lfree)||!contact_response::detail::linear_solve(Lfree,rhs,out.free_velocity)) {out.finite=false;return false;}
    for(int j=0;j<p::dofs;++j) audit.free_velocity[j]=double(out.free_velocity[j]);
    const auto free_trial=contact_response::detail::preview(state,out.free_velocity,h);
    if(!p::finite_body(free_trial)) {out.finite=false;return false;}
    const auto geometry=p::shape(state),predicted=p::shape(free_trial);
    bool active[kRows]{};int index=0;double speed=0;
    for(int body=0;body<3;++body) {
        const auto& link=geometry.link[body];const int endpoints=body==0?1:2;
        for(int face=0;face<6;++face) for(int end=0;end<endpoints;++end) {
            auto& row=out.rows[index];row.body=body;row.face=face;row.end=end;
            const auto n=contact::normal(face),offset=contact::support_offset(link,body,n,end);
            for(int j=0;j<p::dofs;++j) {row.jacobian[j]=p::dot(n,link.linear[j]+p::cross(link.angular[j],offset));audit.jacobian[index][j]=row.jacobian[j];}
            row.gap=contact_response::detail::gap(geometry,row);row.k=a::kStiffness;row.c=a::kDamping;
            row.velocity_free=static_cast<float>(a::detail::product(row.jacobian,out.free_velocity));speed=fmax(speed,fabs(double(row.velocity_free)));
            active[index]=row.gap<=0||row.gap+h*row.velocity_free<=0||contact_response::detail::gap(predicted,row)<=0;
            row.active=active[index];++index;
        }
    }
    out.tolerance=a::kResidualAbsolute+a::kResidualRelative*speed;
    double Lmass[p::dofs][p::dofs]{},response[kRows][p::dofs]{},H[kRows][kRows]{},q[kRows]{},impulse[kRows]{};
    if(!a::detail::mass_factor(system,Lmass)) {out.finite=false;return false;}
    for(int i=0;i<kRows;++i) if(!a::detail::mass_solve(Lmass,out.rows[i].jacobian,response[i])) {out.finite=false;return false;}
    for(int i=0;i<kRows;++i) for(int j=0;j<p::dofs;++j) audit.response[i][j]=response[i][j];
    for(int i=0;i<kRows;++i) {
        auto& row=out.rows[i];
        for(int j=i;j<kRows;++j) {double x=0;for(int k=0;k<p::dofs;++k) x+=double(row.jacobian[k])*response[j][k];H[i][j]=H[j][i]=x;}
        if(!a::detail::finite(H[i][i])||H[i][i]<=0) {out.finite=false;return false;}
        row.effective_mass=static_cast<float>(1./H[i][i]);
        H[i][i]+=1./(double(h)*(a::kDamping+double(h)*a::kStiffness));q[i]=row.velocity_free+a::kStiffness*double(row.gap)/(a::kDamping+double(h)*a::kStiffness);
    }
    for(int pass=0;pass<a::kActivationPasses;++pass) {
        a::SolverStats stats;out.activation_passes=pass+1;
        const bool solved=a::solve_contact(kRows,H,q,active,a::Law::Compliant,impulse,stats,out.tolerance);
        out.iterations+=stats.iterations;out.basis_rank=stats.basis_rank;out.residual=stats.residual;out.finite=stats.finite;out.converged=solved;
        out.active_rows=0;
        for(int j=0;j<p::dofs;++j) {double x=out.free_velocity[j];for(int i=0;i<kRows;++i) x+=response[i][j]*impulse[i];audit.raw_velocity[j]=x;out.after_velocity[j]=static_cast<float>(x);audit.stored_velocity[j]=out.after_velocity[j];}
        for(int i=0;i<kRows;++i) {
            auto& row=out.rows[i];row.active=active[i];audit.raw_impulse[i]=impulse[i];row.impulse=static_cast<float>(impulse[i]);audit.stored_impulse[i]=row.impulse;row.velocity_after=static_cast<float>(a::detail::product(row.jacobian,out.after_velocity));
            double value=q[i];for(int j=0;j<kRows;++j) value+=H[i][j]*impulse[j];row.residual=static_cast<float>(value);if(row.active) ++out.active_rows;
        }
        audit.captured=true;
        if(!solved) return false;
        auto next=state;contact_response::detail::integrate(next,out.after_velocity,action,h);
        out.trial=next;out.pose_ready=true;
        if(!p::finite_body(next)) {out.finite=false;return false;}
        const auto final_geometry=p::shape(next);bool added=false;
        for(int i=0;i<kRows;++i) if(!active[i]) {
            const auto& row=out.rows[i];
            if(row.gap+h*row.velocity_after<=0||contact_response::detail::gap(final_geometry,row)<=0) {active[i]=true;added=true;}
        }
        if(added) {if(pass+1==a::kActivationPasses) {out.missed_activation=true;return false;}continue;}
        out.kinetic_before=a::detail::kinetic(system,out.free_velocity);out.kinetic_after=a::detail::kinetic(system,out.after_velocity);out.kinetic_delta=out.kinetic_after-out.kinetic_before;
        for(int i=0;i<kRows;++i) {
            const auto& row=out.rows[i];out.total_impulse+=row.impulse;if(row.face==2) out.floor_impulse+=row.impulse;
            out.contact_work+=impulse[i]*.5*(double(row.velocity_free)+a::detail::product(row.jacobian,out.after_velocity));
            const double g0=fmin(0.,double(row.gap)),g1=fmin(0.,double(row.gap)+h*a::detail::product(row.jacobian,out.after_velocity));
            const double actual=fmin(0.,double(contact_response::detail::gap(final_geometry,row)));
            out.elastic_before+=.5*a::kStiffness*g0*g0;out.elastic_linear_after+=.5*a::kStiffness*g1*g1;out.elastic_actual_after+=.5*a::kStiffness*actual*actual;
        }
        out.passivity_residual=out.kinetic_delta+out.elastic_actual_after-out.elastic_before;
        const auto proof=contact::inspect(next);
        out.geometry_valid=!proof.invalid&&!proof.escape&&!proof.excessive;
        if(!out.geometry_valid) return false;
        state=next;return true;
    }
    return false;
}
}
#endif
