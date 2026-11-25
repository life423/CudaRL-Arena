#ifndef CUDARL_SIMULATION_POND_RUNTIME_H
#define CUDARL_SIMULATION_POND_RUNTIME_H
#include "motor/joint_motor.h"
#include "environment/pond_body_geometry.h"
#include "environment/pond_swept_clearance.h"

// Product world v4: model6 water/inertia and the saved motor's actuator contract,
// plus physical compliant contact with the same terrain Godot draws. No route,
// heading choice, pose projection, restitution or recovery is implemented here.
namespace cudarl::pond_runtime {
namespace p=physics;
namespace m=movement;
namespace bg=pond_body_geometry;
constexpr uint32_t version=4;
constexpr uint32_t maximum_contact_iterations=32,maximum_contact_backtracks=8;
constexpr double overlap_limit_m=.0005,query_tolerance_m=.00001;
constexpr uint32_t query_visit_limit=131072;
constexpr float stiffness_N_m=.8f;
struct World {
    joint_motor::World motor{};
    pond_geometry_v1::Parameters terrain=pond_geometry_v1::current_visual_parameters();
    bg::Inspection inspection{};
    p::BodyState failure_before{},failure_candidate{};
    double contact_spring_work=0,contact_damping_loss=0,contact_potential_J=0;
    double contact_elastic_residual_J=0,maximum_passive_energy_excess_J=0;
    double minimum_gap_m=1e30,minimum_control_gap_m=1e30;
    uint64_t attempted_controls=0;
    uint64_t contact_system_evaluations=0;
    double initial_kinetic_J=0;
    double initial_potential_J=0,initial_potential_uncertainty_J=0;
    bool prepared=false;
    int failure=0;
};
struct StepInfo {
    m::Telemetry dynamics{};
    double spring_work=0,damping_loss=0,potential_before=0,potential_after=0;
    float maximum_contact_force_N=0;
    uint32_t contact_points=0;
    uint32_t contact_iterations=0;
};
// Opt-in threshold telemetry for training. A finite lower bound describes
// every accepted pose and each certified free sweep. Only a precise control
// also has a measured numerical minimum in World::minimum_control_gap_m.
struct ControlClearance {
    double lower_m=1e30,minimum_exact_lower_m=1e30,minimum_free_lower_m=1e30;
    uint32_t rejected_contacts=0,exact_queries=0,exact_steps=0,trial_free_solves=0;
    bool precise=true;
};
constexpr double bank_threshold_m=.01;
// Optional diagnostic output owned by the caller. Production passes nullptr;
// tracing changes no equation, control, integration order or accepted state.
constexpr uint32_t substep_trace_capacity=20,trace_energy_fields=12;
struct SubstepTrace {
    p::BodyState before{},candidate{},after{};
    float held_targets[4]{},reference_rate[4]{},last_actual_torques[4]{};
    float requested_torque_Nm[4]{},applied_torque_normalized[4]{};
    double energy[trace_energy_fields]{};
    double step_spring_work=0,step_damping_loss=0,step_potential_before=0,step_potential_after=0;
    double minimum_gap_m=1e30,gap_lower_m=1e30,gap_upper_m=1e30;
    uint64_t attempted_substeps=0,accepted_substeps=0,contact_system_evaluations=0;
    uint32_t contact_iterations=0;
    int failure=0;
    bool recorded=false,attempted=false,accepted=false;
};
CUDARL_HD inline void capture_substep_trace(const World& w,SubstepTrace* trace,const p::BodyState& before,
                                            const StepInfo* info,bool attempted,bool accepted) {
    if(!trace)return;
    *trace={};trace->recorded=true;trace->attempted=attempted;trace->accepted=accepted;
    trace->before=before;trace->candidate=attempted?w.failure_candidate:before;trace->after=w.motor.physical.body;
    trace->failure=w.failure;trace->attempted_substeps=w.motor.attempted_substeps;trace->accepted_substeps=w.motor.physical.substeps;
    trace->contact_system_evaluations=w.contact_system_evaluations;
    trace->minimum_gap_m=w.minimum_gap_m;trace->gap_lower_m=w.inspection.lower_gap_m;trace->gap_upper_m=w.inspection.upper_gap_m;
    const double energies[trace_energy_fields]={w.motor.signed_work,w.motor.positive_work,w.motor.fluid_loss,
        w.motor.numerical_loss,w.motor.external_work,w.motor.last_energy_error,w.contact_spring_work,
        w.contact_damping_loss,w.contact_potential_J,w.contact_elastic_residual_J,w.maximum_passive_energy_excess_J,
        w.motor.fluid_loss-w.contact_damping_loss};
    for(uint32_t j=0;j<trace_energy_fields;++j)trace->energy[j]=energies[j];
    for(int j=0;j<4;++j) {
        trace->held_targets[j]=w.motor.physical.held_targets[j];trace->reference_rate[j]=w.motor.reference_rate[j];
        trace->last_actual_torques[j]=w.motor.physical.last_actual_torques[j];
        if(info){trace->requested_torque_Nm[j]=info->dynamics.requested[j];trace->applied_torque_normalized[j]=info->dynamics.torque[j]/p::torque;}
    }
    if(info) {
        trace->contact_iterations=info->contact_iterations;trace->step_spring_work=info->spring_work;
        trace->step_damping_loss=info->damping_loss;trace->step_potential_before=info->potential_before;trace->step_potential_after=info->potential_after;
    }
}
// Time-average contact along g(t)=g0+t*r,0<=t<=h. The spring is the discrete
// gradient of U(g)=.5*k*min(g,0)^2, giving h*Fe*r=-(U1-U0). The dashpot acts
// only during the compressed fraction of an approaching linear path. This
// treats first touch/release continuously without delaying the dashpot or
// adding a physical substep, tensile force, impulse rule or pose correction.
struct ContactLaw {
    int mode=0; //0no contact,1spring average,2spring plus approach damping
    double predicted_gap_m=0,elastic_multiplier_N=0,damping_multiplier_N=0;
    double constant_multiplier_N=0,resistance_N_s_m=0,contact_fraction=0;
};
CUDARL_HD inline ContactLaw contact_law(double old_gap_m,double h,double normal_rate_m_s,double damping_N_s_m) {
    ContactLaw law;law.predicted_gap_m=old_gap_m+h*normal_rate_m_s;
    const double g1=law.predicted_gap_m,k=stiffness_N_m;
    if(old_gap_m<0&&g1<0) {
        law.elastic_multiplier_N=-.5*k*(old_gap_m+g1);
        law.resistance_N_s_m=.5*k*h;law.contact_fraction=1;
    } else if(old_gap_m>=0&&g1<0) {
        const double denominator=old_gap_m-g1,ratio=g1/denominator;
        law.elastic_multiplier_N=.5*k*ratio*g1;
        law.resistance_N_s_m=-.5*k*h*ratio*((2*old_gap_m-g1)/denominator);
        law.contact_fraction=-ratio;
    } else if(old_gap_m<0&&g1>=0) {
        const double denominator=g1-old_gap_m,ratio=old_gap_m/denominator;
        law.elastic_multiplier_N=.5*k*ratio*old_gap_m;
        law.resistance_N_s_m=.5*k*h*ratio*ratio;law.contact_fraction=-ratio;
    }
    if(normal_rate_m_s<0&&law.contact_fraction>0) {
        // During entry, alpha*(-r)=-g1/h; its derivative is exactly-c.
        law.damping_multiplier_N=old_gap_m>=0?-damping_N_s_m*g1/h:-damping_N_s_m*normal_rate_m_s;
        law.resistance_N_s_m+=damping_N_s_m;
    }
    law.mode=law.damping_multiplier_N>0?2:law.elastic_multiplier_N>0?1:0;
    law.constant_multiplier_N=law.elastic_multiplier_N+law.damping_multiplier_N+law.resistance_N_s_m*normal_rate_m_s;
    return law;
}
struct ContactPoint {
    p::V jacobian[p::dofs]{},gradient{};
    double gap_m=0;
    float damping_N_s_m=0;
};
struct ContactSolution {
    double velocity[p::dofs]{};
    ContactLaw law[3];
    uint32_t iterations=0,backtracks=0,residual_evaluations=0;
};
CUDARL_HD inline double contact_rate(const ContactPoint& point,const double* velocity) {
    double value=0;for(int i=0;i<p::dofs;++i)value+=double(p::dot(point.jacobian[i],point.gradient))*velocity[i];return value;
}
struct ContactResidual {double relative=0,merit=0;ContactLaw law[3];};
CUDARL_HD inline bool contact_residual(const double base_A[10][10],const double* base_rhs,
        const ContactPoint* points,float h,const double* velocity,const float* torque,ContactResidual& residual) {
    residual={};double force[10]{};
    for(int b=0;points&&b<3;++b) {
        auto& law=residual.law[b];law=contact_law(points[b].gap_m,h,contact_rate(points[b],velocity),points[b].damping_N_s_m);
        if(!pond_geometry_v1::finite(law.predicted_gap_m)||!pond_geometry_v1::finite(law.elastic_multiplier_N)||
           !pond_geometry_v1::finite(law.damping_multiplier_N)||!pond_geometry_v1::finite(law.resistance_N_s_m)||
           law.elastic_multiplier_N<0||law.damping_multiplier_N<0||law.resistance_N_s_m<0||law.contact_fraction<0||law.contact_fraction>1)return false;
        for(int i=0;i<10;++i)force[i]+=double(h)*(law.elastic_multiplier_N+law.damping_multiplier_N)*double(p::dot(points[b].jacobian[i],points[b].gradient));
    }
    for(int i=0;i<10;++i) {
        double rhs=base_rhs[i];if(i>=6)rhs+=h*torque[i-6];rhs+=force[i];
        double product=0,scale=fabs(rhs)+1e-15;
        for(int j=0;j<10;++j){product+=base_A[i][j]*velocity[j];scale+=fabs(base_A[i][j]*velocity[j]);}
        const double difference=product-rhs;
        residual.relative=fmax(residual.relative,fabs(difference)/scale);residual.merit+=difference*difference;
    }
    return pond_geometry_v1::finite(residual.relative)&&pond_geometry_v1::finite(residual.merit);
}
// Pure finite Newton velocity algebra. Geometry/body never change here. The
// accepted result must meet the TRUE nonlinear momentum residual<=1e-8;
// passing the tangent system alone is insufficient. Backtracking changes only
// an algebraic iterate, never a physical velocity/body state or timestep.
CUDARL_HD CUDARL_DEVICE_NOINLINE bool solve_contacts(const p::BodyState& body,const p::System& base,
        const m::Reference& reference,float h,const ContactPoint* points,ContactSolution& solution,
        m::Telemetry& telemetry,int& failure,bool free_trial=false) {
    // A null list evaluates the tentative water-only system. Its caller must
    // certify separation for both the initial guess and this solved velocity
    // before it may use that solution to advance the body.
    if(!(h>0)||h>.005001f||(!points&&!free_trial)||!p::finite_body(body)){failure=2;return false;}
    double desired_v[4]{};const double omega=2*static_cast<double>(m::pi)*3,zeta=.9;
    for(int j=0;j<4;++j) {
        if(!p::finite_number(reference.angle[j])||fabsf(reference.angle[j])>p::limit||!p::finite_number(reference.rate[j])||
           fabsf(reference.rate[j])>12||!p::finite_number(reference.acceleration[j])||fabsf(reference.acceleration[j])>200){failure=2;return false;}
        const double acceleration=reference.acceleration[j]+omega*omega*(reference.angle[j]-body.angle[j])+2*zeta*omega*(reference.rate[j]-body.velocity[6+j]);
        desired_v[j]=body.velocity[6+j]+h*acceleration;
    }
    double guess[10];for(int i=0;i<10;++i)guess[i]=body.velocity[i];
    for(int b=0;points&&b<3;++b) {
        if(!pond_geometry_v1::finite(points[b].gap_m)||!p::finite_vector(points[b].gradient)||!p::finite_number(points[b].damping_N_s_m)||points[b].damping_N_s_m<0){failure=122;return false;}
        for(const auto& value:points[b].jacobian)if(!p::finite_vector(value)){failure=122;return false;}
    }
    double base_A[10][10]{},base_rhs[10]{};m::discrete_system(base,body,h,base_A,base_rhs);
    for(uint32_t iteration=0;iteration<maximum_contact_iterations;++iteration) {
        solution.iterations=iteration+1;
        double A[10][10]{},rhs[10]{},root_rhs[10]{},root_v[10]{};
        bool tangent_has_contact=false;
        for(int i=0;i<10;++i){rhs[i]=base_rhs[i];for(int j=0;j<10;++j)A[i][j]=base_A[i][j];}
        for(int b=0;points&&b<3;++b) {
            const auto& point=points[b];const auto law=contact_law(point.gap_m,h,contact_rate(point,guess),point.damping_N_s_m);double row[10];
            if(!pond_geometry_v1::finite(law.constant_multiplier_N)||!pond_geometry_v1::finite(law.resistance_N_s_m)||law.resistance_N_s_m<0){failure=122;return false;}
            if(law.constant_multiplier_N==0&&law.resistance_N_s_m==0)continue;
            tangent_has_contact=true;
            for(int i=0;i<10;++i)row[i]=p::dot(point.jacobian[i],point.gradient);
            // Double symmetric rank-one tangent; no contact rounding through
            // the original float water System or change to its arithmetic.
            for(int i=0;i<10;++i) {
                rhs[i]+=double(h)*law.constant_multiplier_N*row[i];
                for(int j=0;j<=i;++j){const double value=double(h)*law.resistance_N_s_m*row[i]*row[j];A[i][j]+=value;if(j!=i)A[j][i]+=value;}
            }
        }
        for(int i=0;i<6;++i){root_rhs[i]=rhs[i];for(int j=0;j<4;++j)root_rhs[i]-=A[i][6+j]*desired_v[j];}
        if(!m::solve(A,root_rhs,root_v,6)){failure=3;return false;}
        float requested[4]{},torque[4]{};
        for(int j=0;j<4;++j){double impulse=-rhs[6+j];for(int i=0;i<6;++i)impulse+=A[6+j][i]*root_v[i];for(int i=0;i<4;++i)impulse+=A[6+j][6+i]*desired_v[i];
            requested[j]=static_cast<float>(impulse/h);torque[j]=p::clamp(requested[j],-p::torque,p::torque);
            if(!p::finite_number(requested[j])){failure=4;return false;}rhs[6+j]+=h*torque[j];}
        double candidate[10];if(!m::solve(A,rhs,candidate,10)){failure=5;return false;}
        ContactResidual next_residual;++solution.residual_evaluations;
        if(!contact_residual(base_A,base_rhs,points,h,candidate,torque,next_residual)){failure=122;return false;}
        bool free_result=true;for(int b=0;b<3;++b)free_result=free_result&&next_residual.law[b].mode==0;
        // If an earlier predictor guessed contact but the solved path is free,
        // finish with the exact original water-only tangent/servo solve. A small
        // nonlinear residual alone must not alter the inherited free trajectory.
        bool converged=next_residual.relative<=1e-8&&!(tangent_has_contact&&free_result);
        if(!converged) {
            ContactResidual previous;++solution.residual_evaluations;
            if(!contact_residual(base_A,base_rhs,points,h,guess,torque,previous)){failure=122;return false;}
            if(next_residual.merit>previous.merit) {
                bool accepted=false;double factor=.5;
                for(uint32_t backtrack=0;backtrack<maximum_contact_backtracks;++backtrack) {
                    double trial[10];for(int i=0;i<10;++i)trial[i]=guess[i]+factor*(candidate[i]-guess[i]);
                    ContactResidual trial_residual;++solution.residual_evaluations;++solution.backtracks;
                    if(!contact_residual(base_A,base_rhs,points,h,trial,torque,trial_residual)){failure=122;return false;}
                    if(trial_residual.merit<=previous.merit) {
                        for(int i=0;i<10;++i)candidate[i]=trial[i];next_residual=trial_residual;accepted=true;break;
                    }
                    factor*=.5;
                }
                if(!accepted){failure=123;return false;}
            }
            free_result=true;for(int b=0;b<3;++b)free_result=free_result&&next_residual.law[b].mode==0;
            converged=next_residual.relative<=1e-8&&!(tangent_has_contact&&free_result);
        }
        if(converged) {
            for(int i=0;i<10;++i)solution.velocity[i]=candidate[i];
            for(int b=0;b<3;++b)solution.law[b]=next_residual.law[b];
            for(int j=0;j<4;++j){telemetry.requested[j]=requested[j];telemetry.torque[j]=torque[j];}
            telemetry.solve_residual=static_cast<float>(next_residual.relative);
            failure=0;return true;
        }
        bool changed=false;for(int i=0;i<10;++i){changed=changed||candidate[i]!=guess[i];guess[i]=candidate[i];}
        if(!changed){failure=121;return false;}
    }
    failure=120;return false;
}
CUDARL_HD inline bool valid_state(const World& w) {
    const auto& s=w.motor.physical;const p::Parameters calm;
    if(!simulation::finite_state(s)||s.invalid||s.unsafe||s.actuation!=simulation::Actuation::JointTarget||
       !p::finite_number(s.mean_squared_torque)||s.mean_squared_torque<0||s.mean_squared_torque>1||s.current.x!=0||s.current.y!=0||s.current.z!=0||
       s.fluid.density!=calm.density||s.fluid.viscosity!=calm.viscosity||s.fluid.gravity!=calm.gravity)return false;
    for(int j=0;j<4;++j)if(fabsf(s.body.angle[j])>p::limit||!p::finite_number(s.held_targets[j])||
       fabsf(s.held_targets[j])>p::limit||!p::finite_number(w.motor.reference_rate[j])||
       fabsf(w.motor.reference_rate[j])>12||!p::finite_number(s.last_actual_torques[j])||fabsf(s.last_actual_torques[j])>1)return false;
    const double diagnostics[]={w.initial_kinetic_J,w.initial_potential_J,w.initial_potential_uncertainty_J,
        w.motor.signed_work,w.motor.positive_work,w.motor.fluid_loss,w.motor.numerical_loss,w.motor.external_work,w.motor.last_energy_error,
        w.contact_spring_work,w.contact_damping_loss,w.contact_potential_J,w.contact_elastic_residual_J,w.maximum_passive_energy_excess_J};
    for(double value:diagnostics)if(!pond_geometry_v1::finite(value))return false;
    return pond_geometry_v1::valid_parameters(w.terrain);
}
CUDARL_HD inline double potential(const bg::Inspection& q) {
    double value=0;for(const auto& link:q.link)if(link.upper_gap_m<0)value+=.5*stiffness_N_m*link.upper_gap_m*link.upper_gap_m;return value;
}
CUDARL_HD inline double potential_uncertainty(const bg::Inspection& q) {
    double upper=0;for(const auto& link:q.link)if(link.lower_gap_m<0)upper+=.5*stiffness_N_m*link.lower_gap_m*link.lower_gap_m;
    return fmax(0.,upper-potential(q));
}
CUDARL_HD inline bool inspect(World& w) {
    w.inspection=bg::inspect(w.motor.physical.body,w.terrain,query_tolerance_m,query_visit_limit);
    if(!w.inspection.valid||!w.inspection.certified){w.failure=100+static_cast<int>(w.inspection.failure_reason);return false;}
    if(w.inspection.lower_gap_m< -overlap_limit_m){w.failure=110;return false;}
    w.contact_potential_J=potential(w.inspection);return true;
}
CUDARL_HD inline bool prepare(World& w) {
    if(!valid_state(w)||!inspect(w))return false;
    w.initial_kinetic_J=m::kinetic(w.motor.physical.body);w.initial_potential_J=potential(w.inspection);
    w.minimum_gap_m=w.minimum_control_gap_m=w.inspection.lower_gap_m;
    w.initial_potential_uncertainty_J=potential_uncertainty(w.inspection);w.prepared=true;return true;
}
CUDARL_HD CUDARL_DEVICE_NOINLINE bool substep(World& w,const m::Reference& reference,float h,StepInfo& info,
                                              bool has_previous_post=false,ControlClearance* clearance=nullptr) {
    auto& body=w.motor.physical.body;info={};w.failure_before=w.failure_candidate=body;
    if(!(h>0)||h>.005001f||!valid_state(w))return false;
    // At control entry the caller may have changed the state. Inside the same
    // control only the servo reference changes between an accepted POST and
    // the next PRE, so both refer to the identical body and terrain.
    const auto system=m::assemble(body,w.motor.physical.fluid,true);
    const auto links=m::shape(body);
    ContactSolution solution;
    pond_swept_clearance::Certificate separation;
    if(clearance&&w.contact_potential_J==0) {
        int trial_failure=0;++clearance->trial_free_solves;
        if(solve_contacts(body,system,reference,h,nullptr,solution,info.dynamics,trial_failure,true))
            separation=pond_swept_clearance::certify(body,links,w.terrain,solution.velocity,h,bank_threshold_m+query_tolerance_m);
    }
    ContactPoint points[3];bool solved=separation.separated;
    if(!separation.separated) {
        if(!has_previous_post||!w.inspection.certified) {
            if(clearance)++clearance->exact_queries;
            if(!inspect(w))return false;
        }
        info.potential_before=potential(w.inspection);
        if(clearance) {
            ++clearance->exact_steps;
            clearance->minimum_exact_lower_m=fmin(clearance->minimum_exact_lower_m,w.inspection.lower_gap_m);
            solution={};info.dynamics={};
        }
        for(int b=0;b<3;++b){const auto& q=w.inspection.link[b];
            const auto& link=links.link[b];const auto offset=q.point_m-link.center;
            points[b].gap_m=q.upper_gap_m;points[b].gradient={static_cast<float>(-q.gradient_x),1,static_cast<float>(-q.gradient_z)};
            points[b].damping_N_s_m=2*sqrtf(stiffness_N_m*link.mass);
            for(int i=0;i<p::dofs;++i)points[b].jacobian[i]=link.linear[i]+p::cross(link.angular[i],offset);
        }
        solved=solve_contacts(body,system,reference,h,points,solution,info.dynamics,w.failure);
    }
    info.contact_iterations=solution.iterations;w.contact_system_evaluations+=solution.iterations;
    if(!solved)return false;
    const auto* next=solution.velocity;
    // Only real water resistance and duration-weighted approach damping are
    // dissipative. The spring tangent matrix is not viscous/fluid loss.
    for(int i=0;i<10;++i){
        info.dynamics.external_work+=h*system.force[i]*next[i];for(int j=0;j<10;++j){
            info.dynamics.fluid_loss+=h*next[i]*system.R[i][j]*next[j];
            info.dynamics.numerical_loss+=.5*(next[i]-body.velocity[i])*system.M[i][j]*(next[j]-body.velocity[j]);}}
    for(int b=0;b<3;++b)if(solution.law[b].mode!=0) {
        const auto& law=solution.law[b];const double rate=contact_rate(points[b],next);
        info.spring_work+=h*law.elastic_multiplier_N*rate;
        info.damping_loss+=-double(h)*law.damping_multiplier_N*rate;
        ++info.contact_points;info.maximum_contact_force_N=fmaxf(info.maximum_contact_force_N,
            static_cast<float>((law.elastic_multiplier_N+law.damping_multiplier_N)*p::norm(points[b].gradient)));
    }
    info.dynamics.external_work+=info.spring_work;info.dynamics.fluid_loss+=info.damping_loss;
    if(info.dynamics.solve_residual>1e-8f||info.dynamics.fluid_loss< -1e-14){w.failure=6;return false;}
    auto candidate=body;candidate.position+=p::V{static_cast<float>(next[0]),static_cast<float>(next[1]),static_cast<float>(next[2])}*(h/p::scale);
    const p::V angular{static_cast<float>(next[3]),static_cast<float>(next[4]),static_cast<float>(next[5])};const float speed=p::norm(angular);
    if(speed>1e-10f)candidate.orientation=p::axis_rotation(angular/speed,h*speed)*candidate.orientation;candidate.orientation.normalize();
    for(int j=0;j<4;++j){candidate.angle[j]+=h*static_cast<float>(next[6+j]);info.dynamics.signed_work+=h*info.dynamics.torque[j]*next[6+j];
        info.dynamics.positive_work+=h*fmax(0.,info.dynamics.torque[j]*next[6+j]);if(fabsf(candidate.angle[j])>p::limit){w.failure_candidate=candidate;w.failure=7;return false;}}
    candidate.work+=static_cast<float>(info.dynamics.positive_work);for(int i=0;i<10;++i)candidate.velocity[i]=static_cast<float>(next[i]);
    candidate.travel+=h*p::norm({candidate.velocity[0],candidate.velocity[1],candidate.velocity[2]})/p::scale;
    w.failure_candidate=candidate;
    if(separation.separated) {
        if(!p::finite_body(candidate)){w.failure=100+bg::InvalidInput;return false;}
        // U=0 and its uncertainty is zero throughout this proved free sweep.
        // No point/normal was measured. Invalidate the old witness explicitly.
        ++clearance->rejected_contacts;
        clearance->minimum_free_lower_m=fmin(clearance->minimum_free_lower_m,separation.lower_m);
        body=candidate;w.inspection={};w.contact_potential_J=0;w.failure=0;return true;
    }
    if(clearance)++clearance->exact_queries;
    const auto after=bg::inspect(candidate,w.terrain,query_tolerance_m,query_visit_limit);
    if(!after.valid||!after.certified){w.failure=100+static_cast<int>(after.failure_reason);return false;}
    if(clearance)clearance->minimum_exact_lower_m=fmin(clearance->minimum_exact_lower_m,after.lower_gap_m);
    // A historical minimum unavailable after threshold-only stepping cannot
    // be recovered from subsequent samples.
    if(pond_geometry_v1::finite(w.minimum_gap_m))w.minimum_gap_m=fmin(w.minimum_gap_m,after.lower_gap_m);
    w.minimum_control_gap_m=fmin(w.minimum_control_gap_m,after.lower_gap_m);
    if(after.lower_gap_m< -overlap_limit_m){w.failure=110;return false;}
    info.potential_after=potential(after);info.dynamics.minimum_gap=static_cast<float>(after.lower_gap_m);
    body=candidate;w.inspection=after;w.contact_potential_J=info.potential_after;w.failure=0;return true;
}
CUDARL_HD inline bool step_impl(World& w,const float* command,float h,SubstepTrace* trace,ControlClearance* clearance,bool precise_endpoint) {
    if(trace)for(uint32_t j=0;j<substep_trace_capacity;++j)trace[j]={};
    auto& s=w.motor.physical;if(!command||!w.prepared||s.paused||!valid_state(w)||w.failure!=0){
        capture_substep_trace(w,trace,s.body,nullptr,false,false);return false;}
    ++w.attempted_controls;for(int j=0;j<4;++j)if(!p::finite_number(command[j])||fabsf(command[j])>1){
        w.failure=10;capture_substep_trace(w,trace,s.body,nullptr,false,false);return false;}
    s.telemetry={};float torque_square_sum=0;
    if(!clearance&&!w.inspection.certified&&!inspect(w))return false;
    w.minimum_control_gap_m=w.inspection.certified?w.inspection.lower_gap_m:1e30;
    if(clearance)clearance->minimum_exact_lower_m=w.minimum_control_gap_m;
    for(int j=0;j<4;++j)s.raw_action[j]=command[j];
    const int substeps=h==.005f?10:(h==.0025f?20:0);if(substeps==0){
        w.failure=11;capture_substep_trace(w,trace,s.body,nullptr,false,false);return false;}
    m::Reference reference;
    for(int k=0;k<substeps;++k){
        p::BodyState trace_before;if(trace)trace_before=s.body;
        // Production target filtering remains at its original5ms cadence.
        // Half-step checks hold the same reference through two physical steps.
        if(k%(substeps/10)==0){if(!joint_motor::filter_targets(s.held_targets,w.motor.reference_rate,command,reference)){
            w.failure=10;capture_substep_trace(w,trace?trace+k:nullptr,trace_before,nullptr,false,false);return false;}}
        ++w.motor.attempted_substeps;StepInfo info;
        if(!substep(w,reference,h,info,k>0,clearance)){
            s.invalid=true;capture_substep_trace(w,trace?trace+k:nullptr,trace_before,&info,true,false);return false;}
        w.motor.signed_work+=info.dynamics.signed_work;w.motor.positive_work+=info.dynamics.positive_work;
        w.motor.fluid_loss+=info.dynamics.fluid_loss;w.motor.numerical_loss+=info.dynamics.numerical_loss;w.motor.external_work+=info.dynamics.external_work;
        w.contact_spring_work+=info.spring_work;w.contact_damping_loss+=info.damping_loss;
        w.contact_elastic_residual_J+=info.spring_work+info.potential_after-info.potential_before;
        const double kinetic=m::kinetic(s.body);
        w.motor.last_energy_error=kinetic-w.initial_kinetic_J-w.motor.signed_work+w.motor.fluid_loss+w.motor.numerical_loss-w.motor.external_work;
        if(!pond_geometry_v1::finite(w.motor.last_energy_error)||!pond_geometry_v1::finite(w.motor.positive_work+w.motor.fluid_loss)||
           fabs(w.motor.last_energy_error)>.02*(w.motor.positive_work+w.motor.fluid_loss)+2e-10){
            w.failure=9;s.invalid=true;capture_substep_trace(w,trace?trace+k:nullptr,trace_before,&info,true,false);return false;}
        const double passive_excess=kinetic+w.contact_potential_J-w.initial_kinetic_J-w.initial_potential_J-w.motor.positive_work-
            (w.motor.external_work-w.contact_spring_work);
        const double passive_allowance=.02*(w.initial_kinetic_J+w.motor.positive_work)+2e-10+
            potential_uncertainty(w.inspection)+w.initial_potential_uncertainty_J;
        if(!pond_geometry_v1::finite(passive_excess)||!pond_geometry_v1::finite(passive_allowance)){
            w.failure=9;s.invalid=true;capture_substep_trace(w,trace?trace+k:nullptr,trace_before,&info,true,false);return false;}
        w.maximum_passive_energy_excess_J=fmax(w.maximum_passive_energy_excess_J,passive_excess);
        if(passive_excess>passive_allowance){
            w.failure=111;s.invalid=true;capture_substep_trace(w,trace?trace+k:nullptr,trace_before,&info,true,false);return false;}
        ++s.substeps;++s.telemetry.executed_substeps;s.time_seconds+=h;
        for(int j=0;j<4;++j){const float normalized=info.dynamics.torque[j]/p::torque;s.last_actual_torques[j]=normalized;
            s.actual_torques[(k/(substeps/10))*4+j]=normalized;s.requested_torques[(k/(substeps/10))*4+j]=info.dynamics.requested[j];torque_square_sum+=normalized*normalized;}
        capture_substep_trace(w,trace?trace+k:nullptr,trace_before,&info,true,true);
    }
    if(clearance) {
        if(precise_endpoint&&!w.inspection.certified) {
            ++clearance->exact_queries;
            if(!inspect(w)){s.invalid=true;return false;}
            clearance->minimum_exact_lower_m=fmin(clearance->minimum_exact_lower_m,w.inspection.lower_gap_m);
        }
        clearance->lower_m=fmin(clearance->minimum_exact_lower_m,clearance->minimum_free_lower_m);
        // An unsampled pose has a 10um-certified lower endpoint at least
        // physical gap minus query tolerance, including the query roundoff.
        // A known lower endpoint below every such endpoint is the exact old
        // numerical minimum. Otherwise only the threshold class is known.
        clearance->precise=!clearance->rejected_contacts||
            clearance->minimum_exact_lower_m+query_tolerance_m<=clearance->minimum_free_lower_m;
        w.minimum_control_gap_m=clearance->precise?clearance->minimum_exact_lower_m:NAN;
        if(pond_geometry_v1::finite(w.minimum_gap_m)) {
            w.minimum_gap_m=fmin(w.minimum_gap_m,clearance->minimum_exact_lower_m);
            if(clearance->rejected_contacts&&w.minimum_gap_m+query_tolerance_m>clearance->minimum_free_lower_m)
                w.minimum_gap_m=NAN;
        }
    }
    s.mean_squared_torque=torque_square_sum/static_cast<float>(4*substeps);++s.control_steps;return true;
}
// Full quantitative geometry, extrema, witnesses and optional substep trace.
CUDARL_HD inline bool step(World& w,const float* command,float h=.005f,SubstepTrace* trace=nullptr) {
    return step_impl(w,command,h,trace,nullptr,false);
}
// Uses the same physical solve, filters, joint limits, energy gates and
// timestep. Every ambiguous substep takes the full quantitative path.
CUDARL_HD inline bool step_classified(World& w,const float* command,ControlClearance& result,
        float h=.005f,bool precise_endpoint=false) {
    result={};
    return step_impl(w,command,h,nullptr,&result,precise_endpoint);
}
}
#endif
