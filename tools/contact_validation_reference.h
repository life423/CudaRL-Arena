#ifndef CUDARL_CONTACT_VALIDATION_REFERENCE_H
#define CUDARL_CONTACT_VALIDATION_REFERENCE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Independent continuous-time scalar oracle. This header deliberately has no
// production physics/solver includes and performs no numerical time integration.
//
// Positive g points away from the wall. m*g'' = F-L, with constant inward load L.
// Before geometric arrival F=0. In contact F=max(0,-k*g-c*g'); release occurs when
// that force first reaches zero on unloading, even while g is still negative.
// A clipped dashpot must not be replaced by a bilateral spring-to-g=0 rebound.
namespace cudarl::contact_validation_reference {

// IDs match the frozen scalar fixture catalog, not an ordering chosen by error.
enum class Case : std::uint32_t { Support=0, Impact=1, Separation=2 };
enum class Phase : std::uint32_t { Flight=0, Contact=1, Released=2 };

inline constexpr double kMass=.003;
inline constexpr double kStiffness=static_cast<double>(320.f);
inline constexpr double kDamping=static_cast<double>(1.9595918f);
inline constexpr double kHorizon=.25;
inline constexpr double kSupportLoad=.0003;
inline constexpr double kImpactGap=25e-6;
inline constexpr double kImpactVelocity=-.05;

// Registered accuracy tolerances. They do not change the contact/safety law.
inline constexpr double kCompressionAbsolute=2e-6;
inline constexpr double kCompressionRelative=.05;
inline constexpr double kRestitutionAbsolute=.02;
inline constexpr double kImpulseAbsolute=1e-8;
inline constexpr double kImpulseRelative=.05;
inline constexpr double kTrajectoryGapAbsolute=10e-6;
inline constexpr double kTrajectoryVelocityAbsolute=.001;
inline constexpr std::uint64_t kReferenceQueryCap=12000;

inline const char* case_name(Case value) {
    switch(value) {
    case Case::Support:return "loaded_support";
    case Case::Impact:return "50mm_s_impact";
    case Case::Separation:return "separating_no_contact";
    }
    return "invalid";
}

struct Sample {
    double time_s=0;
    double gap_m=0,velocity_m_s=0,normal_force_N=0,normal_impulse_N_s=0;
    double kinetic_J=0,spring_J=0,mechanical_J=0;
    double load_work_J=0,load_potential_J=0,total_energy_J=0;
    double dissipated_or_discarded_energy_J=0,max_compression_m=0;
    Phase phase=Phase::Flight;
};

struct Report {
    bool passed=false;
    const char* error="";
    Case case_id=Case::Support;
    double mass_kg=kMass,stiffness_N_m=kStiffness,damping_N_s_m=kDamping;
    double horizon_s=kHorizon,initial_gap_m=0,initial_velocity_m_s=0,inward_load_N=0;
    double contact_start_time_s=-1,force_release_time_s=-1;
    double release_gap_m=0,release_velocity_m_s=0;
    double max_compression_m=0,max_compression_time_s=0;
    bool restitution_defined=false;
    double restitution=0,unrecovered_spring_at_release_J=0;
    double normal_impulse_at_horizon_N_s=0,initial_energy_J=0;
    Sample final_state{};
    std::uint64_t reference_queries=0;
};

namespace detail {
struct State {double gap=0,velocity=0;};

inline double sinc(double x) {
    if(std::abs(x)<1e-3) {const double z=x*x;return 1-z/6+z*z/120-z*z*z/5040;}
    return std::sin(x)/x;
}
inline double sinhc(double x) {
    if(std::abs(x)<1e-3) {const double z=x*x;return 1+z/6+z*z/120+z*z*z/5040;}
    return std::sinh(x)/x;
}

// exp(-a*t) times cos/sin or cosh/sinh fundamental solutions. The analytic
// critical limit S=t avoids cancellation when c is very close to 2*sqrt(k*m).
inline State contact_solution(double t,double gap0,double velocity0,double load) {
    constexpr double a=kDamping/(2*kMass),w2=kStiffness/kMass,d=a*a-w2;
    const double root=std::sqrt(std::abs(d));
    double C=1,S=t;
    if constexpr(d>0) {C=std::cosh(root*t);S=t*sinhc(root*t);}
    else if constexpr(d<0) {C=std::cos(root*t);S=t*sinc(root*t);}
    const double equilibrium=-load/kStiffness,x0=gap0-equilibrium,e=std::exp(-a*t);
    return {equilibrium+e*(x0*C+(velocity0+a*x0)*S),
        e*(velocity0*C-(a*velocity0+w2*x0)*S)};
}

// First positive solution C/S=d for the fixed near-critical impact. Elementary
// event equations, not stepwise search, root iteration, or a physical advance.
inline double event_time(double d) {
    constexpr double a=kDamping/(2*kMass),disc=a*a-kStiffness/kMass;
    if(d<=0) throw std::runtime_error("Reference requires the registered near-critical coefficients");
    if constexpr(disc==0) return 1/d;
    const double b=std::sqrt(std::abs(disc));
    if constexpr(disc<0) return std::atan2(b,d)/b;
    if(!(b<d)) throw std::runtime_error("No finite unloading event for the registered reference");
    return std::atanh(b/d)/b;
}

inline bool finite(const Sample& s) {
    return std::isfinite(s.gap_m)&&std::isfinite(s.velocity_m_s)&&
        std::isfinite(s.normal_force_N)&&std::isfinite(s.normal_impulse_N_s)&&
        std::isfinite(s.kinetic_J)&&std::isfinite(s.spring_J)&&
        std::isfinite(s.total_energy_J)&&std::isfinite(s.dissipated_or_discarded_energy_J);
}
} // namespace detail

class Reference {
    Report report_{};
    double peak_contact_time_=0;
    detail::State release_{};
    std::uint64_t queries_=0;

    detail::State contact(double t) {
        charge();
        const double velocity=report_.case_id==Case::Impact?kImpactVelocity:0.;
        return detail::contact_solution(t,0.,velocity,report_.inward_load_N);
    }
    void charge() {
        if(queries_>=kReferenceQueryCap) throw std::runtime_error("Continuous reference query cap exceeded");
        ++queries_;
    }
    Sample evaluate(double t) {
        charge();
        Sample out;out.time_s=t;
        if(report_.case_id==Case::Separation||t<report_.contact_start_time_s) {
            out.gap_m=report_.initial_gap_m+report_.initial_velocity_m_s*t;
            out.velocity_m_s=report_.initial_velocity_m_s;
        } else if(report_.force_release_time_s>=0&&t>=report_.force_release_time_s) {
            const double dt=t-report_.force_release_time_s;
            out.gap_m=release_.gap+release_.velocity*dt;
            out.velocity_m_s=release_.velocity;
            out.phase=Phase::Released;
        } else {
            // This is one closed-form evaluation; contact() would double-charge.
            const double tau=t-report_.contact_start_time_s;
            const auto state=detail::contact_solution(tau,0.,
                report_.case_id==Case::Impact?kImpactVelocity:0.,report_.inward_load_N);
            out.gap_m=state.gap;out.velocity_m_s=state.velocity;
            out.normal_force_N=std::max(0.,-kStiffness*state.gap-kDamping*state.velocity);
            out.phase=Phase::Contact;
        }
        // Integral of normal force from exact momentum balance, including the
        // actual applied constant-load impulse rather than a split-kick ledger.
        out.normal_impulse_N_s=kMass*(out.velocity_m_s-report_.initial_velocity_m_s)+report_.inward_load_N*t;
        out.kinetic_J=.5*kMass*out.velocity_m_s*out.velocity_m_s;
        const double compressed=std::min(0.,out.gap_m);
        out.spring_J=.5*kStiffness*compressed*compressed;
        out.mechanical_J=out.kinetic_J+out.spring_J;
        out.load_work_J=-report_.inward_load_N*(out.gap_m-report_.initial_gap_m);
        out.load_potential_J=-out.load_work_J;
        out.total_energy_J=out.mechanical_J+out.load_potential_J;
        out.dissipated_or_discarded_energy_J=report_.initial_energy_J+out.load_work_J-out.mechanical_J;
        if(report_.case_id==Case::Impact&&t>=report_.max_compression_time_s)
            out.max_compression_m=report_.max_compression_m;
        else out.max_compression_m=std::max(0.,-out.gap_m);
        if(!detail::finite(out)) throw std::runtime_error("Nonfinite continuous scalar reference");
        return out;
    }

public:
    explicit Reference(Case value,double impact_gap=kImpactGap) {
        report_.case_id=value;
        switch(value) {
        case Case::Support:
            report_.inward_load_N=kSupportLoad;report_.contact_start_time_s=0;
            report_.max_compression_time_s=kHorizon;
            break;
        case Case::Impact: {
            if(!std::isfinite(impact_gap)||impact_gap<0)
                throw std::invalid_argument("Impact gap must be finite and nonnegative");
            report_.initial_gap_m=impact_gap;report_.initial_velocity_m_s=kImpactVelocity;
            report_.contact_start_time_s=-report_.initial_gap_m/kImpactVelocity;
            const double a=kDamping/(2*kMass);
            peak_contact_time_=detail::event_time(a);
            const double release_time=detail::event_time(a-kStiffness/kDamping);
            report_.max_compression_time_s=report_.contact_start_time_s+peak_contact_time_;
            report_.force_release_time_s=report_.contact_start_time_s+release_time;
            if(report_.force_release_time_s>kHorizon)
                throw std::invalid_argument("Impact release must lie within the registered reference horizon");
            release_=contact(release_time);
            report_.release_gap_m=release_.gap;report_.release_velocity_m_s=release_.velocity;
            report_.restitution_defined=true;
            report_.restitution=release_.velocity/(-kImpactVelocity);
            report_.unrecovered_spring_at_release_J=.5*kStiffness*release_.gap*release_.gap;
            report_.max_compression_m=-contact(peak_contact_time_).gap;
            break;
        }
        case Case::Separation:report_.initial_velocity_m_s=.05;break;
        default:throw std::invalid_argument("Unknown continuous scalar reference case");
        }
        report_.initial_energy_J=.5*kMass*report_.initial_velocity_m_s*report_.initial_velocity_m_s;
        report_.final_state=evaluate(kHorizon);
        if(value==Case::Support) report_.max_compression_m=report_.final_state.max_compression_m;
        report_.normal_impulse_at_horizon_N_s=report_.final_state.normal_impulse_N_s;
        report_.passed=detail::finite(report_.final_state)&&report_.restitution>=0&&
            std::isfinite(report_.max_compression_m)&&report_.max_compression_m>=0;
        if(!report_.passed) report_.error="Nonfinite or invalid continuous reference report";
    }

    Sample sample(double time_s) {
        if(!std::isfinite(time_s)||time_s<0||time_s>kHorizon)
            throw std::invalid_argument("Reference time must lie in the registered [0,0.25]s horizon");
        return evaluate(time_s);
    }
    Report report() const {auto result=report_;result.reference_queries=queries_;return result;}
    std::uint64_t query_count() const {return queries_;}
};

inline double compression_tolerance(double reference) {
    return std::max(kCompressionAbsolute,kCompressionRelative*std::abs(reference));
}
inline double impulse_tolerance(double reference) {
    return std::max(kImpulseAbsolute,kImpulseRelative*std::abs(reference));
}

} // namespace cudarl::contact_validation_reference
#endif
