#ifndef CUDARL_TOOLS_CONTACT_PROBE_CASES_H
#define CUDARL_TOOLS_CONTACT_PROBE_CASES_H
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "physics/contact.h"
#include "core/checkpoint.h"

// Fixed, opt-in comparison catalog. No task, training, resets during trials,
// adaptive budgets, parameter sweeps or automatic deployment.
// Primary starts are field-for-field float copies of the ORIGINAL binary:
// build/motor-v2/2026-10-30-v1/contact/protocol.bin
// SHA256 C0488B1E09C89A15023175E654E8F720CF412D9FCCC65F7EDB767CB3E7B510A5.
// Its92-byte BodyState begins at file52+160*case. No old runtime headers needed.
// The original5s protocol is NOT claimed completed by these1.5s comparisons.
namespace cudarl::contact_probe {
namespace p=physics;namespace c=contact;
constexpr uint32_t kProtocolVersion=1;
constexpr uint64_t kArchivedCatalogPayloadHash=16067306998033709655ull;
constexpr uint64_t kArchivedSourceFailureHash=5917264646766711578ull;
constexpr uint64_t kArchivedPhysicsHash=8169187946518266246ull;
inline constexpr const char* kArchivedCatalogSha256="C0488B1E09C89A15023175E654E8F720CF412D9FCCC65F7EDB767CB3E7B510A5";
enum class Stage:uint32_t {Primary=0,Refinement=1,HeldOut=2,FreeWater=3,Passivity=4};
struct Fixture {
    p::BodyState initial{};
    p::Parameters fluid{};
    p::V current{};
    std::array<float,4> action{};
    int id=0,source_case=-1;
    const char* name="";
    bool support=false;
};
struct Trial {Fixture fixture;Stage stage=Stage::Primary;float h=.005f;int steps=0;};
constexpr int kPrimaryCount=8,kHeldOutCount=8,kFreeWaterCount=8,kPassivityCount=3;
constexpr int kMaximumSubsteps=43600;
constexpr int stage_budget(Stage stage) {
    return stage==Stage::Primary?9600:(stage==Stage::Refinement?21600:
        (stage==Stage::HeldOut?9600:(stage==Stage::FreeWater?1600:1200)));
}
inline const char* stage_name(Stage stage) {
    switch(stage) {case Stage::Primary:return "primary";case Stage::Refinement:return "refinement";
        case Stage::HeldOut:return "heldout";case Stage::FreeWater:return "freewater";default:return "passivity";}
}
constexpr bool is_final(Stage stage) {return stage==Stage::HeldOut;}
static_assert(9600+21600+9600+1600+1200==kMaximumSubsteps);

// Historical amplitude thresholds from protocol.json; all geometry/safety is
// checked initially and after EVERY actual substep, never just50ms endpoints.
constexpr float kOverlapTolerance=c::kOverlapTolerance;
constexpr double kPhysicalParityAbs=2e-4,kParityRelative=2e-3;
constexpr double kGapForceParityAbs=2e-6;
constexpr float kPenetrationRefinement=.0001f,kTerminalPositionRefinement=.0005f;
constexpr float kSupportLoadFraction=.8f,kSupportVerticalSpeed=.0005f;
constexpr float kFreeWaterClearance=.002f;
// Actual candidate normal force must be nonadhesive. Candidate solver-specific
// complementarity/convergence guards are additionally mandatory, never replaced.
constexpr float kNormalForceNegativeAllowance=1e-9f;
// No gravity, buoyancy, viscosity, current or motor work in passivity trials.
// E=kinetic+contact springs+joint soft-stop potential must stay <=E0+allowance
// at EVERY sample. This is an absolute energy ceiling, not a claim of exact
// nonlinear energy conservation. No post-run tolerance relaxation is allowed.
constexpr double kEnergyAbsoluteAllowance=1e-10,kEnergyRelativeAllowance=.001;
inline double passive_energy_allowance(double initial_energy) {
    return std::fmax(kEnergyAbsoluteAllowance,kEnergyRelativeAllowance*std::fabs(initial_energy));
}
inline bool parity_close(double reference,double actual,double absolute=kPhysicalParityAbs) {
    return std::isfinite(reference)&&std::isfinite(actual)&&
        std::fabs(actual-reference)<=absolute+kParityRelative*std::fabs(reference);
}
// Final third of1.5s = exactly0.5s:100/200/400 samples at5/2.5/1.25ms.
// Require actual geometric floor contact on EVERY window sample, mean actual
// floor-normal force >=80% buoyancy deficit, max|head_vy|<=0.5mm/s.
// Telemetry must include damping/net impulse, not legacy spring-only fields.
inline int support_window_begin(const Trial& trial) {return trial.steps-trial.steps/3;}
inline float known_buoyancy_deficit(const Fixture& fixture) {
    const auto geometry=p::shape(fixture.initial);float force=0;
    for(const auto& link:geometry.link) force+=(link.mass-fixture.fluid.density*link.volume)*fixture.fluid.gravity;
    return fmaxf(0,force);
}
inline double passive_energy(const p::BodyState& body) {
    double energy=p::energy(body);const auto geometry=p::shape(body);
    for(int link=0;link<3;++link) {
        const auto& shape=geometry.link[link];const int points=link==0?1:2;
        const double stiffness=double(c::kContactStiffness)/points;
        for(int face=0;face<6;++face) for(int end=0;end<points;++end) {
            const auto normal=c::normal(face);
            const float gap=p::dot(normal,shape.center+c::support_offset(shape,link,normal,end))+c::offset(face);
            if(gap<0) energy+=.5*stiffness*double(gap)*gap;
        }
    }
    for(float angle:body.angle) {const float displacement=angle-p::clamp(angle,-p::limit,p::limit);energy+=.5*1e-4*double(displacement)*displacement;}
    return energy;
}
namespace detail {
struct SourceRecord {float body[23],fluid[3],current[3];bool support;const char* name;};
// Ten significant decimal digits round-trip every archived IEEEfloat exactly.
inline constexpr std::array<SourceRecord,8> source{{
    {{-6.351754665e-1f,-3.060680389e+0f,-6.258419752e-1f,-2.298666723e-2f,9.805116653e-1f,-6.820607185e-2f,1.828016490e-1f,-1.039754629e+0f,1.083786845e+0f,-5.136110783e-1f,-3.050808311e-1f,-2.961491235e-3f,9.035510942e-3f,-6.820668932e-3f,-2.035367012e+0f,-2.696887553e-1f,-3.408842953e-3f,-1.879726350e-1f,1.235634446e+0f,3.005467415e+0f,3.025000572e+0f,1.187825575e-3f,3.110179901e+0f},{1.000000000e+3f,1.000000047e-3f,9.810000420e+0f},{0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},false,"protected_v1_fin_breach"},
    {{0.000000000e+0f,-3.837500095e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,1.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,-1.999999955e-2f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},{1.000000000e+3f,1.000000047e-3f,9.810000420e+0f},{0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},false,"neutral_20mm_s_floor_impact"},
    {{0.000000000e+0f,-3.840250015e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,1.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},{9.900000000e+2f,1.000000047e-3f,9.810000420e+0f},{0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},true,"1pct_negative_buoyancy_floor_support"},
    {{0.000000000e+0f,-3.572889805e+0f,0.000000000e+0f,7.468794007e-3f,1.492513865e-1f,4.941795766e-2f,9.875353575e-1f,2.000000030e-1f,2.000000030e-1f,-2.000000030e-1f,-2.000000030e-1f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},{9.900000000e+2f,1.000000047e-3f,9.810000420e+0f},{0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},true,"bent_floor_support"},
    {{0.000000000e+0f,-3.190986156e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,2.231063545e-1f,9.747940898e-1f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},{9.900000000e+2f,1.000000047e-3f,9.810000420e+0f},{0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},true,"pitched_head_floor_support"},
    {{0.000000000e+0f,-3.800250053e+0f,0.000000000e+0f,7.071067691e-1f,0.000000000e+0f,0.000000000e+0f,7.071067691e-1f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},{9.900000000e+2f,1.000000047e-3f,9.810000420e+0f},{0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},true,"rolled_fin_floor_support"},
    {{0.000000000e+0f,-3.837500095e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,1.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,-1.999999955e-2f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},{1.000000000e+3f,1.000000047e-3f,9.810000420e+0f},{4.999999888e-3f,0.000000000e+0f,0.000000000e+0f},false,"neutral_20mm_s_impact_5mm_s_current"},
    {{0.000000000e+0f,-3.840250015e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,1.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f,0.000000000e+0f},{9.900000000e+2f,1.000000047e-3f,9.810000420e+0f},{4.999999888e-3f,0.000000000e+0f,0.000000000e+0f},true,"1pct_negative_buoyancy_support_5mm_s_current"}
}};
inline Fixture archived(int id) {
    const auto& row=source[static_cast<size_t>(id)];Fixture f;f.id=id;f.source_case=id;f.name=row.name;f.support=row.support;
    const auto* b=row.body;f.initial.position={b[0],b[1],b[2]};f.initial.orientation={b[3],b[4],b[5],b[6]};
    for(int j=0;j<4;++j) f.initial.angle[j]=b[7+j];
    for(int j=0;j<10;++j) f.initial.velocity[j]=b[11+j];
    f.initial.work=b[21];f.initial.travel=b[22];
    f.fluid={row.fluid[0],row.fluid[1],row.fluid[2]};f.current={row.current[0],row.current[1],row.current[2]};return f;
}
inline void align_floor(p::BodyState& body,float gap) {body.position.y+=(gap-c::inspect(body).floor_gap)/p::scale;}
inline Fixture heldout(int id) {
    Fixture f=archived(id);f.name="heldout_fixed_perturbation";
    const float gap=c::inspect(f.initial).floor_gap;
    const float sign=id%2?1.f:-1.f;
    f.initial.position.x+=sign*.2f;f.initial.position.z-=sign*.15f; //8/6mm translations
    f.initial.orientation=p::axis_rotation({0,1,0},sign*.11f)*f.initial.orientation*p::axis_rotation({0,0,1},-sign*.03f);
    for(int j=0;j<4;++j) f.initial.angle[j]+=(id+j)%2?.02f:-.02f;
    f.initial.velocity[1]-=.001f;
    if(f.current.x!=0) f.current={0,0,.005f}; // withheld water direction, same speed
    align_floor(f.initial,gap);return f;
}
inline Fixture freewater(int id) {
    Fixture f;f.id=id;f.name="freewater_exact";f.support=false;
    const float sign=id%2?1.f:-1.f;
    f.initial.position={sign*.25f,-2.f,-sign*.2f};
    f.initial.orientation=p::axis_rotation({0,1,0},sign*.23f)*p::axis_rotation({0,0,1},sign*.12f);
    f.initial.angle[0]=sign*.05f;f.initial.angle[1]=-.03f;f.initial.angle[2]=-sign*.04f;f.initial.angle[3]=.02f;
    f.initial.velocity[0]=sign*.002f;f.initial.velocity[1]=-.001f;
    f.initial.velocity[3]=sign*.03f;f.initial.velocity[6]=sign*.04f;f.initial.velocity[9]=-.02f;
    if((id/2)%2) f.current={.005f,0,0};
    if(id>=4) f.action={.1f,-.1f,.05f,-.05f};
    return f;
}
inline Fixture passivity(int id) {
    Fixture f;f.id=id;f.fluid={0,0,0};f.current={};f.support=false;
    if(id==0) {f.name="passive_spring_release";align_floor(f.initial,-.0002f);}
    else if(id==1) {f.name="passive_floor_impact";align_floor(f.initial,.0001f);f.initial.velocity[1]=-.005f;}
    else {f.name="passive_separating_contact";align_floor(f.initial,-.0001f);f.initial.velocity[1]=.005f;}
    return f;
}
inline void write_text(checkpoint::Writer& out,const char* text) {
    const size_t n=std::strlen(text);out.value(static_cast<uint32_t>(n));
    for(size_t i=0;i<n;++i) out.value(static_cast<uint8_t>(text[i]));
}
inline void write_body(checkpoint::Writer& out,const p::BodyState& b) {
    for(float f:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w}) out.value(f);
    for(float f:b.angle) out.value(f);for(float f:b.velocity) out.value(f);out.value(b.work);out.value(b.travel);
}
}
inline std::vector<Trial> trials(Stage stage) {
    std::vector<Trial> out;
    if(stage==Stage::Primary||stage==Stage::HeldOut) {
        for(int id=0;id<8;++id) out.push_back({stage==Stage::Primary?detail::archived(id):detail::heldout(id),stage,.005f,300});
    } else if(stage==Stage::Refinement) {
        // Fixed BEFORE results: impact1, bent3, rolled5. Pair bysource_case.
        for(int id:{1,3,5}) {out.push_back({detail::archived(id),stage,.0025f,600});out.push_back({detail::archived(id),stage,.00125f,1200});}
    } else if(stage==Stage::FreeWater) {
        for(int id=0;id<8;++id) out.push_back({detail::freewater(id),stage,.005f,50});
    } else {
        for(int id=0;id<3;++id) out.push_back({detail::passivity(id),stage,.005f,100});
    }
    return out;
}
// Canonical field serialization binds everyfixedcondition, numericcriterion,
// budget and provenance. Header padding/pointers never enter the protocol.
// Harness adds the frozen candidate algorithm/source bindings separately.
inline std::vector<uint8_t> serialize_protocol() {
    checkpoint::Writer out;out.value(uint32_t{0x43505242u});out.value(kProtocolVersion);
    out.value(kArchivedCatalogPayloadHash);out.value(kArchivedSourceFailureHash);out.value(kArchivedPhysicsHash);
    detail::write_text(out,kArchivedCatalogSha256);out.value(kMaximumSubsteps);
    for(double value:{double(kOverlapTolerance),kPhysicalParityAbs,kParityRelative,kGapForceParityAbs,
        double(kPenetrationRefinement),double(kTerminalPositionRefinement),double(kSupportLoadFraction),
        double(kSupportVerticalSpeed),double(kFreeWaterClearance),double(kNormalForceNegativeAllowance),
        kEnergyAbsoluteAllowance,kEnergyRelativeAllowance}) out.value(value);
    for(int body=0;body<3;++body) {const auto g=p::body_geometry(body);out.value(g.length);out.value(g.half_height);out.value(g.half_width);}
    const auto lo=c::bounds_min_model(),hi=c::bounds_max_model();
    for(float value:{lo.x,lo.y,lo.z,hi.x,hi.y,hi.z,p::scale,p::torque,p::limit,c::kContactStiffness,c::kContactDampingRatio}) out.value(value);
    for(Stage stage:{Stage::Primary,Stage::Refinement,Stage::HeldOut,Stage::FreeWater,Stage::Passivity}) {
        const auto catalog=trials(stage);out.value(static_cast<uint32_t>(stage));out.value(stage_budget(stage));out.value(static_cast<uint32_t>(catalog.size()));
        for(const auto& trial:catalog) {
            const auto& f=trial.fixture;out.value(f.id);out.value(f.source_case);detail::write_text(out,f.name);
            out.value(static_cast<uint8_t>(f.support));out.value(trial.h);out.value(trial.steps);detail::write_body(out,f.initial);
            for(float value:{f.fluid.density,f.fluid.viscosity,f.fluid.gravity,f.current.x,f.current.y,f.current.z}) out.value(value);
            for(float value:f.action) out.value(value);
        }
    }
    return out.data;
}
}
#endif

