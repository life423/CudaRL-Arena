#ifndef CUDARL_CONTACT_VALIDATION_CASES_H
#define CUDARL_CONTACT_VALIDATION_CASES_H
#include <array>
#include <vector>
#include "contact_pilot_cases.h"
#include "contact_probe_cases.h"

// Frozen, fixed-A validation data only. No trial advances, fitting or policy.
namespace cudarl::contact_validation_spec {
namespace p=physics;namespace c=contact;
enum class Stage:uint32_t {ScalarAccuracy=0,RotatingAccuracy=1,Support=2,FreeWater=3,Passive=4,HeldOut=5};
constexpr int kMaximumAdvances=18200;
constexpr int kGeometryQueryCap=256,kKineticQueryCap=256,kStaticSolveCap=96,kContinuousQueryCap=12000;
constexpr int kFixtureGeometryQueriesPerProcess=4,kMaximumFixtureGeometryQueries=8;
constexpr double kScalarCompressionAbsolute=2e-6,kScalarCompressionRelative=.05,kScalarRestitutionAbsolute=.02;
constexpr double kScalarImpulseAbsolute=1e-8,kScalarImpulseRelative=.05,kScalarGapAbsolute=1e-5,kScalarSpeedAbsolute=.001;
constexpr double kRotatingPositionAbsolute=.0001,kRotatingEndpointAbsolute=.0001,kRotatingGapAbsolute=.0001,kRotatingOrientationAbsolute=.02,kRotatingJointAbsolute=.02;
constexpr double kRotatingHeadSpeedAbsolute=.002,kRotatingAngularRateAbsolute=.5,kRotatingJointRateAbsolute=.5;
constexpr double kRotatingImpulseAbsolute=1e-8,kRotatingImpulseRelative=.05;
constexpr double kSupportWindowSeconds=1,kSupportLinkSpeed=.0005,kSupportAngularRate=.05,kSupportJointRate=.05;
constexpr double kMomentumBalanceAbsolute=1e-8,kMomentumBalanceRelative=.05,kSupportBalanceAbsolute=1e-8,kSupportBalanceRelative=.1;
constexpr float kFreshInitialClearance=.00002f;
struct Fixture {
    const char* name="";int id=0;p::BodyState initial{};p::Parameters fluid{};p::V current{};
    std::array<float,4> action{};bool scalar=false;
    // New validation's exact doubles match its independent continuous oracle.
    double scalar_mass=.003,scalar_gap=0,scalar_velocity=0,scalar_force=0;
};
struct Trial {Stage stage=Stage::ScalarAccuracy;Fixture fixture{};float h=.005f;int steps=50;};
inline const char* stage_name(Stage stage) {
    switch(stage){case Stage::ScalarAccuracy:return "scalar-accuracy";case Stage::RotatingAccuracy:return "rotating-accuracy";
        case Stage::Support:return "support";case Stage::FreeWater:return "freewater";case Stage::Passive:return "passive";default:return "heldout";}
}
constexpr int stage_budget(Stage stage) {
    return stage==Stage::ScalarAccuracy?2100:(stage==Stage::RotatingAccuracy?1500:(stage==Stage::Support?6000:
        (stage==Stage::FreeWater?400:(stage==Stage::Passive?200:8000))));
}
constexpr int model_count(Stage stage) {return stage==Stage::FreeWater?2:1;}
constexpr bool accuracy_stage(Stage stage) {return stage==Stage::ScalarAccuracy||stage==Stage::RotatingAccuracy;}
static_assert(2100+1500+6000+400+200+8000==kMaximumAdvances);
namespace detail {
inline int& geometry_queries() {static int value=0;return value;}
inline Fixture copy(const contact_pilot_spec::Fixture& in) {
    Fixture out;out.name=in.name;out.id=in.id;out.initial=in.initial;out.fluid=in.fluid;out.current=in.current;out.action=in.action;return out;
}
inline Fixture copy(const contact_probe::Fixture& in) {
    Fixture out;out.name=in.name;out.id=in.id;out.initial=in.initial;out.fluid=in.fluid;out.current=in.current;out.action=in.action;return out;
}
inline const std::array<Fixture,4>& heldout() {
    static const auto fixtures=[] {
        std::array<Fixture,4> out{};
        const float yaw[4]={.37f,-.51f,.89f,-1.07f};
        const float pitch[4]={.29f,-.34f,.41f,-.47f};
        const float roll[4]={.62f,1.15f,-.78f,2.f};
        const float joints[4][4]={{.23f,-.17f,-.12f,.21f},{-.19f,.24f,.16f,-.13f},{.11f,.28f,-.26f,.09f},{-.27f,.08f,.22f,-.18f}};
        const float x[4]={-.6f,.55f,-.3f,.4f},z[4]={.4f,-.45f,-.6f,.55f};
        for(int index=0;index<4;++index) {
            auto& f=out[static_cast<size_t>(index)];f.id=index;f.name="fresh-fixed-bent-roll-pitch-support";f.fluid.density=990;
            f.initial.position={x[index],0,z[index]};
            f.initial.orientation=p::axis_rotation({0,1,0},yaw[index])*p::axis_rotation({0,0,1},pitch[index])*p::axis_rotation({1,0,0},roll[index]);
            f.initial.orientation.normalize();for(int joint=0;joint<4;++joint)f.initial.angle[joint]=joints[index][joint];
            // Initial fixture construction ONLY, not a runtime projection or a
            // response to test performance. One envelope query per fresh pose.
            const auto inspection=c::inspect(f.initial);++geometry_queries();
            f.initial.position.y+=(kFreshInitialClearance-inspection.floor_gap)/p::scale;
        }return out;
    }();
    return fixtures;
}
}
// Exactly four placement queries per executable process once fresh data are
// materialized; prepare+run register at most eight in the shared static ledger.
inline int fixture_geometry_queries() {return detail::geometry_queries();}
inline std::vector<Trial> trials(Stage stage) {
    std::vector<Trial> out;
    if(stage==Stage::ScalarAccuracy) {
        const float h[3]={.005f,.0025f,.00125f};const int steps[3]={50,100,200};
        for(int id=0;id<3;++id)for(int level=0;level<3;++level) {
            Fixture f;f.id=id;f.scalar=true;f.name=id==0?"exact-3g-loaded-support":(id==1?"exact-50mm-s-impact":"exact-separating-contact");
            if(id==0)f.scalar_force=-.0003;if(id==1){f.scalar_gap=.000025;f.scalar_velocity=-.05;}if(id==2)f.scalar_velocity=.05;
            out.push_back({stage,f,h[level],steps[level]});
        }
    } else if(stage==Stage::RotatingAccuracy) {
        const auto f=detail::copy(contact_pilot_spec::trials(contact_pilot_spec::Stage::Rolling).front().fixture);
        const float h[4]={.005f,.0025f,.00125f,.000625f};const int steps[4]={50,100,200,400};
        for(int level=0;level<4;++level)out.push_back({stage,f,h[level],steps[level]});
    } else if(stage==Stage::Support) {
        const auto original=contact_probe::trials(contact_probe::Stage::Primary);
        for(int id:{3,4,5})out.push_back({stage,detail::copy(original[static_cast<size_t>(id)].fixture),.005f,1000});
    } else if(stage==Stage::FreeWater) {
        for(const auto& trial:contact_pilot_spec::trials(contact_pilot_spec::Stage::FreeWater))out.push_back({stage,detail::copy(trial.fixture),.005f,50});
    } else if(stage==Stage::Passive) {
        for(const auto& trial:contact_pilot_spec::trials(contact_pilot_spec::Stage::Passive))out.push_back({stage,detail::copy(trial.fixture),.005f,50});
    } else {
        for(const auto& f:detail::heldout())out.push_back({stage,f,.005f,1000});
    }
    return out;
}
inline std::vector<uint8_t> serialize_protocol() {
    checkpoint::Writer out;out.value(uint32_t{0x43415631});out.value(uint32_t{1});out.value(kMaximumAdvances);
    for(int value:{kGeometryQueryCap,kKineticQueryCap,kStaticSolveCap,kContinuousQueryCap,kFixtureGeometryQueriesPerProcess,kMaximumFixtureGeometryQueries})out.value(value);
    for(double value:{kScalarCompressionAbsolute,kScalarCompressionRelative,kScalarRestitutionAbsolute,kScalarImpulseAbsolute,kScalarImpulseRelative,kScalarGapAbsolute,kScalarSpeedAbsolute,
        kRotatingPositionAbsolute,kRotatingEndpointAbsolute,kRotatingGapAbsolute,kRotatingOrientationAbsolute,kRotatingJointAbsolute,kRotatingHeadSpeedAbsolute,kRotatingAngularRateAbsolute,kRotatingJointRateAbsolute,kRotatingImpulseAbsolute,kRotatingImpulseRelative,
        kSupportWindowSeconds,kSupportLinkSpeed,kSupportAngularRate,kSupportJointRate,kMomentumBalanceAbsolute,kMomentumBalanceRelative,kSupportBalanceAbsolute,kSupportBalanceRelative,double(kFreshInitialClearance)})out.value(value);
    for(Stage stage:{Stage::ScalarAccuracy,Stage::RotatingAccuracy,Stage::Support,Stage::FreeWater,Stage::Passive,Stage::HeldOut}) {
        out.value(static_cast<uint32_t>(stage));out.value(stage_budget(stage));out.value(model_count(stage));const auto rows=trials(stage);out.value(static_cast<uint32_t>(rows.size()));
        for(const auto& t:rows){const auto& f=t.fixture;contact_pilot_spec::detail::text(out,f.name);out.value(f.id);out.value(t.h);out.value(t.steps);out.value(f.scalar);
            contact_pilot_spec::detail::write_body(out,f.initial);for(float value:{f.fluid.density,f.fluid.viscosity,f.fluid.gravity,f.current.x,f.current.y,f.current.z})out.value(value);for(float value:f.action)out.value(value);
            for(double value:{f.scalar_mass,f.scalar_gap,f.scalar_velocity,f.scalar_force})out.value(value);}
    }
    return out.data;
}
}
#endif
