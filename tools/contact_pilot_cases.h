#ifndef CUDARL_CONTACT_PILOT_CASES_H
#define CUDARL_CONTACT_PILOT_CASES_H
#include <array>
#include <vector>
#include <cstring>
#include "physics/contact.h"
#include "core/checkpoint.h"

namespace cudarl::contact_pilot_spec {
namespace p=physics;
enum class Stage:uint32_t {Parity=0,Scalar=1,FreeWater=2,Passive=3,Rolling=4,Fresh=5};
constexpr int kMaximumAdvances=2024;
constexpr int kGeometryQueryCap=256,kKineticQueryCap=256,kStaticSolveCap=96;
constexpr float kStep=.005f,kSafetyGap=.0005f,kHardGeometryGap=.000002f;
constexpr float kStiffness=320.f,kDamping=1.9595918f,kReferenceMass=.003f;
constexpr float kImpactSpeed=.05f,kLoadEnvelope=.0003f,kDesignCompression=.0004f;
constexpr double kAvailableEnergy=25e-6,kMaximumRestitution=.15;
constexpr double kEnergyAbsolute=1e-10,kEnergyRelative=.001;
constexpr double kParityAbsolute=2e-4,kParityRelative=2e-3,kGapForceAbsolute=2e-6;
constexpr float kNegativeNormalForce=1e-9f;
constexpr const char* kSourceTraceSHA256="F132FA41908D3E792852FE9F7F999E3881E51A7280B506965579A512FF83297D";
struct Fixture {
    const char* name="";int id=0;
    p::BodyState initial{};p::Parameters fluid{};p::V current{};
    std::array<float,4> action{};
    bool scalar=false;
    float scalar_mass=.003f,scalar_gap=0,scalar_velocity=0,scalar_force=0;
};
struct Trial {Stage stage=Stage::Parity;Fixture fixture{};float h=kStep;int steps=50;};
inline const char* stage_name(Stage stage) {
    switch(stage) {case Stage::Parity:return "parity";case Stage::Scalar:return "scalar";
    case Stage::FreeWater:return "freewater";case Stage::Passive:return "passive";
    case Stage::Rolling:return "rolling";default:return "fresh";}
}
constexpr int stage_budget(Stage stage) {
    return stage==Stage::Parity?24:(stage==Stage::Scalar?600:
        (stage==Stage::FreeWater||stage==Stage::Passive||stage==Stage::Fresh?400:200));
}
static_assert(24+600+400+400+200+400==kMaximumAdvances);
namespace detail {
// Exact FP32 round-trip fields of the saved 43,600-call study. No new capture,
// translation, projection, autoreset or simulation is performed to obtain them.
inline constexpr float saved[3][23]={
 {-1.0274076203131699e-06f,-3.8289642333984375f,-7.0453415901283734e-06f,
  -.25355756282806396f,1.525660900369985e-05f,.00041816552402451634f,.96732026338577271f,
  -.00085648614913225174f,-.00053668126929551363f,-.0013743665767833591f,-2.3710459572612308e-05f,
  -7.9166437672029133e-07f,-.014220671728253365f,-1.4075008948566392e-05f,
  -17.420103073120117f,-.024745559319853783f,-.026565788313746452f,
  .0078378869220614433f,-.016407329589128494f,-.0083529390394687653f,.0053853499703109264f,
  0.f,.028713531792163849f},
 {-.20065024495124817f,-3.8257904052734375f,.15247362852096558f,
  -.00071559351636096835f,-.049735881388187408f,.004002199973911047f,.99875420331954956f,
  -.01721465028822422f,.014308422803878784f,-.020040461793541908f,-.010534121654927731f,
  -.00012013601372018456f,.0067020454443991184f,.00049987644888460636f,
  -.0080807153135538101f,.047716576606035233f,.10283981263637543f,
  .0035617134999483824f,.014714044518768787f,-.0013639003736898303f,-.017142852768301964f,
  0.f,.11044026911258698f},
 {-.20065023005008698f,-3.8257896900177002f,.15247362852096558f,
  -.00071548274718225002f,-.049735907465219498f,.0040014064870774746f,.99875420331954956f,
  -.01721465215086937f,.014307225123047829f,-.020040456205606461f,-.010530407540500164f,
  -.00012013747618766502f,.006702205166220665f,.00049987644888460636f,
  -.0080788386985659599f,.047716502100229263f,.10281302779912949f,
  .0035616464447230101f,.014759146608412266f,-.0013637878000736237f,-.017019549384713173f,
  0.f,.11044054478406906f}
};
inline p::BodyState body(int which) {
    const auto* values=saved[which];p::BodyState out;
    out.position={values[0],values[1],values[2]};out.orientation={values[3],values[4],values[5],values[6]};
    for(int j=0;j<4;++j) out.angle[j]=values[7+j];
    for(int j=0;j<10;++j) out.velocity[j]=values[11+j];
    out.work=values[21];out.travel=values[22];return out;
}
inline void text(checkpoint::Writer& out,const char* value) {
    const auto count=std::strlen(value);out.value(static_cast<uint32_t>(count));
    for(size_t i=0;i<count;++i) out.value(static_cast<uint8_t>(value[i]));
}
inline void write_body(checkpoint::Writer& out,const p::BodyState& b) {
    for(float value:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w}) out.value(value);
    for(float value:b.angle) out.value(value);for(float value:b.velocity) out.value(value);
    out.value(b.work);out.value(b.travel);
}
}
inline std::array<p::BodyState,3> reference_states() {return {detail::body(0),detail::body(1),detail::body(2)};}
inline std::vector<Trial> trials(Stage stage) {
    std::vector<Trial> out;
    if(stage==Stage::Parity) {
        for(int origin=0;origin<2;++origin) for(int perturb=0;perturb<3;++perturb) {
            Fixture f;f.id=origin*3+perturb;f.name=origin==0?"CPU-origin-pre77":"CUDA-origin-pre77";
            f.initial=detail::body(origin+1);f.current={0,0,.005f};
            f.initial.velocity[1]+=static_cast<float>(perturb-1)*.000005f;
            out.push_back({stage,f,kStep,1});
        }
    } else if(stage==Stage::Scalar) {
        for(int id=0;id<3;++id) {
            Fixture f;f.id=id;f.scalar=true;
            f.name=id==0?"loaded-support":(id==1?"50mm-s-impact":"separating-nonadhesion");
            if(id==0) f.scalar_force=-kLoadEnvelope;
            if(id==1) {f.scalar_gap=.000025f;f.scalar_velocity=-kImpactSpeed;}
            if(id==2) f.scalar_velocity=kImpactSpeed;
            out.push_back({stage,f,kStep,50});
        }
    } else if(stage==Stage::FreeWater) {
        for(int id=0;id<2;++id) {
            Fixture f;f.id=id;f.name=id==0?"calm-free":"current-free";
            if(id==1) {f.current={0,0,.005f};f.action={.1f,-.1f,.05f,-.05f};}
            out.push_back({stage,f,kStep,50});
        }
    } else if(stage==Stage::Passive) {
        for(int id=0;id<2;++id) {
            Fixture f;f.id=id;f.name=id==0?"zero-energy-touch":"passive-rolling-impact";
            f.fluid={0,0,0};
            if(id==0) f.initial.position.y=(-.16f+.0064f)/p::scale;
            else f.initial=detail::body(0);
            out.push_back({stage,f,kStep,50});
        }
    } else if(stage==Stage::Rolling) {
        Fixture f;f.name="recorded-positive-gap-rolling";f.initial=detail::body(0);f.fluid.density=990;
        out.push_back({stage,f,kStep,50});
    } else {
        for(int id=0;id<2;++id) {
            Fixture f;f.id=id;f.name=id==0?"fresh-calm-momentum":"fresh-current-momentum";
            f.initial=detail::body(0);f.fluid.density=990;
            const float sign=id==0?-1.f:1.f;
            f.initial.velocity[1]+=sign*.001f;f.initial.velocity[3]+=sign*.1f;
            if(id==1) f.current={0,0,.005f};
            out.push_back({stage,f,kStep,50});
        }
    }
    return out;
}
inline std::vector<uint8_t> serialize_protocol() {
    checkpoint::Writer out;out.value(uint32_t{0x43504C54});out.value(uint32_t{1});
    out.value(kMaximumAdvances);out.value(kGeometryQueryCap);out.value(kKineticQueryCap);out.value(kStaticSolveCap);
    detail::text(out,kSourceTraceSHA256);
    for(double value:{double(kStep),double(kSafetyGap),double(kHardGeometryGap),double(kStiffness),double(kDamping),double(kReferenceMass),
        double(kImpactSpeed),double(kLoadEnvelope),double(kDesignCompression),kAvailableEnergy,kMaximumRestitution,
        kEnergyAbsolute,kEnergyRelative,kParityAbsolute,kParityRelative,kGapForceAbsolute,double(kNegativeNormalForce)}) out.value(value);
    for(Stage stage:{Stage::Parity,Stage::Scalar,Stage::FreeWater,Stage::Passive,Stage::Rolling,Stage::Fresh}) {
        out.value(static_cast<uint32_t>(stage));out.value(stage_budget(stage));const auto list=trials(stage);out.value(static_cast<uint32_t>(list.size()));
        for(const auto& t:list) {
            const auto& f=t.fixture;detail::text(out,f.name);out.value(f.id);out.value(t.h);out.value(t.steps);
            detail::write_body(out,f.initial);for(float value:{f.fluid.density,f.fluid.viscosity,f.fluid.gravity,f.current.x,f.current.y,f.current.z}) out.value(value);
            for(float value:f.action) out.value(value);out.value(static_cast<uint8_t>(f.scalar));
            for(float value:{f.scalar_mass,f.scalar_gap,f.scalar_velocity,f.scalar_force}) out.value(value);
        }
    }
    return out.data;
}
}
#endif
