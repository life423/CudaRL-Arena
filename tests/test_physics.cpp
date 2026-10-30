#include "check.h"
#include "simulation/world.h"
#include "fixtures/physics_baseline.h"
#include <cstring>
#include <exception>
#include <limits>
#include <string>

namespace {
namespace p=cudarl::physics;namespace c=cudarl::contact;namespace s=cudarl::simulation;
bool near(float actual,float expected,float absolute=2e-8f,float relative=5e-6f) {
    return std::isfinite(actual)&&std::fabs(actual-expected)<=absolute+relative*std::fabs(expected);
}
void vec(p::V a,const float* b,float absolute=2e-8f) {CHECK(near(a.x,b[0],absolute));CHECK(near(a.y,b[1],absolute));CHECK(near(a.z,b[2],absolute));}
uint32_t bits(float value) {uint32_t out=0;std::memcpy(&out,&value,4);return out;}
p::BodyState from(const physics_baseline::Input& in) {
    p::BodyState b;b.position={in.position[0],in.position[1],in.position[2]};
    b.orientation={in.orientation[0],in.orientation[1],in.orientation[2],in.orientation[3]};
    for(int j=0;j<4;++j) b.angle[j]=in.angle[j];for(int j=0;j<10;++j) b.velocity[j]=in.velocity[j];return b;
}
}
TEST_CASE("physics: independent baseline assembly and geometry oracle (zero integration)") {
    for(const auto& f:physics_baseline::cases) {
        const auto b=from(f.input);const auto geometry=p::shape(b);
        const auto system=p::assemble(b,{f.density,f.viscosity,f.gravity},{f.current[0],f.current[1],f.current[2]});
        for(int link=0;link<3;++link) {
            const auto& a=geometry.link[link];const auto& r=f.links[link];
            CHECK(near(a.q.x,r.q[0]));CHECK(near(a.q.y,r.q[1]));CHECK(near(a.q.z,r.q[2]));CHECK(near(a.q.w,r.q[3]));
            vec(a.center,r.center);vec(a.omega,r.omega);vec(a.acceleration,r.acceleration);vec(a.alpha,r.alpha);vec(a.inertia,r.inertia,1e-15f);
            for(int j=0;j<10;++j) {vec(a.linear[j],r.linear[j]);vec(a.angular[j],r.angular[j]);}
            CHECK(near(a.length,r.length));CHECK(near(a.mass,r.mass,1e-12f));CHECK(near(a.volume,r.volume,1e-15f));CHECK(near(a.vertical_radius,r.vertical_radius));
        }
        for(int i=0;i<10;++i) {
            CHECK(near(system.C[i],f.C[i],2e-12f));CHECK(near(system.force[i],f.force[i],2e-10f));
            float rhs=.005f*(system.force[i]-system.C[i]);
            for(int j=0;j<10;++j) {
                CHECK(near(system.M[i][j],f.M[i][j],1e-15f));CHECK(near(system.R[i][j],f.R[i][j],1e-15f));
                rhs+=system.M[i][j]*b.velocity[j];
            }
            CHECK(near(rhs,f.rhs[i],2e-13f));
        }
    }
}
TEST_CASE("physics: full-envelope floor geometry and spring-force oracle (zero integration)") {
    p::BodyState b;
    b.position.y=(-.16f+.0062f)/p::scale;
    const auto proof=c::inspect(b);CHECK(near(proof.floor_gap,-.0002f,2e-8f));CHECK(proof.floor_contact);CHECK(!proof.excessive);
    c::Telemetry force;const auto system=c::assemble(b,{0,0,0},{},&force);
    CHECK(near(force.floor_force,.8f*.0002f,2e-8f));CHECK(force.points==1);
    CHECK(near(system.force[1],force.floor_force,2e-8f));CHECK(near(system.force[0],0));CHECK(near(system.force[2],0));
    b.position.y-=.0004f/p::scale;CHECK(c::inspect(b).excessive);
    // A 90-degree roll changes the head's downward support from6.4 to8mm.
    b={};b.orientation=p::axis_rotation({1,0,0},1.5707963267948966f);
    const auto g=p::shape(b);const auto support=c::support_offset(g.link[0],0,{0,1,0});
    CHECK(near(support.y,-.008f));
    static_assert(c::kOverlapTolerance==.0005f,"The registered contact tolerance changed");
    static_assert(c::kRolledSupportKnownUnresolved,"Cleanup must retain the unresolved-contact status");
    // Independent tilted elliptical-cylinder support oracle, including the
    // distal fin. Both endpoints matter; this evaluates no integration.
    p::BodyState tilted;
    tilted.orientation=p::axis_rotation({0,1,0},.37f)*p::axis_rotation({0,0,1},-.29f)*p::axis_rotation({1,0,0},.62f);
    tilted.angle[0]=.13f;tilted.angle[1]=-.17f;tilted.angle[2]=-.11f;tilted.angle[3]=.19f;
    const auto tilted_geometry=p::shape(tilted);
    for(int link=1;link<3;++link) for(int face=0;face<6;++face) {
        const auto& cylinder=tilted_geometry.link[link];const auto normal=c::normal(face);
        const auto local=cylinder.q.inverse(normal);
        const double ry=link==2?.006f:.0015f,rz=.0015f,length=.026f;
        const double minimum_projection=-std::fabs(double(local.x))*length*.5-
            std::sqrt(double(local.y)*local.y*ry*ry+double(local.z)*local.z*rz*rz);
        const double expected=p::dot(normal,cylinder.center)+c::offset(face)+minimum_projection;
        float actual=1e6f;
        for(int end=0;end<2;++end) actual=std::fmin(actual,
            p::dot(normal,cylinder.center+c::support_offset(cylinder,link,normal,end))+c::offset(face));
        CHECK(std::fabs(double(actual)-expected)<=2e-8+5e-6*std::fabs(expected));
    }
}
TEST_CASE("physics: manual world pause, fixed actuator units and exact diagnostic continuation") {
    s::PhysicalWorld w;const auto initial=w.serialize();
    CHECK(w.state().paused);CHECK(!w.step({1,1,1,1}));CHECK(w.serialize()==initial);
    w.set_current({.005f,0,0});w.set_actuation(s::Actuation::Torque);w.set_paused(false);
    CHECK(w.step({2,-2,.125f,-.25f}));CHECK(w.state().substeps==10);CHECK(w.state().control_steps==1);
    for(int j=0;j<10;++j) {
        CHECK(w.state().actual_torques[j*4]==1);CHECK(w.state().actual_torques[j*4+1]==-1);
        CHECK(w.state().actual_torques[j*4+2]==.125f);CHECK(w.state().actual_torques[j*4+3]==-.25f);
    }
    const auto saved=w.serialize();s::PhysicalWorld copy;copy.restore(saved);CHECK(copy.serialize()==saved);
    CHECK(w.step({.05f,-.05f,0,0}));CHECK(copy.step({.05f,-.05f,0,0}));CHECK(w.serialize()==copy.serialize());
    const uint32_t kp[4]{0x36772E0B,0x366077A1,0x357478F5,0x35C0F964};
    const uint32_t kd[4]{0x351D5C0E,0x350EE67B,0x341BA2D2,0x3475B3CD};
    for(int j=0;j<4;++j) {CHECK(bits(w.state().gains.kp[j])==kp[j]);CHECK(bits(w.state().gains.kd[j])==kd[j]);}
    const auto body=w.state().body;w.set_actuation(s::Actuation::JointTarget);
    for(float raw:w.state().raw_action) CHECK(raw==0);
    CHECK(w.state().body.position.x==body.position.x);CHECK(w.state().body.angle[0]==body.angle[0]);
    w.reset();CHECK(w.state().paused&&w.state().substeps==0);CHECK(w.state().current.x==.005f);
    w.set_paused(false);CHECK(w.step({2,-2,.25f,-.5f}));
    CHECK(w.state().held_targets[0]==1&&w.state().held_targets[1]==-1);
    for(float actual:w.state().actual_torques) CHECK(std::fabs(actual)<=1);
    CHECK(w.state().telemetry.executed_substeps==10);
}
TEST_CASE("physics: diagnostic format rejects corruption and learner bytes atomically (zero integration)") {
    s::PhysicalWorld w;const auto saved=w.serialize();auto damaged=saved;damaged[32]^=1;
    bool rejected=false;try {w.restore(damaged);} catch(const std::exception&) {rejected=true;}
    CHECK(rejected&&w.serialize()==saved);
    rejected=false;try {w.restore(std::vector<uint8_t>(128,0));} catch(const std::exception&) {rejected=true;}
    CHECK(rejected&&w.serialize()==saved);
    // Version-1 named layout: header+mode are3uint32 and1uint64; body,
    // fluid/current, gains and actuator fields before the clock total130floats.
    constexpr size_t clock_offset=3*sizeof(uint32_t)+sizeof(uint64_t)+130*sizeof(float);
    auto rejects_clock=[&](uint64_t controls,uint64_t substeps,bool unsafe,bool invalid) {
        auto record=saved;
        const double seconds=static_cast<double>(substeps)*static_cast<double>(c::kSubstep);
        std::memcpy(record.data()+clock_offset,&controls,sizeof(controls));
        std::memcpy(record.data()+clock_offset+sizeof(uint64_t),&substeps,sizeof(substeps));
        std::memcpy(record.data()+clock_offset+2*sizeof(uint64_t),&seconds,sizeof(seconds));
        record[clock_offset+2*sizeof(uint64_t)+sizeof(double)+1]=uint8_t(unsafe);
        record[clock_offset+2*sizeof(uint64_t)+sizeof(double)+2]=uint8_t(invalid);
        // Keep the complete physical/actuator state unchanged and supply a
        // legitimate checksum, so rejection exercises clock/schema validation.
        CHECK(std::memcmp(record.data()+16,saved.data()+16,clock_offset-16)==0);
        uint64_t checksum=14695981039346656037ull;
        for(size_t i=0;i<record.size()-sizeof(uint64_t);++i) {checksum^=record[i];checksum*=1099511628211ull;}
        std::memcpy(record.data()+record.size()-sizeof(uint64_t),&checksum,sizeof(checksum));
        bool refused=false;try {w.restore(record);} catch(const std::exception&) {refused=true;}
        CHECK(refused&&w.serialize()==saved);
    };
    rejects_clock(0,10,false,false); // complete substeps without a control
    rejects_clock(1,11,false,false); // safe world cannot have a partial control
    rejects_clock(1,11,true,false);  // unsafe partial final control needs ceil(11/10)
    rejects_clock(0,10,false,true);  // invalid flag does not excuse an impossible count
    rejects_clock(0,std::numeric_limits<uint64_t>::max(),false,true); // ceil must not wrap
    rejected=false;try {w.set_current({0,.001f,0});} catch(const std::exception&) {rejected=true;}CHECK(rejected&&w.serialize()==saved);
    auto invalid=w.state();invalid.paused=false;invalid.body.position.y=1;
    CHECK(!s::step_state(invalid,std::array<float,4>{0,0,0,0}.data()));CHECK(invalid.unsafe&&invalid.substeps==0);
    invalid=w.state();invalid.paused=false;const float nan=std::numeric_limits<float>::quiet_NaN();const float action[4]{nan,0,0,0};
    CHECK(!s::step_state(invalid,action));CHECK(invalid.invalid&&invalid.substeps==0);
}
TEST_CASE("physics: archived rolled support remains KNOWN UNRESOLVED (37 substeps, no learning)") {
    p::BodyState body;body.orientation=p::axis_rotation({1,0,0},1.5707963267948966f);
    body.position.y+=(-.00001f-c::inspect(body).floor_gap)/p::scale;
    const float zero[4]{};p::Parameters water;water.density=990;
    for(int tick=0;tick<37;++tick) c::substep(body,zero,.005f,water);
    const auto proof=c::inspect(body);
    static_assert(c::kRolledSupportKnownUnresolved,"The failed dynamic gate must remain explicit");
    CHECK(p::finite_body(body));CHECK(proof.excessive&&!proof.escape&&!proof.invalid);
    CHECK(proof.floor_gap<-.0005f&&proof.floor_gap>-.0007f);
    // This rolled regression first breaches at the HEAD. The historical v1
    // torque-rollout failure was a different distal-fin/proxy mismatch.
    const auto geometry=p::shape(body);const auto& head=geometry.link[0];
    const float head_gap=p::dot(c::normal(2),head.center+c::support_offset(head,0,c::normal(2)))+c::offset(2);
    CHECK(near(head_gap,proof.floor_gap,2e-8f));
    std::printf("    KNOWN UNRESOLVED rolled-contact gap %.9g m after37substeps; not contact validation\n",proof.floor_gap);
}
