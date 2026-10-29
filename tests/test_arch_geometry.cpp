#include <cmath>
#include "check.h"
#include "envs/swimmer/deep_physics.h"

namespace ag=cudarl::deep;
namespace {
bool finite(ag::V v) {return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);}
bool same_system(const ag::System& a,const ag::System& b) {
    for(int i=0;i<ag::dofs;++i) {
        if(a.C[i]!=b.C[i] || a.force[i]!=b.force[i]) return false;
        for(int j=0;j<ag::dofs;++j) if(a.M[i][j]!=b.M[i][j] || a.R[i][j]!=b.R[i][j]) return false;
    }
    return true;
}
}

TEST_CASE("arch habitat: wider analytic floor and gradient preserve the default basin") {
    ag::Habitat original,wide;wide.scale=1.5f;
    for(int x=-6;x<=6;++x) for(int z=-5;z<=5;++z) {
        const float px=x*.71f,pz=z*.63f;
        CHECK(ag::habitat_floor(px,pz,original)==ag::pond_floor(px,pz));
        CHECK(fabsf(ag::habitat_floor(px*wide.scale,pz*wide.scale,wide)-ag::pond_floor(px,pz))<2e-6f);
        const auto a=ag::habitat_gradient(px,pz,original);
        const auto b=ag::habitat_gradient(px*wide.scale,pz*wide.scale,wide);
        CHECK(ag::norm(a-b*wide.scale)<2e-5f);
    }
    const float x=3.2f,z=1.4f,epsilon=.001f;const auto g=ag::habitat_gradient(x,z,wide);
    CHECK(fabsf(g.x-(ag::habitat_floor(x+epsilon,z,wide)-ag::habitat_floor(x-epsilon,z,wide))/(2*epsilon))<.0003f);
    CHECK(fabsf(g.z-(ag::habitat_floor(x,z+epsilon,wide)-ag::habitat_floor(x,z-epsilon,wide))/(2*epsilon))<.0003f);
    CHECK(ag::habitat_floor(4,0,wide)<ag::pond_floor(4,0));
    CHECK(ag::norm(ag::habitat_gradient(20,20,wide))==0);
}

TEST_CASE("arch habitat: exact torus opening, solid distance, normals and yaw transform") {
    ag::Habitat h;h.arch=true;
    CHECK(fabsf(ag::arch_distance(h.center,h)-(h.major-h.minor))<1e-6f);
    CHECK(fabsf(ag::arch_distance(h.center+ag::V{h.major,0,0},h)+h.minor)<1e-6f);
    CHECK(fabsf(ag::arch_distance(h.center+ag::V{h.major+h.minor,0,0},h))<1e-6f);
    CHECK(ag::arch_distance(h.center+ag::V{h.major+h.minor+.2f,0,0},h)>0);
    ag::V outward;ag::arch_distance(h.center+ag::V{h.major,0,h.minor},h,&outward);
    CHECK(ag::norm(outward-ag::V{0,0,1})<1e-6f);
    const ag::V local{.8f,.7f,.35f};const auto point=ag::arch_to_world(local,h);
    ag::V normal;ag::arch_distance(point,h,&normal);CHECK(fabsf(ag::norm(normal)-1)<1e-6f);
    const float epsilon=.0005f;
    const ag::V numerical{
        (ag::arch_distance(point+ag::V{epsilon,0,0},h)-ag::arch_distance(point-ag::V{epsilon,0,0},h))/(2*epsilon),
        (ag::arch_distance(point+ag::V{0,epsilon,0},h)-ag::arch_distance(point-ag::V{0,epsilon,0},h))/(2*epsilon),
        (ag::arch_distance(point+ag::V{0,0,epsilon},h)-ag::arch_distance(point-ag::V{0,0,epsilon},h))/(2*epsilon)};
    CHECK(ag::norm(normal-numerical)<.0006f);
    ag::Habitat turned=h;turned.yaw=.73f;turned.center={1,-2.5f,-.6f};
    CHECK(ag::norm(ag::arch_to_local(ag::arch_to_world(local,turned),turned)-local)<1e-6f);
    ag::V turned_normal;const float turned_distance=ag::arch_distance(ag::arch_to_world(local,turned),turned,&turned_normal);
    CHECK(fabsf(turned_distance-ag::arch_distance(point,h))<1e-6f);
    CHECK(ag::norm(turned_normal-(ag::arch_to_world(normal,turned)-turned.center))<1e-6f);
    for(const auto p:{h.center,h.center+ag::V{h.major,0,0}}) {
        ag::V n;CHECK(std::isfinite(ag::arch_distance(p,h,&n)));CHECK(finite(n));CHECK(fabsf(ag::norm(n)-1)<1e-6f);
    }
    ag::BodyState swimming;swimming.position=h.center;swimming.orientation=ag::axis_rotation({0,1,0},-1.57079632679f);
    CHECK(ag::arch_body_clearance(swimming,h)>.5f); // Every articulated part fits through the central passage.
}

TEST_CASE("arch habitat: opt-in contact reaches head, both tail links and the distal fin") {
    ag::BodyState state;state.velocity[2]=-.01f;const auto geometry=ag::shape(state);
    for(int body=0;body<3;++body) {
        const auto& link=geometry.link[body];ag::Habitat h;h.arch=true;
        h.center=link.center/ag::scale-ag::V{h.major,0,h.minor+.02f};
        ag::System a;ag::arch_contact(a,link,body,state.velocity,h);
        CHECK(a.force[2]>0);CHECK(a.R[2][2]>0);
        if(body>0) CHECK(fabsf(a.force[6])+fabsf(a.force[8])>0);
        float power=0;for(int i=0;i<ag::dofs;++i) for(int j=0;j<ag::dofs;++j)
            power+=state.velocity[i]*a.R[i][j]*state.velocity[j];
        CHECK(power>=0); // Normal damping cannot add kinetic energy.
        h.arch=false;ag::System disabled;ag::arch_contact(disabled,link,body,state.velocity,h);
        CHECK(same_system(disabled,ag::System{}));
    }
    const auto& fin=geometry.link[2];ag::Habitat h;h.arch=true;
    h.center=fin.center/ag::scale-ag::V{0,h.major+h.minor+.08f,0};
    ag::System core,membrane;ag::arch_contact(core,fin,1,state.velocity,h);ag::arch_contact(membrane,fin,2,state.velocity,h);
    CHECK(core.force[1]==0);CHECK(membrane.force[1]>0); // Fin height, not a centerline-only collision.
    CHECK(ag::arch_body_clearance(state,h)<0);
}

TEST_CASE("arch habitat: legacy assembly is unchanged and larger floor contact is opt-in") {
    ag::BodyState state;state.position={3.7f,-2.4f,0};state.angle[0]=.3f;state.velocity[1]=-.02f;
    ag::Physics legacy;legacy.pond=true;ag::Physics metadata=legacy;
    metadata.habitat.center={20,30,40};metadata.habitat.major=7;metadata.habitat.yaw=.9f;
    CHECK(same_system(ag::assemble(state,legacy),ag::assemble(state,metadata)));
    CHECK(ag::arch_body_clearance(state,metadata.habitat)==1e6f);
    ag::Physics wide=legacy;wide.habitat.scale=1.5f;
    const auto narrow_force=ag::assemble(state,legacy),wide_force=ag::assemble(state,wide);
    CHECK(narrow_force.force[1]>wide_force.force[1]);
    ag::BodyState a=state,b=state;const float action[4]={.2f,-.1f,-.3f,.1f};
    for(int i=0;i<20;++i) {ag::advance_physics(a,action,legacy);ag::advance_physics(b,action,metadata);}
    CHECK(ag::norm(a.position-b.position)==0);
    CHECK(a.orientation.x==b.orientation.x && a.orientation.y==b.orientation.y && a.orientation.z==b.orientation.z && a.orientation.w==b.orientation.w);
    for(int i=0;i<ag::dofs;++i) CHECK(a.velocity[i]==b.velocity[i]);
    for(int i=0;i<4;++i) CHECK(a.angle[i]==b.angle[i]);
}

TEST_CASE("arch habitat: compliant contact stays finite and resolves shallow penetration without projection") {
    ag::Physics p;p.pond=true;p.habitat.scale=1.5f;p.habitat.arch=true;
    ag::BodyState state;state.position=p.habitat.center+ag::V{p.habitat.major,0,p.habitat.minor+.08f};
    const float initial=ag::arch_body_clearance(state,p.habitat);CHECK(initial<0);
    const ag::V position=state.position;const float action[4]{};
    ag::substep(state,action,.0001f,p);
    CHECK(ag::norm(state.position-position)<.001f); // A force step, not geometric relocation.
    for(int tick=0;tick<80;++tick) {
        ag::advance_physics(state,action,p);CHECK(finite(state.position));CHECK(std::isfinite(ag::energy(state)));
        for(float velocity:state.velocity) CHECK(std::isfinite(velocity));
        CHECK(ag::arch_body_clearance(state,p.habitat)>-.4f); // A compliant solver permits finite small penetration.
    }
    CHECK(ag::arch_body_clearance(state,p.habitat)>initial);
    p.boundaries=false;const auto free=ag::assemble(state,p);p.habitat.arch=false;
    CHECK(same_system(free,ag::assemble(state,p))); // Disabling all boundaries also disables the new solid.
}
