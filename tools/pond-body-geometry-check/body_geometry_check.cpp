#include "environment/pond_body_geometry.h"
#include <cstdio>
#include <limits>
namespace bg=cudarl::pond_body_geometry;
namespace pg=cudarl::pond_geometry_v1;
namespace p=cudarl::physics;
uint64_t assertions=0,queries=0,visits=0,reference_queries=0;
bool good=true;
void check(bool value,const char* message) {++assertions;if(!value){good=false;std::fprintf(stderr,"FAIL: %s\n",message);}}
bg::Inspection inspect(const p::BodyState& state,uint32_t cap=32768){++queries;auto r=bg::inspect(state,pg::current_visual_parameters(),1e-5,cap);visits+=r.visits;return r;}
double gap(bg::detail::V point){++reference_queries;const auto q=pg::sample(pg::current_visual_parameters(),point.x,point.z);return point.y-q.height_m;}
double dense(const bg::detail::Shape& s) {
    double result=1e30;
    const int U=240,V=480;
    if(s.box) {
        for(int face=0;face<6;++face)for(int u=0;u<=U;++u)for(int v=0;v<=V;++v) {
            const int fixed=face/2;
            const double hu=fixed==0?s.b:s.a,hv=fixed==2?s.b:s.c;
            const bg::detail::Patch patch{-hu+2*hu*u/U,-hv+2*hv*v/V,0,0,face};
            result=bg::detail::minimum(result,gap(bg::detail::patch_geometry(s,patch).point));
        }
    } else if(s.ellipsoid) {
        for(int u=0;u<=U;++u)for(int v=0;v<V;++v) {
            bg::detail::Patch patch{2*pg::pi*v/V,pg::pi*u/U,0,0,0};
            result=bg::detail::minimum(result,gap(bg::detail::patch_geometry(s,patch).point));
        }
    } else {
        for(int u=0;u<=U;++u)for(int v=0;v<V;++v) {
            for(int kind=0;kind<3;++kind) {
                const double axial=kind==0?-s.a+2*s.a*u/U:double(u)/U;
                bg::detail::Patch patch{axial,2*pg::pi*v/V,0,0,kind};
                result=bg::detail::minimum(result,gap(bg::detail::patch_geometry(s,patch).point));
            }
        }
    }
    return result;
}
int main() {
    const auto params=pg::current_visual_parameters();
    p::BodyState state;
    auto r=inspect(state);
    check(r.valid&&r.certified,"initial complete body is certified");
    check(r.visits==4,"exact flat fast path covers head, rods and membrane box union");
    check(fabs(r.lower_gap_m-(.08-.0064))<2e-8,"head controls level flat clearance");
    check(r.upper_gap_m-r.lower_gap_m<2e-8,"flat proof is exact within roundoff guard");
    check(fabs(r.max_body_y_m-(-.08+.0064))<2e-8,"whole-body surface scope uses actual highest envelope");
    state.angle[1]=.2f;state.angle[3]=-.2f;r=inspect(state);
    check(r.certified&&r.link[2].lower_gap_m<r.link[0].lower_gap_m,"pitched composite fin envelope is included in the complete minimum");
    state.orientation=p::axis_rotation({1,0,0},.14f);r=inspect(state);
    const auto fin_links=cudarl::movement::shape(state);
    auto fin_box=bg::detail::shape(fin_links.link[2],2);fin_box.box=true;fin_box.c=cudarl::movement::Geometry::fin_thickness*.5;
    const auto lowest_fin=bg::detail::support(fin_box,{0,1,0});
    check(r.certified&&r.link[2].lower_gap_m<=lowest_fin.y-params.floor_y_m,"rolled membrane rim corners are inside the certified union");
    for(int a=-1;a<=1;a+=2)for(int b=-1;b<=1;b+=2)for(int c=-1;c<=1;c+=2) {
        const auto corner=bg::detail::transform(fin_box,{a*fin_box.a,b*fin_box.b,c*fin_box.c});
        check(r.link[2].lower_gap_m<=gap(corner),"every rendered membrane box corner is covered");
    }
    check(bg::collision_envelope_version==2,"membrane union correction has an explicit collision version");
    state={};
    state.position.x=.25f/p::scale;
    r=inspect(state);
    check(r.valid&&r.certified&&r.lower_gap_m>.01,"translation past removed small ellipse is certified");
    for(int pose=0;pose<6;++pose) {
        state={};state.position.x=(pose%2?.37f:-.37f)/p::scale;
        state.position.z=(pose/2-1)*.03f/p::scale;
        state.orientation=p::axis_rotation({0,0,1},pose%2?.13f:-.17f)*p::axis_rotation({1,0,0},(pose-2)*.11f);
        state.angle[0]=.21f;state.angle[1]=-.08f;state.angle[2]=-.31f;state.angle[3]=.12f;
        r=inspect(state);
        std::printf("pose %d valid %d certified %d reason %d visits %u gap [%.9g,%.9g]\n",pose,r.valid,r.certified,r.failure_reason,r.visits,r.lower_gap_m,r.upper_gap_m);
        check(r.valid&&r.certified,"translated articulated rolled body has a complete certificate");
        if(!r.certified)continue;
        const auto links=cudarl::movement::shape(state);
        for(int b=0;b<3;++b) {
            auto envelope=bg::detail::shape(links.link[b],b);
            double reference=dense(envelope);
            if(b==2){envelope.box=true;envelope.c=cudarl::movement::Geometry::fin_thickness*.5;reference=bg::detail::minimum(reference,dense(envelope));}
            check(r.link[b].lower_gap_m<=reference+1e-9,"certified bound does not exceed independent dense sample");
            check(r.link[b].upper_gap_m-reference<2e-5,"adaptive minimum agrees with independent dense search");
            check(r.link[b].upper_gap_m-r.link[b].lower_gap_m<=1.0001e-5,"each envelope meets the declared precision");
        }
    }
    // A cylinder aligned with the tangent of a concave radial bank has clear
    // end corners while its middle intersects the bank. Corners are not proof.
    state={};
    const double x=.419;
    ++reference_queries;const auto surface=pg::sample(params,x,0);
    const double hx=-surface.normal.x/surface.normal.y,hz=-surface.normal.z/surface.normal.y,slope=hypot(hx,hz),angle=atan(slope);
    const double yaw=atan2(-hz,hx);
    state.orientation=p::axis_rotation({0,1,0},float(yaw))*p::axis_rotation({0,0,1},float(angle));
    state.position.x=float((x+.023*cos(angle)*cos(yaw))/p::scale);
    state.position.z=float((-.023*cos(angle)*sin(yaw))/p::scale);
    state.position.y=float((surface.height_m+.0015*sqrt(1+slope*slope)-.00008+.023*sin(angle))/p::scale);
    r=inspect(state);
    check(r.valid&&r.link[1].certified,"concave-bank rod envelope is certified");
    check(r.certified&&r.failure_reason==bg::None&&r.visits<=32768,"correct gap gradients certify the complete concave-bank body within the unchanged cap");
    const auto links=cudarl::movement::shape(state);const auto rod=bg::detail::shape(links.link[1],1);
    std::printf("corner details: input x %.9g H %.9g slope %.9g angle %.9g rod center %.9g %.9g axes %.9g %.9g\n",x,surface.height_m,slope,angle,rod.center.x,rod.center.y,rod.axis[0].x,rod.axis[0].y);
    double corners=1e30;
    for(int a=-1;a<=1;a+=2)for(int b=-1;b<=1;b+=2)for(int c=-1;c<=1;c+=2)
        corners=bg::detail::minimum(corners,gap(bg::detail::transform(rod,{a*rod.a,b*rod.b,c*rod.c})));
    std::printf("corner fixture: corners %.9g rod [%.9g,%.9g] visits %u\n",corners,r.link[1].lower_gap_m,r.link[1].upper_gap_m,r.visits);
    check(corners>0&&r.link[1].upper_gap_m<0,"end-box corners miss real curved-bed penetration");
    state={};state.position.y=-.004f/p::scale;r=inspect(state);
    check(!r.certified&&r.failure_reason==bg::SurfaceScope,"actual surface crossing fails scope without a10mm ceiling");
    state={};state.position.x=.6f/p::scale;r=inspect(state);
    check(!r.certified&&r.failure_reason==bg::DryLandScope,"dry-land scope is explicit");
    state={};state.position.x=.37f/p::scale;r=inspect(state,1);
    check(!r.certified&&r.failure_reason==bg::VisitLimit,"budget exhaustion never returns a safe certificate");
    state={};state.position.x=std::numeric_limits<float>::quiet_NaN();r=inspect(state);
    check(!r.valid&&!r.certified&&r.failure_reason==bg::InvalidInput,"invalid state fails explicitly");
    const auto bounds=bg::detail::wet_bounds(params);
    check(bounds.gradient>0&&bounds.hessian>0,"global wet derivatives have finite positive bounds");
    for(int angle_index=0;angle_index<120;++angle_index)for(int radius_index=0;radius_index<20;++radius_index) {
        const double angle_value=2*pg::pi*angle_index/120,radial=.45+.50*radius_index/19;
        const auto at=pg::outline(params,angle_value,radial).point_m;
        constexpr double h=1e-7;
        reference_queries+=5;
        const auto q=pg::sample(params,at.x,at.z),xp=pg::sample(params,at.x+h,at.z),xm=pg::sample(params,at.x-h,at.z),zp=pg::sample(params,at.x,at.z+h),zm=pg::sample(params,at.x,at.z-h);
        const double gx=-q.normal.x/q.normal.y,gz=-q.normal.z/q.normal.y;
        const double xx=(-xp.normal.x/xp.normal.y+xm.normal.x/xm.normal.y)/(2*h);
        const double zx=(-xp.normal.z/xp.normal.y+xm.normal.z/xm.normal.y)/(2*h);
        const double xz=(-zp.normal.x/zp.normal.y+zm.normal.x/zm.normal.y)/(2*h);
        const double zz=(-zp.normal.z/zp.normal.y+zm.normal.z/zm.normal.y)/(2*h);
        check(hypot(gx,gz)<=bounds.gradient,"analytic gradient remains inside proof bound");
        check(sqrt(xx*xx+zx*zx+xz*xz+zz*zz)<=bounds.hessian,"numerical Hessian reference remains inside proof bound");
    }
    std::printf("{\"passed\":%s,\"assertions\":%llu,\"body_queries\":%llu,\"query_visits\":%llu,\"independent_reference_queries\":%llu,\"gradient_bound\":%.17g,\"hessian_bound\":%.17g,\"physical_controls\":0,\"physical_substeps\":0,\"CUDA_kernels\":0,\"optimizer_steps\":0}\n",good?"true":"false",static_cast<unsigned long long>(assertions),static_cast<unsigned long long>(queries),static_cast<unsigned long long>(visits),static_cast<unsigned long long>(reference_queries),bounds.gradient,bounds.hessian);
    return good?0:1;
}
