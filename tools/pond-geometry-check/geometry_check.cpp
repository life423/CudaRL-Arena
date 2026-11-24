// Geometry/algebra checks only. No simulation, native actor, kernel or optimizer.
#include "environment/pond_geometry_v1.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace g=cudarl::pond_geometry_v1;
namespace {
int assertions=0,queries=0;
double maximum_height_error=0,maximum_normal_error=0;
void require(bool condition,const char* message) {++assertions;if(!condition)throw std::runtime_error(message);}
double clip_weight(double x) {return std::min(1.,std::max(0.,x));}
double smooth_reference(double low,double high,double x) {const double t=clip_weight((x-low)/(high-low));return t*t*(3-2*t);}
// Direct transcription of the pre-existing GDScript scalar equations, using
// model-space coordinates. Production queries independently use SI coordinates
// and analytic derivatives. This reference never changes a physical body.
double drawn_height_reference(const g::Parameters& p,double x_m,double z_m) {
    const double scale=p.reference_scale_m,x=x_m/scale,z=z_m/scale;
    const double wx=p.water_radius_x_m/scale,wz=p.water_radius_z_m/scale;
    const double fx=p.flat_radius_x_m/scale,fz=p.flat_radius_z_m/scale;
    const double angle=std::atan2(z/wz,x/wx);
    const double edge=1+.022*std::sin(angle*3+.8)+.014*std::cos(angle*5-.4)+.009*std::sin(angle*7);
    const double radius=std::hypot(x/wx,z/wz)/edge;
    const double flat=1/(std::hypot(std::cos(angle)*wx/fx,std::sin(angle)*wz/fz)*edge);
    if(radius<=flat)return p.floor_y_m;
    if(radius<=1) {
        const double t=clip_weight((radius-flat)/(1-flat)),slope=t*t*t*(t*(t*6-15)+10);
        return scale*((p.floor_y_m/scale)*(1-slope));
    }
    const double rise=smooth_reference(1,1.075,radius)*.35;
    const double undulation=std::sin(x*.57)*std::cos(z*.68)*.07*smooth_reference(1.02,1.2,radius);
    const double open=smooth_reference(1.32,2.4,radius);
    const double hills=1.7+1.2*std::sin(x*.055+.6)*std::cos(z*.045-.3)+.55*std::sin(x*.11-z*.08);
    const double west=3.8*std::exp(-((x+36)*(x+36)/540+(z+24)*(z+24)/230));
    const double east=2.8*std::exp(-((x-37)*(x-37)/400+(z-18)*(z-18)/330));
    return scale*(rise+undulation+open*(hills+west+east));
}
g::Query query(const g::Parameters& p,double x,double z) {++queries;return g::sample(p,x,z);}
void verify_surface(const g::Parameters& p,double x,double z,bool derivative=true) {
    const auto s=query(p,x,z);
    require(s.valid,"Finite geometry fixture rejected");
    const double expected=drawn_height_reference(p,x,z),height_error=std::abs(s.height_m-expected);
    maximum_height_error=std::max(maximum_height_error,height_error);
    require(height_error<=1e-12+std::abs(expected)*1e-11,"SI height differs from original drawn equation");
    const double norm=std::hypot(std::hypot(s.normal.x,s.normal.y),s.normal.z);
    require(std::abs(norm-1)<1e-12&&s.normal.y>0,"Surface normal must be a finite upward unit vector");
    require(s.region!=g::Invalid,"Valid geometry lacks a region");
    if(!derivative)return;
    constexpr double h=1e-7;
    const auto xp=query(p,x+h,z),xm=query(p,x-h,z),zp=query(p,x,z+h),zm=query(p,x,z-h);
    require(xp.valid&&xm.valid&&zp.valid&&zm.valid,"Derivative fixture rejected");
    const double dx=(xp.height_m-xm.height_m)/(2*h),dz=(zp.height_m-zm.height_m)/(2*h);
    const double nx=-dx,ny=1,nz=-dz,length=std::hypot(std::hypot(nx,ny),nz);
    const double error=std::max({std::abs(s.normal.x-nx/length),std::abs(s.normal.y-ny/length),std::abs(s.normal.z-nz/length)});
    maximum_normal_error=std::max(maximum_normal_error,error);
    require(error<2e-7,"Analytic normal differs from independent height differences");
}
void fixtures(const g::Parameters& p) {
    require(g::valid_parameters(p),"Fixture parameter set invalid");
    const auto center=query(p,0,0);
    require(center.valid&&center.region==g::FlatBed&&center.height_m==p.floor_y_m,"Center floor must remain flat");
    require(center.normal.x==0&&center.normal.y==1&&center.normal.z==0,"Center normal is not vertical");
    const double radii[]={.08,.3,.55,.72,.9,.99,1.01,1.04,1.1,1.24,1.5,2.,3.25,7.,12.};
    for(int i=0;i<192;++i) {
        const double angle=2*g::pi*double(i)/192;
        const auto edge=g::outline(p,angle);
        require(edge.valid,"Shoreline query rejected");
        const auto rim=query(p,edge.point_m.x,edge.point_m.z);
        require(rim.valid&&std::abs(rim.shoreline_coordinate-1)<1e-14,"Outlined point does not lie on shared waterline");
        require(std::abs(rim.height_m)<2e-14,"Waterline height differs from zero");
        require(std::abs(edge.edge_factor-g::edge(angle))<1e-14,"Outline harmonic changed");
        for(double radius:radii) {
            const auto at=g::outline(p,angle,radius);
            require(at.valid,"Scaled outline rejected");
            verify_surface(p,at.point_m.x,at.point_m.z);
        }
        const double fx=p.flat_radius_x_m*std::cos(angle),fz=p.flat_radius_z_m*std::sin(angle);
        const auto inside=query(p,fx*.99999,fz*.99999),outside=query(p,fx*1.00001,fz*1.00001);
        require(inside.valid&&outside.valid&&inside.height_m==p.floor_y_m,"Flat-bed join invalid");
        require(std::abs(outside.height_m-p.floor_y_m)<1e-10,"Flat-bed height is discontinuous");
        const auto before=query(p,edge.point_m.x*(1-1e-7),edge.point_m.z*(1-1e-7));
        const auto after=query(p,edge.point_m.x*(1+1e-7),edge.point_m.z*(1+1e-7));
        require(before.valid&&after.valid&&std::abs(before.height_m-after.height_m)<1e-10,"Waterline height is discontinuous");
        require(std::abs(before.normal.y-1)<1e-10&&std::abs(after.normal.y-1)<1e-8,"Waterline tangent is discontinuous");
    }
}
}
int main() {
    try {
        const auto original=g::current_visual_parameters();
        require(g::version==1&&!g::physical_response_implemented&&!g::selects_actions,"Geometry must not claim physical or behavioral response");
        fixtures(original);
        auto other=original;other.water_radius_x_m=.31;other.water_radius_z_m=.21;other.flat_radius_x_m=.11;other.flat_radius_z_m=.08;
        other.floor_y_m=-.12;other.reference_scale_m=.05;fixtures(other);
        const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
        require(!query(original,nan,0).valid&&!query(original,0,inf).valid,"Nonfinite positions must be rejected");
        auto invalid=original;invalid.flat_radius_x_m=invalid.water_radius_x_m;
        require(!g::valid_parameters(invalid)&&!query(invalid,0,0).valid,"Invalid sloped annulus must be rejected");
        invalid=original;invalid.reference_scale_m=0;require(!g::valid_parameters(invalid),"Zero reference scale accepted");
        invalid=original;invalid.floor_y_m=nan;require(!g::valid_parameters(invalid),"Nonfinite parameters accepted");
        require(!g::outline(original,nan).valid&&!g::outline(original,0,-1).valid,"Invalid outline inputs accepted");
        require(!g::outline(original,1e308).valid,"Overflowing angle must reject rather than produce an invalid point");
        std::cout.precision(17);
        std::cout<<"{\"passed\":true,\"assertions\":"<<assertions<<",\"geometry_queries\":"<<queries
            <<",\"maximum_height_error_m\":"<<maximum_height_error<<",\"maximum_normal_error\":"<<maximum_normal_error
            <<",\"physical_controls\":0,\"physical_substeps\":0,\"CUDA_kernels\":0,\"optimizer_steps\":0}\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"Geometry check failed after "<<assertions<<" assertions: "<<e.what()<<'\n';return 1;
    }
}
