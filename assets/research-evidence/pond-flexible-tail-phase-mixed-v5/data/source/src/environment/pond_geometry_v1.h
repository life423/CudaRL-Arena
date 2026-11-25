#ifndef CUDARL_ENVIRONMENT_POND_GEOMETRY_V1_H
#define CUDARL_ENVIRONMENT_POND_GEOMETRY_V1_H
#include "core/hd.h"
#include <cmath>
#include <cstdint>

// Geometry queries only. These functions do not integrate a body, resolve
// contact, change a pose, select an action or invoke an optimizer. Version 1
// describes the existing drawn pond; it does not qualify a physical bank.
namespace cudarl::pond_geometry_v1 {
constexpr uint32_t version=1;
constexpr bool physical_response_implemented=false;
constexpr bool selects_actions=false;
constexpr double pi=3.141592653589793238462643383279502884;
struct Parameters {
    double water_radius_x_m=0,water_radius_z_m=0;
    double flat_radius_x_m=0,flat_radius_z_m=0;
    double floor_y_m=0,reference_scale_m=0;
};
CUDARL_HD constexpr Parameters current_visual_parameters() {
    return {.52,.34,.22,.14,-.16,.04};
}
struct Vec3 {double x=0,y=0,z=0;};
enum Region : uint32_t {Invalid=0,FlatBed=1,SubmergedSlope=2,DryLand=3};
struct Outline {bool valid=false;Vec3 point_m{};double edge_factor=0;};
struct Query {
    bool valid=false;
    double height_m=0;
    Vec3 normal{0,1,0};
    double shoreline_coordinate=0,flat_coordinate=0;
    Region region=Invalid;
};
CUDARL_HD inline bool finite(double value) {return value==value&&fabs(value)<=1.7976931348623157e308;}
CUDARL_HD inline bool valid_parameters(const Parameters& p) {
    const double values[]={p.water_radius_x_m,p.water_radius_z_m,p.flat_radius_x_m,
        p.flat_radius_z_m,p.floor_y_m,p.reference_scale_m};
    for(double value:values)if(!finite(value))return false;
    // 0.955 is a conservative lower bound for the three-harmonic outline.
    // A genuine sloped annulus needs the entire flat ellipse inside that rim.
    return p.water_radius_x_m>0&&p.water_radius_z_m>0&&p.flat_radius_x_m>0&&p.flat_radius_z_m>0&&
        p.reference_scale_m>0&&p.floor_y_m<0&&
        p.flat_radius_x_m/p.water_radius_x_m<.955&&p.flat_radius_z_m/p.water_radius_z_m<.955;
}
CUDARL_HD inline double edge(double angle) {
    return 1+.022*sin(angle*3+.8)+.014*cos(angle*5-.4)+.009*sin(angle*7);
}
CUDARL_HD inline double edge_derivative(double angle) {
    return .066*cos(angle*3+.8)-.070*sin(angle*5-.4)+.063*cos(angle*7);
}
CUDARL_HD inline Outline outline(const Parameters& p,double angle,double normalized_radius=1) {
    Outline out;
    if(!valid_parameters(p)||!finite(angle)||!finite(normalized_radius)||normalized_radius<0)return out;
    out.edge_factor=edge(angle);
    out.point_m={cos(angle)*p.water_radius_x_m*normalized_radius*out.edge_factor,0,
        sin(angle)*p.water_radius_z_m*normalized_radius*out.edge_factor};
    out.valid=finite(out.edge_factor)&&finite(out.point_m.x)&&finite(out.point_m.z);
    if(!out.valid)return {};
    return out;
}
namespace detail {
// This bounded scalar is an interpolation weight, never a body-state clamp.
CUDARL_HD inline double unit_weight(double value) {return value<0?0:value>1?1:value;}
struct Smooth {double value=0,derivative=0;};
CUDARL_HD inline Smooth smooth(double low,double high,double value) {
    const double u=unit_weight((value-low)/(high-low));
    return {u*u*(3-2*u),value>low&&value<high?6*u*(1-u)/(high-low):0};
}
CUDARL_HD inline bool set_surface(Query& q,double height,double dx,double dz,Region region) {
    if(!finite(height)||!finite(dx)||!finite(dz))return false;
    const double length=hypot(hypot(dx,1.),dz);
    if(!finite(length)||length<=0)return false;
    q.height_m=height;q.normal={-dx/length,1/length,-dz/length};q.region=region;q.valid=true;
    return true;
}
}
#if defined(CUDARL_POND_GEOMETRY_RDC) && defined(__CUDA_ARCH__)
extern __device__ __noinline__ Query sample(const Parameters& p,double x_m,double z_m)
#else
CUDARL_HD CUDARL_DEVICE_NOINLINE Query sample(const Parameters& p,double x_m,double z_m)
#endif
#if defined(CUDARL_POND_GEOMETRY_RDC) && defined(__CUDA_ARCH__) && !defined(CUDARL_DEFINE_POND_SAMPLE)
;
#else
{
    Query out;
    if(!valid_parameters(p)||!finite(x_m)||!finite(z_m))return out;
    const double a=x_m/p.water_radius_x_m,b=z_m/p.water_radius_z_m;
    const double radial=hypot(a,b),angle=atan2(b,a),c=cos(angle),s=sin(angle),e=edge(angle);
    const double flat_x=p.water_radius_x_m/p.flat_radius_x_m,flat_z=p.water_radius_z_m/p.flat_radius_z_m;
    const double flat_scale=hypot(c*flat_x,s*flat_z);
    const double r=radial/e,f=1/(flat_scale*e);
    const double flat_coordinate=hypot(x_m/p.flat_radius_x_m,z_m/p.flat_radius_z_m);
    if(!finite(radial)||!finite(r)||!finite(f)||!finite(flat_coordinate)||f>=1)return out;
    out.shoreline_coordinate=r;out.flat_coordinate=flat_coordinate;
    if(r<=f) {detail::set_surface(out,p.floor_y_m,0,0,FlatBed);return out;}
    if(radial<=0)return out; // The origin was necessarily handled by FlatBed.
    const double angle_x=-b/(radial*radial*p.water_radius_x_m),angle_z=a/(radial*radial*p.water_radius_z_m);
    const double radial_x=a/(radial*p.water_radius_x_m),radial_z=b/(radial*p.water_radius_z_m);
    const double e_prime=edge_derivative(angle);
    const double r_angle=-radial*e_prime/(e*e);
    const double rx=radial_x/e+r_angle*angle_x,rz=radial_z/e+r_angle*angle_z;
    if(r<=1) {
        const double flat_prime=c*s*(flat_z*flat_z-flat_x*flat_x)/flat_scale;
        const double f_prime=-f*(flat_prime/flat_scale+e_prime/e);
        const double denominator=1-f,t=detail::unit_weight((r-f)/denominator);
        const double q=t*t*t*(t*(t*6-15)+10),q_prime=30*t*t*(t-1)*(t-1);
        const double tx=rx/denominator+(r-1)*f_prime*angle_x/(denominator*denominator);
        const double tz=rz/denominator+(r-1)*f_prime*angle_z/(denominator*denominator);
        detail::set_surface(out,p.floor_y_m*(1-q),-p.floor_y_m*q_prime*tx,-p.floor_y_m*q_prime*tz,SubmergedSlope);
        return out;
    }
    // Convert the old artistic dry-land formula from model units to SI.
    // reference_scale_m describes that formula only; it never rescales a body.
    const double scale=p.reference_scale_m,x=x_m/scale,z=z_m/scale;
    const auto lip=detail::smooth(1,1.075,r),ripple=detail::smooth(1.02,1.2,r),open=detail::smooth(1.32,2.4,r);
    const double undulation=sin(x*.57)*cos(z*.68)*.07;
    const double ux=cos(x*.57)*cos(z*.68)*(.07*.57)/scale;
    const double uz=-sin(x*.57)*sin(z*.68)*(.07*.68)/scale;
    const double hill=1.7+1.2*sin(x*.055+.6)*cos(z*.045-.3)+.55*sin(x*.11-z*.08);
    const double hx=(1.2*.055*cos(x*.055+.6)*cos(z*.045-.3)+.55*.11*cos(x*.11-z*.08))/scale;
    const double hz=(-1.2*.045*sin(x*.055+.6)*sin(z*.045-.3)-.55*.08*cos(x*.11-z*.08))/scale;
    const double west=3.8*exp(-((x+36)*(x+36)/540+(z+24)*(z+24)/230));
    const double east=2.8*exp(-((x-37)*(x-37)/400+(z-18)*(z-18)/330));
    const double hills=hill+west+east;
    const double hills_x=hx+west*(-2*(x+36)/540)/scale+east*(-2*(x-37)/400)/scale;
    const double hills_z=hz+west*(-2*(z+24)/230)/scale+east*(-2*(z-18)/330)/scale;
    const double height=scale*(.35*lip.value+undulation*ripple.value+open.value*hills);
    const double dr=scale*(.35*lip.derivative+undulation*ripple.derivative+open.derivative*hills);
    detail::set_surface(out,height,dr*rx+scale*(ux*ripple.value+open.value*hills_x),
        dr*rz+scale*(uz*ripple.value+open.value*hills_z),DryLand);
    return out;
}
#endif
}
#endif
