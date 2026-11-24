// Pure saved-state geometry/normal regression. No physical step, force assembly,
// inference, CUDA context or optimizer entry point is invoked.
#include "simulation/pond_runtime.h"
#include "core/checkpoint.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
namespace bg=cudarl::pond_body_geometry;
namespace pg=cudarl::pond_geometry_v1;
namespace p=cudarl::physics;
namespace cp=cudarl::checkpoint;
namespace pr=cudarl::pond_runtime;
namespace {
constexpr double pi=3.141592653589793238462643383279502884;
constexpr double reference_gradient_bound=3,reference_hessian_bound=150,reference_roundoff=2e-12;
uint64_t assertions=0,body_queries=0,visits=0,height_evaluations=0,normal_queries=0,dense_points=0;
uint64_t flat_cases=0,bank_cases=0;
bool good=true;
double maximum_normal_error=0,maximum_wrong_sign_error=0,maximum_reference_width=0;
void check(bool passed,const char* why){++assertions;if(!passed){good=false;std::fprintf(stderr,"FAIL: %s\n",why);}}
struct V {double x=0,y=0,z=0;};
struct Shape {V center;double R[3][3]{};double a=0,b=0,c=0,norm_bound=0;int kind=0;};
struct Interval {double lower=1e30,upper=1e30;};
struct Frame {pr::World world;float command[4];pr::SubstepTrace trace[20];bool advanced;};
double height(double x,double z) {
    ++height_evaluations;
    // Independently written polar interpolation, with no terrain normal or
    // query helper used in this oracle. Only the fixed current wet pond applies.
    const double a=x/.52,b=z/.34,r=std::hypot(a,b),theta=std::atan2(b,a);
    const double s=1/std::hypot((.52/.22)*std::cos(theta),(.34/.14)*std::sin(theta));
    if(r<=s)return -.16;
    const double edge=1+.022*std::sin(3*theta+.8)+.014*std::cos(5*theta-.4)+.009*std::sin(7*theta);
    const double t=(r-s)/(edge-s),Q=t*t*t*(10+t*(-15+6*t));
    return -.16*(1-Q);
}
double gap(V point){return point.y-height(point.x,point.z);}
Shape shape(const p::Link& link,int component) {
    Shape s;s.center={link.center.x,link.center.y,link.center.z};
    const double x=link.q.x,y=link.q.y,z=link.q.z,w=link.q.w;
    // Independent explicit rotation matrix; do not call the implementation's
    // quaternion rotation, support, patch or minimum-linear helpers.
    s.R[0][0]=1-2*(y*y+z*z);s.R[0][1]=2*(x*y-z*w);s.R[0][2]=2*(x*z+y*w);
    s.R[1][0]=2*(x*y+z*w);s.R[1][1]=1-2*(x*x+z*z);s.R[1][2]=2*(y*z-x*w);
    s.R[2][0]=2*(x*z-y*w);s.R[2][1]=2*(y*z+x*w);s.R[2][2]=1-2*(x*x+y*y);
    double frobenius=0;for(const auto& row:s.R)for(double v:row)frobenius+=v*v;
    s.norm_bound=std::sqrt(frobenius);
    const auto geometry=cudarl::movement::body_geometry(component);
    s.a=geometry.length*.5;s.b=geometry.half_height;s.c=geometry.half_width;s.kind=component==0?0:1;
    return s;
}
V transform(const Shape& s,V local) {
    return {s.center.x+s.R[0][0]*local.x+s.R[0][1]*local.y+s.R[0][2]*local.z,
            s.center.y+s.R[1][0]*local.x+s.R[1][1]*local.y+s.R[1][2]*local.z,
            s.center.z+s.R[2][0]*local.x+s.R[2][1]*local.y+s.R[2][2]*local.z};
}
double extent_y(const Shape& s) {
    if(s.kind==0)return std::hypot(std::hypot(s.a*s.R[1][0],s.b*s.R[1][1]),s.c*s.R[1][2]);
    if(s.kind==2)return std::abs(s.a*s.R[1][0])+std::abs(s.b*s.R[1][1])+std::abs(s.c*s.R[1][2]);
    return std::abs(s.a*s.R[1][0])+std::hypot(s.b*s.R[1][1],s.c*s.R[1][2]);
}
bool wholly_wet(const Shape& s) {
    double local=s.kind==0?std::max({s.a,s.b,s.c}):std::hypot(s.a,std::max(s.b,s.c));
    if(s.kind==2)local=std::hypot(std::hypot(s.a,s.b),s.c);
    return std::hypot(s.center.x/.52,s.center.z/.34)+s.norm_bound*local/.34<=.955;
}
Interval dense(const Shape& s) {
    check(wholly_wet(s),"independent enclosing ball certifies every reference surface is wet");
    double minimum=1e30,error=0;
    const auto sample=[&](V local){++dense_points;minimum=std::min(minimum,gap(transform(s,local)));};
    const double G=std::sqrt(1+reference_gradient_bound*reference_gradient_bound),H=reference_hessian_bound;
    if(s.kind==0) {
        constexpr int Nu=1024,Nv=512;
        const double U=s.norm_bound*std::max(s.a,s.b),W=s.norm_bound*std::max({s.a,s.b,s.c});
        error=(G*U+H*U*U)*std::pow(2*pi/Nu,2)/8+(G*W+H*W*W)*std::pow(pi/Nv,2)/8;
        for(int i=0;i<=Nu;++i)for(int j=0;j<=Nv;++j) {
            const double u=2*pi*i/Nu,v=pi*j/Nv;
            sample({s.a*std::cos(u)*std::sin(v),s.b*std::sin(u)*std::sin(v),s.c*std::cos(v)});
        }
    } else if(s.kind==1) {
        constexpr int Nu=256,Nv=512,Nr=128;
        const double R=s.norm_bound*std::max(s.b,s.c),N=s.norm_bound;
        const double side_error=H*N*N*std::pow(2*s.a/Nu,2)/8+(G*R+H*R*R)*std::pow(2*pi/Nv,2)/8;
        const double disc_error=H*R*R*std::pow(1./Nr,2)/8+(G*R+H*R*R)*std::pow(2*pi/Nv,2)/8;
        error=std::max(side_error,disc_error);
        for(int i=0;i<=Nu;++i)for(int j=0;j<=Nv;++j) {
            const double a=-s.a+2*s.a*i/Nu,v=2*pi*j/Nv;
            sample({a,s.b*std::cos(v),s.c*std::sin(v)});
        }
        for(int end:{-1,1})for(int i=0;i<=Nr;++i)for(int j=0;j<=Nv;++j) {
            const double radius=double(i)/Nr,v=2*pi*j/Nv;
            sample({end*s.a,radius*s.b*std::cos(v),radius*s.c*std::sin(v)});
        }
    } else {
        constexpr int Nu=256,Nv=128;
        for(int face=0;face<6;++face) {
            const int axis=face/2;const double sign=face%2?1.:-1.;
            const double u=axis==0?s.b:s.a,v=axis==2?s.b:s.c;
            error=std::max(error,H*s.norm_bound*s.norm_bound*(std::pow(2*u/Nu,2)+std::pow(2*v/Nv,2))/8);
            for(int i=0;i<=Nu;++i)for(int j=0;j<=Nv;++j) {
                const double a=-u+2*u*i/Nu,b=-v+2*v*j/Nv;
                sample(axis==0?V{sign*s.a,a,b}:axis==1?V{a,sign*s.b,b}:V{a,b,sign*s.c});
            }
        }
    }
    // Sequential one-dimensional linear-interpolation error bounds imply the
    // bilinear grid error Muu*du²/8+Mvv*dv²/8; no mixed derivative is needed.
    // Thus every complete parameter cell is covered, not only its grid points.
    maximum_reference_width=std::max(maximum_reference_width,error+2*reference_roundoff);
    check(error+2*reference_roundoff<2.5e-6,"independent dense certificate meets its predeclared2.5um precision");
    return {minimum-error-reference_roundoff,minimum+reference_roundoff};
}
void normals(const pg::Parameters& parameters,V point) {
    ++normal_queries;const auto q=pg::sample(parameters,point.x,point.z);constexpr double h=1e-6;
    check(q.valid&&q.region!=pg::DryLand,"normal test is inside wet geometry scope");
    const double Hx=(height(point.x+h,point.z)-height(point.x-h,point.z))/(2*h);
    const double Hz=(height(point.x,point.z+h)-height(point.x,point.z-h))/(2*h);
    const auto gradient=bg::detail::gap_gradient(q);
    const double error=std::hypot(gradient.x+Hx,gradient.z+Hz);
    maximum_normal_error=std::max(maximum_normal_error,error);
    maximum_wrong_sign_error=std::max(maximum_wrong_sign_error,std::hypot(-q.normal.x/q.normal.y+Hx,-q.normal.z/q.normal.y+Hz));
    check(error<=2e-8&&gradient.y==1,"gap gradient agrees with independent height finite differences");
    const double norm=std::sqrt(Hx*Hx+1+Hz*Hz);
    check(std::hypot(std::hypot(q.normal.x+Hx/norm,q.normal.y-1/norm),q.normal.z+Hz/norm)<=2e-8,"upward terrain normal matches independent finite-difference normal");
    check(std::abs(q.height_m-height(point.x,point.z))<=2e-14,"independent wet height formula agrees");
}
void body(const p::BodyState& state,const pg::Parameters& parameters,int tick,int id) {
    ++body_queries;const auto query=bg::inspect(state,parameters,1e-5,131072);visits+=query.visits;
    std::printf("body tick=%d case=%d certified=%d visits=%u gap=[%.17g,%.17g]\n",tick,id,query.certified,query.visits,query.lower_gap_m,query.upper_gap_m);
    check(query.valid&&query.certified,"saved complete body has a corrected whole-body certificate");
    if(!query.certified)return;
    const auto links=cudarl::movement::shape(state);std::array<Shape,4> shapes;
    for(int i=0;i<3;++i)shapes[static_cast<size_t>(i)]=shape(links.link[i],i);
    shapes[3]=shapes[2];shapes[3].kind=2;shapes[3].c=cudarl::movement::Geometry::fin_thickness*.5;
    double highest=-1e30;for(const auto& s:shapes)highest=std::max(highest,s.center.y+extent_y(s));
    check(std::abs(highest-query.max_body_y_m)<2e-14,"whole-body highest envelope agrees with independent exact support");
    if(id<4) {
        ++flat_cases;double lowest=1e30;
        for(const auto& s:shapes)lowest=std::min(lowest,s.center.y-extent_y(s)+.16);
        check(query.visits==4,"flat saved bodies retain their exact four-envelope fast path");
        check(query.lower_gap_m<=lowest&&lowest<=query.upper_gap_m,"flat whole-body certificate contains independent exact minimum");
    } else {
        ++bank_cases;std::array<Interval,4> reference;
        for(int i=0;i<4;++i)reference[static_cast<size_t>(i)]=dense(shapes[static_cast<size_t>(i)]);
        reference[2]={std::min(reference[2].lower,reference[3].lower),std::min(reference[2].upper,reference[3].upper)};
        for(int i=0;i<3;++i) {
            const auto& q=query.link[i];const auto& r=reference[static_cast<size_t>(i)];
            check(q.lower_gap_m<=r.upper&&r.lower<=q.upper_gap_m,"each complete corrected envelope overlaps independent dense certificate");
            check(q.upper_gap_m-q.lower_gap_m<=1e-5,"each corrected link retains unchanged10um precision");
            normals(parameters,{q.exact_point_m.x,q.exact_point_m.y,q.exact_point_m.z});
            const double exact_gap=gap({q.exact_point_m.x,q.exact_point_m.y,q.exact_point_m.z});
            check(std::abs(q.upper_gap_m-(exact_gap+q.arithmetic_margin_m))<2e-14,"reported upper bound is an independently evaluated actual surface sample");
            const auto point_query=pg::sample(parameters,q.exact_point_m.x,q.exact_point_m.z);++normal_queries;
            const double Hx=-point_query.normal.x/point_query.normal.y,Hz=-point_query.normal.z/point_query.normal.y;
            check(std::abs(q.gradient_x-Hx)<2e-14&&std::abs(q.gradient_z-Hz)<2e-14,"returned gradient_x/z remain height derivatives for contact solver");
        }
    }
}
}
int main(int argc,char** argv) {
    if(argc!=4)return 2;static_assert(sizeof(Frame)==12448);
    static_assert(bg::collision_envelope_version==2&&bg::query_arithmetic_version==2);
    const auto parameters=pg::current_visual_parameters();
    // The independent fixed-pond derivative bounds are conservative roundings
    // above the already-derived analytic wet bounds; no numerical Hessian
    // sampling is substituted for the certificate proof.
    const auto proved=bg::detail::wet_bounds(parameters);
    check(proved.gradient<reference_gradient_bound&&proved.hessian<reference_hessian_bound,"independent oracle constants conservatively cover analytic global derivative bounds");
    for(int angle=0;angle<64;++angle)for(int radius=0;radius<8;++radius) {
        const double a=2*pi*angle/64,r=.46+.49*radius/7;
        const double e=1+.022*std::sin(3*a+.8)+.014*std::cos(5*a-.4)+.009*std::sin(7*a);
        normals(parameters,{.52*e*r*std::cos(a),-.08,.34*e*r*std::sin(a)});
    }
    try {
        for(int file=1;file<4;++file) {
            const auto bytes=cp::read(argv[file]);cp::Reader r{bytes};
            check(r.value<uint32_t>()==0x32474350&&r.value<uint32_t>()==1,"saved snapshot format is preserved");const int tick=r.value<int>();
            check(r.value<uint32_t>()==sizeof(Frame)&&r.value<uint32_t>()==8,"saved snapshot layout is preserved");
            for(int backend=0;backend<2;++backend)for(int id=0;id<8;++id){const auto saved=r.value<Frame>();if(backend==1)body(saved.world.motor.physical.body,saved.world.terrain,tick,id);}
            r.finish();
        }
    } catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 2;}
    check(body_queries==24&&flat_cases==12&&bank_cases==12,"all predeclared saved states ran exactly once");
    check(maximum_wrong_sign_error>.05,"normal regression rejects the former opposite horizontal signs");
    std::printf("{\"passed\":%s,\"assertions\":%llu,\"body_queries\":%llu,\"query_visits\":%llu,\"flat_cases\":%llu,\"bank_cases\":%llu,\"independent_dense_points\":%llu,\"independent_height_evaluations\":%llu,\"normal_queries\":%llu,\"maximum_gap_gradient_error\":%.17g,\"maximum_old_sign_error\":%.17g,\"maximum_reference_interval_width_m\":%.17g,\"physical_controls\":0,\"physical_substeps\":0,\"CUDA_kernels\":0,\"optimizer_updates\":0}\n",
        good?"true":"false",static_cast<unsigned long long>(assertions),static_cast<unsigned long long>(body_queries),static_cast<unsigned long long>(visits),
        static_cast<unsigned long long>(flat_cases),static_cast<unsigned long long>(bank_cases),static_cast<unsigned long long>(dense_points),
        static_cast<unsigned long long>(height_evaluations),static_cast<unsigned long long>(normal_queries),maximum_normal_error,maximum_wrong_sign_error,maximum_reference_width);
    return good?0:1;
}
