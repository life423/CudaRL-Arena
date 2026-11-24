#ifndef CUDARL_ENVIRONMENT_POND_BODY_GEOMETRY_H
#define CUDARL_ENVIRONMENT_POND_BODY_GEOMETRY_H
#include "environment/pond_geometry_v1.h"
#include "motor/movement_model.h"

// Pure queries of the EXISTING model6 collision envelopes. No force, pose,
// actuator, integration or optimization operation is performed here.
namespace cudarl::pond_body_geometry {
namespace terrain=pond_geometry_v1;
namespace p=physics;
// v2 adds the rendered thin membrane box to the original composite elliptical
// cylinder. Its rim corners extend outside that ellipse; no mass/inertia or
// original proxy dimensions are changed.
constexpr uint32_t collision_envelope_version=2;
// Arithmetic v2 uses the gradient of gap=y-height for support, tangent
// certificates and downhill traversal. Collision envelopes are unchanged.
constexpr uint32_t query_arithmetic_version=2;
enum FailureReason {None=0,InvalidInput=1,SurfaceScope=2,DryLandScope=3,VisitLimit=4,InvalidQuery=5};
struct LinkInspection {
    double lower_gap_m=1e30,upper_gap_m=1e30;
    p::V point_m{},normal{0,1,0};
    terrain::Vec3 exact_point_m{};
    double point_rounding_error_m=0,arithmetic_margin_m=0;
    double gradient_x=0,gradient_z=0;
    bool certified=false;
};
struct Inspection {
    bool valid=false,certified=false;
    int failure_reason=None;
    uint32_t visits=0;
    double lower_gap_m=1e30,upper_gap_m=1e30,max_body_y_m=-1e30;
    LinkInspection link[3];
};
namespace detail {
struct V {
    double x=0,y=0,z=0;
    CUDARL_HD V operator+(V b) const {return {x+b.x,y+b.y,z+b.z};}
    CUDARL_HD V operator-(V b) const {return {x-b.x,y-b.y,z-b.z};}
    CUDARL_HD V operator*(double n) const {return {x*n,y*n,z*n};}
};
// Terrain normals are proportional to {-height_x,1,-height_z}; these ratios
// are the horizontal derivatives of gap, not the returned height derivatives.
CUDARL_HD inline V gap_gradient(const terrain::Query& q) {
    return {q.normal.x/q.normal.y,1,q.normal.z/q.normal.y};
}
CUDARL_HD inline double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
CUDARL_HD inline double length(V a){return hypot(hypot(a.x,a.y),a.z);}
CUDARL_HD inline double maximum(double a,double b){return a>b?a:b;}
CUDARL_HD inline double minimum(double a,double b){return a<b?a:b;}
struct Shape {
    V center{},axis[3]{};
    double a=0,b=0,c=0,gain=1;
    bool ellipsoid=false,box=false;
};
CUDARL_HD inline V rotate(p::Q q,V v) {
    // Evaluate the same linear quaternion rotation as physics::Q in double.
    const V n{q.x,q.y,q.z};
    const V c{n.y*v.z-n.z*v.y,n.z*v.x-n.x*v.z,n.x*v.y-n.y*v.x};
    const V t=c+v*double(q.w);
    const V d{n.y*t.z-n.z*t.y,n.z*t.x-n.x*t.z,n.x*t.y-n.y*t.x};
    return v+d*2.;
}
CUDARL_HD inline Shape shape(const p::Link& link,int body) {
    const auto g=movement::body_geometry(body);Shape s;
    s.center={link.center.x,link.center.y,link.center.z};
    s.axis[0]=rotate(link.q,{1,0,0});s.axis[1]=rotate(link.q,{0,1,0});s.axis[2]=rotate(link.q,{0,0,1});
    s.a=g.length*.5;s.b=g.half_height;s.c=g.half_width;s.ellipsoid=body==0;
    const double n2=double(link.q.x)*link.q.x+double(link.q.y)*link.q.y+double(link.q.z)*link.q.z;
    // Exact operator norm of Q::rotate, including an accepted nonunit Q.
    s.gain=maximum(1.,sqrt((1-2*n2)*(1-2*n2)+4*double(link.q.w)*link.q.w*n2));
    return s;
}
CUDARL_HD inline V transform(const Shape& s,V local){return s.center+s.axis[0]*local.x+s.axis[1]*local.y+s.axis[2]*local.z;}
CUDARL_HD inline V support(const Shape& s,V n) {
    const double x=dot(s.axis[0],n),y=dot(s.axis[1],n),z=dot(s.axis[2],n);
    if(s.box)return transform(s,{x>=0?-s.a:s.a,y>=0?-s.b:s.b,z>=0?-s.c:s.c});
    if(s.ellipsoid) {
        const double d=hypot(hypot(s.a*x,s.b*y),s.c*z);
        if(d<=0)return s.center;
        return transform(s,{-s.a*s.a*x/d,-s.b*s.b*y/d,-s.c*s.c*z/d});
    }
    const double d=hypot(s.b*y,s.c*z);
    return transform(s,{x>=0?-s.a:s.a,d>0?-s.b*s.b*y/d:0,d>0?-s.c*s.c*z/d:0});
}
struct Bounds {double gradient=0,hessian=0;};
CUDARL_HD inline Bounds wet_bounds(const terrain::Parameters& p) {
    // rho = |(x/Rx,z/Rz)|, s(theta) = 1/sqrt(A²cos²+B²sin²),
    // tau=(rho-s)/(edge-s). Outside the flat bed and inside the wet outline,
    // rho>=smin and d=edge-s>=dmin. Bounds follow the chain/product rules;
    // quintic Q has |Q'|<=1.875, |Q''|<6. Flat joins are C2, so these bounds
    // also cover any line segment crossing between flat and sloped bed.
    const double A=p.water_radius_x_m/p.flat_radius_x_m,B=p.water_radius_z_m/p.flat_radius_z_m;
    const double m=minimum(A,B),M=maximum(A,B),delta=fabs(A*A-B*B);
    const double smin=1/M,smax=1/m,dmin=.955-smax,R=minimum(p.water_radius_x_m,p.water_radius_z_m);
    const double S1=delta/(2*m*m*m),S2=.75*delta*delta/(m*m*m*m*m)+delta/(m*m*m);
    const double D1=.199+S1,D2=.989+S2;
    const double Tr=1/dmin,Tt=(S1+D1)/dmin,Trt=D1/(dmin*dmin);
    const double Ttt=(S2+D2)/dmin+2*(S1+D1)*D1/(dmin*dmin);
    const double Gr=1/R,Gt=1/(smin*R),Hr=1/(smin*R*R),Ht=1/(smin*smin*R*R);
    const double G=Tr*Gr+Tt*Gt,H=2*Trt*Gr*Gt+Ttt*Gt*Gt+Tr*Hr+Tt*Ht;
    return {fabs(p.floor_y_m)*1.875*G,fabs(p.floor_y_m)*(6*G*G+1.875*H)};
}
struct Patch {double u=0,v=0,hu=0,hv=0;int kind=0;};
struct PatchGeometry {V point{},du{},dv{};double radius=0,remainder=0;};
CUDARL_HD inline PatchGeometry patch_geometry(const Shape& s,const Patch& p) {
    PatchGeometry g;
    if(s.box) {
        const int fixed=p.kind/2;const double sign=p.kind%2?1.:-1.;
        V local;
        if(fixed==0){local={sign*s.a,p.u,p.v};g.du=s.axis[1];g.dv=s.axis[2];}
        else if(fixed==1){local={p.u,sign*s.b,p.v};g.du=s.axis[0];g.dv=s.axis[2];}
        else {local={p.u,p.v,sign*s.c};g.du=s.axis[0];g.dv=s.axis[1];}
        g.point=transform(s,local);g.radius=length(g.du)*p.hu+length(g.dv)*p.hv;
    } else if(s.ellipsoid) {
        const double cu=cos(p.u),su=sin(p.u),cv=cos(p.v),sv=sin(p.v),r=maximum(s.a,maximum(s.b,s.c))*s.gain;
        g.point=transform(s,{s.a*cu*sv,s.b*su*sv,s.c*cv});
        g.du=s.axis[0]*(-s.a*su*sv)+s.axis[1]*(s.b*cu*sv);
        g.dv=s.axis[0]*(s.a*cu*cv)+s.axis[1]*(s.b*su*cv)+s.axis[2]*(-s.c*sv);
        g.radius=r*(p.hu+p.hv);g.remainder=.5*r*(p.hu+p.hv)*(p.hu+p.hv);
    } else if(p.kind==0) {
        const double cv=cos(p.v),sv=sin(p.v),r=maximum(s.b,s.c)*s.gain;
        g.point=transform(s,{p.u,s.b*cv,s.c*sv});g.du=s.axis[0];
        g.dv=s.axis[1]*(-s.b*sv)+s.axis[2]*(s.c*cv);
        g.radius=s.gain*p.hu+r*p.hv;g.remainder=.5*r*p.hv*p.hv;
    } else {
        const double cv=cos(p.v),sv=sin(p.v),r=maximum(s.b,s.c)*s.gain;
        g.point=transform(s,{p.kind==1?-s.a:s.a,p.u*s.b*cv,p.u*s.c*sv});
        g.du=s.axis[1]*(s.b*cv)+s.axis[2]*(s.c*sv);
        g.dv=s.axis[1]*(-p.u*s.b*sv)+s.axis[2]*(p.u*s.c*cv);
        g.radius=r*(p.hu+p.hv);g.remainder=r*(p.hu*p.hv+.5*p.hv*p.hv);
    }
    return g;
}
CUDARL_HD inline double min_sincos(double cosine,double sine,double low,double high) {
    double result=minimum(cosine*cos(low)+sine*sin(low),cosine*cos(high)+sine*sin(high));
    const double critical=atan2(sine,cosine)+terrain::pi;
    const double candidate=critical+ceil((low-critical)/(2*terrain::pi))*(2*terrain::pi);
    if(candidate<=high)result=minimum(result,-hypot(cosine,sine));
    return result;
}
CUDARL_HD inline double minimum_linear(const Shape& s,const Patch& patch,V gradient) {
    if(s.box){const auto g=patch_geometry(s,patch);return dot(gradient,g.point)-fabs(dot(gradient,g.du))*patch.hu-fabs(dot(gradient,g.dv))*patch.hv;}
    const double x=dot(s.axis[0],gradient),y=dot(s.axis[1],gradient),z=dot(s.axis[2],gradient);
    if(s.ellipsoid) {
        // sin(polar)>=0, so azimuth minimizes independently before polar.
        const double azimuth=min_sincos(s.a*x,s.b*y,patch.u-patch.hu,patch.u+patch.hu);
        return dot(gradient,s.center)+min_sincos(s.c*z,azimuth,patch.v-patch.hv,patch.v+patch.hv);
    }
    const double radial=min_sincos(s.b*y,s.c*z,patch.v-patch.hv,patch.v+patch.hv);
    if(patch.kind==0)return dot(gradient,s.center)+x*(x>=0?patch.u-patch.hu:patch.u+patch.hu)+radial;
    return dot(gradient,s.center)+x*(patch.kind==1?-s.a:s.a)+radial*(radial>=0?patch.u-patch.hu:patch.u+patch.hu);
}
CUDARL_HD inline bool wet_ball(const terrain::Parameters& p,V center,double radius) {
    const double rho=hypot(center.x/p.water_radius_x_m,center.z/p.water_radius_z_m);
    const double dr=radius/minimum(p.water_radius_x_m,p.water_radius_z_m);
    double edge_min=.955;
    if(rho>dr)edge_min=maximum(edge_min,terrain::edge(atan2(center.z/p.water_radius_z_m,center.x/p.water_radius_x_m))-.199*asin(minimum(1.,dr/rho)));
    return rho+dr<=edge_min;
}
CUDARL_HD inline bool flat_box(const Shape& s,const terrain::Parameters& p) {
    const auto lo_x=support(s,{1,0,0}),hi_x=support(s,{-1,0,0});
    const auto lo_z=support(s,{0,0,1}),hi_z=support(s,{0,0,-1});
    const double x=maximum(fabs(lo_x.x),fabs(hi_x.x))/p.flat_radius_x_m;
    const double z=maximum(fabs(lo_z.z),fabs(hi_z.z))/p.flat_radius_z_m;
    return x*x+z*z<=1.;
}
CUDARL_HD inline double arithmetic_margin(const terrain::Parameters& p,V point) {
    const double A=p.water_radius_x_m/p.flat_radius_x_m,B=p.water_radius_z_m/p.flat_radius_z_m;
    const double dmin=.955-1/minimum(A,B);
    const double magnitude=maximum(1.,fabs(point.x)+fabs(point.y)+fabs(point.z)+fabs(p.floor_y_m)+maximum(p.water_radius_x_m,p.water_radius_z_m));
    return 2e-9*magnitude*maximum(1.,1/(dmin*dmin));
}
CUDARL_HD inline bool sample(const terrain::Parameters& p,V point,LinkInspection& out,Inspection& total,
                            uint32_t visit_limit,terrain::Query& q) {
    if(total.visits>=visit_limit){total.failure_reason=VisitLimit;return false;}
    ++total.visits;q=terrain::sample(p,point.x,point.z);
    if(!q.valid){total.failure_reason=InvalidQuery;return false;}
    if(q.region==terrain::DryLand||q.shoreline_coordinate>=1){total.failure_reason=DryLandScope;return false;}
    const double gap=point.y-q.height_m;
    // Certificate arithmetic uses the double point. Its conservative margin
    // scales with coordinates and annulus conditioning. Float contact-point
    // conversion is separate, measured and exposed, not hidden in the margin.
    const double roundoff=arithmetic_margin(p,point);
    if(gap+roundoff<out.upper_gap_m) {
        out.upper_gap_m=gap+roundoff;out.point_m={float(point.x),float(point.y),float(point.z)};
        out.exact_point_m={point.x,point.y,point.z};
        out.point_rounding_error_m=length(point-V{out.point_m.x,out.point_m.y,out.point_m.z});
        out.arithmetic_margin_m=roundoff;
        out.normal={float(q.normal.x),float(q.normal.y),float(q.normal.z)};
        out.gradient_x=-q.normal.x/q.normal.y;out.gradient_z=-q.normal.z/q.normal.y;
    }
    return true;
}
CUDARL_HD CUDARL_DEVICE_NOINLINE bool inspect_link(const Shape& s,const terrain::Parameters& p,Bounds bounds,double tolerance,
                                  uint32_t visit_limit,LinkInspection& out,Inspection& total) {
    if(flat_box(s,p)) {
        const auto point=support(s,{0,1,0});terrain::Query q;
        if(!sample(p,point,out,total,visit_limit,q))return false;
        out.lower_gap_m=point.y-p.floor_y_m-arithmetic_margin(p,point);
        out.certified=out.upper_gap_m-out.lower_gap_m<=tolerance;
        if(!out.certified)total.failure_reason=InvalidQuery;
        return out.certified;
    }
    // Good upper bounds reduce work; no finite search is used as a certificate.
    V n{0,1,0};
    for(int k=0;k<6;++k) {
        terrain::Query q;if(!sample(p,support(s,n),out,total,visit_limit,q))return false;
        n=gap_gradient(q);
    }
    double terminal_lower=1e30;
    // Each head patch covers an azimuth/polar rectangle. Cylinder patches cover
    // the side and both COMPLETE end discs, not only axial end rings.
    const int roots=s.box?6:s.ellipsoid?32:24;
    Patch root_patches[32];
    for(int root=0;root<roots;++root) {
        if(s.box) {
            const int fixed=root/2;
            root_patches[root]={0,0,fixed==0?s.b:s.a,fixed==2?s.b:s.c,root};
        }
        else if(s.ellipsoid)root_patches[root]={(root%8+.5)*terrain::pi/4,(root/8+.5)*terrain::pi/4,terrain::pi/8,terrain::pi/8,0};
        else if(root<8)root_patches[root]={0,(root+.5)*terrain::pi/4,s.a,terrain::pi/8,0};
        else root_patches[root]={.5,((root-8)%8+.5)*terrain::pi/4,.5,terrain::pi/8,root<16?1:2};
        terrain::Query q;
        if(!sample(p,patch_geometry(s,root_patches[root]).point,out,total,visit_limit,q))return false;
    }
    for(int root=0;root<roots;++root) {
        Patch stack[64];int size=1;stack[0]=root_patches[root];
        while(size>0) {
            const Patch patch=stack[--size];const auto g=patch_geometry(s,patch);terrain::Query q;
            if(!sample(p,g.point,out,total,visit_limit,q))return false;
            const double roundoff=arithmetic_margin(p,g.point);
            const double horizontal_radius=minimum(g.radius,hypot(g.du.x,g.du.z)*patch.hu+hypot(g.dv.x,g.dv.z)*patch.hv+g.remainder);
            const bool wet=wet_ball(p,g.point,horizontal_radius);
            double lower=-1e30;
            if(wet) {
                const V gradient=gap_gradient(q);
                // Exact minimum of the linear tangent potential over this
                // complete shape patch; Taylor bounds only the terrain term.
                const double linear=minimum_linear(s,patch,gradient)-dot(gradient,g.point);
                const double taylor=g.point.y-q.height_m+linear-.5*bounds.hessian*horizontal_radius*horizontal_radius-roundoff;
                const double lipschitz=g.point.y-q.height_m-sqrt(1+bounds.gradient*bounds.gradient)*g.radius-roundoff;
                lower=maximum(taylor,lipschitz);
            }
            if(wet&&lower>=out.upper_gap_m-tolerance) {terminal_lower=minimum(terminal_lower,lower);continue;}
            if(size+2>64){total.failure_reason=VisitLimit;return false;}
            bool split_u;
            if(s.box)split_u=length(g.du)*patch.hu>=length(g.dv)*patch.hv;
            else if(s.ellipsoid)split_u=patch.hu>=patch.hv;
            else if(patch.kind==0)split_u=s.gain*patch.hu>=maximum(s.b,s.c)*s.gain*patch.hv;
            else split_u=patch.hu>=patch.hv;
            Patch first=patch,second=patch;
            if(split_u){first.hu=second.hu=patch.hu*.5;first.u-=first.hu;second.u+=second.hu;}
            else {first.hv=second.hv=patch.hv*.5;first.v-=first.hv;second.v+=second.hv;}
            const V local_gradient=gap_gradient(q);
            const double direction=dot(local_gradient,split_u?g.du:g.dv);
            // Visit the locally downhill half first to tighten the upper
            // bound before certifying its siblings. This changes no bound.
            if(direction>=0){stack[size++]=second;stack[size++]=first;}
            else {stack[size++]=first;stack[size++]=second;}
        }
    }
    out.lower_gap_m=terminal_lower;out.certified=terrain::finite(terminal_lower)&&out.upper_gap_m-out.lower_gap_m<=tolerance;
    if(!out.certified){total.failure_reason=InvalidQuery;return false;}
    return true;
}
}
CUDARL_HD CUDARL_DEVICE_NOINLINE Inspection inspect(const p::BodyState& body,
        terrain::Parameters parameters=terrain::current_visual_parameters(),
        double tolerance_m=1e-5,uint32_t visit_limit=32768) {
    Inspection result;
    if(!p::finite_body(body)||!terrain::valid_parameters(parameters)||!terrain::finite(tolerance_m)||tolerance_m<1e-7||tolerance_m>.001||visit_limit==0) {
        result.failure_reason=InvalidInput;return result;
    }
    const auto links=movement::shape(body);detail::Shape shapes[3];
    for(int link=0;link<3;++link) {
        shapes[link]=detail::shape(links.link[link],link);
        const auto highest=detail::support(shapes[link],{0,-1,0});
        result.max_body_y_m=detail::maximum(result.max_body_y_m,highest.y);
    }
    auto membrane=shapes[2];membrane.box=true;membrane.c=movement::Geometry::fin_thickness*.5;
    result.max_body_y_m=detail::maximum(result.max_body_y_m,detail::support(membrane,{0,-1,0}).y);
    if(result.max_body_y_m>=0){result.failure_reason=SurfaceScope;return result;}
    const auto bounds=detail::wet_bounds(parameters);
    if(!terrain::finite(bounds.gradient)||!terrain::finite(bounds.hessian)){result.failure_reason=InvalidInput;return result;}
    result.valid=true;
    for(int link=0;link<3;++link) {
        if(!detail::inspect_link(shapes[link],parameters,bounds,tolerance_m,visit_limit,result.link[link],result))return result;
        if(link==2) {
            LinkInspection fin;
            if(!detail::inspect_link(membrane,parameters,bounds,tolerance_m,visit_limit,fin,result))return result;
            const double lower=detail::minimum(result.link[link].lower_gap_m,fin.lower_gap_m);
            if(fin.upper_gap_m<result.link[link].upper_gap_m)result.link[link]=fin;
            result.link[link].lower_gap_m=lower;
        }
        result.lower_gap_m=detail::minimum(result.lower_gap_m,result.link[link].lower_gap_m);
        result.upper_gap_m=detail::minimum(result.upper_gap_m,result.link[link].upper_gap_m);
    }
    result.certified=true;return result;
}
}
#endif
