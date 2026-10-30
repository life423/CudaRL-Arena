#ifndef CUDARL_TOOLS_CONTACT_PILOT_REFERENCE_H
#define CUDARL_TOOLS_CONTACT_PILOT_REFERENCE_H
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "physics/contact.h"

// CPU-double oracle only. It never advances a physical state. The independent
// path does NOT call native shape/assemble/support/J. Explicitly separated native
// comparisons below do call them. Matching the prescribed inertia model does not
// prove biological mass, fluid law, or Coriolis assumptions are physically exact.
namespace cudarl::contact_pilot_reference {
using D10=std::array<double,10>;using D25=std::array<double,25>;using D100=std::array<double,100>;
constexpr int kGeometryCap=256,kKineticCap=256,kSolveCap=64;
constexpr double kTranslationEps[2]={1e-5,5e-6},kAngularEps[2]={1e-4,5e-5};
constexpr double kGapAbs=2e-6,kTranslationJAbs=1e-5,kAngularJAbs=2e-7,kJRelative=1e-4;
constexpr double kMassScaledTolerance=1e-4,kEffectiveMassRelative=1e-4,kEnergyRelative=1e-4,kRankPivot=1e-12;
constexpr double kStep=.005,kAvailableEnergyLimit=25e-6,kCalibrationSpring=320,kMaximumNormalMass=.003;
namespace detail {
struct V {double x=0,y=0,z=0;V operator+(V b)const{return{x+b.x,y+b.y,z+b.z};}V operator-(V b)const{return{x-b.x,y-b.y,z-b.z};}V operator*(double a)const{return{x*a,y*a,z*a};}};
inline double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V cross(V a,V b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
struct Q {
    double x=0,y=0,z=0,w=1;
    Q operator*(Q b)const{return{w*b.x+x*b.w+y*b.z-z*b.y,w*b.y-x*b.z+y*b.w+z*b.x,w*b.z+x*b.y-y*b.x+z*b.w,w*b.w-x*b.x-y*b.y-z*b.z};}
    Q unit()const{const double n=std::sqrt(x*x+y*y+z*z+w*w);if(!std::isfinite(n)||n<=0)throw std::runtime_error("Invalid reference quaternion");return{x/n,y/n,z/n,w/n};}
    V rotate(V v)const{const V q{x,y,z};return v+cross(q,v)*(2*w)+cross(q,cross(q,v))*2;}
    V inverse(V v)const{return Q{-x,-y,-z,w}.rotate(v);}
};
inline Q rotation(int axis,double angle){const double a=std::sin(angle/2),b=std::cos(angle/2);return axis==0?Q{a,0,0,b}:(axis==1?Q{0,a,0,b}:Q{0,0,a,b});}
struct Pose {V position;Q orientation;std::array<double,4> angle{};};
struct Link {V center,anchor,yaw,pitch,inertia;Q orientation;double length=0,height=0,width=0,mass=0,volume=0;};
struct Geometry {Pose pose;std::array<Link,3> link;};
inline Pose pose(const physics::BodyState& b){
    Pose q;q.position={double(b.position.x)*.04,double(b.position.y)*.04,double(b.position.z)*.04};
    q.orientation=Q{b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w}.unit();for(int i=0;i<4;++i)q.angle[i]=b.angle[i];return q;
}
inline Geometry fk(Pose pose){
    Geometry g;g.pose=pose;constexpr double pi=3.1415926535897932384626433832795;
    for(int b=0;b<3;++b){
        auto& l=g.link[b];l.length=b==0?.020:.026;l.height=b==0?.0064:(b==1?.0015:.006);l.width=b==0?.008:.0015;
        if(b==0){l.center=pose.position;l.orientation=pose.orientation;}
        else{
            const auto& parent=g.link[b-1];const int j=2*(b-1);
            l.anchor=parent.center+parent.orientation.rotate({-parent.length/2,0,0});
            l.yaw=parent.orientation.rotate({0,1,0});const Q yaw=parent.orientation*rotation(1,pose.angle[j]);
            l.pitch=yaw.rotate({0,0,1});l.orientation=(yaw*rotation(2,pose.angle[j+1])).unit();
            l.center=l.anchor+l.orientation.rotate({-l.length/2,0,0});
        }
        if(b==0){
            const double a=.010,ry=.0064,rz=.008;l.volume=4*pi*a*ry*rz/3;l.mass=1000*l.volume;
            l.inertia={l.mass*(ry*ry+rz*rz)/5,l.mass*(a*a+rz*rz)/5,l.mass*(a*a+ry*ry)/5};
        }else{
            constexpr double r=.0015,membrane=.0097;
            l.volume=pi*r*r*l.length+(b==2?membrane*l.length*.0001:0);l.mass=1000*l.volume;
            const double axial=l.mass*(r*r+(b==2?membrane*membrane/12:0))*.5;
            const double transverse=l.mass*(l.length*l.length+3*r*r)/12;l.inertia={axial,transverse,transverse};
        }
    }return g;
}
inline V floor_offset(const Link& l,int body,int end){
    const V n=l.orientation.inverse({0,1,0});V r;
    if(body==0){
        const double a=l.length/2,b=l.height,c=l.width,d=std::sqrt(a*a*n.x*n.x+b*b*n.y*n.y+c*c*n.z*n.z);
        if(d<=0)throw std::runtime_error("Degenerate head support");r={-a*a*n.x/d,-b*b*n.y/d,-c*c*n.z/d};
    }else{
        const double d=std::sqrt(l.height*l.height*n.y*n.y+l.width*l.width*n.z*n.z);
        r.x=(end==0?-.5:.5)*l.length;
        if(d>0){r.y=-l.height*l.height*n.y/d;r.z=-l.width*l.width*n.z/d;}
    }return l.orientation.rotate(r);
}
constexpr int bodies[5]={0,1,1,2,2},ends[5]={0,0,1,0,1};
inline double gap(const Geometry& g,int row){const int b=bodies[row];return g.link[b].center.y+floor_offset(g.link[b],b,ends[row]).y+.16;}
inline double kinetic(const Geometry& g,const D10& v){
    std::array<V,3> linear{},angular{};linear[0]={v[0],v[1],v[2]};angular[0]={v[3],v[4],v[5]};
    double energy=0;
    for(int b=0;b<3;++b){
        const auto& l=g.link[b];
        if(b>0){const int j=2*(b-1);angular[b]=angular[b-1]+l.yaw*v[6+j]+l.pitch*v[7+j];
            const V at_anchor=linear[b-1]+cross(angular[b-1],l.anchor-g.link[b-1].center);
            linear[b]=at_anchor+cross(angular[b],l.center-l.anchor);}
        const V w=l.orientation.inverse(angular[b]);
        energy+=.5*(l.mass*dot(linear[b],linear[b])+l.inertia.x*w.x*w.x+l.inertia.y*w.y*w.y+l.inertia.z*w.z*w.z);
    }return energy;
}
inline bool same_pose(const physics::BodyState&a,const physics::BodyState&b){
    if(a.position.x!=b.position.x||a.position.y!=b.position.y||a.position.z!=b.position.z||
       a.orientation.x!=b.orientation.x||a.orientation.y!=b.orientation.y||a.orientation.z!=b.orientation.z||a.orientation.w!=b.orientation.w)return false;
    for(int i=0;i<4;++i)if(a.angle[i]!=b.angle[i])return false;return true;
}
inline double relative(double a,double b){return std::fabs(a-b)/std::fmax(std::fabs(a),1e-30);}
inline void charge(int& count,int cap){if(count>=cap)throw std::runtime_error("Static reference query cap exceeded");++count;}
struct Factor {D100 L{};D10 scale{};double minimum=1;};
inline Factor factor(const D100& m){
    Factor f;
    for(int i=0;i<10;++i){if(!std::isfinite(m[i*10+i])||m[i*10+i]<=0)throw std::runtime_error("Nonpositive reference mass diagonal");f.scale[i]=std::sqrt(m[i*10+i]);}
    for(int i=0;i<10;++i)for(int j=0;j<=i;++j){
        double a=m[i*10+j]/(f.scale[i]*f.scale[j]);for(int k=0;k<j;++k)a-=f.L[i*10+k]*f.L[j*10+k];
        if(!std::isfinite(a)||(i==j&&a<=kRankPivot))throw std::runtime_error("Reference mass rank/finite check failed");
        if(i==j)f.minimum=std::fmin(f.minimum,a);f.L[i*10+j]=i==j?std::sqrt(a):a/f.L[j*10+j];
    }return f;
}
inline D10 solve(const Factor&f,const D10&rhs){
    D10 x{};for(int i=0;i<10;++i){double a=rhs[i]/f.scale[i];for(int j=0;j<i;++j)a-=f.L[i*10+j]*x[j];x[i]=a/f.L[i*10+i];}
    for(int i=9;i>=0;--i){double a=x[i];for(int j=i+1;j<10;++j)a-=f.L[j*10+i]*x[j];x[i]=a/f.L[i*10+i];}
    for(int i=0;i<10;++i){x[i]/=f.scale[i];if(!std::isfinite(x[i]))throw std::runtime_error("Nonfinite mass inverse query");}return x;
}
inline double dot10(const D10&a,const D10&b){double x=0;for(int i=0;i<10;++i)x+=a[i]*b[i];return x;}
}
struct EnergyBound {
    double kinetic=0,softstop=0,contact_spring=0,potential=0,lower_potential=0,available=0;
    bool fully_submerged=false,safe_floor=false,calm=false,applicable=false,passed=false;
};
struct FloorRowReport {
    int body=0,end=0;D10 jacobian_fd_coarse{},jacobian_fd{},jacobian_native{};
    double gap_reference=0,gap_native=0,gap_abs_error=0,jacobian_scaled_error=0,fd_scaled_error=0;
    double mass_M_reference=0,mass_M_native=0,mass_A_hybrid=0,mass_A_native=0,mass_M_relative_error=0,mass_A_relative_error=0;
    double normal_velocity_reference=0,normal_velocity_native_double=0,normal_velocity_native_float=0;
};
struct StateReport {
    int id=0;std::string name;bool passed=false;physics::BodyState source{};
    std::array<FloorRowReport,5> floor_rows{};
    D100 mass_reference{},mass_native{};D25 W_M_reference{},W_M_native{},W_A_hybrid{},W_A_native{};
    double mass_scaled_error=0,coupling_M_scaled_error=0,coupling_A_scaled_error=0;
    double energy_reference=0,energy_native=0,energy_relative_error=0,min_normalized_pivot=1,max_normal_mass=0;
    EnergyBound energy_bound{};detail::Geometry cached_geometry{};
};
struct Report {bool passed=false;std::string error;int geometry_queries=0,kinetic_evaluations=0,matrix_solves=0,matrix_factorizations=0;std::vector<StateReport> cases;};
inline EnergyBound bound_from(const StateReport&s,double density,double gravity,double energy,physics::V current={}){
    EnergyBound b;b.kinetic=energy;b.fully_submerged=true;b.safe_floor=true;
    b.calm=current.x==0&&current.y==0&&current.z==0;
    for(int i=0;i<4;++i){const double d=s.cached_geometry.pose.angle[i]-std::fmax(-1.,std::fmin(1.,s.cached_geometry.pose.angle[i]));b.softstop+=.5*1e-4*d*d;}
    for(int r=0;r<5;++r){const double gap=s.floor_rows[r].gap_reference;b.safe_floor=b.safe_floor&&gap>=-.0005;if(gap<0)b.contact_spring+=.5*kCalibrationSpring*gap*gap;}
    for(int i=0;i<3;++i){
        const auto& l=s.cached_geometry.link[i];const double weight=(l.mass-density*l.volume)*gravity;
        double lower_offset=detail::floor_offset(l,i,0).y;
        if(i>0)lower_offset=std::fmin(lower_offset,detail::floor_offset(l,i,1).y);
        b.fully_submerged=b.fully_submerged&&l.center.y-lower_offset<0;
        b.potential+=weight*l.center.y;b.lower_potential+=weight*(-.16+(i==0?.0064:.0015)-.0005);
        if(i==0){b.potential-=density*l.volume*gravity*l.orientation.rotate({0,.0004,0}).y;b.lower_potential-=density*l.volume*gravity*.0004;}
    }
    b.available=b.kinetic+b.softstop+b.contact_spring+std::fmax(0.,b.potential-b.lower_potential);
    b.applicable=std::isfinite(density)&&density>=0&&density<=1000&&std::isfinite(gravity)&&gravity>=0&&b.fully_submerged&&b.safe_floor&&b.calm;
    b.passed=b.applicable&&std::isfinite(b.available)&&b.available<=kAvailableEnergyLimit;
    return b;
}
inline EnergyBound fresh_energy_bound(Report&report,size_t index,const physics::BodyState&state,physics::V current={}){
    if(index>=report.cases.size()||!detail::same_pose(report.cases[index].source,state))throw std::runtime_error("Fresh energy query must reuse the exact cached pose");
    detail::charge(report.kinetic_evaluations,kKineticCap);D10 v{};for(int i=0;i<10;++i)v[i]=state.velocity[i];
    const double energy=detail::kinetic(report.cases[index].cached_geometry,v);
    return bound_from(report.cases[index],index==0?990:1000,9.8100004196167,energy,current);
}
inline Report reference_report(const std::vector<physics::BodyState>&states){
    Report report;
    try{
        if(states.size()!=3)throw std::runtime_error("Exactly three preregistered saved prestates required");
        for(int index=0;index<3;++index){
            StateReport s;s.id=index;s.source=states[static_cast<size_t>(index)];
            s.name=index==0?"rolled_legacy_CPU_post33":(index==1?"parity_legacy_CPU_post76":"parity_legacy_CUDA_post76");
            detail::charge(report.geometry_queries,kGeometryCap);const auto pose=detail::pose(s.source);s.cached_geometry=detail::fk(pose);
            for(int r=0;r<5;++r){s.floor_rows[r].body=detail::bodies[r];s.floor_rows[r].end=detail::ends[r];s.floor_rows[r].gap_reference=detail::gap(s.cached_geometry,r);}
            for(int column=0;column<10;++column)for(int level=0;level<2;++level){
                const double eps=column<3?kTranslationEps[level]:kAngularEps[level];detail::Pose minus=pose,plus=pose;
                if(column<3){double* a=column==0?&minus.position.x:(column==1?&minus.position.y:&minus.position.z);double* b=column==0?&plus.position.x:(column==1?&plus.position.y:&plus.position.z);*a-=eps;*b+=eps;}
                else if(column<6){minus.orientation=(detail::rotation(column-3,-eps)*pose.orientation).unit();plus.orientation=(detail::rotation(column-3,eps)*pose.orientation).unit();}
                else{minus.angle[column-6]-=eps;plus.angle[column-6]+=eps;}
                detail::charge(report.geometry_queries,kGeometryCap);const auto gm=detail::fk(minus);
                detail::charge(report.geometry_queries,kGeometryCap);const auto gp=detail::fk(plus);
                for(int r=0;r<5;++r){const double derivative=(detail::gap(gp,r)-detail::gap(gm,r))/(2*eps);(level==0?s.floor_rows[r].jacobian_fd_coarse:s.floor_rows[r].jacobian_fd)[column]=derivative;}
            }
            D10 diagonal_energy{};
            for(int i=0;i<10;++i){D10 v{};v[i]=1;detail::charge(report.kinetic_evaluations,kKineticCap);diagonal_energy[i]=detail::kinetic(s.cached_geometry,v);s.mass_reference[i*10+i]=2*diagonal_energy[i];}
            for(int i=0;i<10;++i)for(int j=i+1;j<10;++j){D10 v{};v[i]=v[j]=1;detail::charge(report.kinetic_evaluations,kKineticCap);const double e=detail::kinetic(s.cached_geometry,v)-diagonal_energy[i]-diagonal_energy[j];s.mass_reference[i*10+j]=s.mass_reference[j*10+i]=e;}
            D10 velocity{};for(int i=0;i<10;++i)velocity[i]=s.source.velocity[i];
            detail::charge(report.kinetic_evaluations,kKineticCap);s.energy_reference=detail::kinetic(s.cached_geometry,velocity);
            // Native COMPARISON only: none of these values feeds oracle FK/J/M.
            detail::charge(report.geometry_queries,kGeometryCap);const auto native=physics::shape(s.source);
            const physics::Parameters fluid{index==0?990.f:1000.f,.001f,9.81f};const physics::V current=index==0?physics::V{}:physics::V{0,0,.005f};
            detail::charge(report.geometry_queries,kGeometryCap);auto system=physics::assemble(s.source,fluid,current);
            for(int i=0;i<4;++i){const float displacement=s.source.angle[i]-physics::clamp(s.source.angle[i],-physics::limit,physics::limit);system.R[6+i][6+i]+=physics::torque/12.f+(displacement!=0?3e-6f:0.f);}
            D100 hybrid_A{},native_A{};
            for(int i=0;i<10;++i)for(int j=0;j<10;++j){
                s.mass_native[i*10+j]=system.M[i][j];hybrid_A[i*10+j]=s.mass_reference[i*10+j]+kStep*system.R[i][j];native_A[i*10+j]=double(system.M[i][j])+kStep*system.R[i][j];
                s.mass_scaled_error=std::fmax(s.mass_scaled_error,std::fabs(s.mass_reference[i*10+j]-system.M[i][j])/std::sqrt(s.mass_reference[i*10+i]*s.mass_reference[j*10+j]));
                s.energy_native+=.5*velocity[i]*system.M[i][j]*velocity[j];
            }
            s.energy_relative_error=detail::relative(s.energy_reference,s.energy_native);
            for(int r=0;r<5;++r){
                auto& row=s.floor_rows[r];const auto& l=native.link[row.body];const auto n=contact::normal(2),offset=contact::support_offset(l,row.body,n,row.end);
                row.gap_native=physics::dot(n,l.center+offset)+contact::offset(2);row.gap_abs_error=std::fabs(row.gap_reference-row.gap_native);
                physics::V J[10];for(int i=0;i<10;++i){J[i]=l.linear[i]+physics::cross(l.angular[i],offset);row.jacobian_native[i]=physics::dot(n,J[i]);}
                row.normal_velocity_native_float=physics::dot(n,physics::multiply(J,s.source.velocity));
                for(int i=0;i<10;++i){
                    const double tol=(i<3?kTranslationJAbs:kAngularJAbs)+kJRelative*std::fabs(row.jacobian_fd[i]);
                    row.jacobian_scaled_error=std::fmax(row.jacobian_scaled_error,std::fabs(row.jacobian_native[i]-row.jacobian_fd[i])/tol);
                    row.fd_scaled_error=std::fmax(row.fd_scaled_error,std::fabs(row.jacobian_fd_coarse[i]-row.jacobian_fd[i])/tol);
                    row.normal_velocity_reference+=row.jacobian_fd[i]*velocity[i];row.normal_velocity_native_double+=row.jacobian_native[i]*velocity[i];
                }
            }
            const D100 matrices[4]={s.mass_reference,s.mass_native,hybrid_A,native_A};D25* coupling[4]={&s.W_M_reference,&s.W_M_native,&s.W_A_hybrid,&s.W_A_native};
            for(int m=0;m<4;++m){
                const auto factor=detail::factor(matrices[m]);++report.matrix_factorizations;s.min_normalized_pivot=std::fmin(s.min_normalized_pivot,factor.minimum);
                for(int r=0;r<5;++r){const auto& jacobian=m%2==0?s.floor_rows[r].jacobian_fd:s.floor_rows[r].jacobian_native;detail::charge(report.matrix_solves,kSolveCap);const auto response=detail::solve(factor,jacobian);
                    for(int q=0;q<5;++q)(*coupling[m])[q*5+r]=detail::dot10(m%2==0?s.floor_rows[q].jacobian_fd:s.floor_rows[q].jacobian_native,response);}
            }
            s.passed=s.mass_scaled_error<=kMassScaledTolerance&&s.energy_relative_error<=kEnergyRelative;
            for(int r=0;r<5;++r){
                auto& row=s.floor_rows[r];const double values[4]={s.W_M_reference[r*5+r],s.W_M_native[r*5+r],s.W_A_hybrid[r*5+r],s.W_A_native[r*5+r]};
                for(double value:values)if(!std::isfinite(value)||value<=0)throw std::runtime_error("Nonpositive normal inverse mass");
                row.mass_M_reference=1/values[0];row.mass_M_native=1/values[1];row.mass_A_hybrid=1/values[2];row.mass_A_native=1/values[3];
                row.mass_M_relative_error=detail::relative(row.mass_M_reference,row.mass_M_native);row.mass_A_relative_error=detail::relative(row.mass_A_hybrid,row.mass_A_native);
                s.max_normal_mass=std::fmax(s.max_normal_mass,row.mass_M_reference);
                s.passed=s.passed&&row.gap_abs_error<=kGapAbs&&row.jacobian_scaled_error<=1&&row.fd_scaled_error<=1&&
                    row.mass_M_relative_error<=kEffectiveMassRelative&&row.mass_A_relative_error<=kEffectiveMassRelative;
            }
            for(int i=0;i<5;++i)for(int j=0;j<5;++j){
                s.coupling_M_scaled_error=std::fmax(s.coupling_M_scaled_error,std::fabs(s.W_M_reference[i*5+j]-s.W_M_native[i*5+j])/
                    std::sqrt(s.W_M_reference[i*5+i]*s.W_M_reference[j*5+j]));
                s.coupling_A_scaled_error=std::fmax(s.coupling_A_scaled_error,std::fabs(s.W_A_hybrid[i*5+j]-s.W_A_native[i*5+j])/
                    std::sqrt(s.W_A_hybrid[i*5+i]*s.W_A_hybrid[j*5+j]));
            }
            s.passed=s.passed&&s.coupling_M_scaled_error<=kMassScaledTolerance&&s.coupling_A_scaled_error<=kMassScaledTolerance;
            s.energy_bound=bound_from(s,fluid.density,fluid.gravity,s.energy_reference,current);
            report.cases.push_back(s);
            if(!s.passed){report.error="Independent reference gate failed at "+s.name;return report;}
        }
        report.passed=true;
    }catch(const std::exception&e){report.error=e.what();}
    return report;
}
inline std::string to_json(const Report&report){
    std::ostringstream out;out<<std::setprecision(17);
    auto number=[&](double x){if(std::isfinite(x))out<<x;else out<<"null";};
    auto array=[&](const auto&a){out<<'[';bool first=true;for(double v:a){if(!first)out<<',';first=false;number(v);}out<<']';};
    out<<"{\"passed\":"<<(report.passed?"true":"false")<<",\"error\":\"";for(char c:report.error){if(c=='"'||c=='\\')out<<'\\';out<<c;}
    out<<"\",\"geometry_queries\":"<<report.geometry_queries<<",\"kinetic_evaluations\":"<<report.kinetic_evaluations<<",\"matrix_solves\":"<<report.matrix_solves
       <<",\"matrix_factorizations\":"<<report.matrix_factorizations<<",\"caps\":[256,256,64]"
       <<",\"settings\":{\"translation_eps_m\":[1e-5,5e-6],\"angular_eps_rad\":[1e-4,5e-5],\"gap_abs_m\":"<<kGapAbs
       <<",\"translation_J_abs\":"<<kTranslationJAbs<<",\"angular_J_abs_m\":"<<kAngularJAbs<<",\"J_relative\":"<<kJRelative
       <<",\"M_W_scaled_error\":"<<kMassScaledTolerance<<",\"meff_relative\":"<<kEffectiveMassRelative<<",\"energy_relative\":"<<kEnergyRelative
       <<",\"normalized_rank_pivot\":"<<kRankPivot<<",\"hybrid_A_step_s\":"<<kStep<<",\"available_energy_J\":"<<kAvailableEnergyLimit
       <<",\"calibrated_spring_N_m_per_patch\":"<<kCalibrationSpring<<",\"normal_mass_bound_kg\":"<<kMaximumNormalMass<<"}"
       <<",\"reference_basis\":\"independent double FK, finite-difference support J, recursive link kinetic energy and55-query mass polarization; native R is reused ONLY for explicitly hybrid A\",\"cases\":[";
    for(size_t i=0;i<report.cases.size();++i){
        if(i)out<<',';const auto&s=report.cases[i];out<<"{\"id\":"<<s.id<<",\"name\":\""<<s.name<<"\",\"passed\":"<<(s.passed?"true":"false")<<",\"mass_scaled_error\":";number(s.mass_scaled_error);
        out<<",\"energy_reference\":";number(s.energy_reference);out<<",\"energy_native\":";number(s.energy_native);out<<",\"energy_relative_error\":";number(s.energy_relative_error);
        out<<",\"min_normalized_pivot\":";number(s.min_normalized_pivot);out<<",\"max_normal_mass\":";number(s.max_normal_mass);
        out<<",\"coupling_M_scaled_error\":";number(s.coupling_M_scaled_error);out<<",\"coupling_A_scaled_error\":";number(s.coupling_A_scaled_error);
        out<<",\"available_energy\":";number(s.energy_bound.available);out<<",\"energy_bound_passed\":"<<(s.energy_bound.passed?"true":"false")
           <<",\"energy_bound_applicable\":"<<(s.energy_bound.applicable?"true":"false")<<",\"energy_bound_calm\":"<<(s.energy_bound.calm?"true":"false")
           <<",\"energy_bound_fully_submerged\":"<<(s.energy_bound.fully_submerged?"true":"false")<<",\"energy_bound_safe_floor\":"<<(s.energy_bound.safe_floor?"true":"false")
           <<",\"energy_T\":";number(s.energy_bound.kinetic);out<<",\"energy_joint_softstop\":";number(s.energy_bound.softstop);out<<",\"energy_A_springs\":";number(s.energy_bound.contact_spring);
        out<<",\"potential\":";number(s.energy_bound.potential);out<<",\"safe_floor_lower_potential\":";number(s.energy_bound.lower_potential);
        out<<",\"mass_reference\":";array(s.mass_reference);out<<",\"mass_native\":";array(s.mass_native);
        out<<",\"W_M_reference\":";array(s.W_M_reference);out<<",\"W_M_native\":";array(s.W_M_native);out<<",\"W_A_hybrid\":";array(s.W_A_hybrid);out<<",\"W_A_native\":";array(s.W_A_native);out<<",\"floor_rows\":[";
        for(int r=0;r<5;++r){if(r)out<<',';const auto&f=s.floor_rows[r];out<<"{\"body\":"<<f.body<<",\"end\":"<<f.end<<",\"gap_reference\":";number(f.gap_reference);out<<",\"gap_native\":";number(f.gap_native);
            out<<",\"gap_abs_error\":";number(f.gap_abs_error);out<<",\"jacobian_scaled_error\":";number(f.jacobian_scaled_error);out<<",\"fd_scaled_error\":";number(f.fd_scaled_error);
            out<<",\"jacobian_fd_coarse\":";array(f.jacobian_fd_coarse);out<<",\"jacobian_fd\":";array(f.jacobian_fd);out<<",\"jacobian_native\":";array(f.jacobian_native);
            out<<",\"mass_M_reference\":";number(f.mass_M_reference);out<<",\"mass_M_native\":";number(f.mass_M_native);out<<",\"mass_A_hybrid\":";number(f.mass_A_hybrid);out<<",\"mass_A_native\":";number(f.mass_A_native);
            out<<",\"mass_M_relative_error\":";number(f.mass_M_relative_error);out<<",\"mass_A_relative_error\":";number(f.mass_A_relative_error);
            out<<",\"normal_velocity_reference\":";number(f.normal_velocity_reference);out<<",\"normal_velocity_native_double\":";number(f.normal_velocity_native_double);out<<",\"normal_velocity_native_float\":";number(f.normal_velocity_native_float);out<<'}';}
        out<<"]}";
    }out<<"]}";return out.str();
}
}
#endif

