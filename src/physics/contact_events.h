#ifndef CUDARL_CONTACT_EVENTS_H
#define CUDARL_CONTACT_EVENTS_H
#include <cstdint>
#include "contact_pilot.h"

// Opt-in fixed-law numerical correction. No production contact change,
// coefficient tuning, projection or behavioral controller.
namespace cudarl::contact_events {
using ScalarState=contact_pilot::ScalarState;
constexpr double kMaximumInternalStep=.0001;
constexpr int kReleaseBisections=24,kMaximumEvents=64;
constexpr double kSpring=double(contact_pilot::kStiffness),kDamper=double(contact_pilot::kDamping);
struct Counter {std::uint64_t used=0,limit=0;};
struct Stats {
    std::uint64_t trials=0;
    int outer_calls=0,accepted_segments=0,arrivals=0,releases=0,exits=0,reactivations=0,events=0,bisections=0;
    bool finite=true,budget_exhausted=false,event_limit=false,negative_force=false,completed=false;
    double elapsed=0,maxcompression=0,impulse=0,contact_work=0,external_work=0;
    double kinetic_before=0,kinetic_after=0,spring_before=0,spring_after=0,energy_balance=0,max_energy=0;
    double max_event_gap_error=0,max_release_force_error=0;
    ScalarState proposed{};
};
namespace detail {
CUDARL_HD inline bool finite(double x) {return x==x&&fabs(x)<1e300;}
CUDARL_HD inline double force(ScalarState s) {return -kSpring*s.gap-kDamper*s.velocity;}
CUDARL_HD inline double elastic(double gap) {const double x=fmin(0.,gap);return .5*kSpring*x*x;}
CUDARL_HD inline bool charge(Counter& counter,Stats& stats,bool trial=true) {
    if(counter.used>=counter.limit) {stats.budget_exhausted=true;return false;}
    ++counter.used;if(trial) ++stats.trials;else ++stats.outer_calls;return true;
}
CUDARL_HD inline bool valid(ScalarState s) {return finite(s.gap)&&finite(s.velocity);}
CUDARL_HD inline bool flight(ScalarState before,double mass,double load,double h,
    ScalarState& after,Stats& stats,Counter& counter) {
    if(!charge(counter,stats)) return false;
    const double acceleration=-load/mass;
    after={before.gap+h*(before.velocity+.5*h*acceleration),before.velocity+h*acceleration};
    if(!valid(after)) {stats.finite=false;return false;}return true;
}
CUDARL_HD inline bool contact(ScalarState before,double mass,double load,double h,
    ScalarState& after,Stats& stats,Counter& counter) {
    if(!charge(counter,stats)) return false;
    const double denominator=mass+.5*kDamper*h+.25*kSpring*h*h;
    const double numerator=(mass-.5*kDamper*h-.25*kSpring*h*h)*before.velocity-kSpring*h*before.gap-h*load;
    after.velocity=numerator/denominator;
    after.gap=before.gap+.5*h*(before.velocity+after.velocity);
    if(!valid(after)||!finite(denominator)||denominator<=0) {stats.finite=false;return false;}return true;
}
// First strictly positive polynomial root with the requested crossing direction.
// This is event algebra, not an uncounted physical advance. Every state trial
// made at the returned time is separately charged by flight()/contact().
CUDARL_HD inline double crossing(double A,double B,double C,double duration,int direction) {
    double candidates[2]={-1,-1};
    if(A==0) {if(B!=0)candidates[0]=-C/B;}
    else {
        const double discriminant=B*B-4*A*C;
        if(discriminant<0||!finite(discriminant))return -1;
        const double q=-.5*(B+copysign(sqrt(discriminant),B));
        if(q==0)candidates[0]=-B/(2*A);
        else {candidates[0]=q/A;candidates[1]=C/q;}
    }
    double first=-1;
    for(double t:candidates) if(finite(t)&&t>0&&t<=duration) {
        const double derivative=B+2*A*t;
        if((direction<0&&derivative>=0)||(direction>0&&derivative<=0))continue;
        if(first<0||t<first)first=t;
    }
    return first;
}
CUDARL_HD inline bool record(ScalarState before,ScalarState after,double mass,double load,double h,
    bool engaged,Stats& stats,double initial_energy) {
    double pulse=0;
    if(engaged) {
        const double midvelocity=.5*(before.velocity+after.velocity),midgap=.5*(before.gap+after.gap);
        const double normal=-kSpring*midgap-kDamper*midvelocity;
        if(!finite(normal)||normal<0) {stats.negative_force=true;stats.finite=finite(normal);return false;}
        pulse=h*normal;stats.contact_work+=pulse*midvelocity;
    }
    stats.impulse+=pulse;stats.external_work-=load*(after.gap-before.gap);
    stats.maxcompression=fmax(stats.maxcompression,fmax(0.,-after.gap));
    stats.kinetic_after=.5*mass*after.velocity*after.velocity;stats.spring_after=elastic(after.gap);
    const double energy=stats.kinetic_after+stats.spring_after+load*after.gap;
    stats.max_energy=fmax(stats.max_energy,energy);stats.energy_balance=energy-initial_energy;
    stats.proposed=after;stats.elapsed+=h;++stats.accepted_segments;
    if(!finite(stats.impulse)||!finite(stats.contact_work)||!finite(stats.external_work)||!finite(energy)) {stats.finite=false;return false;}
    return true;
}
}

// Each outer request and every trial advance/root candidate consumes the fixed
// shared counter. A rejected request preserves its input state; counters and
// diagnostic proposals remain available. No retry or automatic budget growth.
CUDARL_HD inline bool scalar_step(ScalarState& state,double mass,double inward_load,double outer_h,
    Stats& stats,Counter& counter) {
    stats={};stats.proposed=state;
    if(!detail::charge(counter,stats,false))return false;
    if(!detail::valid(state)||!detail::finite(mass)||mass<=0||!detail::finite(inward_load)||
       !detail::finite(outer_h)||outer_h<=0) {stats.finite=false;return false;}
    stats.kinetic_before=stats.kinetic_after=.5*mass*state.velocity*state.velocity;
    stats.spring_before=stats.spring_after=detail::elastic(state.gap);
    stats.maxcompression=fmax(0.,-state.gap);
    const double initial_energy=stats.kinetic_before+stats.spring_before+inward_load*state.gap;
    stats.max_energy=initial_energy;
    ScalarState current=state;bool arrival_boundary=false;
    const double time_roundoff=32*2.2204460492503131e-16*outer_h;
    while(outer_h-stats.elapsed>time_roundoff) {
        const double h=fmin(kMaximumInternalStep,outer_h-stats.elapsed);
        const double raw=detail::force(current),raw_derivative=-kSpring*current.velocity+kDamper*inward_load/mass;
        const bool engaged=arrival_boundary||(current.gap<=0&&(raw>0||(raw==0&&raw_derivative>0)));
        arrival_boundary=false;ScalarState proposal;
        if(engaged) {
            if(!detail::contact(current,mass,inward_load,h,proposal,stats,counter))return false;
            double accepted_h=h;bool release=false;
            if(raw>0&&detail::force(proposal)<=0) {
                double left=0,right=h;ScalarState right_state=proposal;
                for(int i=0;i<kReleaseBisections;++i) {
                    const double middle=.5*(left+right);ScalarState candidate;
                    if(!detail::contact(current,mass,inward_load,middle,candidate,stats,counter))return false;
                    ++stats.bisections;
                    if(detail::force(candidate)>0)left=middle;else {right=middle;right_state=candidate;}
                }
                proposal=right_state;accepted_h=right;release=true;
                stats.max_release_force_error=fmax(stats.max_release_force_error,fabs(detail::force(proposal)));
            }
            if(!detail::record(current,proposal,mass,inward_load,accepted_h,true,stats,initial_energy))return false;
            current=proposal;
            if(release) {++stats.releases;++stats.events;}
        } else {
            if(!detail::flight(current,mass,inward_load,h,proposal,stats,counter))return false;
            const double acceleration=-inward_load/mass;double event_h=-1;int event=0;
            if(current.gap>=0) {
                event_h=detail::crossing(.5*acceleration,current.velocity,current.gap,h,-1);if(event_h>0)event=1;
            } else {
                event_h=detail::crossing(.5*acceleration,current.velocity,current.gap,h,1);if(event_h>0)event=2;
                // A released, still-overlapping patch can re-engage under a
                // constant inward load. This remains the same clipped law.
                const double activate=detail::crossing(-.5*kSpring*acceleration,
                    -kSpring*current.velocity-kDamper*acceleration,raw,h,1);
                if(activate>0&&(event_h<0||activate<event_h)) {event_h=activate;event=3;}
            }
            double accepted_h=h;
            if(event_h>0) {
                accepted_h=event_h;
                if(event_h<h&&!detail::flight(current,mass,inward_load,event_h,proposal,stats,counter))return false;
            }
            if(!detail::record(current,proposal,mass,inward_load,accepted_h,false,stats,initial_energy))return false;
            current=proposal;
            if(event) {
                ++stats.events;
                if(event==1){++stats.arrivals;arrival_boundary=true;stats.max_event_gap_error=fmax(stats.max_event_gap_error,fabs(current.gap));}
                else if(event==2)++stats.exits;
                else {++stats.reactivations;arrival_boundary=true;}
            }
        }
        if(stats.events>kMaximumEvents) {stats.event_limit=true;return false;}
    }
    // Only accumulated floating-point clock roundoff is discarded here. This
    // is not a state projection, event displacement or omitted finite interval.
    stats.elapsed=outer_h;stats.proposed=current;stats.completed=true;state=current;return true;
}
// The following articulated path is a separate, opt-in fixed-fixture screen.
// The scalar implementation above is unchanged.
struct BodyCoordinate {
    double raw_residual=0,stored_residual=0,raw_allowance=0,stored_allowance=0;
    double raw_score=0,stored_score=0,velocity_storage_bound=0,impulse_storage_bound=0,arithmetic_bound=0;
};
struct BodyStats {
    physics::BodyState trial{};
    bool finite=true,completed=false,solver=true,momentum=true,geometry=true,energy=true;
    bool budget_exhausted=false,event_limit=false,negativeforce=false;
    std::uint64_t trials=0;
    int outer_calls=0,accepted_segments=0,events=0,bisections=0,arrivals=0,releases=0,reactivations=0,solver_iterations=0;
    double elapsed=0,min_gap=1,min_trial_gap=1,minforce=0,impulsefloor=0,impulse_total=0;
    double energy_start=0,energy_end=0,max_energy=0,max_energy_gain=0,frozen_energy_residual=0;
    double momentum_raw_score=0,momentum_stored_score=0,storage_bound=0;
    double raw_velocity[physics::dofs]{},free_velocity[physics::dofs]{},raw_impulse[contact_pilot::kRows]{};
    BodyCoordinate coordinate[physics::dofs]{};
};
namespace body_detail {
namespace p=physics;namespace a=contact_pilot;
constexpr int rows=contact_pilot::kRows;
struct Frame {
    p::BodyState before{};p::System system{};
    contact_response::Row row[rows]{};
    double force_start[rows]{};bool active[rows]{};
};
struct Trial {
    p::BodyState after{};
    double velocity[p::dofs]{},free[p::dofs]{},impulse[rows]{},gap[rows]{},force[rows]{};
};
CUDARL_HD inline bool charge(Counter& counter,BodyStats& stats,bool trial=true) {
    if(counter.used>=counter.limit){stats.budget_exhausted=true;return false;}
    ++counter.used;if(trial)++stats.trials;else ++stats.outer_calls;return true;
}
CUDARL_HD inline double dot(const float* J,const double* v) {
    double value=0;for(int k=0;k<p::dofs;++k)value+=double(J[k])*v[k];return value;
}
CUDARL_HD inline double half_ulp(double value) {
    const float converted=static_cast<float>(value);
    const double up=double(nextafterf(converted,3.4028234663852886e38f))-converted;
    const double down=double(converted)-nextafterf(converted,-3.4028234663852886e38f);
    return .5*fmax(fabs(up),fabs(down));
}
CUDARL_HD inline bool factor(const p::System& s,double h,double L[p::dofs][p::dofs]) {
    for(int i=0;i<p::dofs;++i)for(int j=0;j<=i;++j){
        double value=double(s.M[i][j])+.5*h*s.R[i][j];
        for(int k=0;k<j;++k)value-=L[i][k]*L[j][k];
        if(!detail::finite(value)||(i==j&&value<=0))return false;
        L[i][j]=i==j?sqrt(value):value/L[j][j];
    }return true;
}
CUDARL_HD inline bool solve(const double L[p::dofs][p::dofs],const double* rhs,double* x) {
    for(int i=0;i<p::dofs;++i){double value=rhs[i];for(int j=0;j<i;++j)value-=L[i][j]*x[j];x[i]=value/L[i][i];}
    for(int i=p::dofs-1;i>=0;--i){double value=x[i];for(int j=i+1;j<p::dofs;++j)value-=L[j][i]*x[j];x[i]=value/L[i][i];if(!detail::finite(x[i]))return false;}
    return true;
}
CUDARL_HD inline Frame frame(const p::BodyState& state,const float* action,p::Parameters fluid,p::V current,const bool* boundary) {
    Frame f;f.before=state;f.system=p::assemble(state,fluid,current);
    for(int j=0;j<4;++j){const float displacement=state.angle[j]-p::clamp(state.angle[j],-p::limit,p::limit);
        f.system.force[6+j]+=p::torque*p::clamp(action[j],-1,1)-1e-4f*displacement;
        f.system.R[6+j][6+j]+=p::torque/12.f+(displacement!=0?3e-6f:0.f);}
    const auto shape=p::shape(state);int index=0;double velocity[p::dofs]{};for(int k=0;k<p::dofs;++k)velocity[k]=state.velocity[k];
    for(int body=0;body<3;++body)for(int face=0;face<6;++face)for(int end=0;end<(body==0?1:2);++end){
        auto& row=f.row[index];row.body=body;row.face=face;row.end=end;const auto n=contact::normal(face);
        const auto& link=shape.link[body];const auto offset=contact::support_offset(link,body,n,end);
        row.gap=p::dot(n,link.center+offset)+contact::offset(face);
        for(int k=0;k<p::dofs;++k)row.jacobian[k]=p::dot(n,link.linear[k]+p::cross(link.angular[k],offset));
        f.force_start[index]=-kSpring*row.gap-kDamper*dot(row.jacobian,velocity);
        f.active[index]=boundary[index]||(row.gap<=0&&f.force_start[index]>0);++index;
    }return f;
}
CUDARL_HD inline p::BodyState pose(const p::BodyState& before,const double* v1,double h,const float* action) {
    auto out=before;double mid[p::dofs]{};for(int k=0;k<p::dofs;++k)mid[k]=.5*(double(before.velocity[k])+v1[k]);
    out.position.x=static_cast<float>(double(before.position.x)+h*mid[0]/p::scale);
    out.position.y=static_cast<float>(double(before.position.y)+h*mid[1]/p::scale);
    out.position.z=static_cast<float>(double(before.position.z)+h*mid[2]/p::scale);
    const double speed=sqrt(mid[3]*mid[3]+mid[4]*mid[4]+mid[5]*mid[5]);
    double dx=0,dy=0,dz=0,dw=1;if(speed>0){const double s=sin(.5*h*speed)/speed;dx=mid[3]*s;dy=mid[4]*s;dz=mid[5]*s;dw=cos(.5*h*speed);}
    const auto q=before.orientation;
    double x=dw*q.x+dx*q.w+dy*q.z-dz*q.y,y=dw*q.y-dx*q.z+dy*q.w+dz*q.x;
    double z=dw*q.z+dx*q.y-dy*q.x+dz*q.w,w=dw*q.w-dx*q.x-dy*q.y-dz*q.z;
    const double length=sqrt(x*x+y*y+z*z+w*w);out.orientation={static_cast<float>(x/length),static_cast<float>(y/length),static_cast<float>(z/length),static_cast<float>(w/length)};
    double work=before.work;for(int j=0;j<4;++j){out.angle[j]=static_cast<float>(double(before.angle[j])+h*mid[6+j]);work+=h*fmax(0.,double(p::torque*p::clamp(action[j],-1,1))*mid[6+j]);}
    out.work=static_cast<float>(work);out.travel=static_cast<float>(double(before.travel)+h*sqrt(mid[0]*mid[0]+mid[1]*mid[1]+mid[2]*mid[2])/p::scale);
    for(int k=0;k<p::dofs;++k)out.velocity[k]=static_cast<float>(v1[k]);return out;
}
// Whole-step certificate. The stored residual uses stored terminal velocity and
// stored impulses. Its exact conversion term is (M+hR/2)*delta_v-J^T*delta_p.
// No free-velocity or assembled-physics uncertainty is relabeled as storage.
CUDARL_HD inline bool certify(const Frame& f,const Trial& t,double h,BodyStats& stats) {
    constexpr double unit=0x1p-53;const double gamma=160*unit/(1-160*unit);
    for(int i=0;i<p::dofs;++i){
        BodyCoordinate c;double lhs=0,stored_lhs=0,load=h*(double(f.system.force[i])-f.system.C[i]);
        double stored_load=load,impulse=0,stored_impulse=0,scale=0,stored_scale=0,arithmetic_scale=fabs(load);
        for(int k=0;k<p::dofs;++k){const double old=f.before.velocity[k],raw=t.velocity[k],stored=t.after.velocity[k];
            const double term=double(f.system.M[i][k])*(raw-old),sterm=double(f.system.M[i][k])*(stored-old);
            lhs+=term;stored_lhs+=sterm;scale+=fabs(term);stored_scale+=fabs(sterm);
            load-=h*f.system.R[i][k]*.5*(old+raw);stored_load-=h*f.system.R[i][k]*.5*(old+stored);
            c.velocity_storage_bound+=fabs(double(f.system.M[i][k])+.5*h*f.system.R[i][k])*half_ulp(raw);
            arithmetic_scale+=(fabs(double(f.system.M[i][k]))+.5*h*fabs(double(f.system.R[i][k])))*(fabs(raw)+fabs(stored)+2*fabs(old));
        }
        for(int row=0;row<rows;++row){const double J=f.row[row].jacobian[i],raw=t.impulse[row],stored=static_cast<float>(raw);
            impulse+=J*raw;stored_impulse+=J*stored;c.impulse_storage_bound+=fabs(J)*half_ulp(raw);arithmetic_scale+=fabs(J)*(fabs(raw)+fabs(stored));}
        c.raw_residual=lhs-load-impulse;c.stored_residual=stored_lhs-stored_load-stored_impulse;
        c.arithmetic_bound=gamma*arithmetic_scale;
        c.raw_allowance=1e-13+2e-5*fmax(scale,fabs(load+impulse))+c.arithmetic_bound;
        c.stored_allowance=c.raw_allowance+c.velocity_storage_bound+c.impulse_storage_bound+c.arithmetic_bound;
        c.raw_score=fabs(c.raw_residual)/c.raw_allowance;c.stored_score=fabs(c.stored_residual)/c.stored_allowance;
        if(!detail::finite(c.raw_score)||!detail::finite(c.stored_score)){stats.finite=false;stats.momentum=false;return false;}
        stats.momentum_raw_score=fmax(stats.momentum_raw_score,c.raw_score);stats.momentum_stored_score=fmax(stats.momentum_stored_score,c.stored_score);
        stats.storage_bound=fmax(stats.storage_bound,c.velocity_storage_bound+c.impulse_storage_bound+c.arithmetic_bound);
        if(fmax(c.raw_score,c.stored_score)>=fmax(stats.coordinate[i].raw_score,stats.coordinate[i].stored_score))stats.coordinate[i]=c;
        if(c.raw_score>1||c.stored_score>1){stats.momentum=false;return false;}
    }return true;
}
CUDARL_HD inline bool integrate(const Frame& f,const float* action,double h,Trial& t,BodyStats& stats,Counter& counter) {
    if(!charge(counter,stats))return false;
    double L[p::dofs][p::dofs]{},rhs[p::dofs]{},response[rows][p::dofs]{},H[rows][rows]{},q[rows]{};
    if(!factor(f.system,h,L)){stats.solver=false;return false;}
    for(int i=0;i<p::dofs;++i){rhs[i]=h*(double(f.system.force[i])-f.system.C[i]);for(int k=0;k<p::dofs;++k)rhs[i]+=(double(f.system.M[i][k])-.5*h*f.system.R[i][k])*f.before.velocity[k];}
    if(!solve(L,rhs,t.free)){stats.solver=false;return false;}
    double speed=0;
    for(int i=0;i<rows;++i){double J[p::dofs]{};for(int k=0;k<p::dofs;++k)J[k]=f.row[i].jacobian[k];
        if(f.active[i]&&!solve(L,J,response[i])){stats.solver=false;return false;}
        speed=fmax(speed,fabs(dot(f.row[i].jacobian,t.free)));}
    const double b=.5*kDamper+.25*h*kSpring;
    for(int i=0;i<rows;++i){
        for(int j=i;j<rows;++j){double W=0;if(f.active[i]&&f.active[j])for(int k=0;k<p::dofs;++k)W+=double(f.row[i].jacobian[k])*response[j][k];H[i][j]=H[j][i]=W;}
        H[i][i]+=1./(h*b);q[i]=dot(f.row[i].jacobian,t.free)+kSpring*f.row[i].gap/b;
        for(int k=0;k<p::dofs;++k)q[i]+=double(f.row[i].jacobian[k])*f.before.velocity[k];
    }
    a::SolverStats solve_stats;const double tolerance=a::kResidualAbsolute+a::kResidualRelative*speed;
    if(!a::solve_contact(rows,H,q,f.active,a::Law::Compliant,t.impulse,solve_stats,tolerance)){stats.solver=false;return false;}
    stats.solver_iterations+=solve_stats.iterations;
    for(int k=0;k<p::dofs;++k){t.velocity[k]=t.free[k];for(int i=0;i<rows;++i)t.velocity[k]+=response[i][k]*t.impulse[i];stats.raw_velocity[k]=t.velocity[k];stats.free_velocity[k]=t.free[k];}
    for(int i=0;i<rows;++i){stats.raw_impulse[i]=t.impulse[i];stats.minforce=fmin(stats.minforce,t.impulse[i]/h);if(t.impulse[i]<0){stats.negativeforce=true;return false;}}
    t.after=pose(f.before,t.velocity,h,action);stats.trial=t.after;
    if(!p::finite_body(t.after)){stats.finite=false;return false;}
    const auto geometry=p::shape(t.after);double mid[p::dofs]{};
    for(int k=0;k<p::dofs;++k)mid[k]=.5*(double(f.before.velocity[k])+t.velocity[k]);
    double kinetic_delta=0,force_work=0,contact_work=0;
    for(int i=0;i<p::dofs;++i){double delta=0;for(int k=0;k<p::dofs;++k)delta+=double(f.system.M[i][k])*(t.velocity[k]-f.before.velocity[k]);kinetic_delta+=mid[i]*delta;
        force_work+=h*(double(f.system.force[i])-f.system.C[i])*mid[i];for(int k=0;k<p::dofs;++k)force_work-=h*mid[i]*f.system.R[i][k]*mid[k];}
    for(int i=0;i<rows;++i){const auto& row=f.row[i];const auto& link=geometry.link[row.body];const auto n=contact::normal(row.face),offset=contact::support_offset(link,row.body,n,row.end);
        t.gap[i]=p::dot(n,link.center+offset)+contact::offset(row.face);float J[p::dofs]{};double velocity=0;
        for(int k=0;k<p::dofs;++k){J[k]=p::dot(n,link.linear[k]+p::cross(link.angular[k],offset));velocity+=double(J[k])*t.after.velocity[k];}
        t.force[i]=-kSpring*t.gap[i]-kDamper*velocity;contact_work+=t.impulse[i]*dot(row.jacobian,mid);
        stats.min_trial_gap=fmin(stats.min_trial_gap,t.gap[i]);}
    stats.frozen_energy_residual=fmax(stats.frozen_energy_residual,fabs(kinetic_delta-force_work-contact_work));
    return certify(f,t,h,stats);
}
CUDARL_HD inline int event_kind(const Frame& f,const Trial& trial,int row) {
    if(!f.active[row]&&f.row[row].gap>0&&trial.gap[row]<=0)return 1;
    if(f.active[row]&&f.force_start[row]>0&&trial.force[row]<=0)return 2;
    if(!f.active[row]&&f.row[row].gap<=0&&f.force_start[row]<=0&&trial.gap[row]<=0&&trial.force[row]>0)return 3;
    return 0;
}
CUDARL_HD inline bool event(const Frame& f,const Trial& trial) {for(int i=0;i<rows;++i)if(event_kind(f,trial,i))return true;return false;}
CUDARL_HD inline double mechanical_energy(const p::BodyState& state,p::Parameters fluid) {
    double value=p::energy(state);const auto shape=p::shape(state);
    for(int body=0;body<3;++body){const auto& link=shape.link[body];
        value+=(link.mass-fluid.density*link.volume)*fluid.gravity*link.center.y;
        if(body==0)value-=fluid.density*link.volume*fluid.gravity*link.q.rotate({0,.0004f,0}).y;
        for(int face=0;face<6;++face)for(int end=0;end<(body==0?1:2);++end){const auto n=contact::normal(face);const double g=p::dot(n,link.center+contact::support_offset(link,body,n,end))+contact::offset(face);if(g<0)value+=.5*kSpring*g*g;}}
    for(float angle:state.angle){const double d=angle-p::clamp(angle,-p::limit,p::limit);value+=.5e-4*d*d;}return value;
}
}

// Bounded midpoint/event integration for the one preregistered rotating case.
// Geometry/forces are refreshed after every accepted interval; this is not a
// claim of a globally second-order method for arbitrary nonlinear scenes.
CUDARL_HD inline bool body_step(physics::BodyState& state,const float* action,double outer_h,
    physics::Parameters physical,physics::V current,BodyStats& stats,Counter& counter) {
    namespace p=physics;stats={};stats.trial=state;
    if(!body_detail::charge(counter,stats,false))return false;
    if(!p::finite_body(state)||!p::finite_vector(current)||!detail::finite(outer_h)||outer_h<=0||
       !p::finite_number(physical.density)||physical.density<0||!p::finite_number(physical.viscosity)||physical.viscosity<0||!p::finite_number(physical.gravity)||physical.gravity<0){stats.finite=false;return false;}
    bool no_actuation=true;for(int j=0;j<4;++j){if(!p::finite_number(action[j])){stats.finite=false;return false;}no_actuation=no_actuation&&action[j]==0;}
    const auto initial=contact::inspect(state);stats.min_gap=stats.min_trial_gap=initial.min_gap;
    if(initial.invalid||initial.excessive||initial.escape){stats.geometry=false;return false;}
    stats.energy_start=stats.energy_end=stats.max_energy=body_detail::mechanical_energy(state,physical);
    const bool conservative=current.x==0&&current.y==0&&current.z==0&&no_actuation;
    const double ceiling=stats.energy_start+fmax(1e-10,.001*fabs(stats.energy_start));
    p::BodyState current_state=state;bool boundary[contact_pilot::kRows]{};
    const double time_roundoff=32*2.2204460492503131e-16*outer_h;
    while(outer_h-stats.elapsed>time_roundoff){
        const double h=fmin(kMaximumInternalStep,outer_h-stats.elapsed);
        const auto frame=body_detail::frame(current_state,action,physical,current,boundary);
        for(bool& enabled:boundary)enabled=false;
        body_detail::Trial proposal;
        if(!body_detail::integrate(frame,action,h,proposal,stats,counter))return false;
        double accepted_h=h;bool is_event=body_detail::event(frame,proposal);
        if(is_event){double left=0,right=h;auto right_trial=proposal;
            for(int iteration=0;iteration<kReleaseBisections;++iteration){const double middle=.5*(left+right);body_detail::Trial candidate;
                if(!body_detail::integrate(frame,action,middle,candidate,stats,counter))return false;++stats.bisections;
                if(body_detail::event(frame,candidate)){right=middle;right_trial=candidate;}else left=middle;}
            accepted_h=right;proposal=right_trial;++stats.events;
            for(int i=0;i<contact_pilot::kRows;++i){const int kind=body_detail::event_kind(frame,proposal,i);if(kind==1){++stats.arrivals;boundary[i]=true;}else if(kind==2)++stats.releases;else if(kind==3){++stats.reactivations;boundary[i]=true;}}
        }
        stats.trial=proposal.after;const auto proof=contact::inspect(proposal.after);stats.min_gap=fmin(stats.min_gap,double(proof.min_gap));
        if(proof.invalid||proof.excessive||proof.escape){stats.geometry=false;return false;}
        const double energy=body_detail::mechanical_energy(proposal.after,physical);
        stats.energy_end=energy;stats.max_energy=fmax(stats.max_energy,energy);stats.max_energy_gain=fmax(stats.max_energy_gain,energy-stats.energy_start);
        if(!detail::finite(energy)){stats.finite=false;stats.energy=false;return false;}
        if(conservative&&energy>ceiling){stats.energy=false;return false;}
        for(int i=0;i<contact_pilot::kRows;++i){stats.impulse_total+=proposal.impulse[i];if(frame.row[i].face==2)stats.impulsefloor+=proposal.impulse[i];}
        ++stats.accepted_segments;stats.elapsed+=accepted_h;current_state=proposal.after;
        if(stats.events>kMaximumEvents){stats.event_limit=true;return false;}
    }
    stats.elapsed=outer_h;stats.trial=current_state;stats.completed=true;state=current_state;return true;
}
}
#endif
