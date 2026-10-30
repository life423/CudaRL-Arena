#ifndef CUDARL_CONTACT_EVENTS_H
#define CUDARL_CONTACT_EVENTS_H
#include <cstdint>
#include "contact_pilot.h"

// Scalar-only numerical correction. No articulated implementation, production
// contact change, coefficient tuning, projection or behavioral controller.
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
}
#endif
