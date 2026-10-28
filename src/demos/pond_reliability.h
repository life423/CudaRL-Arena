#ifndef CUDARL_POND_RELIABILITY_H
#define CUDARL_POND_RELIABILITY_H
#include "pond_playback.h"
#include <algorithm>
#include <array>
#include <vector>
namespace cudarl::demos {
struct ImmobilityEvent {int start=0,end=0;};
// Observes the creature; never changes its actions, food or physical state.
class PondReliability {
    static constexpr int window=60; // three seconds at the shared 50-ms step
    std::array<deep::V,window+1> positions_{};
    int samples_=0,last_tick_=-1,start_=-1,confirmed_=0;bool active_confirmed_=false;
    void close(int end) {if(active_confirmed_) events.push_back({start_,end});start_=-1;active_confirmed_=false;}
public:
    int invalid_states=0,invalid_foods=0,nonsequential_steps=0,max_food_gap=0;
    float max_immobility_seconds=0;
    std::vector<ImmobilityEvent> events;
    bool observe(const PondFrame& frame) {
        const auto& s=frame.state;const int tick=s.steps;
        if(last_tick_>=0 && tick!=last_tick_+1) ++nonsequential_steps;last_tick_=tick;
        max_food_gap=std::max(max_food_gap,s.since_food);
        bool finite=std::isfinite(s.body.position.x) && std::isfinite(s.body.position.y) && std::isfinite(s.body.position.z) &&
            std::isfinite(s.body.orientation.x) && std::isfinite(s.body.orientation.y) && std::isfinite(s.body.orientation.z) && std::isfinite(s.body.orientation.w);
        for(float v:s.body.velocity) finite=finite && std::isfinite(v);
        for(float a:s.body.angle) finite=finite && std::isfinite(a);
        for(auto food:s.food) if(!deep_pond::water(food)) ++invalid_foods;
        if(!finite) {++invalid_states;return false;}
        positions_[static_cast<size_t>(samples_)%positions_.size()]=s.body.position;++samples_;
        bool still=samples_>=window+1;
        if(still) for(auto position:positions_) if(deep::norm(position-s.body.position)>.1f) {still=false;break;}
        if(still) {
            if(start_<0) start_=tick-window;
            const float seconds=(tick-start_)*deep::dt;max_immobility_seconds=std::max(max_immobility_seconds,seconds);
            if(!active_confirmed_ && seconds>=5.f) {active_confirmed_=true;++confirmed_;return true;}
        } else if(start_>=0) close(tick);
        return false;
    }
    void finish() {if(start_>=0) close(last_tick_);}
    int confirmed_events() const {return confirmed_;}
    bool passes() const {return confirmed_==0 && invalid_states==0 && invalid_foods==0 && nonsequential_steps==0;}
};
}
#endif
