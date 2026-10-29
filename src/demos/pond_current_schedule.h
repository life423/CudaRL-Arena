#ifndef CUDARL_POND_CURRENT_SCHEDULE_H
#define CUDARL_POND_CURRENT_SCHEDULE_H
#include "pond_playback.h"
namespace cudarl::demos {
class PondCurrentSchedule {
    deep::V initial_;
    double period_;
public:
    explicit PondCurrentSchedule(deep::V initial,double signed_period_seconds=0):initial_(initial),period_(signed_period_seconds) {
        validate_pond_current(initial);
        if(!std::isfinite(period_) || (period_!=0 && std::fabs(period_)<1))
            throw std::invalid_argument("Rotation period must be zero or at least one second in magnitude");
    }
    deep::V at_step(int completed_steps) const {
        if(completed_steps<0) throw std::invalid_argument("Current schedule needs nonnegative simulation steps");
        if(period_==0) return initial_;
        const double angle=6.2831853071795864769*completed_steps*static_cast<double>(deep::dt)/period_;
        const double c=std::cos(angle),s=std::sin(angle);
        return {static_cast<float>(initial_.x*c-initial_.z*s),0,static_cast<float>(initial_.x*s+initial_.z*c)};
    }
};
}
#endif
