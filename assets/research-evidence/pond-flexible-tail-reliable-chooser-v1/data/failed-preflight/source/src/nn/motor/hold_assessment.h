#ifndef CUDARL_HOLD_ASSESSMENT_H
#define CUDARL_HOLD_ASSESSMENT_H

#include <cmath>

#include "motor/straight_task.h"

namespace cudarl::straight_motor {
// ForwardTask still owns bookkeeping and its historical swimming outcome.
// A hold consumer can end at the same control without taking swim success as
// a hold verdict. The full RestHoldAssessment is the only hold success gate.
CUDARL_HD inline Transition record_hold_control(ForwardTask& task, const World& world,
                                               const ForwardConfig& config) {
    auto transition = record_control(task, world, config);
    if (config.position_hold && config.stationary_request && transition.outcome == Outcome::Success) {
        transition.outcome = Outcome::Timeout;
        task.outcome = Outcome::Timeout;
    }
    return transition;
}

// Frozen 64-second, still-water rest-hold assessment. Supply measured control
// boundaries in order, including the reset sample, at the existing 50 ms cadence.
// This never drives physics; ForwardTask's terminal outcome remains swim-only.
class RestHoldAssessment {
public:
    static constexpr int controls = 1280;
    static constexpr int final_window_start = 1120;
    static constexpr float max_speed_m_s = .0005f;
    static constexpr float max_anchor_m = .002f;
    static constexpr float max_depth_m = .001f;
    static constexpr float max_lane_m = .002f;
    static constexpr float max_tilt_deg = 10.f;
    static constexpr float max_excursion_m = .002f;

    struct Gates {
        bool speed = true, anchor = true, depth = true, lane = true, tilt = true, excursion = true;
        bool all() const { return speed && anchor && depth && lane && tilt && excursion; }
    };

    bool record(int control, const Metrics& sample) {
        if (!valid_ || control != next_control_ || control > controls) {
            valid_ = false;
            return false;
        }
        const float speed = p::norm(sample.velocity_m_s), anchor = p::norm(sample.hold_error_m);
        if (!std::isfinite(speed) || !std::isfinite(anchor) || !std::isfinite(sample.depth_m) ||
            !std::isfinite(sample.lateral_m) || !std::isfinite(sample.tilt_deg) ||
            !std::isfinite(sample.forward_m) ||
            sample.depth_m < 0.f || sample.lateral_m < 0.f || sample.tilt_deg < 0.f) {
            valid_ = false;
            return false;
        }

        peak_speed_ = std::fmax(peak_speed_, speed);
        peak_anchor_ = std::fmax(peak_anchor_, anchor);
        peak_depth_ = std::fmax(peak_depth_, sample.depth_m);
        peak_lane_ = std::fmax(peak_lane_, sample.lateral_m);
        peak_tilt_ = std::fmax(peak_tilt_, sample.tilt_deg);
        gates_.speed = peak_speed_ <= max_speed_m_s;
        gates_.anchor = peak_anchor_ <= max_anchor_m;
        gates_.depth = peak_depth_ <= max_depth_m;
        gates_.lane = peak_lane_ <= max_lane_m;
        gates_.tilt = peak_tilt_ <= max_tilt_deg;
        if (control != 0)
            speed_squared_sum_ += static_cast<double>(speed) * speed;
        if (control == final_window_start)
            forward_at_window_start_ = sample.forward_m;
        if (control >= final_window_start) {
            excursion_ = std::fmax(excursion_, std::fabs(sample.forward_m - forward_at_window_start_));
            gates_.excursion = excursion_ <= max_excursion_m;
        }
        ++next_control_;
        return true;
    }

    bool complete() const { return valid_ && next_control_ == controls + 1; }
    bool passed() const { return complete() && gates_.all(); }
    const Gates& gates() const { return gates_; }
    float peak_speed_m_s() const { return peak_speed_; }
    float peak_anchor_m() const { return peak_anchor_; }
    float peak_depth_m() const { return peak_depth_; }
    float peak_lane_m() const { return peak_lane_; }
    float peak_tilt_degrees() const { return peak_tilt_; }
    float final_excursion_m() const { return excursion_; }
    double rms_speed_m_s() const {
        return next_control_ > 1 ? std::sqrt(speed_squared_sum_ / (next_control_ - 1)) : 0.;
    }

private:
    int next_control_ = 0;
    bool valid_ = true;
    Gates gates_{};
    float peak_speed_ = 0, peak_anchor_ = 0, peak_depth_ = 0, peak_lane_ = 0, peak_tilt_ = 0;
    float excursion_ = 0, forward_at_window_start_ = 0;
    double speed_squared_sum_ = 0;
};

} // namespace cudarl::straight_motor
#endif
