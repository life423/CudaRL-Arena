#pragma once

#include "rl/flexible_tail_transition.h"

namespace cudarl::flexible_tail {

struct BeatCourseSample {
    physics::V velocity_m_s{};
    float period_s = 1.f;
    bool period_measured = false;
    bool fresh = false;
    bool reliable = false;
};

// Causal course feedback. Root-rate crossings measure the latest full beat;
// current COM displacement over that duration is interpolated in the measured
// 20ms history. This avoids waiting for another beat before course can change.
// All fields are ordinary value state and accompany body/servo/RNG checkpoints.
struct BeatCourseFeedback {
    static constexpr int capacity = 128;
    static constexpr float minimum_period_s = .4f;
    static constexpr float maximum_period_s = 2.f;
    static constexpr float acquisition_window_s = 1.f;
    static constexpr float reliable_speed_m_s = .003f;
    static constexpr float crossing_deadband_rad_s = .02f;

    physics::V positions[capacity]{};
    int newest = 0, count = 0, controls = 0, initial_control = 0;
    int previous_sample = -1, direction = 0;
    float previous_rate = 0.f;
    double candidate_sample = -1., last_positive_sample = -1., last_valid_sample = -1.;
    float period_s = acquisition_window_s;
    uint32_t complete_periods = 0, rejected_periods = 0;
    bool pending_positive = false;

    CUDARL_HD void initialize(const TurnWorld& motion) {
        *this = {};
        // The archive holds real COM observations over the preceding two
        // seconds. No destination, requested heading or head-facing substitute
        // supplies travel when acquiring the first live beat period.
        for (int i = 0; i < turn_history_controls; ++i)
            positions[i] = motion.history[(motion.history_next + i) % turn_history_controls];
        newest = turn_history_controls;
        positions[newest] = snapshot(motion.body).com_m;
        count = turn_history_controls + 1;
        controls = initial_control = motion.controls;
        period_s = acquisition_window_s;
        previous_sample = controls * turn_substeps_per_control;
        previous_rate = motion.body.velocity[6];
        direction = previous_rate > crossing_deadband_rad_s ? 1
            : previous_rate < -crossing_deadband_rad_s ? -1 : 0;
    }

    CUDARL_HD void add_substep(float rate, int sample, physics::V com) {
        if (sample <= previous_sample || !physics::finite_number(rate) ||
            !physics::finite_vector(com)) return;
        if (previous_sample >= 0 && previous_rate <= 0.f && rate > 0.f) {
            const float fraction = -previous_rate / (rate - previous_rate);
            candidate_sample = static_cast<double>(previous_sample) +
                static_cast<double>(fraction) * (sample - previous_sample);
            pending_positive = true;
        }
        const int next = rate > crossing_deadband_rad_s ? 1
            : rate < -crossing_deadband_rad_s ? -1 : 0;
        if (next) {
            if (direction < 0 && next > 0 && pending_positive) {
                if (last_positive_sample >= 0.) {
                    const float measured = static_cast<float>(candidate_sample - last_positive_sample)
                        * motor_substep_s;
                    if (measured >= minimum_period_s && measured <= maximum_period_s) {
                        period_s = measured;
                        last_valid_sample = candidate_sample;
                        ++complete_periods;
                    } else {
                        ++rejected_periods;
                    }
                }
                last_positive_sample = candidate_sample;
            }
            if (direction != next) pending_positive = false;
            direction = next;
        }
        previous_rate = rate;
        previous_sample = sample;
    }

    CUDARL_HD void add_control(physics::V com) {
        newest = (newest + 1) % capacity;
        positions[newest] = com;
        count = count < capacity ? count + 1 : capacity;
        ++controls;
    }

    CUDARL_HD bool position(float lag_controls, physics::V& result) const {
        if (!(lag_controls >= 0.f) || !physics::finite_number(lag_controls)) return false;
        const int recent = static_cast<int>(floorf(lag_controls));
        const int older = static_cast<int>(ceilf(lag_controls));
        if (older >= count) return false;
        const auto a = positions[(newest + capacity - recent) % capacity];
        const auto b = positions[(newest + capacity - older) % capacity];
        result = a + (b - a) * (lag_controls - recent);
        return physics::finite_vector(result);
    }

    CUDARL_HD BeatCourseSample sample(int back_controls = 0) const {
        BeatCourseSample out{};
        out.period_s = period_s;
        out.period_measured = complete_periods != 0;
        physics::V begin{}, end{};
        if (back_controls < 0 || !(period_s >= minimum_period_s) ||
            !(period_s <= maximum_period_s) ||
            !position(static_cast<float>(back_controls), end) ||
            !position(static_cast<float>(back_controls) + period_s / motor_control_s, begin)) return out;
        out.velocity_m_s = (end - begin) / period_s;
        out.velocity_m_s.y = 0.f;
        // During acquisition this is explicitly a one-second measured COM
        // window. After acquisition require a recently observed root beat.
        const double now = static_cast<double>((controls - back_controls) * turn_substeps_per_control);
        out.fresh = out.period_measured
            ? (last_valid_sample >= 0. && now - last_valid_sample <= 2. * period_s / motor_substep_s)
            : (controls - back_controls - initial_control) * motor_control_s <= 2.f * maximum_period_s;
        out.reliable = out.fresh && physics::norm(out.velocity_m_s) >= reliable_speed_m_s;
        return out;
    }
};

} // namespace cudarl::flexible_tail
