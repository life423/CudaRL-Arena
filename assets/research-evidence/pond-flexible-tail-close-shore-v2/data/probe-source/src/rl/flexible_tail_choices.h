#pragma once

#include "physics/dynamics.h"
#include "rl/ppo_math.h"

namespace cudarl::flexible_tail {

// The chooser selects a movement request. A separately learned motor
// converts the held request to the two root joint targets.
enum class Choice : int { Swim, TurnLeft, TurnRight, Hold };
constexpr int choice_count = 4;
constexpr uint32_t choice_schema = 1;
constexpr float requested_speed_m_s = .008f;
constexpr float requested_turn_rad = .5235987756f;

struct MovementRequest {
    Choice choice = Choice::Swim;
    physics::V origin_m, heading{1, 0, 0}, terminal_velocity_m_s;
    bool initialized = false;
};

struct ChoiceSample {
    int action = -1;
    float log_probability = 0;
};

// A caller invokes this at a chooser decision boundary with policy logits
// and the measured horizontal course. Neither network runs here. Reissuing
// Hold keeps the same fixed anchor throughout that holding request.
CUDARL_HD inline bool accept_choice(const float (&logits)[choice_count], float uniform, bool greedy,
                                    physics::V com_m, physics::V course, physics::V facing, MovementRequest& held,
                                    ChoiceSample& sample) {
    if (!physics::finite_vector(com_m) || !physics::finite_vector(course) || !physics::finite_vector(facing) ||
        (!greedy && (!(uniform > 0.f) || !(uniform < 1.f)))) return false;
    float maximum = logits[0];
    for (const auto logit : logits) {
        if (!physics::finite_number(logit)) return false;
        maximum = fmaxf(maximum, logit);
    }
    float centered[choice_count], logp[choice_count];
    for (int i = 0; i < choice_count; ++i) centered[i] = logits[i] - maximum;
    rl::log_softmax(centered, choice_count, logp);
    const int action = rl::sample_action(logp, choice_count, uniform, greedy);
    const auto choice = static_cast<Choice>(action);

    auto next = held;
    course.y = 0;
    const float speed = physics::norm(course);
    if (!held.initialized) {
        facing.y = 0;
        const float facing_length = physics::norm(facing);
        if (!(speed > 0) && !(facing_length > 0)) return false;
        next.heading = speed > 0 ? course / speed : facing / facing_length;
        next.origin_m = com_m;
        next.initialized = true;
    }
    if (choice == Choice::TurnLeft || choice == Choice::TurnRight) {
        const auto direction = speed > 0 ? course / speed : next.heading;
        const float delta = choice == Choice::TurnLeft ? requested_turn_rad : -requested_turn_rad;
        next.heading = physics::axis_rotation({0, 1, 0}, delta).rotate(direction);
        next.origin_m = com_m;
    } else if (choice != held.choice) {
        next.origin_m = com_m;
    }
    next.choice = choice;
    next.terminal_velocity_m_s = choice == Choice::Hold ? physics::V{} :
        next.heading * requested_speed_m_s;
    held = next;
    sample = {action, logp[action]};
    return true;
}

} // namespace cudarl::flexible_tail
