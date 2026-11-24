#include "rl/flexible_tail_choices.h"
#include <iostream>
#include <limits>

namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;

int main() {
    int passed = 0, total = 0;
    const auto check = [&](bool pass, const char* label) {
        ++total;
        if (pass) ++passed;
        else std::cerr << label << '\n';
    };
    tail::MovementRequest request{};
    tail::ChoiceSample selected{};
    const physics::V course{1, 0, 0}, first{.05f, -.08f, -.02f}, drift{.06f, -.08f, -.02f};
    const float swim[]{4, 0, 0, 0}, left[]{0, 4, 0, 0}, right[]{0, 0, 4, 0}, hold[]{0, 0, 0, 4};
    const auto choose = [&](const float (&logits)[tail::choice_count], physics::V position) {
        return tail::accept_choice(logits, .5f, true, position, course, course, request, selected);
    };
    check(choose(swim, first) && request.choice == tail::Choice::Swim &&
        request.terminal_velocity_m_s.x == tail::requested_speed_m_s, "Swim request");
    check(choose(left, first) && request.choice == tail::Choice::TurnLeft && request.heading.z < -.49f,
        "Left course request");
    check(choose(right, first) && request.choice == tail::Choice::TurnRight && request.heading.z > .49f,
        "Right course request");
    check(choose(hold, first) && request.choice == tail::Choice::Hold &&
        physics::norm(request.terminal_velocity_m_s) == 0 && request.origin_m.x == first.x,
        "First hold anchors at current COM");
    check(choose(hold, drift) && request.origin_m.x == first.x, "Repeated hold keeps its anchor");
    check(choose(swim, drift) && request.heading.z > .49f && request.origin_m.x == drift.x,
        "Leaving hold preserves course and releases anchor");
    const float balanced[]{0, 0, 0, 0};
    check(tail::accept_choice(balanced, .9f, false, drift, course, course, request, selected) &&
        selected.action == static_cast<int>(tail::Choice::Hold) &&
        std::abs(selected.log_probability + std::log(4.f)) < 1e-6f, "Categorical sample and PPO log-probability");
    const float invalid[]{0, std::numeric_limits<float>::quiet_NaN(), 0, 0};
    check(!choose(invalid, first) && request.choice == tail::Choice::Hold &&
        request.origin_m.x == drift.x, "Invalid inference leaves request untouched");
    tail::MovementRequest from_rest{};
    check(tail::accept_choice(hold, .5f, true, first, {}, course, from_rest, selected) &&
        from_rest.choice == tail::Choice::Hold && from_rest.heading.x == 1.f,
        "A stationary start can take a hold request");
    std::cout << passed << '/' << total << " choice contract checks passed\n";
    return passed == total ? 0 : 1;
}
