#include "check.h"
#include "rl/pond_ppo_schedule.h"
#include <array>
#include <cstring>
#include <type_traits>

namespace {
namespace schedule = cudarl::rl::pond_ppo_schedule;
namespace task = cudarl::rl::pond_ppo_task;

struct Practice {
    schedule::SavedTask saved{};
    explicit Practice(int kind = task::Turn) {
        saved.task.kind = kind;
        saved.task.turn_radians = 15.f * task::obs::pi / 180.f;
        saved.task.request.request.heading = {1, 0, 0};
        saved.task.request.request.terminal_velocity = {.008f, 0, 0};
    }
    void prepare() { schedule::before(saved.schedule, saved.task, 1, 17); }
    void accept(bool good, bool terminal = false, bool at_cap = false) {
        const auto heading = saved.task.request.request.heading;
        const schedule::Sample sample{.008f * heading.x, .008f * heading.z, good ? 0.f : .002f, 0.f};
        ++saved.task.episode_controls;
        schedule::after(saved.schedule, saved.task, sample, terminal, at_cap);
    }
};
}

TEST_CASE("turn practice: settle once and qualify only after five seconds plus a continuous second") {
    Practice p;
    for (int i = 0; i < 40; ++i) {
        p.prepare();
        CHECK(p.saved.schedule.onsets == 0);
        p.accept(true);
    }
    p.prepare();
    CHECK(p.saved.schedule.onsets == 1);
    CHECK(p.saved.schedule.onset_control == 41);
    p.saved.schedule.long_hold = false;
    for (int i = 1; i <= 119; ++i) {
        p.accept(true);
        CHECK(p.saved.schedule.successes == 0);
        p.prepare();
        CHECK(p.saved.schedule.onsets == 1);
    }
    p.accept(false);
    CHECK(p.saved.schedule.streak == 0);
    for (int i = 1; i <= 20; ++i) {
        p.prepare();
        p.accept(true);
        CHECK(p.saved.schedule.successes == (i == 20 ? 1u : 0u));
    }
    p.prepare();
    CHECK(p.saved.schedule.onsets == 2);
    CHECK(p.saved.task.episode_controls == 180);
}

TEST_CASE("turn practice: timeout and held success each issue once without ending the episode") {
    for (bool qualifies : {false, true}) {
        Practice p;
        p.saved.task.episode_controls = 40;
        p.prepare();
        p.saved.schedule.long_hold = true;
        for (int age = 1; age <= 400; ++age) {
            p.prepare();
            CHECK(p.saved.schedule.onsets == 1);
            p.accept(qualifies);
            CHECK(p.saved.schedule.active == (age < 400));
            CHECK(p.saved.schedule.successes == (qualifies && age >= 120 ? 1u : 0u));
        }
        CHECK(p.saved.schedule.timeouts == (qualifies ? 0u : 1u));
        CHECK(p.saved.schedule.interrupted == 0);
        p.prepare();
        CHECK(p.saved.schedule.onsets == 2);
        CHECK(p.saved.task.episode_controls == 440);
    }
}

TEST_CASE("turn practice: a checkpoint preserves the streak and a terminal does not create a phantom turn") {
    static_assert(std::is_trivially_copyable_v<schedule::SavedTask>);
    Practice p;
    p.saved.task.episode_controls = 40;
    p.prepare();
    p.saved.schedule.long_hold = false;
    for (int age = 1; age <= 111; ++age) p.accept(true);
    std::array<unsigned char, sizeof(p.saved)> bytes{};
    std::memcpy(bytes.data(), &p.saved, bytes.size());
    Practice restored;
    std::memcpy(&restored.saved, bytes.data(), bytes.size());
    for (int i = 0; i < 9; ++i) {
        p.prepare(); restored.prepare();
        p.accept(true); restored.accept(true);
    }
    CHECK(p.saved.schedule.successes == 1);
    CHECK(restored.saved.schedule.successes == 1);
    p.prepare(); restored.prepare();
    CHECK(restored.saved.task.request.request.heading.x == p.saved.task.request.request.heading.x);
    CHECK(restored.saved.task.request.request.heading.z == p.saved.task.request.request.heading.z);
    p.saved.task.episode_controls = 1199;
    p.accept(false, true, true);
    CHECK(p.saved.schedule.interrupted == 1);
    CHECK(p.saved.schedule.timeouts == 0);
    const auto starts = p.saved.schedule.onsets;
    p.prepare();
    CHECK(p.saved.schedule.onsets == starts);
    schedule::reset_episode(p.saved.schedule);
    CHECK(p.saved.schedule.onsets == starts);
    CHECK(p.saved.schedule.interrupted == 1);
    for (int kind : {task::Hold, task::Straight}) {
        Practice other(kind);
        other.saved.task.episode_controls = 100;
        other.prepare(); other.accept(true);
        CHECK(other.saved.schedule.onsets == 0);
    }
}

TEST_CASE("turn practice: the last eligible tick can complete a request exactly at its timeout") {
    Practice p;
    p.saved.task.episode_controls = 800;
    p.prepare();
    p.saved.schedule.long_hold = true;
    for (int age = 1; age <= 400; ++age) {
        p.prepare();
        p.accept(age > 380, age == 400, age == 400);
    }
    CHECK(p.saved.schedule.successes == 1);
    CHECK(p.saved.schedule.timeouts == 0);
    CHECK(p.saved.schedule.interrupted == 0);
    CHECK(!p.saved.schedule.active);
    p.prepare();
    CHECK(p.saved.schedule.onsets == 1);
}
