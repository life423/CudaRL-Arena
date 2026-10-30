// Generic background publication must retain the validated actor when newer
// learner weights regress. This requires no legacy demo or GPU training.
#include <chrono>
#include <thread>
#include <vector>
#include "check.h"
#include "demos/ppo_session.h"

namespace {
using namespace cudarl;

template <typename Predicate>
bool wait_for(Predicate predicate, double seconds = 5.0) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
}

struct ScoredPublicationTrainer {
    struct Config {int total_updates=4;};
    struct Stats {int update;};
    explicit ScoredPublicationTrainer(const Config&) {}
    int update_id=0;
    rl::EvaluatedPolicy best{0,0,64,1200,0x51A7E123u,1,{0.f}};
    Stats update() {
        ++update_id;
        if(update_id==1) best={1,10,64,1200,0x51A7E123u,1,{1.f}};
        return {update_id};
    }
    std::vector<float> actor_params() const {return {static_cast<float>(update_id)};}
    const rl::EvaluatedPolicy& best_policy() const {return best;}
};
}
TEST_CASE("training session: best publication keeps validated weights through learner regression") {
    cudarl::demos::TrainingSession<ScoredPublicationTrainer> session({},0.0);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!session.finished()&&session.error().empty()&&std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(session.finished()&&session.error().empty());
    uint64_t version=0;cudarl::rl::EvaluatedPolicy best;
    CHECK(session.latest_best_policy(version,best));
    CHECK(best.update==1&&best.foods==10&&best.parameters==std::vector<float>{1.f});
    uint64_t latest_version=0;std::vector<float> latest;
    CHECK(session.latest_policy(latest_version,latest)&&latest==std::vector<float>{4.f});
    session.stop();
    CHECK(!session.latest_best_policy(version,best));
    std::vector<ScoredPublicationTrainer::Stats> history;
    CHECK(session.history_since(0,history)==4);
}

TEST_CASE("training session: background updates publish current weights and preserve initialization") {
    using Session = cudarl::demos::TrainingSession<ScoredPublicationTrainer>;
    Session session({8}, 0.0);
    CHECK(wait_for([&] { return session.finished() || !session.error().empty(); }));
    CHECK(session.error().empty());
    std::vector<ScoredPublicationTrainer::Stats> history;
    CHECK(session.history_since(0, history) == 8);
    CHECK(history.size() == 8 && history.back().update == 8);
    uint64_t version = 0;
    std::vector<float> current;
    CHECK(session.latest_policy(version, current) && version == 9 && current == std::vector<float>{8.f});
    CHECK(!session.latest_policy(version, current));
    std::vector<float> initial, initial_again;
    CHECK(session.initial_policy(initial) && initial == std::vector<float>{0.f});
    CHECK(initial != current);
    CHECK(session.initial_policy(initial_again) && initial_again == initial);
}

TEST_CASE("training session: throttle bounds update rate and pause stops publication") {
    using Session = cudarl::demos::TrainingSession<ScoredPublicationTrainer>;
    Session session({0}, 10.0);
    auto update_count = [&] {
        std::vector<ScoredPublicationTrainer::Stats> history;
        return session.history_since(0, history);
    };
    CHECK(wait_for([&] { return update_count() >= 1 || !session.error().empty(); }));
    CHECK(session.error().empty());
    const size_t before = update_count();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    // An unthrottled fake trainer would publish thousands of updates here.
    // Permit an update already in flight and coarse host scheduling.
    CHECK(update_count() <= before + 3);
    session.set_paused(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const size_t paused_at = update_count();
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    CHECK(update_count() == paused_at);
    CHECK(session.paused());
}

TEST_CASE("training session: paused start publishes initialization before any update") {
    using Session = cudarl::demos::TrainingSession<ScoredPublicationTrainer>;
    Session session({2}, 0.0, true);
    std::vector<float> initial;
    CHECK(wait_for([&] { return session.initial_policy(initial) || !session.error().empty(); }));
    CHECK(session.error().empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    std::vector<ScoredPublicationTrainer::Stats> history;
    CHECK(session.history_since(0, history) == 0);
    uint64_t version = 0;
    std::vector<float> current;
    CHECK(session.latest_policy(version, current) && version == 1 && current == initial);
    session.set_paused(false);
    CHECK(wait_for([&] { return session.finished() || !session.error().empty(); }));
    CHECK(session.error().empty());
    CHECK(session.history_since(0, history) == 2);
    CHECK(session.latest_policy(version, current) && current == std::vector<float>{2.f});
    CHECK(current != initial);
}
