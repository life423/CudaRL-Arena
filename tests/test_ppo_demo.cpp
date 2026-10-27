// The live CartPole demo's pieces: the background training session (throttle,
// pause, published policy) and the CPU showcase driven by a trained policy.

#include <chrono>
#include <thread>
#include <vector>

#include "check.h"
#include "demos/cartpole_showcase.h"
#include "demos/ppo_session.h"

using namespace cudarl;
using namespace cudarl::demos;

namespace {

template <typename Pred>
bool wait_for(Pred pred, double seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

} // namespace

namespace {
struct ScoredPublicationTrainer {
    struct Config {int total_updates=4;};
    struct Stats {int update;};
    explicit ScoredPublicationTrainer(const Config&) {}
    int update_id=0;
    rl::EvaluatedPolicy best{0,0,64,1200,0x51A7E123u,1,{0.f}};
    Stats update() {
        ++update_id;
        if(update_id==1) best={1,10,64,1200,0x51A7E123u,1,{1.f}};
        // Updates 2--4 represent a decline, a tied result, then another decline.
        return {update_id};
    }
    std::vector<float> actor_params() const {return {static_cast<float>(update_id)};}
    const rl::EvaluatedPolicy& best_policy() const {return best;}
};
}
TEST_CASE("ppo_demo: best publication keeps the evaluated actor rather than newer weights") {
    TrainingSession<ScoredPublicationTrainer> session({},0.0);
    CHECK(wait_for([&] {return session.finished();},5));
    uint64_t version=0;rl::EvaluatedPolicy best;
    CHECK(session.latest_best_policy(version,best));
    CHECK(best.update==1 && best.foods==10 && best.parameters==std::vector<float>{1.f});
    uint64_t latest_version=0;std::vector<float> latest;
    CHECK(session.latest_policy(latest_version,latest) && latest==std::vector<float>{4.f});
    session.stop();
    CHECK(!session.latest_best_policy(version,best));
    std::vector<ScoredPublicationTrainer::Stats> history;
    CHECK(session.history_since(0,history)==4);
}

GPU_TEST_CASE("ppo_demo: session trains in the background and publishes the policy") {
    rl::PpoConfig cfg;
    cfg.total_updates = 8;
    PpoSession session(cfg);
    session.set_updates_per_second(0);   // unthrottled
    CHECK(wait_for([&] { return session.finished() || !session.error().empty(); }, 30));
    CHECK(session.error().empty());
    std::vector<rl::PpoStats> history;
    CHECK(session.history_since(0, history) == 8);
    CHECK(history.size() == 8 && history.back().update == 8);
    uint64_t version = 0;
    std::vector<float> params;
    CHECK(session.latest_policy(version, params) && version == 9 && !params.empty());
    CHECK(!session.latest_policy(version, params));   // nothing new since
    std::vector<float> initial, initial_again;
    CHECK(session.initial_policy(initial));
    rl::PpoTrainer reference(cfg);
    CHECK(initial == reference.actor_params()); // actual initialized weights, before any update
    CHECK(initial != params);                  // training must not overwrite the checkpoint
    CHECK(session.initial_policy(initial_again) && initial_again == initial);
}

GPU_TEST_CASE("ppo_demo: throttle and pause limit the update rate") {
    rl::PpoConfig cfg;
    PpoSession session(cfg);
    session.set_updates_per_second(4);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::vector<rl::PpoStats> h;
    const size_t after_one_second = session.history_since(0, h);
    CHECK_MSG(after_one_second >= 2 && after_one_second <= 6, "updates in 1 s at 4/s: " + std::to_string(after_one_second));
    session.set_paused(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let an in-flight update land
    const size_t paused_at = session.history_since(0, h);
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    CHECK(session.history_since(0, h) == paused_at);
}

GPU_TEST_CASE("ppo_demo: a paused start publishes update zero without taking an optimizer step") {
    rl::PpoConfig cfg;
    cfg.num_envs=128;
    cfg.rollout_steps=16;
    cfg.total_updates=2;
    PpoSession session(cfg,0.0,true);
    std::vector<float> initial;
    CHECK(wait_for([&] { return session.initial_policy(initial) || !session.error().empty(); },10));
    CHECK(session.error().empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::vector<rl::PpoStats> history;
    CHECK(session.history_since(0,history)==0);
    uint64_t version=0;
    std::vector<float> current;
    CHECK(session.latest_policy(version,current));
    CHECK(version==1 && current==initial);
    session.set_paused(false);
    CHECK(wait_for([&] {return session.finished() || !session.error().empty();},10));
    CHECK(session.error().empty());
    CHECK(session.history_since(0,history)==2);
    CHECK(session.latest_policy(version,current) && current!=initial);
}

GPU_TEST_CASE("ppo_demo: a trained policy balances every showcase cart for 500 steps") {
    rl::PpoConfig cfg;
    rl::PpoTrainer trainer(cfg);
    for (int u = 0; u < 30; ++u) trainer.update();

    CartPoleShowcase showcase(12, cfg.hidden, 99);
    showcase.set_policy(trainer.actor_params());
    bool any_failed = false;
    for (int t = 0; t < 500; ++t) {
        showcase.step(/*greedy=*/true);
        for (int i = 0; i < showcase.count(); ++i) any_failed |= showcase.just_failed(i);
    }
    CHECK(!any_failed);
    for (int i = 0; i < showcase.count(); ++i) CHECK(showcase.last_length(i) == 500);   // every cart hit the time limit
}

TEST_CASE("ppo_demo: an untrained policy drops the poles") {
    CartPoleShowcase showcase(12, 64, 99);   // zero weights: a uniform-random policy
    int failures = 0;
    for (int t = 0; t < 200; ++t) {
        showcase.step(false);
        for (int i = 0; i < showcase.count(); ++i) failures += showcase.just_failed(i) ? 1 : 0;
    }
    CHECK(failures > 0);
}
