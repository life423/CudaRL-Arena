#include "maze_env.h"

namespace cudarl::maze {

namespace {
constexpr float kWallPenalty = -0.1f;
constexpr float kWinReward   = 1.0f;
} // namespace

void MazeEnv::clear_walls() {
    for (auto& row : walls_)
        for (bool& cell : row) cell = false;
}

void MazeEnv::set_wall(int x, int y) {
    if (x >= 0 && x < kSize && y >= 0 && y < kSize) walls_[y][x] = true;
}

bool MazeEnv::is_wall(int x, int y) const {
    if (x < 0 || x >= kSize || y < 0 || y >= kSize) return true;
    return walls_[y][x];
}

void MazeEnv::reset() {
    human_ = kHumanStart;
    ai_    = kAIStart;
}

float MazeEnv::move(Cell& pos, int action) const {
    const Cell next{pos.x + action_dx(action), pos.y + action_dy(action)};
    if (is_wall(next.x, next.y)) return kWallPenalty;
    pos = next;
    return 0.0f;
}

StepResult MazeEnv::step(int human_action, int ai_action) {
    StepResult r{};
    r.human.reward = move(human_, human_action);
    r.ai.reward    = move(ai_, ai_action);

    const bool human_won = human_ == kGoal;
    const bool ai_won    = ai_ == kGoal;
    if (human_won && !ai_won) {
        r.human.reward += kWinReward;
        r.ai.reward    -= kWinReward;
    } else if (ai_won && !human_won) {
        r.human.reward -= kWinReward;
        r.ai.reward    += kWinReward;
    }
    r.human.done = r.ai.done = human_won || ai_won;
    r.human.pos  = human_;
    r.ai.pos     = ai_;
    return r;
}

} // namespace cudarl::maze
