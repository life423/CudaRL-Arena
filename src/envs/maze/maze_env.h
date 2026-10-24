#ifndef CUDARL_ENVS_MAZE_ENV_H
#define CUDARL_ENVS_MAZE_ENV_H

namespace cudarl::maze {

// Screen-space directions (y grows downward), matching Godot's 2D convention.
enum Action : int { Up = 0, Right = 1, Down = 2, Left = 3 };
constexpr int kNumActions = 4;
constexpr int kSize = 10;

constexpr int action_dx(int a) { return a == Right ? 1 : (a == Left ? -1 : 0); }
constexpr int action_dy(int a) { return a == Down  ? 1 : (a == Up   ? -1 : 0); }

struct Cell {
    int x, y;
    friend constexpr bool operator==(Cell a, Cell b) { return a.x == b.x && a.y == b.y; }
};

struct AgentStep {
    Cell  pos;
    float reward;
    bool  done;
};

struct StepResult {
    AgentStep human;
    AgentStep ai;
};

// Two agents race from opposite corners to a shared goal, 7 moves from each.
// Bumping a wall costs 0.1; reaching the goal first is +1 for the winner and -1
// for the loser, and simultaneous arrival is a draw. Invalid actions don't move.
class MazeEnv {
public:
    static constexpr Cell kHumanStart{1, 1};
    static constexpr Cell kAIStart{8, 8};
    static constexpr Cell kGoal{1, 8};

    void clear_walls();
    void set_wall(int x, int y);           // out-of-range cells are ignored
    bool is_wall(int x, int y) const;      // outside the board counts as a wall

    void       reset();
    StepResult step(int human_action, int ai_action);

    Cell human() const { return human_; }
    Cell ai() const { return ai_; }

private:
    float move(Cell& pos, int action) const;

    bool walls_[kSize][kSize] = {};
    Cell human_ = kHumanStart;
    Cell ai_    = kAIStart;
};

} // namespace cudarl::maze

#endif // CUDARL_ENVS_MAZE_ENV_H
