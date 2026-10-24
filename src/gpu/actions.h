#ifndef CUDARL_ACTIONS_H
#define CUDARL_ACTIONS_H

// Single source of truth for the action encoding, shared by the CUDA kernel
// and the Godot bindings. Coordinates are screen-space (y grows downward),
// matching Godot's 2D convention.

#ifdef __CUDACC__
#define CUDARL_HD __host__ __device__
#else
#define CUDARL_HD
#endif

namespace cudarl {

enum Action : int { Up = 0, Right = 1, Down = 2, Left = 3 };
constexpr int kNumActions = 4;

CUDARL_HD inline bool is_valid_action(int a) { return a >= 0 && a < kNumActions; }

// Invalid actions map to a zero move.
CUDARL_HD inline int action_dx(int a) { return a == Right ? 1 : (a == Left ? -1 : 0); }
CUDARL_HD inline int action_dy(int a) { return a == Down  ? 1 : (a == Up   ? -1 : 0); }

} // namespace cudarl

#endif // CUDARL_ACTIONS_H
