# CudaRL creature motor

Native C++/CUDA reinforcement learning with PPO and one Godot viewer. Physical
motion comes from torque-limited internal joints and water forces; Godot supplies
presentation only. The motor controls stroke strength and signed cadence.

## Current state

The prior imitation-trained controllers and dedicated supervised training paths
were removed by user request. No verified PPO checkpoint is installed. The viewer
shows the native creature paused with its motion controls unavailable. It does
not substitute random weights, a teacher or another algorithm.

The native PPO primitives live in src/rl: Gaussian sampling/log probabilities,
clipped policy loss, GAE and minibatch gathers. Shared MLP, Adam, GEMM and critic
loss code lives in src/nn. The ppo_motor executable trains a fresh actor/critic
on the forward8mm/s lesson. Training worlds and rollout/update buffers stay on
the GPU; frozen evaluation uses the same motor equations on the CPU.

```powershell
./tools/view-live.ps1             # incremental native build; open the viewer
./tools/view-live.ps1 -SkipBuild  # use the existing extension
```

CMake3.24+, Visual Studio2022 C++ tools, CUDA for learning and matching prebuilt
Godot C++ bindings are required. Live stepping and numerical kernels are native;
there is no Python runtime or exporter in the active project.

```powershell
cmake --preset windows
cmake --build --preset release
./build/Release/foundation_tests.exe gaussian:
./build/Release/foundation_tests.exe ppo:
./tools/run-ppo.ps1 -Name NEW_UNIQUE_NAME # build, zero-step checks, freeze inputs
# Its -Run switch executes that identity only after an explicit bounded approval.
```

The zero-step viewer check is:
`Godot_console.exe --headless --path live --script ../tests/godot/viewer_controls.gd`.
It verifies the missing-policy state and never advances physics. New physical or
training runs require their own finite approved protocol; all old budgets closed.

## Architecture and limitations

One native world owner, one bridge and one plain scene. live/main.gd directly
extends Node3D; FrozenSwimmer composes FrozenController. Shared motor code lives
in src/nn/motor and physics in src/physics. Keep training independent of Godot.
Preserve model6, two universal joints/four axes and +/-10microN*m torque limits.
Contact, current, stopping/holding and arbitrary transitions remain unvalidated.

Tracked historical source/assets remain recoverable through Git (a812960 and
prior checkpoints). Unique local research results/checkpoints remain in ignored
build/ as evidence, not runtime dependencies. Git does not save ignored evidence.
The removal inventory is build/supervised-removal-inventory.json. The first fresh
PPO result is pinned in assets/ppo-forward-v1:64updates,4/4four-second frozen
forward cases, with no imported weights/examples. Provenance, fresh initial state,
protocol and compact results accompany the checkpoint. Full ignored evidence is
in build/ppo/fresh-forward-ppo-v1. This is a research checkpoint, not a Godot
inference asset; it has not been installed in the viewer. Speed changes remain a
later lesson.

Follow [AGENTS.md](AGENTS.md). Preserve the original checkout, editor changes and
personal data. Third-party notices remain in docs/third-party/.
