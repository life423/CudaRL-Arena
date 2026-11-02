# CudaRL creature motor

Native C++/CUDA reinforcement learning with PPO and one Godot viewer. Physical
motion comes from torque-limited internal joints and water forces; Godot supplies
presentation only. The motor supplies four independent joint-angle targets through bounded servos.

## Current state

Training is stopped. The sustained-depth milestone is saved in
`assets/ppo-four-axis-checkpoints/update-336-episode64`:4/4 short swims and3/3
64-second recovery cases pass. Both existing60-second swims hold depth, lane and
posture;28/30 windows pass. Formal long-case acceptance is0/2 because the2–4s
speed windows miss7.2mm/s. The earlier32s settling deadline also remains failed.
This is not full formal motor acceptance or a runtime-installed policy.
JSON provenance records hashes, exact protocol, results and remaining limits.
Original288 and the converted-parent/path-scale320 milestone remain preserved.
The first stopping trial is saved in
`assets/ppo-four-axis-checkpoints/stopping-336to352-rejected`. It slowed after
the zero-speed request but never stopped and regressed to0/4 short swims. Those
checkpoints are rejected;336 remains the preserved sustained-swimming baseline.
[HANDOFF.md](HANDOFF.md) describes the earlier baseline campaign.
Reward5 separates roll/pitch and scores depth-dependent vertical settling from
the first control. Observation schema2 scales all three body-frame path-error
inputs by2mm; old checkpoints retain explicit72mm scaling. Checkpoint schema4
records that choice and rejects mismatched frozen evaluators.

The prior imitation-trained controllers and dedicated supervised training paths
were removed by user request. No verified PPO checkpoint is installed. The viewer
shows the native creature paused with its motion controls unavailable. It does
not substitute random weights, a teacher or another algorithm.

The native PPO primitives live in src/rl: Gaussian sampling/log probabilities,
clipped policy loss, GAE and minibatch gathers. Shared MLP, Adam, GEMM and critic
loss code lives in src/nn. The ppo_motor executable trains a fresh actor/critic
on the forward8mm/s lesson:40observations, four outputs, eight- or32-second episodes.
There is no oscillator in this action path. Old two-output checkpoints are
preserved as evidence and rejected by the new explicit schema. Training worlds and rollout/update buffers stay on
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

The historical frozen headless evaluation loaded the exact prior PPO checkpoint
without restoring optimizers. The two60second cases use an explicit1.2x0.8m horizontal
arena, with no wrapping or resets. First4second parity passed, but later speed,
lane and COM-path gates failed; only2/30windows passed. The policy is therefore
not validated for long-duration tracking. Compact evidence is long-evaluation.json
and long-windows.csv beside the checkpoint; full control traces remain in ignored
build/ppo/frozen-long-forward-v1. No weights changed during this check.

Follow [AGENTS.md](AGENTS.md). Preserve the original checkout, editor changes and
personal data. Third-party notices remain in docs/third-party/.
