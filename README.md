# Tadpole Pond

**Mission: create an evolving reinforcement-learning creature that users can
interact with, drawing on what it has learned and learning from new experiences
to adapt and thrive.**

The intended experience is its own persistent pond world: a creature to feed,
observe and interact with as it explores, rests and learns. The current motor
inspector is an intermediate tool; that integrated world remains to be built.

Native C++/CUDA reinforcement learning with PPO and one Godot viewer. Physical
motion comes from torque-limited internal joints and water forces; Godot supplies
presentation only. The motor supplies four independent joint-angle targets through bounded servos.

Dependable movement is a foundation for that experience. Core aquatic abilities
include swimming and steering in still water and currents, braking, holding a
position against flow, and recovering while retaining previously learned skills.
User interaction should lead to measurable adaptation, with actual RL updates
distinguished from the use of existing training. The longer journey includes
physical growth, underwater limbs, shallow crawling, land movement and eventually
manipulation. One learned goal chooser and one reusable body motor remain the
intended design. The interactive learning loop and later body stages remain to
build; current-capable swimming is not yet qualified.

## Current state

**Current work:** the explicitly approved [corrected GPU current study](docs/CURRENT_SWIM_V2_RESULTS_2026-11-22.md)
passed its fresh physics gate and completed 32 native PPO updates. Original
current-swimming cases improved from **4/16 to 8/16**, short of the unchanged
16/16 gate. The candidate is not qualified or installed; further withheld/current
endurance tests did not run. The viewer keeps the qualified calm policy and
discloses the latest result in Details. The selected 16,384-world CUDA workload
measured about 500,294 control steps/second including evidence writes. The study
used 89,566,320 controls and 344 Adam steps, then closed without retries or an
extension. [The first numerical failure](docs/CURRENT_SWIM_RESULTS_2026-11-22.md)
and all prior evidence remain preserved.

**November 22:** [learned persistent amplitude](docs/HOLD_LEARNED_AMPLITUDE_RESULTS_2026-11-22.md)
now passes all six original and six withheld 64-second calm holds, plus both
uninterrupted five-minute holds. Native PPO learned two amplitude-distribution
parameters in seven updates; the initial deterministic actor failed all six holds.
All four short swims and the same 28 long windows stayed byte-identical through
the frozen backbone. Independent verification passes. This is calm holding from
rest, with restricted parameter learning; moving-start braking and disturbance
recovery remain unvalidated. Training is stopped, checkpoint 336 is preserved,
and no production adoption follows automatically. [AI_STATUS.md](docs/AI_STATUS.md)
records current evidence and scope. Viewer inspection is recorded or frozen
physics, not ongoing learning.

### Earlier milestones

The approved balanced reward-scale comparison is complete and training is stopped.
Both arms ran exactly8updates. Gain1 lost short swimming(0/4); holding gain1/1024
retained4/4short and28/30long windows and improved critic conditioning, but both
passed0/6holding cases. The scaled arm still travels~8.1mm/s with~64mm drift in
the final8s and reaches10.42degrees tilt in one case, failing the10degree limit.
Neither candidate is accepted or installed. Checkpoint336/all prior data remain
unchanged. [Measured results](docs/HOLD_SCALE_RESULTS_2026-11-04.md) and
[frozen protocol](docs/HOLD_SCALE_2026-11-04.md) contain exact gates and budgets.

Training is stopped. The compatible41-input migration and exact8-update pilot are
saved in `assets/ppo-four-axis-checkpoints/append41-hold-336to344`.
Both same-seed replays preserve336observations/actions/critic/trajectories exactly.
After8updates,4/4short swims and28/30long windows remain, but0/6holding cases pass:
the creature continues~8.6mm/s and drifts~68mm in8s. Depth and10degree tilt gates
pass; stopping/position/lane do not. Original336 remains unchanged; no adoption.
The independent reward check found~334x cost scale, a separate learning concern.
See [protocol and measured results](docs/APPEND41_HOLD_2026-11-04.md).

The earlier November4 position/depth-hold lesson is preserved as
rejected in `assets/ppo-four-axis-checkpoints/position-hold-r8-336to368-rejected`.
It completed32updates but passed0/6holding cases at every assessment and regressed
short swimming to0/4. Low final speed was accompanied by large position/depth
error and~120degree tilt, so this is not successful holding. Original336 remains
unchanged. [Protocol and results](docs/POSITION_HOLD_2026-11-04.md) identify the
new full-position input's incompatibility with the inherited policy.

The earlier November4 half-rate startup trial is preserved in
`assets/ppo-four-axis-checkpoints/startup-speed-r7-half-rate-336to352-rejected`.
It completed16updates and improved2-4s speed, but344 still exceeds the10degree
tilt limit (10.0813degrees);352 regresses further. Neither candidate qualifies.
Both are rejected; no viewer adoption or new stopping run occurred. See
[the frozen protocol and outcome](docs/STARTUP_HALF_RATE_2026-11-04.md).

The sustained-depth milestone is saved in
`assets/ppo-four-axis-checkpoints/update-336-episode64`:4/4 short swims and3/3
64-second recovery cases pass. Both existing60-second swims hold depth, lane and
posture;28/30 windows pass. Formal long-case acceptance is0/2 because the2–4s
speed windows miss7.2mm/s. The earlier32s settling deadline also remains failed.
The additional `five-minute-evaluation` record passes148/150 windows over two
300-second frozen swims; only those same startup-speed windows fail. This is
bounded endurance evidence, not proof of indefinite operation.
This is not full formal motor acceptance or a runtime-installed policy.
JSON provenance records hashes, exact protocol, results and remaining limits.
Original288 and the converted-parent/path-scale320 milestone remain preserved.
The first stopping trial is saved in
`assets/ppo-four-axis-checkpoints/stopping-336to352-rejected`. It slowed after
the zero-speed request but never stopped and regressed to0/4 short swims. Those
checkpoints are rejected;336 remains the preserved sustained-swimming baseline.
The subsequent startup-speed trial is preserved in
`assets/ppo-four-axis-checkpoints/startup-speed-r7-336to352-rejected`.
Reward7 scores horizontal speed from the first control. At the previous actor
rate0.00025 it fixes the2-4s speed miss but exceeds the unchanged10degree tilt
limit. Update344 passes2/4 short and1/2 long cases;352 passes0/4 and0/2.
Both are rejected;336 remains unchanged. Original uncommitted files and hashes
are backed up in `build/review-preservation/2026-11-04-before-startup-half-rate`.
Unrelated Godot editor changes and earlier smooth-exploration evidence are
preserved separately and remain outside this source checkpoint.
[HANDOFF.md](HANDOFF.md) describes an older campaign; the current state above
and asset provenance supersede its choice of baseline.
Reward5 separates roll/pitch and scores depth-dependent vertical settling from
the first control. Observation schema2 scales all three body-frame path-error
inputs by2mm; old checkpoints retain explicit72mm scaling. Checkpoint schema4
records that choice and rejects mismatched frozen evaluators.

## Run and inspect the creature

The current single Godot frontend loads the qualified saved research policy from
`res://research_data/learned-amplitude-v1.bin`, with its matching `viewer.json`
metadata. It starts paused; Play runs actual native physics with frozen inference.
Rest/Swim selection restarts the isolated task from rest, so switching is not a
braking test. The viewer has no optimizer or background learning worker. See
[the interface and verification guide](docs/POND_UI_REVIEW_2026-11-22.md).
It starts in a close aerial creature view; Side view changes the angle, and
scroll/pinch or + / - adjusts zoom. Title/status use an opaque high-contrast surface.

The earlier imitation-trained controllers and dedicated supervised training paths
were removed by user request. Missing or incompatible policies produce a usable
error; the app does not substitute random weights, a teacher or another algorithm.

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
Godot C++ bindings are required. Simulation, neural inference and learning are
native. Offline Python utilities package evidence metadata; they are not part of
the running creature or its learning system.

```powershell
cmake --preset windows
cmake --build --preset release
./build/Release/foundation_tests.exe gaussian:
./build/Release/foundation_tests.exe ppo:
./tools/run-ppo.ps1 -Name NEW_UNIQUE_NAME # build, zero-step checks, freeze inputs
# Its -Run switch executes that identity only after an explicit bounded approval.
```

The zero-physics frontend check runs rendered, because headless Godot uses a dummy
viewport here: `Godot_console.exe --path live --resolution 1280x720 --script
../tests/godot/pond_ui_smoke.gd`. It tests controls with native stepping disabled.
New physical or training runs require their own finite approved protocol; old
budgets remain closed.

## Architecture and limitations

One native world owner, one bridge and one scene: `live/main.tscn`. Its frontend
is `live/main.gd`; `live/pond_world.gd` only presents native body transforms.
HoldLabPreview owns frozen policy playback. Shared motor code lives
in src/nn/motor and physics in src/physics. Keep training independent of Godot.
Preserve model6, two universal joints/four axes and +/-10microN*m torque limits.
Calm holding from rest is qualified only within the declared original, withheld
and five-minute suites. Contact, moving-start braking, disturbance/current holding
and arbitrary transitions remain unvalidated.

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
