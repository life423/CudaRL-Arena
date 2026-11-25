# Flexible-tail station keeping

The one-root flexible-tail prototype held a fixed world-space position in a
0.75 mm/s current while moving its tail. The flow was uniform and the head
started facing upstream. Positive root work and the matched powered-off
controls distinguish this from passive drift. All trials used intact native
body dynamics in the flat central pond and the same joint and torque limits.

The stop-to-hold diagnostic freezes the whole-body center-of-mass anchor at
the request and preserves the complete body state. It checks a four-second
braking allowance and the next eight simulated seconds. The hold window
requires 3D COM speed at most 0.5 mm/s, 3D distance to anchor and axial
excursion at most 2 mm, depth error at most 1 mm, and tilt at most 10°.
Whole-case lateral error is capped at 2 mm. Current confirmation uses a
four-second allowance and a sixteen-second hold window.

## Calm water

[Calm traces and controls](calm-brake-and-rest.csv) show zero drift and zero
motor work for a centered body at rest. Both ±0.75 mm/s initial-drift controls
wandered 5.84 mm from the anchor. From a swimming state with instantaneous
COM speed 9.94 mm/s, **0/28** brake-to-hold probes passed. Neutral root
targets ended at 2.28 mm/s and 58.31 mm from the anchor. A disconnected
moving root hit the joint-domain guard. All 24 mirrored tail flicks and
the two folded-tail tests failed to improve the anchor error over neutral.
The process completed in 0.671 s, excluding compilation.

## Gentle current

The [first comparison](current-stroke-sweep.csv) tested 45 powered strokes,
and 17 passed. It selected a 3 Hz yaw stroke with a 0.05 rad base amplitude
and 35 rad/m longitudinal feedback from world-space anchor error. The
native servo converts those requests to physical root torques. These are
test commands in the standalone executable.

The [20-second confirmation](current-hold-confirmation.csv) evaluated only
that frozen stroke. The mirrored case reverses the flow and initial heading
and begins 0.5 mm downstream of the anchor with a 0.25 mm/s downstream
velocity. Passive controls use the same initial states.

| Case | Peak anchor error | Peak 3D COM speed in hold | Pass |
| --- | ---: | ---: | ---: |
| Centered, powered | 0.58 mm | 0.197 mm/s | Yes |
| Centered, passive | 7.38 mm | 0.587 mm/s | No |
| Mirrored and drifting, powered | 1.11 mm | 0.293 mm/s | Yes |
| Mirrored and drifting, passive | 10.35 mm | 0.638 mm/s | No |
| Mirrored and drifting, powered, half timestep | 1.10 mm | 0.294 mm/s | Yes |

The confirmation took 0.582 s of process time including startup and evidence
writing. The current sweep took 0.564 s. All three local executions stayed
under the four-second cap. Powered current cases stayed upright at the
requested depth with at least 68 mm of certified center-water clearance.
The half step compares 0.625 ms against 1.25 ms on the CPU. CUDA parity and
PPO training for this body are still pending.

The separate [Godot recording](current-station.json) shows the mirrored
current case, including the native tail shape, COM path, speed and hold
target. The regular pond project loads its existing motor and controller.
The rebuilt diagnostic also reproduced the archived forward result and all
four ±15° and ±30° moving-turn cases with a byte-identical case CSV.
Braking from full swimming, arbitrary flow directions and stronger currents
remain open.

## Watch the recorded tests

In Godot Project Manager, select **Tadpole Pond | Flexible Tail** and click
**Run**. Choose **Swim**, **Turn**, or **Swim in place** in the preview. If the entry
is missing, import [project.godot](../../../flexible-tail-preview/project.godot)
from the worktree. Each clip is a recording of native diagnostic physics.

To open the specific current case directly from the worktree, use the
following command.

```powershell
$trace = (Resolve-Path assets/research-evidence/pond-flexible-tail-hold-v1/current-station.json).Path
& 'C:/Users/aiand/Godot/Godot_console.exe' --path flexible-tail-preview -- "--flex-tail-trace=$trace"
```

## Reproduce the confirmations

Run the confirmation from the minimal-foundation worktree with a fresh
output folder.

```powershell
cmake --build build/flexible-tail/native --config Release
& build/flexible-tail/native/Release/pond_flexible_tail_check.exe build/flexible-tail/current-check-new --verify-current-hold
```
