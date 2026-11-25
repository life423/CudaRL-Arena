# Sustained native PPO session

The workflow completed 64 CUDA PPO updates on 1,024 worlds in one native
process. It collected 524,288 genuine navigation decisions in 562.309 s,
averaging 932.4 decisions/s and 3.06 million accepted physics substeps/s.
There were 1,720,333,558 accepted physical substeps, 2,048 actor Adam
steps and 2,048 critic Adam steps.

The final chooser passed two of the twelve 120-second cases. The best
scheduled checkpoint, update 56, passed three. No candidate passed the
full navigation gate, so the endurance prerequisite remained closed.
Godot continues to show the verified live manual motor.

## Workflow delivered

- A single sustained entry command owns configuration, build, training,
  periodic independent evaluation, source manifests and receipts.
- The native session uses 1,024 CUDA worlds and eight decisions per
  world per rollout, with GPU-resident collection, neural inference,
  PPO, GAE and Adam.
- Full atomic checkpoints preserve actor, critic, both Adam states,
  all worlds, applied trim, heading and course history, visitation
  state, episode counters and sampling progress.
- Immutable per-update exports feed CPU evaluations while CUDA
  collection continues. The evaluator never mutates training worlds.
- The workflow records policy and value loss, entropy, clipping, KL,
  explained variance, movement, coverage, failures and execution timing.
- Resume rejects a changed frozen motor, task, request interface or
  learning configuration. A session has a single writer.

The initial actor was imported once from the previous bank-approach
candidate. The legacy file contained the actor alone. All subsequent
updates used one continuing critic, Adam state and population.

The registered first session was bounded at 64 updates and a 1,800 s
cooperative budget. It completed every update. The current command's
64-update ceiling is an administrative limitation for this registered
session. Extending beyond that ceiling requires the explicit budget
revision and a compatible checkpoint migration,
while preserving all learning state. The full final checkpoint remains
available for that continuation.

## Steering and restart verification

The selected persistent-heading interface increased a turn request
against the held heading. It produced actual 60-degree course changes
in all 16 matched left/right cases, compared with zero under the old
measured-course interface. Mean effective radius fell from 426.7 to
329.9 mm. Eight trajectories in each diagnostic mode eventually lost
native bank clearance under the fixed diagnostic schedule.

Before training, a 1,024-world verification executed three actual PPO
updates. The uninterrupted two-update state matched an update-one
save/load followed by update two byte for byte. It compared actor,
critic, both optimizers, all worlds, counters and sampling progress.
That verification accepted 84,784,155 physical substeps and took
26.897 s. Its complete update-two checkpoint also matches this main
session's update-two checkpoint exactly.

An independent parser audited checkpoint 64 and found all 1,024 worlds
and complete Adam state. The actor matched the immutable actor export.
A separate process then restored the completed session through the
normal resume command in 0.305 s, with zero additional policy updates.

## Scheduled navigation measurements

| Update | Full passes / 12 | Median duration | Pooled coverage |
| --- | ---: | ---: | ---: |
| 8 | 0 | 52.12 s | 32.5% |
| 16 | 1 | 66.40 s | 42.9% |
| 24 | 0 | 53.07 s | 34.2% |
| 32 | 0 | 60.64 s | 40.3% |
| 40 | 1 | 52.30 s | 39.4% |
| 48 | 1 | 54.07 s | 38.3% |
| 56 | 3 | 67.10 s | 43.8% |
| 64 | 2 | 62.25 s | 41.4% |

All eight evaluations together accepted 4,640,517 of 4,640,605 attempted
substeps. They recorded every case with zero resets and optimizer calls.
Final passing cases were the north and south shelf tangents. The exact
Godot rest start stopped at 70.19 s. The final mean swimming speeds
were 9.86–10.20 mm/s, with zero measured tilt or depth error.

Coverage uses the existing 345-cell, fixed-posture estimate at the
Godot start's 72 mm depth. Held-out depth requests span 72–89 mm.
The percentages describe pooled coverage of that reference map.
They do not establish whole-volume exploration.

## Remaining control work

The frozen steering network divides course error by 5 degrees and
clips that observation at 20 degrees. Its dominant coefficient is
−0.08764146. The saturated course-feedback term produces only about
0.077 requested trim with aligned facing and zero yaw rate and prior
trim, despite a decoder limit of 0.25. The steering diagnostic used
about 0.104 trim in motion. Faster response has to be physically
measured before a stronger candidate can be accepted.

Repeated persistent requests can also wind the target past the actual
course and reverse the wrapped signed error after 180 degrees. The
chooser needs an effective way to manage that demand. Current
directional probes reach 130 mm. At roughly 10 mm/s, this represents
about 13 seconds of straight-line travel, while the diagnostic took
about 34 seconds for a 60-degree course change.

Use measured COM speed and course rate to estimate radius and maneuver
time, then choose a warning horizon with whole-body room and an explicit
margin. For approximately steady speed and rate, radius is v/abs(omega)
and maneuver time is angle/abs(omega). The native trajectories remain
the source of truth as speed and course rate vary.

The fitted gait's pitch readout is zero in these symmetric calm-water
fixtures. The zero tilt and depth results cover those tested conditions.
Current flow, disturbed posture, physical bank contact recovery, and
moving-to-hold skill remain outside this navigation qualification.

## Timing interpretation

GPU physics and frozen-motor work used 559.533 s, chooser inference
0.072 s, optimization 0.734 s and checkpoint evidence writing about
1.645 s. The host synchronization/readback measurement includes
waiting for GPU work and overlaps the physics interval. It cannot be
added as an independent transfer cost. Total elapsed process time is
the authoritative wall measurement.

[Workflow](../../../tools/pond-flexible-tail-chooser/README.md) ·
[Session protocol](data/protocol.md) · [Learning curve](data/training/updates.csv) ·
[Evaluation curve](data/evaluation-curve.csv) · [Final full checkpoint](data/training/latest.checkpoint) ·
[Resume verification](data/verification/verification.json)

