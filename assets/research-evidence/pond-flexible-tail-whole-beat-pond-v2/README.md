# Earlier learned bank avoidance — closed, not admitted to Godot

The preceding whole-beat motor session reduced measured feedback noise and
produced checkpoint 56, which preserved all eight gentle-turn and all 12
swimming/from-rest pass keys. This conditional stage kept that motor frozen.
It used the same native feedback in the motor, collector and evaluator.

The preserved chooser first passed **6/12** fixed pond cases with the new motor
and original sensing. A separate native CUDA PPO chooser then completed all
**64 updates / 524,288 macro decisions in 522.101 seconds**, using 1,024 worlds
and longer bank sensing. The final policy passed **8/12**, with four clearance
stops. No candidate reached 12/12, so **zero endurance cases ran** and Godot's
working original motor was not replaced. Indefinite navigation is unproven.

## Fixed results and behavior

| Chooser state | Fixed navigation passes | Pooled cells | Clearance stops |
|---|---:|---:|---:|
| Preserved chooser / original sensing | 6/12 | 185/345 (53.62%) | 6 |
| New task update 8 | 7/12 | 200/345 (57.97%) | 5 |
| New task update 16 | 8/12 | 207/345 (60.00%) | 4 |
| New task update 32 | 8/12 | 205/345 (59.42%) | 4 |
| New task update 64 | 8/12 | 193/345 (55.94%) | 4 |

All eight scheduled checks completed the same 12 cases. None attempted the
four 300-second endurance cases because the navigation prerequisite failed.
The coverage denominator remains the fixed upright 72 mm-depth geometry
estimate, not the full three-dimensional navigable volume.

The final failures were the exact Godot rest start (80.95 s), center-west
(56.4825 s), northwest diagonal (24.02875 s) and southeast diagonal (22.88875 s).
All ended at the native whole-body clearance guard. Depth/tilt and speed stayed
healthy; there were no other native faults and maximum measured swim stall was
zero. Mean post-launch speeds across final cases were 10.57–11.16 mm/s.
A clearance stop is not a demonstrated resolved bank collision.

Decisions changed meaningfully. Across held-out cases, keep/left/right choices
went from **16/232/0 to 93/129/55**. The northeast/southwest-to-center cases
now finished 120 seconds with learned right turns. The remaining diagonals also
changed direction choices, but still ran out of clearance, in some cases sooner.
The final pooled perimeter time fell 20.48 → 12.06 seconds. These are separate
finite trajectories, not one uninterrupted pond lifetime or proof against
perimeter looping over arbitrary durations.

Attribution is limited: the frozen baseline used original sensing, whereas the
trained task used longer sensing. The 6→8 comparison combines new information
and PPO learning; it does not isolate the contribution of either alone.

## Task and preservation

The chooser retained its three learned actions and persistent heading interface.
Forward terrain distances were 40/200/400 mm and side distances 150/350 mm.
They expose native clearance information; no action was prescribed from those
readings. The actor, critic, Adam, world/visit/beat/servo state and RNG progress
were saved after every update. Only scored training episode terminals reset
training worlds. Evaluation trajectories had zero resets.

The original live gait, update-eight offset and native Godot library hashes
remain unchanged. Candidate motor 56 was also unchanged throughout this stage.
No Godot files, active model paths or Project Manager entry were replaced.
No additional sharp-turn experiment ran; the user's next request is a brainstorm.

## Actual execution

| Phase | Decisions | Attempted substeps | Accepted substeps |
|---|---:|---:|---:|
| Main 64-update chooser | 524,288 | 1,687,058,034 | 1,687,054,210 |
| Three-update resume proof | 24,576 | 82,668,240 | 82,667,732 |
| Frozen + eight held-out suites | — | 8,101,603 | 8,101,563 |
| Conditional stage total | — | **1,777,827,877** | **1,777,823,505** |

Warmup is included. The main attempted/accepted difference is exactly its
3,824 counted training clearance terminals. The diagnostic difference is 508
terminals and held-out suites account for 40. No numerical, pose, cue or native
integration fault occurred. Main Adam calls were 2,048 per optimizer; the
three-update proof added 96 per optimizer. Its complete continuous/restored
payloads matched byte for byte.

The stage stayed within 64 updates, 1,200 cooperative wall seconds and the
preregistered physical limits. Before any chooser physics, the protocol's
worst-case allocation was corrected to count episode warmup; the optimizer
and wall budgets did not change. The independent audit checked all 65 full
checkpoints, actor exports, model/mode metadata, optimizer ages and source/input
bindings with zero new physics or optimization.

GPU physics/motor time was **519.003 s**, chooser inference **0.078 s**, and
PPO/Adam **0.760 s**. Training evidence took **2.012 s**. All nine CPU evaluator
invocations totaled **6.704 s** including their writing, and overlapped training.
The complete stage workflow took **551.439 s**, including its exact-resume proof,
initial fixed gate and recorded incremental build. Host waits overlap GPU work.

One wrapper metadata-path error was fixed before any native executable launched;
its separate zero-physics/zero-optimizer receipt is retained. It did not restart
a physical study or rerun the completed proof. No native process timed out.

## Evidence

- [Frozen pond-stage protocol](data/protocol.md)
- [All fixed case results](data/learning-curve.csv)
- [Final training accounting](data/training/result.json)
- [Final assessment](data/evaluations/update-0064/assessment/summary.json)
- [Final case-level failures](data/evaluations/update-0064/assessment/cases.csv)
- [Resume proof](data/verification/data/verification.json)
- [Independent saved-state audit](data/saved-data-audit/report.json)
- [Model, source and mode bindings](data/manifest.json)
- [Original source and file identities](data/original-files.json)

Full originals remain at `build/flexible-tail/whole-beat-pond-v2-execution`.
The compact archive keeps all actor exports and fixed case scores, selected
control/full-state traces, the full resume proof and source snapshots. It does
not overwrite the earlier closed sessions or their checkpoints.
