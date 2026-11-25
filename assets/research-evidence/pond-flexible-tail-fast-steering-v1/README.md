# Faster steering PPO — completed, no candidate adopted

The authorized step-one session completed all **128 updates with 1,024 CUDA
worlds**. It collected **8,388,608 actuator decisions in 97.963 seconds**
(85,630 decisions/s). The workflow finished in 104.271 seconds including its
recorded incremental build, baseline, resume verification and final evaluation.
The fitted gait, original update-eight offset, habitat and physics stayed exact.
Godot continues to load the original fitted gait and update-eight offset.

## Actual movement

These are 24 fixed cases, four moving phases in each direction at 15°,30°,60°.
Each case has 13 seconds turning, two seconds onward and two seconds of beat
closure. They are distinct from the older rigid-body 16-turn suite.

| Controller | 15° complete passes | 30° complete passes | 60° complete passes | Straight/from-rest retention |
|---|---:|---:|---:|---:|
| Original update-eight offset | 8/8 | 0/8 | 0/8 | 12/12 |
| Candidate update 8 | 3/8 | 2/8 | 0/8 | 12/12 |
| Candidate update 128 | 3/8 | 0/8 | 0/8 | 12/12 |

Update eight ranks first under the preregistered sharper-turn ordering, but its
two 30° passes are both right turns and it loses original full pass keys.
It is an experimental checkpoint. No candidate passed the all-case gate.

For 60° requests, the final candidate's mean actual COM course change in the
final two turn seconds increased **24.28° → 30.90°**; course error decreased
**35.72° → 29.10°**. Mean projected speed increased **8.29 → 8.85 mm/s**,
response delay decreased **1.79 → 1.42 seconds**, and effective complete-beat
radius decreased **318.74 → 235.99 mm**. The subsequent two-second onward
course change was 35.12°. All eight cases still underturned substantially.
Zero tilt/depth error and positive certified clearance were retained in these
calm planar fixtures; this does not qualify disturbed three-dimensional swimming.

Separate movement from rhythm criteria. Original/update-eight-candidate/final
candidate scored **8/24,11/24,8/24** on movement plus physical envelopes alone.
Final 15° cases all met movement and envelope requirements; five lost their
complete pass on the passive-beat frequency criterion. Early larger-bias
checkpoints also shrank root swing (at update eight's 60° cases, mean
0.964 → 0.427 rad). The final candidate recovered root swing to 1.003 rad,
but still did not turn far enough. Frozen gait weights do not imply identical
physical gait under a learned trim. The registered criteria were not changed.
The passive-frequency requirement was explicitly frozen for this study before
physics, alongside root frequency and both swings. Passive crossings averaged
5.21 Hz in the original 15° cases versus 0.91 Hz for the root; that passive
vibration count needs interpretation before treating it as a propulsion skill.
It does not explain the 60° failures: their actual COM course missed the target.

Half-second course feedback remains beat-oscillatory. For example, final-stage
15° traces can pass two-second course windows while half-second estimates leave
the 5° band repeatedly. Neither original nor candidate recorded a full second
continuously inside that instantaneous band. Therefore a blank completion-time
field is distinct from failing the declared final two-second motion windows.

## Learning and retention

The actor retained its historical four steering features and zero intercept.
Only four coefficients changed; both the inactive coefficients and their Adam
moments remained zero across all 129 checkpoints. The training-only critic used
36 observations and 32 hidden neurons. At update 128 explained variance was
0.99884 and value loss 0.0160. The critic fit its rollout targets well; that does
not establish good deterministic turning or remove possible task limitations.

The initial coefficient vector was
`[-0.08764146,0.008203539,0.043857753,0.030495565]`; the final vector was
`[-0.08417093,-0.09703398,0.000131105,-0.03619089]`.
All 16 scheduled assessments preserved 12/12 swimming cases, including all
eight starts from rest. Inactive commands and paired final native states matched
bit for bit. No new braking/holding experiment ran, and no current, disturbance,
repeated-sharp-turn or autonomous pond qualification follows from these results.

The next focused problem is steering feedback/control that preserves the
tail's useful motion. Complete-beat course estimation during steering and
phase-aware trim handling merit a separately bounded comparison. In particular,
the gait's phase observation uses raw root angle about zero while the trim
changes the commanded mean. That coupling is a source-level hypothesis for
the early swing loss, not an established physical cause. The traces
establish oscillatory feedback and trim-dependent gait changes; they do not
prove which correction will succeed. Do not repeat or extend this closed
session implicitly, or publish either experimental actor.

## Execution and finite accounting

| Work | Actual spend |
|---|---:|
| Main PPO | 128 updates, 8,388,608 sampled decisions |
| Main physics | 535,461,888 attempted and accepted substeps |
| Main Adam | 8,192 actor and 8,192 critic calls |
| Exact resume diagnostic | 3 executed updates, 196,608 decisions |
| Resume physics / Adam | 12,582,912 accepted substeps; 192 calls per optimizer |
| Baseline + 16 assessments | 1,224 trajectories, 15,014,400 accepted substeps |
| Total physical work | 563,059,200 attempted and accepted substeps |
| Numeric, native, clearance, posture failures | 0 |

Training issued 22,528 left and 22,528 right episode starts, with 44,032
completed horizons. Worlds and optimizer history continued across rollouts;
new training bodies occurred only at scored episode horizons. No evaluation
body was reset within its trajectory. No process timed out or restarted.
Maximum registered spend was 128 updates or 1,200 cooperative wall seconds.

GPU physics/observation took **90.285 s**, network inference **0.417 s**, and
PPO/Adam **2.821 s**. Training evidence writing took **3.992 s**. Across all 17
CPU evaluations, physics/scoring took **8.699 s** and writing **4.788 s**.
The CPU assessments overlapped GPU training. Host synchronization/readback
included waiting for physics and cannot be added to GPU time as an exclusive
cost. The final source rebuild before the recorded workflow took 9.704 seconds;
the workflow's unchanged incremental configure/build took 0.889/0.434 seconds.

## Reproduction and preserved evidence

The current authoritative study is preserved in
`build/flexible-tail/fast-steering-v1`; its 389.8 MiB originals remain untouched.
This compact archive contains frozen protocol/source and input identities,
all learning curves and actor exports, selected complete session checkpoints,
all assessment scores/beat records, baseline and update-eight/final control
traces, verification checkpoints and an independent saved-state audit. The
original-files manifest hashes every original artifact, including omitted
intermediate control CSVs and complete checkpoint history.

- [Registered protocol](data/protocol.md)
- [Training result](data/training/result.json) and [update curve](data/training/updates.csv)
- [All scheduled scores](data/learning-curve.csv)
- [Movement/rhythm breakdown](data/analysis/evaluator-offline-metrics.csv)
- [Exact resume result](data/verification/data/verification.json)
- [Independent 129-checkpoint audit](data/saved-data-audit/report.json)
- [Final assessment](data/evaluations/update-128/assessment/result.json)
- [Bound sources and inputs](data/manifest.json)

The source runner is `tools/pond-flexible-tail-steering/run-session.ps1`.
It builds once, verifies the frozen controller, runs baseline/full-state resume
checks, then trains and evaluates immutable checkpoints automatically. Closed
study replay must use its bound source snapshot and binaries; a new learning
study requires a new finite protocol and explicit authorization. The checkpoint
contains full actor/critic/Adam/world/RNG state and is bound to this task.
