# Phase-aware steering — measured 30-degree improvement, not adopted

The authorized session completed128 GPU PPO updates with1,024 worlds and
8,388,608 sampled actuator decisions in99.663seconds (99.722seconds externally).
The complete runner took107.314seconds, excluding the initial source compilation
performed before its incremental build. Original Godot models, motor56, physics,
torque limits, trim magnitude/slew and historical evidence are unchanged.

## Coherent implementation

One existing motor policy is extended, using the same CUDA rollout, MLP/GEMM,
PPO and Adam path. `flexible_tail_steering_policy.h` is the single authority for
feature schema, model specification, trainable mask, validation and serialization.
Four original steering coefficients stay frozen. Nine initially zero coefficients
learn from actual root-joint phase, full-angle whole-body course error and COM
speed, including phase/error interactions. The original fitted gait stays frozen.
The initial phase model reproduces the protected motor at all tested angles;
there is no pretraining large-angle gain increase as in the previous trial.

FTYO2 explicitly binds the phase layout and frozen base. Old FTYO1 loaders reject
it. FTSSv4 saves actor, critic, optimizer, body/servo/history, sampling and session
identity together. CPU inference and CUDA collection use the same observer.

## Actual fixed-test results

Training practised balanced +/-30-degree requests only. Each scheduled assessment
tested four held-out phases, both signs and15/30/60/90 degrees, plus12 swimming
cases. All course, speed, posture, clearance and complete-beat gates stayed fixed.

| Policy | Complete15 | Complete30 | Complete60 | Complete90 | Straight/from-rest |
|---|---:|---:|---:|---:|---:|
| Protected whole-beat56 / initial phase model |8/8|0/8|0/8|0/8|12/12|
| Candidate16 |7/8|5/8|0/8|0/8|12/12|
| Candidate112 |3/8|6/8|0/8|0/8|12/12|
| Candidate128 |4/8|5/8|0/8|0/8|12/12|

No candidate meets the declared lesson gate of8/8 complete30,8/8 complete15
and12/12 swimming. Update8 retains all gentle keys but passes no30 case.
Updates64 onward meet all eight30-degree motion/posture cases; rhythm failures
remain and are not waived. Candidate112 is the best30-progress checkpoint, not
a qualified replacement. No60/90 case passes; those angles were not trained.

At update112, final-turn mean COM course error fell from5.89 to1.44degrees on
left30 requests and5.54 to1.92degrees on right30 requests. Projected speed rose
from about11.60 to13.21mm/s. Onward errors were0.42/1.50degrees. Minimum passive
beat-frequency ratios were0.627/0.677 in the late-turn windows against the fixed
0.80 threshold. Thus movement improved while some rhythm keys regressed.
The frozen gait's weights alone do not imply an unchanged executed stroke.

All12 straight/from-rest cases retain exact inactive commands and paired final
states. These include8 from-rest cases, not a fresh station-holding qualification.
Tilt/depth errors were zero in these calm, submerged planar fixtures; this is
not evidence for general disturbed posture/depth recovery. Repeated-turn, bank,
current and endurance tests did not run for this new schema. Godot remains on
its protected live motor and autonomous pond exploration remains unqualified.

## Accounting and verification

Main training used535,461,888 accepted/attempted physics substeps and8,192 Adam
calls per optimizer, with22,528 starts per turn sign and zero physical/numerical
faults. Full-state resume verification executed3 updates,196,608 samples,
12,582,912 substeps and192 Adam calls per optimizer. Baseline plus16 evaluations
used18,713,600 substeps across1,496 trajectories. Total566,758,400 substeps.
Zero-physics feature/model checks add no simulation or optimizer work.

The initial check passed3,072 exact actor-mean and27,648 exact decoded-command
comparisons across the tested angle range. Mirrored new features matched exactly;
full-angle numerical error was5.96e-8. CPU/CUDA feature checks stayed below1e-6
and complete resume payloads matched exactly. Checkpoint audits are archived below.

GPU physics/observation took91.000s, inference0.407s and PPO/Adam2.756s.
Training evidence writing took5.035s. CPU assessments summed11.671s physics/
scoring and10.732s evidence writing, overlapping training. Host waits overlap GPU
execution and are not additive. Main throughput was84,169 actuator decisions/s.
The runner's configure/incremental-build took0.868/0.783s; initial compilation
is outside that workflow measurement. See receipts for exact process boundaries.

The next proposed step is retain gentle turns and complete-beat rhythm while
learning30-degree steering, using mixed-angle practice and an explicitly defined
rhythm objective. This session does not isolate which individual new feature
caused improvement. No follow-up run or automatic publication occurred.

## Preserved evidence

- [Frozen protocol](data/protocol.md)
- [Initial equivalence and schema checks](data/equivalence/data/report.json)
- [Learning curve and movement/rhythm classification](data/analysis/learning-curve.csv)
- [Phase contribution and metrics](data/analysis/turn-analysis.json)
- [GPU accounting](data/training/result.json)
- [Exact resume and CPU/CUDA agreement](data/verification/data/verification.json)
- [Saved-data audit](data/saved-data-audit/report.json)
- [Candidate112 results](data/evaluations/update-112/assessment/result.json)
- [Final results](data/evaluations/update-128/assessment/result.json)

All originals remain at `build/flexible-tail/phase-aware-v4`. The archive keeps
all actor exports, assessment score/beat records, selected complete states and
control traces, frozen sources and file identities. No checkpoint was overwritten,
and no Godot publication, physics change, commit or push occurred in this study.
