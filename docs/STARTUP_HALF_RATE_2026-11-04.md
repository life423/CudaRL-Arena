# Startup continuation: half actor rate, 2026-11-04

Status: protocol frozen before preparation or physical execution. User approved
proceeding after the read-only review. This is one finite study; no retries,
extensions, additional seeds, online runtime adoption, push or merge.

Preservation commit b3addfb contains the prior reward7 code and rejected trial.
All 40 previously dirty/untracked files were copied and SHA256 verified under
build/review-preservation/2026-11-04-before-startup-half-rate. Unrelated Godot
edits and older smooth-exploration evidence remain untouched outside that commit.

## Evidence and single change

Baseline336 passes4/4 short swims and holds depth/lane/tilt in two300s swims.
It fails only the2-4s startup-speed windows in those long cases. The rejected
reward7 trial at actor rate0.00025 fixes startup speed but reaches10.3255degrees
at update344 and10.5756/10.9606degrees at352, exceeding10degrees. Its first
same-batch KL was0.00418 and clip fraction0.0406. A smaller actor step is a
focused hypothesis to retain acceleration gains with less posture displacement;
it is not a guarantee and does not alter the tilt criterion.

The only treatment change relative to that rejected reward7 trial is actor
learning rate0.00025 ->0.000125. Restore baseline336, both networks, both Adam
moments/counters, fixed sigma/correlation and seed. Change only the actor learning
rate after restore. Keep critic rate0.001, reward7 (horizontal speed charged from
first control), all other rewards and masks, observations, torque limits, filters,
model6 body/water physics, starts, action mapping and numerical checks unchanged.
Protocol14 already records actor rate and optimizer configs; schema4 unchanged.
The smaller rate is accepted by CLI only in explicit startup-study mode.

## Fixed budget and execution

Run identity: build/ppo/startup-speed-r7-half-rate-336to352-v1.
Source checkpoint: assets/ppo-four-axis-checkpoints/update-336-episode64/ppo-update-336.bin.
SHA256:810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
Source protocol SHA256:FCAA74C5DA6ADC1E000282776CEBFC51F007B00633E7A6A29ED9D8A34E485E1E.

One native CUDA PPO run:16 additional updates336->352,32worlds x1280controls,
64s true episodes,0.05s controls/0.005s physics. Training655360controls,
6553600substeps; at most5120 additional Adam steps per network. Seed1247037243,
rho0.8, log_sigma[-1,-1,-1,-1], actor/critic40->32->32->4/1, two epochs,
minibatch256, gamma0.99/lambda1. Fixed-target starts retain original alternating
pose0/3 with balanced0,+/-0.5mm depth offsets and global-update phase.
Training arena1.2x0.8m. Resume starts fresh true episodes, not mid-rollout.

Use one uniquely prepared directory and frozen executable/source/input hashes;
write run-started before launch. Maximum combined controls686144 and
substeps6861440 includes64 axis checks, baseline4x8s, two candidate4x8s,
two candidate2x60s, and at most two candidate2x300s assessments. These maxima
are shared across CUDA training and CPU frozen evaluation; all attempted
controls/substeps and optimizer updates count. Native wall cap90s, own process
cap75s; stop first safety/nonfinite/numerical failure without retry or tuning.

Before launch: build PPO/tests, focused nonphysical startup/checkpoint/observation
checks with actual CUDA required, synthetic CUDA reward verification, zero-step
resume readiness; verify parent hashes and preserved source. Rejected CLI checks
must not create a run or advance physics. Record actual launch/training timestamps.

## Evaluation and selection frozen in advance

Save/evaluate updates344 and352. Frozen CPU evaluation uses the same physical
equations; no optimizer updates during evaluation. Compare each with unchanged
baseline336 and the previous rejected rate0.00025 trial.

Four8s short cases; two60s long cases from poses0/3. Gates unchanged:
speed7.2-8.8mm/s,3DvelocityRMSE<=2.5mm/s,lane<=2mm,depth<=1mm,
combined tilt<=10degrees,path<=5degrees,work<=0.001J/m,clearance>=10mm.
Keep original startup scoring and cumulative tilt/path bookkeeping. Every long
window must pass, including2-4s. No revised thresholds or discarded startup.

Only a candidate passing4/4short and2/2long receives its two300s frozen cases
in6x0.8m arena, without wrapping/reset/recentering, same gates and startup.
Record every candidate and failure. A qualifying startup milestone requires
4/4short,2/2long,150/150endurance windows and short meanRMSE better than336;
if both qualify retain the earlier344 as the smaller completed change.
No automatic viewer installation. Depth recovery at32/64s is not retested in
this study and cannot be claimed newly validated. Stopping remains rejected;
revisit its design only after the startup result, without launching another run
under this budget. Finite two-start300s results never prove indefinite behavior.

## Completed result

Training actually started2026-11-04T16:31:55.684Z. First completed update337
was recorded16:31:57.513Z. Process ended16:32:21.630Z,exit0,26.265s wall.
Completed16updates,655360training+6784evaluation=662144total controls,
6621440physical substeps and5120additional Adam steps per network. No safety or
numerical stops; no retries, extensions or300s evaluations. All frozen inputs
and baseline336 hashes remained unchanged. Candidate hashes, actual execution,
checks and comparisons are in the durable rejected-asset provenance.

| Candidate | Short passes | Long passes/windows |2-4s speed poses0/3 | Peak tilt poses0/3 |
|---|---|---|---|---|
| Baseline336 |4/4 |0/2,28/30 |7.0260/7.1162mm/s |8.8279/9.3577deg |
| Half-rate344 |2/4 |1/2,15/30 |7.2402/7.3130mm/s |9.5950/10.0813deg |
| Half-rate352 |0/4 |0/2,0/30 |7.3598/7.4078mm/s |10.2432/10.6386deg |

Both new candidates pass the startup-speed windows but fail tilt. At344,
poses1/3 fail only tilt; at352 all four short poses fail only tilt. The first
60s-case tilt exits are0.65s for344pose3 and0.70/0.65s for352poses0/3.
Cumulative peak bookkeeping means an early excursion fails later windows even
after posture recovers. Depth/lane remain within limits across all four60s
cases. Maximum depth errors at344 are0.3708/0.3473mm and lane0.7158/0.7136mm;
at352 depth0.3199/0.2809mm and lane0.9767/0.8442mm.

The first update's same-batch KL_k3 drops0.0041765->0.0029891 and clip fraction
0.040576->0.018628 relative to the prior rate. Initial return mean/std match
exactly. Restored actor/critic parameters, both Adam moments/counters and critic
config match336 exactly; only actor rate differs. This supports that the smaller
step was exercised, but it did not solve the startup/posture tradeoff.

Validation: native PPO/test build passes,15/15 focused checks with0skips,
actual CUDA reset/observation checks and4synthetic reward cases pass; two CLI
scope rejections create no run; both durable candidates reload at344/352 with
24260/26820Adam steps each and zero physical/optimizer steps. Full64s GPU reward
verification includes20480post32s rows and126late minibatch rows with nonzero
policy gradients; maximum GAE discrepancy7.07e-7. No new32/64s recovery trials.

Decision: reject both; retain336. The frozen study is closed. Stopping remains
unresolved and no stopping trial was launched. The next design review should
address startup acceleration versus combined tilt, rather than another implicit
rate sweep. Saved traces lack signed head roll/pitch and instantaneous joint
torques, so do not claim which tilt component causes the excursion. Add those
diagnostics to a future separately bounded evaluation and consider aligning its
training posture objective with the unchanged combined-tilt gate. No additional
training, viewer replacement, push or merge is part of this completed study.