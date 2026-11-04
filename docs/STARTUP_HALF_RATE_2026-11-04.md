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