# Approved balanced holding reward-scale comparison, 2026-11-04

Identity: `build/ppo/balanced-hold-scale-336to344-v1`. Branch:
`codex/minimal-foundation`. Approved paired comparison, no adoption, push or merge.
Original checkpoint 336 SHA256:
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
Preserve every earlier checkpoint and experiment; unrelated Godot/exploration
files backed up and hashed before implementation. The prior budgets stay closed.

## Fixed intervention and controls

Two independent copies of migrated336; no continuation from344. Original40
observations retained, appended signed error `tanh(forward_error/.02m)`; new
actor/critic first-layer columns and moments zero, new age0, old age21700.
No critic reset, warmup, hidden optimizer step or change to advantage normalization.
All four learned joint controls, model6 water/body physics, torque10microNm,
filters, original lane/heading/depth reference and one-time32s anchor unchanged.

Both arms use32 worlds x1280controls,8 updates336->344. Pose=i%2?3:0.
Hold episode=(i/2)%2==1. Each pose/goal has8 worlds/update. Depth offsets retain
the original `(i/2+update%3)%3` center,-.5,+.5mm mixture. Each pose/goal/depth has
21 or22 episodes across8updates. Every start is at rest, goal8mm/s; swim-only
episodes retain8 for64s, stopping episodes request0 at32s and hold through64s.
No hold-from-rest curriculum. Both arms use the same counter-keyed random draws.

ArmA holds gain1. ArmB multiplies the entire summed reward and all5components
by exactly1/1024 only for the active zero-speed phase, control641 onward in
hold episodes. Swim and pre-stop rewards unchanged. All internal weights within
the holding cost stay unchanged. Frozen physical assessments report unscaled
costs for comparability; their rewards never select actions or train a policy.
The distinct arm protocols/checksums bind gain and balanced mix to checkpoints.

Seed1247037243, sigma exp(-1), rho.8; actor rate.000125, critic.001; inherited
Adam configs/moments/old clocks retained; independent new-column clock. Gamma.99,
lambda1, clip.2, gradientclip.5, eps1e-5, minibatch256, at most2epochs/update,
native epoch KL stop>.01. Exactly8 PPO updates per arm absent hard failure.

## Integrity gates

Nonphysical preparation checks exact parameter/moment migration, checkpoint
roundtrip/rejection, signed observation/output fixtures, independent Adam age
reference and CPU/CUDA goal/gain boundary probes. Synthetic Adam fixture is1
step/network/arm, not a trained-policy update and integrates no physics.
Build both legacy trainers and new entry with warnings treated as errors.
Focused CPU/CUDA tests must run on an actual CUDA device, no skipped GPU pass.

Run each arm's64control CPU/CUDA axis check and baseline assessments, collect
and cache the first40960row training rollout for both arms before any PPO update.
Compare ALL observations, histories, sampled actions, critic values, old logp,
dones and final physical state fields; require exact equality. Observations
across every control are the trajectory evidence; no extra simulation replay.
Check every actual reward/component: B=A before holding; B=A/1024 during holding.
Exactly10240 active-holding rows per first batch. The cached batch becomes
update337's training batch, so these controls count once, not twice.

On actual first-rollout states verify original40/new41 CPU actor/critic errors
<=1e-6 and CPU/CUDA <=2e-5; synthetic fixtures verify unchanged first40 meanings.
Actual reset checks retain fixed -80mm depth, offsets and request references.
Every training batch checks actual goals/counts, terminal-only-at1280, component
sum, independent GAE <=.002+.0002*abs(return), old-logp rescore<1e-4, and complete
permutation coverage including all10240 active-holding rows/epoch. Existing
first optimizer-batch check verifies actual gathered returns and nonzero late
policy gradients. Optimizer ages must match actual minibatch steps. Final
checkpoints reload with exact network parameters and both saved ages.

Save all16 rollout batches,18 numbered checkpoints, initial migrated checkpoints,
source/executable/input hashes, reset logs, per-update PPO diagnostics and
per-pose/goal/pre-post32s statistics plus pooled active-holding diagnostics.
Those include return moments, value bias/halfMSE/EV, variance-normalized MSE,
normalized advantage moments/sign fractions, and5component integrated costs.

## Physical assessments and decisions

Before/after each arm:4 original8s swims,6 stop64s tests(two poses x3depth
offsets),2 long60s swims. No intermediate physical candidate assessment.
Short gates:7.2..8.8mm/s,3DvelocityRMSE<=2.5mm/s,lane<=2mm,depth<=1mm,
tilt<=10degrees,path<=5degrees,work<=.001J/m. Need4/4short and retention of
all28 baseline-passing4..60s long windows. Report full30window/formalcase
results separately; baseline's two2..4s startup misses are not full acceptance.

All6stops must pass:8mm/s0..32s thenzero32..64s; original pre-stop2..32s
speed/RMSE/path/work criteria. Actual3DCOMspeed<=.5mm/s at EVERY50ms sample
48..64s. Fixed-anchor3Derror<=2mm throughout56..64s, forward excursion from56s
<=2mm. Whole64s depth<=1mm,lane<=2mm,combined head tilt<=10degrees. The existing
evaluator now includes the exact56s endpoint in its anchor check, matching this
already-approved criterion; no tolerance change. Record sustained ranges,
RMSspeed, drift/overshoot/reversal, first1s low-speed streak/permanent stop,
signed depth drift, roll/pitch and all4joint commands/rates/torques. Joints may
move while holding; no posture clamp or scripted assistance.

Usable bounded candidate requires holding6/6 plus swimming regression gates.
No automatic installation even on success; finite checks never prove indefinite
performance. Reward-conditioning diagnostic passes only if B's update344
active-holding variance-normalized MSE <=.75*A and EVafter>=A+.20, with finite
defined variance>1e-8. These are experimental decision thresholds, not physical
laws; diagnostics cannot substitute for physical acceptance.

## Budget and stop rules

Per arm:327680training+64axis+2*10720assessment=349184controls,
3491840substeps. Combined698368controls/6983680substeps. Exactly16PPOupdates
total, at most2560Adam steps/network/arm; record actual early-stop counts.
At most180 native seconds/arm accumulated across its phases; own-process monitor
enforces each accumulated arm cap and360s combined. Build/config cap300s and
zero-step checks/preparation cap60s. No retry, extension, extra seed, tolerance
relaxation, curriculum/input/physics change or reuse of unused budget.
First integrity/nonfinite/numerical/physics fault stops entire study, preserving
all partial evidence. Behavioral gate failure is a reported result and permits
finishing the originally specified eight updates while physics remains valid.

Both first rollouts are collected before optimizing A, then run A's eight
updates/final assessments followed by B's eight/final assessments, consuming
the cached first batches once. This changes scheduling only, not samples,
updates, seeds or learning rules. All native time including diagnostics/I/O is
charged to its arm; the combined process cap also covers orchestration overhead.

This is one matched seed and a small trial. A/B tests gain under the same
balanced mix. Comparisons against the old confounded pilot are historical,
not a full2x2 factorial test. Reward/value mismatch, input saturation, missing
time/history in the critic and limited hold-start coverage remain hypotheses.
