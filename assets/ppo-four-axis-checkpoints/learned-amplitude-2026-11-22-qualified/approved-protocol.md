# Learned persistent amplitude — frozen execution protocol, November 22

The user explicitly authorized the proposed learnable-amplitude and matched-duration
holding change, requesting substantial steps toward a working result. This is one
new finite native C++/CUDA PPO study. Earlier runs stay closed. Preserve the original
checkout, all previous policies, full saves, ignored evidence and current viewer.
No merge, publication, deployment, morphology/current/contact change or unrelated
training. The native research viewer may inspect the resulting saved candidate
under the user's existing request to see it; this is separate from production adoption.

## Change being tested

Previous quiet trials mostly relied on a random amplitude outside the learned
policy, and training ended after one or three seconds. This study makes the
persistent amplitude the sampled policy decision and trains the full 64-second
holding episode. Reward credits that one decision once. Neither a per-control
likelihood repeated 1,280 times nor a baseline conditioned on the sampled amplitude
is valid for this design.

Keep canonical checkpoint 336's frozen MLP, verified zero-column 41-input migration
and complete original optimizer state unchanged. Parent SHA-256:
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
The four physical joint outputs still come from that same body motor. For a holding
request, multiply its four means by one learned persistent amplitude; swimming
requests use the original computation directly. No separate behavioral network,
teacher, neutral action label, scripted recovery or global propulsion is added.

This is restricted policy-parameter learning: it teaches suppression of an inherited
gait for calm holding from rest, with swimming structurally preserved. It is not
proof of a fully trainable shared network retaining skills, a general motor, braking
from motion, disturbance/current recovery or arbitrary mode switching.

## Policy and exact likelihood

Learn two fresh parameters: a Gaussian mean `mu` and a raw variance coordinate `v`.
Define `sigma = 0.05 + 7.95*sigmoid(v)`. Initialize `mu=logit(0.99)` and `v` so
`sigma=6`. Initial deterministic holding amplitude is 0.99; its failure must be
measured, not replaced with a neutral initialization. The variance bounds are a
smooth policy parameterization, not an optimizer projection or physical rescue.

At each complete episode reset independently draw exactly one
`u ~ Normal(mu,sigma)` using the fixed counter-based seeded native RNG. Calibration
uses seed 314159265 and stream/index 0/world; fresh PPO batches use seed
`0x6192a781`, stream=the one-based PPO update, index=world. These identifiers are
fixed before collection; there is no seed search or replay after a failed draw. Apply
`g=sigmoid(u)` throughout all 1,280 controls. At each control, evaluate the frozen
MLP on actual native observations and apply four latent joint targets
`z[j]=g*b[j](x)`, then the unchanged tanh/physical target filter/servo. There is
**no additional per-control Gaussian or AR action noise** in the holding lesson.
The learned stochastic amplitude is itself the exploration. No gain-bin or
successful-episode resampling is allowed. The scalar gain is positive and bounded
by one, avoiding unbounded amplification or reversed feedback. Floating-point
rounding may yield exactly one; score stored `u`, never invert rounded `g`.

PPO stores and scores the original Gaussian latent `u` once per episode. Its
Gaussian log probability is the policy density; the deterministic sigmoid and
downstream physical motor are the action transform. No additional sigmoid Jacobian
is inserted into a latent-action PPO ratio. For independent verification:

- `d logp/d mu = (u-mu)/sigma^2`;
- `d logp/d v = (((u-mu)^2/sigma^2)-1) * 7.95*s*(1-s)/sigma`, `s=sigmoid(v)`.

Fresh native GPU Adam optimizes only these two parameters, initially age zero:
learning rate 0.05, existing beta/epsilon settings, global gradient norm cap 0.5,
PPO clip 0.2, zero entropy bonus/critic loss, minibatches of 64 episodes and at
most four epochs per update. Check epoch KL against 0.01; stop further epochs
if exceeded. Maximum 16 Adam steps per update, 512 in the study. Do not reset
the old MLP optimizer clock or mislabel it as the new head's optimizer.

Deterministic tests and the viewer use `g=sigmoid(learned mu)`, with no sampling.
They **retain** the learned amplitude; they must not reset it to one or force
it to zero. Store both policy parameters and the derived amplitude explicitly.

## Physical task and return

Same Model 6 body/water equations, two universal joints/four axes, +/-10 microN*m
torque caps, 1 Hz target filter, ten 5 ms substeps per 50 ms control, observations,
fixed anchor and original physical reward components. The goal chooser is excluded.

Each update has **256 holding worlds**, all holding, no accidental half-swimming
mixture. Each independent world completes one uninterrupted 64-second/1,280-control
episode. Case `world%6` selects original pose 0/3 crossed with depth offset
0/-0.5/+0.5 mm in the established case order. Four case groups contain 43 worlds,
two contain 42. Start/filter/joint/AR reset semantics and fixed reference depth
remain exact. No intermediate reset, teleport, changing anchor or time-feature reset.
There is one true episode terminal, done=1/truncated=0, at control 1,280.

Episode return is `R=sum(original per-control holding reward including 1/1024)/64`.
This is an undiscounted whole-episode mean cost-rate objective: no per-tick GAE,
discount bootstrap, reward-component change or successful-example imitation.
Store all original per-control rewards so the sum can be independently checked.

For episode `i`, the current-batch baseline is the leave-one-out mean return of
**other** episodes in the same initial pose/depth case. Require group size >=2.
Exclude the episode's own return and never condition this baseline on `u`, `g`,
an amplitude bin or any resulting trajectory property. Freeze these advantages
for the PPO epochs and do not differentiate the baseline. Normalize by one
positive global within-case residual RMS fitted on the independent calibration
batch and kept fixed. It is not six different case weights or a current-batch
scale fitted on the episode's own action.

## Preparation and data readiness

Use new identity `build/ppo/learned-amplitude-2026-11-22-v1`; refuse an existing run.
Freeze protocol, executable, transitive sources, linked native libraries, parent,
migrated weights and initial policy/Adam states by checksums and SHA-256 before
physical collection. Preserve exact original bytes before any necessary source edit.
No physical or optimizer step in preparation fixtures.

Preflight CPU-double/actual-CUDA checks cover Gaussian samples/densities, sigmoid
variance chain, both parameter gradients and finite differences, same-policy
ratios one, episode-score counted once, leave-one-out exclusion, case counts,
all-world holding setup and no intermediate resets, unchanged swimming/backbone,
and explicit new checkpoint/optimizer round trips. Any old/new critic forward
comparison in migration preparation is separate from collection, which has no critic.

Phase A, with zero learning:

1. Original 336 assessment at gain one: 4 short swims, 6 full 64-second holds,
   2 long swims, 10,720 controls. Require original 4 short passes and exact 28
   passing long-window keys, and record original failed holding.
2. Six full deterministic initial-policy holds at amplitude 0.99: 7,680 controls.
   Require 0/6 full passes to establish that holding was not installed initially.
3. One independent calibration batch: 256 complete holding episodes, 327,680
   controls. Use fixed native calibration seed 314159265, separate from PPO draws.

Calibration gate: at least four complete 64-second physically passing holding
episodes, including at least one for each original pose. Every endpoint, the
initial sample and the final-eight-second excursion count. Require all values,
returns, density rescores and the positive global scale finite; episode return
absolute p99 <=5, max <=20; stored latent old-logp error <1e-4 and independent
return sums within existing floating-point tolerance. No calibration optimizer,
gain-dependent baseline, manual amplitude assignment, retry or reseed. Report the
measured coverage and close phase A before launching training.

Existing user authorization permits the separate training launch if all frozen
gates pass. The first fresh PPO batch repeats the coverage, integrity and return
scale gates before its first optimizer step. Calibration data are never optimized.

## Training, acceptance and withheld validation

At most 32 PPO updates, each 256 fresh complete episodes. Save actual sampled
latents, old likelihoods, parameters, amplitudes, complete returns, grouped
baselines/advantages, physical counters and relevant control observations/actions.
Save complete two-parameter optimizer state after every update and preserve the
full unchanged inherited weights/optimizer. No Python/PyTorch training runtime.

After **every** completed update, run the original 10,720-control assessment with
the learned deterministic amplitude and no sampled noise. Keep swimming untouched
and test its actual path: any lost original short pass or exact previously passing
long-window key stops/rejects. Hard numerical, integrity or physical fault stops
the entire attempt before further optimization. Behavioral holding failures are
recorded and never authorize a retry or tolerance relaxation.

Stop PPO immediately when original holding passes 6/6 while all four short swims
and original 28 long-window keys remain. Then perform the following **one-shot
reserved validation**, without further policy updates:

- Six withheld 64-second cases: `initial_world(1)` and `initial_world(2)`, each
  crossed with depth offsets 0/-0.25/+0.25 mm. These cases do not enter training
  or calibration.
- Two uninterrupted 300-second holds: centered original poses 0 and 3. These are
  not repeated 64-second resets. Keep the original anchor, depth and time history
  for all 6,000 controls.

Use unchanged instantaneous limits throughout: 3D speed <=0.5 mm/s, 3D anchor
error <=2 mm, absolute depth <=1 mm, lane <=2 mm and tilt <=10 degrees. Joint
motion remains allowed. Forward excursion from the start of the **final eight
seconds** must stay <=2 mm. A generalized scorer must use N-160 as that boundary
(1,120 for 64 seconds; 5,840 for 300 seconds), include the reset sample in peaks,
and never reuse a fixed 1,280-control scorer beyond its valid range.

Full qualification requires original 6/6, withheld 6/6, both 300-second cases and
all swimming retention. If reserved validation fails, reject the candidate and
close; do not resume PPO with leftover budget. If no original full pass occurs by
update 32, close/reject without spending the reserve. No production replacement
is automatic even if the registered research gates pass.

## Finite maximum accounting

| Maximum component | Controls |
|---|---:|
| Original baseline assessment | 10,720 |
| Initial deterministic 0.99 holds | 7,680 |
| Independent 256-episode calibration | 327,680 |
| 32 fresh 256-episode PPO batches | 10,485,760 |
| 32 post-update assessments | 343,040 |
| Withheld 6x64-second validation | 7,680 |
| Two 300-second endurance holds | 12,000 |
| **Combined cap** | **11,194,560** |

Maximum 111,945,600 attempted physical substeps, 32 PPO updates, 512 new-head Adam
steps, zero collection/training critic calls or updates and 900 combined active
native/process seconds. Count failures and all internal/backend attempts. Report
actual phase counts and unused reserve. One attempt only: no retry, extra seed,
extension, threshold relaxation, unused-budget reuse or automatic follow-on study.

Keep raw evidence under the new build directory, compact original/final policies
and measured results in a unique assets directory, and exact native head/quaternion/
joint poses for UI inspection. Use compiled read-only checksum verification for
large files if needed; it must not advance physics or apply an optimizer. The
viewer and its fidelity checks are separate from this research budget, with their
own explicitly counted controls. A recorded or frozen policy preview is never
reported as ongoing learning. Update current status with actual outcomes, including
failures and scope limits. Five minutes is finite evidence, not indefinite holding.
