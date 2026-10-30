# Registered conditioned-motor supervised diagnostic

Registered October 29, 2026, before execution. This is an expressibility,
conditioning and feedback diagnostic on disposable native C++/CUDA copies.
It does not continue PPO, change the architecture/physics/rewards, adopt a
controller, or modify personal saves. A failed finite fit is not a capacity proof.

## References and initialization

- Feeding teacher: the original 55-input accepted motor embedded in the protected
  `build/predator-shelter/source-boulder.bin`, SHA-256
  `d9a5c0fd07163c6558fb11c30e6db9cd65ca98dd5711f20e631f080ce668d8a6`.
  Use its deterministic, unbounded physical Gaussian means, not the damaged
  holding-only actor. Preserve the shared original goal selector.
- All three students start from the exact actor in protected protocol-5
  update-300 `motor-v5-training.bin`, SHA-256
  `13dadfe3eb0b76251a7ea93870f9e373e77c97589f5843dae2abf10174e5b7b5`.
  Architecture remains 59→32 tanh→32 tanh→4 linear. The existing task-dependent
  physical-mean transform and gains `[8,4,1024,2]` remain intact.
- Each student gets identical **fresh zero Adam moments/counter**, learning rate
  0.0003, beta1/beta2 0.9/0.999, epsilon 0.00001, global gradient norm cap 0.5.
  No inherited PPO momentum, critic/std learning, Gaussian noise or new brain.

## Data and grouping

Food TRAIN: 64 complete original-teacher trajectories, 32 calm and 32 with
0.0075 m/s steady horizontal current, seed `0xF17A2026`. Each runs 600 physical
ticks and records observations/teacher means at ticks 0,10,…,590: 3,840 rows.
Food TEST: 32 separate trajectories, 16 per current, seed `0xF17B2026`, same
sampling: 1,920 rows. Use the actual native goal-selection/physical path. The
first 55 observation fields feed the original teacher; the four added food-mode
fields must be zero. Check original-55 and normalized-expanded-59 teacher
equivalence. Whole trajectories belong to one split; neighboring frames never
cross splits.

Cover candidate pools are finite: 32 candidates per original cover goal per
split, four goals. TRAIN seed `0xC07A2026` includes the exact nominal pose as the
first candidate, then small position/yaw perturbations. TEST seed `0xC07B2026`
has only new perturbations. Position bounds are ±0.12 model units horizontally,
±0.06 vertically; yaw bounds ±0.06 radians, relative to each nominal pose.
Initial joint angles, linear/angular velocities and joint rates are zero.

Admit a candidate only if the unchanged actual shelter Batch, in calm layout2,
with zero torques and no autoreset/practice, verifies **all 200 ticks / ten
seconds** finite and safe: fresh 216-sample exposure ≤0.05 and full-body
bed/solid/surface clearance ≥0.08 model units. Check initial safety too. No
exposed or moving start receives a zero target. Reject failures without
changing geometry, clearance thresholds or the finite candidate pool.

Select the first eight qualified TRAIN starts per goal (32 trajectories) and
first four qualified TEST starts per goal (16 trajectories). Abort if the fixed
pool cannot supply them. Use **only the initial observation** per qualified
start: 32 TRAIN and16 TEST cover rows. Each zero target is therefore backed by
the entire ten-second passive proof from that exact state. Do not label later
possibly moving frames or count repeated equilibrium frames as extra coverage.
Minibatches may sample these rows with replacement. This is stationary holding,
not braking or recovery.

Freeze datasets and record provenance/hashes. Check food/cover mode and identical
input target consistency; report duplicates rather than treating them as added
coverage. TEST never enters optimizer updates, scale selection or model selection.

## Loss, budget and numerical criteria

Exactly **1,000 gradient steps per arm**, three arms: cover-only, food-only,
joint. Batch size 256 contains 128 rows per task. All arms use the same fixed
index schedule, seed `0x51F12026`, and the same initial parameters. Final policy
is step1,000 by count. Log training minibatch pre-update loss at the first step
and every100 completed steps, then full TRAIN prediction errors at the final
checkpoint. There is no early stopping, test-based tuning, retry sweep or
automatic extension.

Define `s_food[j]=max(1,RMS_TRAIN(teacher physical mean[j]))`, computed once from
TRAIN only. Cover scale is1. Freeze/report all eight task/dimension scales.
For each batch optimize:

`L = sum(active(task) * ((physical_mean-target)/s_task,j)^2) / (256*4)`.

Task masks are1 for included rows,0 otherwise. Because each task occupies half
the batch, joint loss has equal half-task contributions; single-task fits keep
that same contribution/denominator. This is explicitly **standardized physical
mean MSE**, not equal unscaled physical MSE. Backprop includes both `1/s²` and
the mathematically required food gain Jacobian. Do not remove gains or match
clipped torques alone. Verify the derivative numerically before interpreting fits.

Final held-out prediction gates, required for **every joint**:

- FOOD physical-mean RMSE ≤0.05, normalized RMSE ≤0.01, clipped-torque RMSE ≤0.02.
- COVER physical-mean RMSE ≤0.01 and maximum absolute error ≤0.025.

Report TRAIN/TEST raw-network output errors, physical-mean RMSE/max, normalized
RMSE and clipped-mean torque RMSE/max separately. These are preregistered
diagnostic tolerances, not product-quality guarantees or a PPO distribution fit.

## Frozen behavioral evaluations and decision

After all three fits finish, evaluate final students, unchanged v5 baseline and
the original teacher with deterministic physical means in the actual simulator.
No action overrides, scripted rescues or resets during a holding trial.

- Cover: four nominal training starts and all16 qualified held-out starts,
  ten seconds each. Record first failed safety condition, initial safety,
  continuous longest safe time, five-second hold count and full-ten-second
  safety count. Closed-loop gate: all16 held-out and four nominal cases remain
  strictly safe for ten seconds. Exposed starts are excluded from zero labels;
  recovery/arrival/braking are not established by this experiment.
- Feeding: 16 worlds×600 ticks per cell, calm and0.75 cm/s, reference seed
  `0x571EF001` plus fresh final seed `0x5F152629`. Retention gate: ≥90% of the
  original teacher's food/minute in **each** of the four matched cells. Keep
  finite-state checks. These are bounded paired tests, not broad reliability.
- Save every final actor and fresh Adam state in a distinct checksummed
  diagnostic artifact with source/data/optimizer provenance and exact step
  count. Verify readback. None is a player checkpoint or deployment candidate.

Joint held-out prediction plus behavioral success supports representability of
these tested skills, not successful PPO learning. Good prediction with failed
closed-loop behavior points first to coverage/feedback sensitivity. Separate
task fits succeeding while joint fails calls for conditioning/interference
analysis, not a capacity claim. Any matched-budget head/adapter comparison is
only a proposal to discuss afterward; no such architecture change is included.

Shared exploration, unobserved cached-exposure reward state and the fin/contact
discrepancy remain separate unresolved issues and are unchanged throughout.
