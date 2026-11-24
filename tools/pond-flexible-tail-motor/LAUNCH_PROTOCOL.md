# Flexible-tail straight swimming PPO pilot

Run one native C++/CUDA candidate from the protected fitted flexible-tail
actor. The task is to start under learned control or continue from an existing
swim, travel straight, and sustain at least 6 mm/s upright. The observation
includes signed half-second forward speed, signed lateral speed and lateral
deviation from the original lane. The body, native dynamics, servo limits,
water, clearance and 1.25 ms substep remain as in the previous comparison.

The trained candidate starts from the fitted yaw policy and zeroes the pitch
output weights and bias at initialization. Both axes remain trainable. The
new input columns start at zero. The candidate checkpoint uses its own mode
identifier for the new observations and decision cadence. Existing
checkpoints serve as read-only sources.

Reward per 20 ms control is 0.020 times the following terms.

- Signed forward transport weighted 0.70 for the latest control and 1.30
  for the half-second average. Scaling uses the 8 mm/s request and the
  existing bounded overspeed cost. Small real displacements earn credit.
- Squared lateral speed divided by 4 mm/s and squared lane deviation divided
  by 20 mm, weighted 0.25 and 0.30 respectively. Both scaled errors cap at 2.
- Squared tilt divided by 10° and squared depth error divided by 1 mm,
  weighted 0.35 and 0.30. Additional squared excess beyond those thresholds
  is weighted 0.55 and 0.60. Both scaled errors cap at 3.

The 256 worlds start from 64 exact-rest states, 128 slightly perturbed
rest states with paired bend directions, and 64 previously captured moving
states. A single heading and depth request persists throughout. All controls
come from the policy, and all movement comes from the existing physical
actuators. The trained and evaluated policy chooses its next action
every four controls (80 ms). The simulator executes all four controls and
their full native substeps. An action's Gaussian likelihood is scored once.

## Fixed budget

- One candidate and at most 12 updates of 256 worlds × 64 physical controls.
  This permits 196,608 controls, 49,152 sampled policy decisions and 3,145,728
  physical substeps. Preserve the complete body and history throughout.
- Two PPO passes over 4,096 decision rows per update, minibatch 1,024.
  Seed 20261124. Yaw and pitch latent log standard deviations are −1.8 and
  −2.8. Use base gamma 0.998 and lambda 0.9797, discounted control rewards
  within each action block and gamma and lambda raised to the fourth power
  across blocks. Actor and critic learning rates are 0.00035 and 0.0003.
  Clip is 0.2 and the exact KL cap is 0.05.
- One launch each for fixture preparation, trainer, and each available
  assessment at update 0, 4, 8 and 12. Cap each process at 4.000 wall seconds.
  The trainer and evaluator also check internal 3.5-second deadlines.
  Physics, numerical, KL or timeout faults end the affected run.
- Held-out suite has 12 clear-water cases, with four exact-rest, four
  perturbed-rest and four moving starts. Each runs 12 simulated seconds.
  In both final two-second windows require projected COM speed ≥6 mm/s
  and course error ≤5°. Require peak tilt ≤10°, whole-case depth error
  ≤1 mm and positive certified whole-body wet clearance. Retain the older
  final-window-only score as a separate comparison metric.
- Report passes by start type, steady speed, course, peak tilt/depth,
  physical work and actual execution time. Record the fresh run outputs
  and keep all historical evidence. End after the bounded run and checks.
