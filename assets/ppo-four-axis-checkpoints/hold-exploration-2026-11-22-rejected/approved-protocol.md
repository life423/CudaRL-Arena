# Sustained quiet-action exploration — frozen protocol, November 22

The user explicitly authorized addressing exploration and one further training
exercise to learn still-water holding while retaining swimming. The subsequent
menu/GUI overhaul and visible inspection are also authorized, separate from
physical acceptance. All previous experiments remain closed. This protocol
defines one new attempt; no retries, extra seeds, adaptive tuning or automatic
follow-on study. No production policy adoption, publication, merge or deployment.

## Hypothesis and restricted policy

The [saved exploration analysis](HOLD_EXPLORATION_DESIGN_2026-11-22.md) finds that
widening independent action noise barely improves simultaneous small four-axis
commands. Use persistent, positive stochastic amplitude exploration to create
sustained small-command trials. Whether those trials physically hold must be
measured before any PPO update.

Use the canonical checkpoint-336 actor and its verified zero-column 41-input
migration. Preserve its weights and full optimizer state byte-for-byte; do not
update that backbone. Add four learned output-head coefficients `a[j]`, initially
**one**, with their own fresh four-parameter native CUDA Adam state. This remains
one motor with four actuator outputs, not an additional behavioral network.

For frozen network output `b(x)` and a stationary request indicator `h`,
`mu[j] = b[j] + h*(a[j]-1)*b[j]`. Define `h=1` for the tested zero-terminal-velocity
holding request and `h=0` for the original 8 mm/s request; validate it against
actual observation inputs 3..5. Swimming uses the original `b` directly, with no
floating-point multiply-by-one detour. The new coefficients are not zeroed or
pretrained into a neutral solution. Every changed coefficient must come from
native PPO/Adam. Existing backbone and old Adam bytes are retained unchanged.

This is **restricted policy-parameter learning**: swimming retention is protected
by freezing its existing computation. A successful result would demonstrate
learning a request-conditioned quiet gait amplitude in the original still-water
starts. It would not demonstrate that a fully trainable shared network learned
without forgetting, or that the animal can brake from motion, recover to an
overshot anchor, handle currents, switch modes arbitrarily or hold indefinitely.

## Exploration distribution and likelihood

During holding collection sample one exogenous positive amplitude `g` for each
true 20- or 60-control segment, and retain it until that segment terminates.
Use four equal-width log bins with endpoints
`[0.01, 0.0316227766, 0.1, 0.3162277660, 1]`. Within each bin draw continuously
log-uniformly using a fixed independent seeded counter stream. Set
`bin=(floor(world/8)+floor(segment_index/2))%4` for holding worlds, where
segment_index is 0..7. This balances pose, duration, fit/held-out split and bins.
Swimming uses `g=1`, the original action distribution and original seeded stream.
Do not resample an unsuccessful segment, force zero targets, use a teacher,
clamp posture, alter the physical filters or substitute a recovery controller.

Store actual physical latent actions `z`, sampled `g`, bin, actual conditional
log probability and frozen current/previous backbone means. At a history row,
`c = g*a*(b_current-rho*b_previous) + rho*z_previous`, with conditional standard
deviation `g*sigma*sqrt(1-rho^2)`. At a true reset, omit previous state/action and
use `g*a*b_current`, standard deviation `g*sigma`. Here `rho=0.8`, and sigma is
the preserved checkpoint's original value. The true density includes the
`-4*log(g)` Jacobian. The exogenous bin-conditioned density is independent of
policy coefficients and cancels in PPO ratios. Do not count it repeatedly as a
learned action or silently use the unscaled action density.

The gain gradient includes both means:
`d(c[j])/d(a[j]) = h*g*(b_current[j]-rho*b_previous[j])`, with history omitted
at reset. Native CUDA PPO must use actual sampled `z` and actual old density.
Swimming gain gradients are exactly zero. Use shared Gaussian PPO clipping and
native Adam math; no Python training or runtime. Offline Python verification is
permitted and never applies updates or advances physics.

**All deterministic acceptance tests use g=1 and no sampled action noise.**
Exploratory attenuation cannot be presented as learned holding. The frozen
baseline with coefficients one must retain the old behavior and fail holding
as previously measured.

## Unchanged physical/task contract

Canonical parent SHA-256:
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
Preserve original Model 6 body/water equations, two universal joints/four axes,
+/-10 microN*m torque caps, 1 Hz joint-target filter, 50 ms controls and ten
5 ms substeps. Same observation meanings, reward and holding gain 1/1024,
gamma 0.99, PPO clip 0.2, minibatches 256, at most two epochs and epoch KL guard
0.01. The four fresh coefficients use fixed Adam learning rate **0.0025**, with
the existing native numerical settings. Their optimizer ages start at zero;
the inherited 21,700-step optimizer is preserved, not relabeled as their history.
No coefficient projection or retuning after seeing results. Nonfinite policy
parameters or physical faults stop the entire attempt.

Same 32 worlds, 16 holding and 16 swimming, both original poses balanced.
Holding alternates 20/60 controls over four 80-control cycles; true boundaries
reset original zero-velocity centered body/task/filter/AR state. Swimming remains
uninterrupted for 320 controls. All segments end with true terminal done=1,
truncated=0, without return leakage. Same original swim offset mixture.

## Preparation, calibration and data gate

Identity: `build/ppo/hold-exploration-2026-11-22-v1`. Refuse an existing identity.
Freeze source, executable, libraries, protocol, canonical/migrated weights and
initial coefficient/Adam state with explicit checksummed schemas and SHA-256.
Preserve all previous experiment artifacts and unrelated changes. GUI changes
are separately owned and must not change physical/training inputs.

Zero-physics, zero-policy-update preparation must verify CPU-double and actual
CUDA action densities/samples at g=0.01/0.1/1, with and without AR history;
same-policy ratios equal one; four gain finite differences; frozen swimming
outputs and zero gain gradients; initial all-one output parity; segment gain
persistence and true-boundary history clearing; fit-world exclusion and cell
counts; explicit checkpoint/optimizer clock round trips. Existing nonphysical
probability/GAE/checkpoint suites remain separate. Old/new CPU critic output
comparison may occur in migration fixtures only; collection/PPO has no critic.

Phase A: one original 10,720-control assessment (4 short swims, 6 full 64-second
holds, 2 long swims), then one independent 10,240-control calibration batch with
coefficients one, new amplitude sampling and fixed calibration action seed
314159265. No optimizer. Require original 4/4 short passes, the exact 28 passing
long-window keys, and the original failed holding baseline.

Fit only worlds `world%8<4`; hold out the other worlds completely. Fixed baseline
means are indexed by task, pose, duration/local age and amplitude bin. There are
1,280 cells: 640 swimming, 640 holding; four fit rows per cell. Per-task residual
RMS uses fit rows only and remains fixed. Return absolute p99 <=5, max <=20;
finite positive residual RMS; independent return error within prior numerical
tolerance; actual old log-probability rescoring error <1e-4.

Coverage gate before PPO: at least four COMPLETE holding segments must stay
within the original instantaneous speed, anchor, depth, lane and tilt limits
from reset through their final post-action sample, with at least one segment
in each pose x duration combination. Also require >=10% held-out holding
pre-action samples at speed <=0.5 mm/s and holding share of squared normalized
held-out signal in [10%,90%]. Record small-command row counts and contiguous
sequence lengths at +/-0.03 and +/-0.10, physical quiet segments, per-bin and
per-duration residual bias fractions. Bias fractions are diagnostics, not new
pass thresholds. No retuning, resampling or new attempt if coverage fails.

Close phase A and report its actual coverage before launching the separate
training phase. Existing user authorization permits that phase if all frozen
gates pass. Repeat the coverage, occupancy, signal and integrity gates on its
first fresh batch before the first optimizer step. Never train on calibration.

## Training, deterministic acceptance and limits

At most **32** PPO updates, only the four new coefficients. After each update,
save its complete coefficient/Adam state and run all original tests with no
exploration scaling or action noise. Re-evaluate actual unchanged swimming,
not only parameter identity. Stop at any lost original short swim or any lost
one of the exact 28 long-window passes. Stop early if all six full holds pass
alongside retained swimming; no automatic adoption. Otherwise close at update
32. Any integrity/nonfinite/physics fault stops before further learning.

Holding criteria remain speed <=0.5 mm/s, 3D fixed-anchor error <=2 mm, absolute
depth error <=1 mm, lane <=2 mm, tilt <=10 degrees throughout each 64-second
case, plus final-eight-second forward excursion <=2 mm. Both original poses
and offsets 0/-0.5/+0.5 mm, no grace period; joint motion is allowed.

| Maximum component | Controls |
|---|---:|
| Baseline assessment | 10,720 |
| Independent calibration | 10,240 |
| 32 fresh PPO batches | 327,680 |
| 32 post-update assessments | 343,040 |
| **Combined** | **691,680** |

Caps: 6,916,800 attempted physical substeps, 32 PPO updates, 2,560 new-parameter
Adam steps, zero collection/training critic calls/updates and 120 combined active
native/process seconds. Count all attempted controls and actual substeps,
including failures. No retries, additional seed, extension, leftover-budget
reuse, threshold relaxation or automatic subsequent exercise.

Preserve all sampled batches, post-action physical metrics, gains/bins/densities,
full new optimizer states, unchanged backbone/full original optimizer, all tests,
first unsafe pre-state/candidate, exact actual head/body poses and joint angles
for visible recorded inspection, accounting, logs and hash checks. Pose logging
observes existing controls and must not add integration. A GUI recording must
be labeled recorded; frozen native playback is not background learning.
