# Holding credit-assignment exercise — frozen November 22, 2026

User authorization: saved-data audit, status-documentation updates and another
training exercise in the current chat on November 22. This is one new finite
exercise, not a reopening of a prior attempt. No runtime adoption, publication,
merge, deployment, body change or broader physical study is authorized here.

## Question and single intervention

The completed duration-pair study pooled the baseline across different return
horizons. The saved-data audit finds opposite holding residual offsets that
account for most residual energy and change many advantage signs. It also finds
almost no explored near-neutral four-axis commands. These are separate concerns.
This exercise tests whether correcting baseline conditioning improves holding
credit while retaining swimming. It does not change exploration to address the
second concern, and it may fail to discover holding.

Use one copy of canonical checkpoint 336 and the same mixed 20/60-control holding
curriculum as the prior B arm. Replace the pooled baseline-conditioning package with a frozen
fit-only baseline indexed by task, initial pose, holding-segment duration and
local control age. Swimming uses pose and its original 320-control local age.
Do not pool returns across durations. Per-task residual RMS is fit on the same
disjoint calibration worlds and remains fixed. Both mean subtraction and residual
RMS change when conditioning changes: this is a joint centering/normalization
exercise, not a causal isolation of centering alone. Its results cannot separate
those two effects. No critic is evaluated or fitted during physical collection
or PPO. Zero-step migration fixtures may compare the preserved old/new CPU
critic outputs; those preparation-only forwards are separate from run accounting.
Saved gradients do not
establish shared-task opposition as the cause of the prior B-arm regression.

## Unchanged contracts

- Canonical parent SHA-256:
  `810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
  Preserve its original protocol and complete actor Adam state. Migrate 40 inputs
  to the already verified 41-input schema with a zero appended column and zero
  appended moments; inherited Adam age 21,700, new-column age zero.
- One native learned motor, four independent network outputs. Original latent
  Gaussian actions, sigma and AR correlation 0.8; original tanh decoding,
  1 Hz target filter, torque-limited servos, Model 6 body/water equations,
  two universal joints/four axes and +/-10 microN*m limits. No teacher, neutral
  action injection, forced posture, scripted recovery or propulsion.
- Original 40 observation meanings, appended signed forward hold error, request,
  anchor, elapsed-time and history-reset meanings remain unchanged.
- Still water. Six original 64-second holding cases: poses 0/3 crossed with depth
  offsets 0/-0.5/+0.5 mm. Holding succeeds only with speed <=0.5 mm/s,
  3D fixed-anchor error <=2 mm, absolute depth error <=1 mm, lane error <=2 mm,
  tilt <=10 degrees throughout, and final-eight-second forward excursion <=2 mm.
  Joint motion is allowed. No grace period or tolerance relaxation.
- Same 32 training worlds: 16 holding and 16 unchanged swimming worlds, balanced
  across poses. Holding alternates lengths 20 and 60 controls over four 80-control
  cycles. Each actual holding boundary resets body/task/filter/AR state to the
  original centered, zero-velocity start. Swimming is uninterrupted for 320
  controls. Every finite segment has a true terminal; no return leakage.
- Same reward and gains (holding 1/1024, swimming unchanged), gamma 0.99, actor
  learning rate 0.000125, clip 0.2, two epochs maximum, minibatches 256 and epoch
  KL guard 0.01. Preserve seeded collection and shuffle schedules from the prior
  native study. All actor weights may learn; no per-skill network is introduced.

## Preparation and phase A

Use the new identity `build/ppo/hold-credit-336to344-2026-11-22-v1`.
Preparation refuses an existing directory. Compile the native source before
collection; freeze the executable, all transitive local sources, parent inputs,
protocol and migrated checkpoint with hashes. Preserve every prior asset and
study artifact. Rebuilds and numerical fixtures before the first physical run
are preparation, not retries of physical collection.

Zero-physics checks cover original/migrated outputs on CPU and actual CUDA,
segment schedules, finite terminals, history clearing, held-out exclusion,
baseline cell occupancies, fixed statistics and checkpoint round trips. They
must not perform policy optimizer updates or physical integration. Focused
probability/GAE/checkpoint tests remain separate nonphysical validation.
An opt-in failure-evidence hook in the shared CPU budget wrapper records the
first unsafe pre-state and rejected candidate in short/long evaluations as well
as holding. The historical wrapper's source bytes are backed up before this
instrumentation; the hook changes no physical equations or acceptance limits.

Phase A runs exactly one baseline assessment (4 short swims, 6 full holds,
2 long swims: 10,720 controls) and one independent 320x32 calibration batch
(10,240 controls), using calibration action seed 314159265. It does not update
the actor. Baseline must reproduce all 4 accepted short swims and the same 28
accepted long-window keys; its failed holding remains recorded.

Fit only worlds where `world % 8 < 4`; hold out all other worlds. There are 800
baseline cells: 640 swimming cells and 160 holding cells. Each swimming cell
has four fit rows; each duration/age holding cell has sixteen fit rows. Compute
positive finite task residual RMS using fit rows only. All held-out rows remain
excluded from means, scales and initialization.

Calibration gates: actual CPU/CUDA and old/new initial function checks;
independent segmented-return error within the existing numeric tolerance;
old-policy rescoring error <1e-4; return absolute p99 <=5 and maximum <=20;
at least 10% of held-out holding samples at speed <=0.5 mm/s; holding share
of squared normalized held-out residuals in [10%,90%]. For each holding
duration, squared held-out residual bias / mean squared residual <=25%, so a
constant offset cannot dominate the stated signal. Nonfinite/zero denominator
fails. No tuning or retry on a failed gate.

Phase A closes before learning and saves its measured report and hash-bound
ready marker. Report its actual results to the user before launching phase B.
The user's existing authorization permits phase B if all frozen gates pass;
no invented receipt identifier or additional confirmation is required.

## Phase B, acceptance and hard caps

At most eight PPO updates (336->344), each one fresh 320x32 rollout. Never train
on the calibration batch. Independently check every batch's actions, old log
probabilities, rewards, segmented returns, terminal/history masks and counters
before its optimizer. Repeat the calibration occupancy/signal gates on the first
fresh batch before any optimizer step. Preserve every full sampled batch and
actor/optimizer checkpoint with explicit new schema and protocol identity.

After EVERY completed update, run the original 10,720-control assessment.
Stop and reject immediately if any of the four original short swims or any of
the exact 28 passing baseline long-window keys is lost. Holding failures are
recorded with first violation times and do not extend the budget. Success
requires 6/6 full holds plus all retained swimming passes. Stop early on success;
do not adopt the actor automatically.

| Budget component | Maximum controls |
|---|---:|
| Baseline assessment | 10,720 |
| Independent calibration batch | 10,240 |
| Eight fresh PPO batches | 81,920 |
| Eight post-update assessments | 85,760 |
| **Combined** | **188,640** |

Maximum 1,886,400 physical substeps (ten 5 ms substeps per 50 ms control),
eight PPO updates, 640 actor Adam steps, zero critic calls/steps and 120 active
native seconds combined across phases. Count all attempted controls and actual
internal substeps, including a failing attempt. First integrity, nonfinite or
physical fault stops the whole exercise before further optimization. Native
and process watchdogs apply. No retry, extra seed, extension, retuning,
unused-budget reuse or automatic follow-on exercise.

Preserve raw traces, baseline/statistics, all policy/optimizer checkpoints,
inputs, source/executable hashes, gates, actual counts and preservation checks.
Publish a local result report even if readiness fails or no PPO update runs.
Finite results do not establish current handling, braking from motion,
disturbance recovery, indefinite holding or product readiness.
