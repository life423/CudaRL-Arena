# Continuous pond: control feasibility audit

This is a source and saved-data audit, not a new physical experiment. No body
steps, GPU kernels or optimizer updates were executed. The first continuous
study remains closed and unqualified. V2 preparation does not establish a clean
turning skill or authorize another run.

Machine-readable evidence and its reproducible, simulator-free reader are in
`build/continuous-pond-control-audit-2026-11-23/`. The JSON records exact input
SHA-256 hashes, representative saved rows and zero new-work counters. It checks
the native checkpoint envelopes/FNV bindings of the saved baseline arrays.

## What the proposed residual can actually command

V2 uses the unchanged four joint targets, with
`z_j = b_j + 0.75*tanh(sum_k tanh(u_jk)*f_k)` followed by `command_j=tanh(z_j)`.
Consequently, the possible requested normalized target is bounded by
`[tanh(b_j-0.75), tanh(b_j+0.75)]`. If `abs(b_j)>0.75`, no residual can reverse
that target's sign. These are requested targets before the unchanged physical
filter and servo; they are not held angles or torques.

| Saved input distribution | Actual recorded controls | Joint0 sign locked | Joint2 sign locked | Distal pitch3 sign locked |
|---|---:|---:|---:|---:|
| Two retained fixed-heading 60-second CPU swims | 2,400 | 62.375% | 62.417% | 42.833% |
| V1 autonomous zero-head baseline, all active rows | 4,180 | 50.335% | 51.818% | 46.651% |

The autonomous baseline has all sixteen V1 coefficients exactly zero, so its
recorded motor latent is the backbone output. Padding is excluded. This avoids
treating the fixed-heading retention observations as identical to autonomous
local-forward observations.

For example, retained case0/control20/joint2 has `b=2.449347734`: inherited
target `0.985198`, possible interval only `[0.935327,0.996678]`. Autonomous
case4/control19/joint2 has `b=1.870748281`: target `0.953662`, interval
`[0.807829,0.989471]`. The JSON includes both extrema for every joint.

This is a concrete authority restriction, not proof that turning requires a
target sign reversal or that V2 cannot work. Swimming can steer through waveform
asymmetry without reversing every joint. Saved V1 states also do not predict
the V2 state distribution. What is unsupported is the claim that allowing four
residual outputs automatically supplies adequate physical turning authority.

## The backbone receives no persistent turning intention

`turning_task_v2::observe()` rebuilds requested heading and terminal velocity
from the current physical head azimuth at every control. Its start X/Z is the
current COM X/Z. In `straight_motor::observe()`, this removes horizontal lane
error and sustained desired-yaw error. The original world-space depth remains.

The residual therefore has to create the turn itself while the frozen motor
continues to receive its local-forward request and observes the resulting joint
and angular-rate changes. Straight-swimming feedback could oppose some steering
perturbations; cancellation is a hypothesis, not a measured causal finding.
The current code does not demonstrate requested-heading tracking.

## State and function-class limits

Depth is signed actual-minus-original depth; vertical COM velocity is positive
upward. The existing inverse-up roll/pitch formulas and joint angle/rate scales
are used consistently. No obvious sign or axis wiring defect was found.

The new head receives attitude and joint phase, but not body roll/pitch angular
rates. The backbone observes those rates, so existing damping may help; adequate
damping under new all-axis corrections is an assumption. Two states with the
same twenty head features can require different corrections when those omitted
rates differ.

Each output correction is a single `tanh` of an additive feature projection,
not a general state network. There are no explicit bank-by-gait-phase interaction
terms. A correction whose sign must follow the product of independent signed
bank and phase features cannot generally be represented by that one projection.
Whether this body needs that particular coordination is unproved. A reactive
orbit remains representable in principle; the audit establishes restrictions,
not an impossibility theorem.

## Concrete decision before another learning attempt

A learned persistent heading request is legitimate: it changes motor intent,
not the body's transform, and does not require scripted navigation. It also
matches the intended chooser/motor ownership. However, saved336 qualifications
initialize the requested heading from the initial body and hold it; depth
recovery does not test a moving horizontal heading switch. A frozen motor's
heading-tracking ability must not be assumed.

The narrow unresolved hypothesis is whether small persistent heading requests
produce useful moving turns through the intended motor interface while retaining
depth and propulsion. If that capability exists, a learned request chooser is a
more direct steering interface than making a bounded residual fight or reshape
the gait. If it does not, motor training must explicitly teach requested turns;
another larger residual map is not evidence that the missing skill has been
supplied. Any physical test or new learning attempt needs its own approval and
fixed acceptance criteria. This audit authorizes neither.

## Offline persistent-heading sensitivity follow-up

The adjacent `heading-request-sensitivity.json` and reader evaluate offsets
`-90,-60,-30,-15,-5,0,+5,+15,+30,+60,+90` degrees with the original 41-input
meanings and preserved2532 weights. Input checksums/SHA hashes and zero new-work
counters are recorded. Existing checkpoint readers validate the preserved
parents. Reconstructing the actual saved backbone gives maximum absolute error
`6.77e-7` on the2,400 retained CPU states and `3.51e-6` on the4,180 active
autonomous GPU states, before changing any request inputs.

Two questions are deliberately separate. Immediate switches anchor X/Z at each
actual saved state, keep its original depth, and change heading/terminal velocity
without a growing lateral error. Persistent queries hold one episode-initial
heading and lane anchor while reading the unchanged old recorded trajectory.
Neither query advances the world or constitutes a measured turn.

Concrete findings:

- Removing the persistent reference is already a substantial controller change.
  On retained straight-swim states, replacing the original heading/lane by the
  local-current-heading request changes requested targets RMS by
  `[0.07595,0.16924,0.04878,0.04278]` across joints0..3. Conversely, restoring the
  initial persistent reference on the autonomous baseline states changes them
  by `[0.11395,0.27070,0.12595,0.08192]`. Same saved weights and zero residual do
  not preserve the learned input feedback when the task changes its reference.
- The existing motor is sensitive to small heading requests; those input columns
  are not disconnected. On autonomous saved states, an immediate positive5-degree
  request changes normalized targets RMS by `[0.04254,0.10921,0.01826,0.01947]`.
  At15degrees the values are `[0.12923,0.31047,0.06479,0.05185]`. Proximal pitch
  changes more than either yaw output. This shows coupled output sensitivity,
  not correct turning direction, depth control or successful tracking.
- Persistent queries on old straight trajectories eventually create large lane
  errors: even5degrees reaches41.5mm in the retained60-second recordings. Large
  request/output changes and near-limit targets in that replay are therefore
  counterfactual out-of-distribution responses, not evidence that a real
  feedback-controlled turn would continue down the old straight path.

These results strengthen the case for exercising the intended persistent
request interface before expanding another residual. They do not establish that
training only a chooser will work. A small requested-turn motor curriculum is
justified if clean small-angle tracking is absent; that should teach this same
motor to execute the request, with preserved straight-swim/holding assessments.
The current evidence cannot substitute offline target sensitivity for that
physical capability check. All target saturation fractions in the JSON use
`abs(command)>=0.95` solely as a diagnostic, not a new acceptance threshold.
