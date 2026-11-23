# Requested-turn full-motor PPO lesson — preparation proposal

**DRAFT. Preparation only; no physical controls, optimizer steps or grant
creation are authorized.** The completed frozen requested-course probe and all
earlier studies remain closed. The held continuous-pond V2 proposal remains
held. This is a distinct, finite proposed experiment with a distinct grant.
Freeze this document only after the source, schemas, numerical fixtures and
independent readiness checks agree with every item below.

## Checkpoint readiness — unapproved and incomplete

This draft is being saved for handoff, not execution. New task/policy source
and the native driver exist; the task's CPU-only synthetic suite passed
2,615 assertions with zero integration/Adam. The private driver compilation
and zero-only preparation are separate work and do not approve physics.

The fresh 2,480-control matched-command gate implementation and exact fixture
selection are not yet complete. Its allocation is a proposed budget, not a
passed prerequisite. The physical backend gate cadence still needs source
reconciliation. Independent 42-input/per-action-AR calibration-readiness,
training and optimizer-ledger auditors, plus their release writers, have not
been completed or run. Existing frozen-probe and continuous-pond audit proofs
do not validate this new full-network lesson. No new source
freeze, actual grant, physical phase or Adam step exists for this lesson.
The described release formats are interfaces for later implementation, not
already-issued releases. Do not reuse any prior study's approval or readiness
marker. Do not create an authorization marker automatically from this draft.

## Question and limits

The frozen update-336 swimmer executes small course changes, but larger
requests reduce propulsion and disturb depth/posture. At 30 degrees it can
point approximately toward the requested velocity while slowing to about
2.2 mm/s. All tested 60/90-degree requests reached the 20-degree tilt guard.
The [verified frozen-motor probe](FROZEN_REQUEST_TURN_RESULTS_2026-11-23.md)
supports teaching coordinated turning with the existing whole motor.

Train one reusable four-output motor, not another behavior-specific motor
head. The experiment measures a requested-turn primitive. It contains no food,
current, boundary navigator, automatic request route, goal-chooser training,
online player learning, or claim of indefinite swimming. Even a qualified
primitive does not establish a safe continuous pond creature. No automatic
adoption, Godot physical preview or deployment is included.

The 42 actor inputs contain body state, requested motion and intent; they do
**not** contain bank distance, object locations or terrain geometry. In
heading-only mode, translating an otherwise identical swimmer toward a bank
does not reveal that bank to this actor. A bank penalty alone cannot supply
the missing observation or establish general turn timing. This lesson can
learn **how to execute a supplied course change**, not independently learn
**when and where to choose it** around varied banks/objects.

The user's newest direction is a creature that senses its surroundings,
receives credit for safe continuing movement, and can eventually climb a bank
out of water. That direction needs a separate reviewed environment/sensor and
physical-capability design within the planned chooser-plus-one-motor
architecture. This lesson's inset termination and hard active-ellipse escape
explicitly prevent bank climbing. Submerged model 6 and these free-water
guards are not evidence of a physically valid shoreline, water exit, support
or land locomotion. Do not present a heading-only motor result as completing
that broader goal, and do not substitute scripted navigation.

## Actor, critic and actual retained behavior

The only behavioral network is a fully trainable 42–32–32–4 MLP with
tanh/tanh/identity activations. Migrate the exact original 41-input holding
backbone (2,532 parameters) by appending one zero input column, yielding 2,564
parameters. Inputs 0–40 retain their old meanings. Input 41 is an explicit
heading-only intent flag. Use the exact update-336 swimmer and holding source 7
identified in the frozen probe; preserve their original files and checksums.

Old requested Swim and Rest use flag zero and the original 41-input observer.
Rest retains the saved fixed holding gain and stationary-request decoder.
Crucially, **all modes execute the current trained 42-input actor**. There is no
frozen old-network bypass for training, inference or any retention fixture.
The old gain is a decoder parameter, not a substitute motor. No teacher-output
matching loss counts as retained physical skill.

The critic is a separate trainable 43–32–32–1 value network (2,497 parameters).
It receives the actor's 42 inputs plus remaining-time fraction, and is a
training auxiliary, not another controller. Initialize it afresh with
deterministic orthogonal weights with layer gains sqrt(2), sqrt(2), 1; it does
not inherit an old negative-cost
value function for the new positive utility. Both networks have fresh Adam
states with beta1 .9, beta2 .999 and epsilon 1e-5.

For heading-only intent, remove the world-X/Z lane-error vector **before**
rotating it into body coordinates. Preserve the complete original global
depth-error vector and persistent world heading/terminal velocity. Do not
zero body-frame channels 0/2 after rotation: doing so would erase depth
information during tilt. This observer is read-only; it applies no physical
force or pose correction.

## Cases, requests and fixed curriculum

There are 24 training cases: poses 0 and 3, each with stationary Rest,
unchanged Swim at zero offset, and ±5, ±15, ±30, ±60, ±90-degree turns. Each
case has 128 replications in 3,072 CUDA worlds, at most 240 controls (12 s).
Rest is stationary from control 1, flag zero. Swim zero remains flag zero for
the whole episode. Nonzero turn cases run the **current candidate actor** with
flag zero for 80 real warmup controls. Immediately before control 81, rotate
the original persistent requested course once and set heading-only flag one.
The course is not relative to the head's oscillating gait angle.

Maintain original depth, clocks, elapsed controls, servos, filter targets,
reference rates, physical body state and AR history at the intent change.
Do not restart `begin_forward`, the world or the startup observation. No
frozen parent supplies warmup actions after learning begins.

The maximum absolute training angle is fixed by update number:

| Updates | Angle cap |
| --- | ---: |
| 1–8 | 15 degrees |
| 9–16 | 30 degrees |
| 17–24 | 60 degrees |
| 25–32 | 90 degrees |

Clamp each canonical signed request to that cap, preserving case identity and
its 128 replications. Repeated capped angles are intentional. No success-based
curriculum change, extra update, retrospective angle tuning or restart is
permitted. Calibration uses the first curriculum stage and the initial actor.

Development uses poses 0/3 and the full 22-case 0/±5/±15/±30/±60/±90 grid at
updates 4,8,12,16,20,24,28,32. It is diagnostic only: no early selection,
checkpoint ranking or tuning follows those results. The only final candidate
is completed update 32. If the bounded run stops before update 32 there is no
final qualification or adoption. The final 22 withheld cases use poses 1/2
with the same request grid, once; these turning starts do not enter training
or development.

Heading-only intent changes the motor interface compared with the completed
frozen probe, which retained rigid horizontal lane tracking. Therefore this
new task must collect its own matched 22-case initial baseline before PPO.
The earlier small-turn result is rationale, not this lesson's before score.

## Bounded utility and genuine terminals

Use SI units and actual accepted interval `dt`. Let `h` be the persistent
horizontal desired heading, `v` actual COM velocity and
`v_goal = .008*h`. During movement:

```
speed_credit = clamp(dot(v,h)/.008, 0, 1)
velocity_quality = exp(-.5 * dot(v-v_goal, v-v_goal) / .0025^2)
lane_cost = .25*(horizontal_lane_error/.002)^2  # flag0 only; zero for flag1
positive_power = (World.positive_work_POST - World.positive_work_PRE) / dt
u_move = speed_credit * velocity_quality /
  (1 + lane_cost + .05*(signed_depth/.001)^2
     + .05*(roll_degrees/10)^2 + .01*(pitch_degrees/10)^2
     + 6250*positive_power)
```

The vector velocity quality includes vertical velocity; no extra lateral term
is added. The same formula applies during warmup and after the request, with
only the persistent request and intent flag changing.

For stationary Rest:

```
u_rest = 1 /
  (1 + (norm(COM_velocity)/.0005)^2
     + (norm(COM_position-anchor)/.002)^2
     + (signed_depth/.001)^2 + (roll_degrees/10)^2
     + (pitch_degrees/10)^2 + 6250*positive_power)
```

Both utilities lie in [0,1]. Nonfinite values or a negative positive-work
increment are numerical faults, not trainable failures. Record the actual
double positive-work ledger before/after each control so this reward can be
independently reconstructed. Do not substitute a float work delta.

Each valid interval reward is `dt*u/12`. At an ordinary failure, additionally
subtract `(remaining_seconds_AFTER_the_transition + 4)/12` **once**. Do not
divide returns by survived time or shorten the registered denominator.
Continuing from the same prefix with nonnegative future utility therefore
always beats immediate termination. This corrects the held proposal's cheap
failure incentive; it does not claim every failed high-quality prefix has
lower total return than every low-quality survivor.

The 12 s horizon is a true finite-task terminal (`done=1`, `truncated=0`),
with zero bootstrap. Ordinary bank/depth/tilt/stall failure is also true done.
No automatic reset or recollection occurs within a batch. Inactive padding
after termination supplies no physical controls, loss, statistics or gradients.
The interval reward is already normalized by 12; do not divide summed reward,
GAE or value targets by 12 a second time.

## Physical rules and stopping

Preserve model 6, dimensions, masses, drag, fluid, contact, four actuator axes,
1 Hz reference filter, 5 ms mechanics and torque caps. Native numerical,
solver, energy, contact or invalid-state failures stop the entire attempt
before GAE or Adam. Preserve the first unsafe PRE/candidate and the actual
attempted/accepted counters. Reject faulty data; never train around a solver
or physical defect.

The active pond ellipse has radii 0.20/0.12 m. All body corners participate.
Ordinary bank termination uses inset radii 0.195/0.115 m. Accepted 5 ms states
with absolute global COM-depth error over 5 mm or head tilt over 20 degrees
are ordinary behavioral terminals, with priority bank, depth, tilt. Hard
active-ellipse escape remains a global fault. The same prospective guards
must be checked by zero-integration edge fixtures before any physical phase.

Moving cases additionally terminate for stall when the last 40 controls'
horizontal COM arc divided by their actual accepted duration is below
2 mm/s, first checked at completed control 140, inclusive (7 s total), then
at each completed control. Rest ignores this guard.
Ordinary guard thresholds are not successful-skill standards.

## Per-control stochastic policy and PPO

Use four raw joint-target latents, AR correlation `rho=.8` and fixed
`log_sigma=-1` for every dimension. Score the actual latent action conditional
on current/previous means and previous raw latent. At a true episode start
there is no history term. Maintain history through the control-81 request;
intent changes are not episode resets. No tanh-command or decoder Jacobian
is added when scoring the recorded raw latent policy.

For history-bearing rows the conditional mean is
`mu_current + rho*(z_previous-mu_previous)` and innovation standard deviation
is `exp(-1)*sqrt(1-rho^2)`. The first row uses mean `mu_current` and standard
deviation `exp(-1)`. Recompute both means with the same candidate actor during
PPO. Backpropagate both current-mean and `-rho` previous-mean derivatives.
The Gaussian is around raw NN means in all modes. Rest decodes its recorded
raw sample with `command=tanh(saved_gain*raw_z)`; movement uses
`command=tanh(raw_z)`. Thus the gain scales physical Rest exploration as well
as its mean, while probability/history retain the raw latent. Do not instead
add unscaled Gaussian noise to a gain-scaled Rest mean.

Use GAE `gamma=1`, `lambda=.98`, clipped PPO ratio epsilon .2, fixed entropy
coefficient zero, actor learning rate .0003, critic .001, gradient norm .5,
and KL diagnostic threshold .01. The critic uses value clipping .2 and
value-loss coefficient .5. With `vc=Vold+clamp(Vnew-Vold,-.2,.2)`, its
unweighted loss is `.5*max((Vnew-R)^2,(vc-R)^2)` and the coefficient multiplies
that loss once. Store `Vold`; `R` is the fixed GAE target.
Two epochs, minibatches of 8,192 over 737,280 nominal slots
give 90 minibatches per epoch. Pool approximate-KL statistics at the changing
minibatch parameters throughout each epoch, divided by actual active rows.
Skip the second epoch if the first epoch's pooled diagnostic exceeds .01.
This is the established early-stop diagnostic, **not** a strict bound on
final-epoch policy KL. No extra final-policy likelihood sweep is implied.
Never change optimizer settings after
freeze. Advantages use statistics from actual active rows only. Divide each
minibatch's losses, gradients and KL statistics by that minibatch's active row
count; skip zero-active minibatches without advancing either Adam counter.
The same raw per-action sample is scored once, even though both observation
halves contribute mean gradients.

Fixed seeds are calibration `0x6b514c39`, training actions `0xd37a8021`,
permutations `0x19b6e40d` and critic initialization `0x7238af15`. Sample stream
is `batch_index*240+zero_based_tick`, with calibration index 0 and training
indices 1–32; normal RNG element is `world_index*4+joint`.
Use `make_permutation(737280, permutation_seed+update*2+epoch)` for update
1–32 and epoch 0/1. No seeding or resampling may change after collection.
The critic uses `orthogonal_params` with the fixed seed and gains above.

At most 180 actor **and** 180 critic Adam steps per update: **5,760 actor +
5,760 critic = 11,520 combined steps** over 32 updates. Actual early KL stops,
empty masked minibatches and actual optimizer ages must be recorded honestly.
Every active sample retains its previous observation/action/history needed
for conditional likelihood, directly or through independently reconstructible
recorded state. Never normalize advantages over padding or allow padding to
acquire a nonzero gradient after normalization.

## Frozen phase budget and independent barriers

All counts are upper bounds including failed attempts and partial controls.
No unused budget from an earlier study is transferred.

| Phase | Maximum physical controls |
| --- | ---: |
| Fresh matched-command CPU/CUDA physical gate | 2,480 |
| Original-skill baseline | 10,720 |
| Initial requested-turn baseline, 22 × 240 | 5,280 |
| One calibration collection, 3,072 × 240 | 737,280 |
| Training, 32 × 3,072 × 240 | 23,592,960 |
| Eight development assessments, 22 × 240 | 42,240 |
| One final primitive assessment, 22 × 240 | 5,280 |
| Full final original-skill retention | 46,240 |
| **Total** | **24,442,480** |

At most **244,424,800 attempted 5 ms substeps**, with actual accepted substeps
reported separately. One combined **900 s native physical/process budget**
covers all physical-phase native process wall times, including native loading,
math and evidence I/O, not 900 s per phase. Compilation, pure preparation and
independent offline audits execute outside that physical-process sum and
cannot launch additional simulation. Require at most **8 GiB of new
evidence plus an 8 GiB free-space reserve**, with a complete schema-derived
worst-case projection checked before the first physical attempt and before
each allocation/collection. No row sampling, evidence deletion or old-study
file modification may be used to fit the cap. The compact record has 45
float32 values (PRE filter12, POST Body23, raw Gaussian action4, old logp,
old value, reward, three 5 ms extrema), six float64 POST energy-ledger values,
and two uint8 accepted-substep/done fields: **230 bytes per nominal slot**,
with separate arrays to avoid padding. Initial/final full World/task records,
counts and all phase/checkpoint/assessment overhead are additional. The 33
maximum-sized calibration/training batches use 5,595,955,200 bytes before
overhead. The final source-derived whole-study projection must prove 8 GiB.

Preparation, source checks, pure CUDA tensor/math fixtures and compilation
perform zero physical controls and zero optimizer steps. Freeze the full
causal closure and exact approved protocol before physical entry. Every
physical entry point requires the new study-specific explicit grant. It must
reject grants for the frozen probe, closed V1 or held V2 before creating its
one-shot directory.

The execution owner advances through prerequisite, baseline, calibration,
independent full-calibration audit and its hash-bound readiness release before
the first Adam step. The old-skill baseline must pass four short swims, six
original holds and all 28 original successful 60 s keys. The new 22-case
baseline is descriptive: zero initial turning passes does not prohibit PPO.

Before GAE/Adam, calibration and every training batch must contain 128 worlds
for each of 24 canonical case IDs, finite valid metadata and at least one
valid active postrequest flag-one row for **each of the 20 nonzero turn case
IDs**. Report all these counts. If all 128 copies of a case stop during warmup,
that batch has no learning signal for its turning intent: close for data
coverage failure, without resampling or optimizer. Individual ordinary
failures and zero successful turns remain valid data and do not themselves
trigger a global stop. Do not impose a positive-advantage or successful-action
coverage gate.

Safety or registered data-coverage failure closes the attempt; no automatic retry or
extension is allowed. Final original-skill tests run once even after bounded
learning completion, even if primitive gates fail, unless a physical/numerical
fault has globally stopped the study. Withheld and full retention are run only
after completed update 32. No physical viewer allowance is included.

The one-shot grant envelope has magic `0x31555452`, version 1, this protocol's
payload checksum, the whole frozen-input-manifest file checksum and the actual
fresh user grant bytes. `requested_turn_ppo_authorization_v1.h` rejects a
design hold, absent files, wrong study magic, changed protocol/freeze or empty
grant before physical-phase directory creation.

The independent baseline release is a separate checksummed envelope:
`0x31565254`, version 1, protocol checksum, whole-file checksum of
`independent-baseline-verification.json`, then that of `baseline-result.json`.
The calibration release similarly contains `0x314b5254`, version 1, protocol,
then whole-file checksums of `independent-calibration-verification.json`,
`calibration-ready.bin`, `calibration-started/batch/meta.bin` and
`calibration-started/batch/trace.bin`.
These are offline verification releases, not substitute user grants. An
auditor may create them only after its saved-data checks pass and its executed
code/import snapshots are bound. Failed pure/audit attempts remain preserved;
fixing an offline parser never authorizes recollecting physical data.

## Final movement and retention standards

For each complete 240-control/12 s primitive, compute the mean actual POST
COM velocity over controls 161–240. Its projection onto persistent desired
heading must be at least 6 mm/s and its horizontal course error at most
5 degrees. Vector velocity RMSE over the same 80 POST samples, relative to
the full 8 mm/s desired vector, must be at most 2.5 mm/s. Across **all accepted
5 ms states including warmup**, depth error must remain within 1 mm and head
tilt within 10 degrees, with no ordinary/hard terminal. No head-mean, head-RMS
or work threshold is added to these new turning gates; report those diagnostics
separately. Never replace COM movement with head orientation.
Primitive baseline/development/withheld assessments use CUDA deterministic
actor means. Old-skill retention uses the current candidate's CPU inference.
Report both backend scopes; the matched-command prerequisite does not by
itself prove independently closed-loop CPU turning or long pond survival.

Training records native per-control extrema, endpoints and first unsafe PRE,
not every intermediate 5 ms pose. Its independent audit reconstructs all
recorded endpoints and checks recorded extrema, while first-crossing mechanics
rely on the frozen guarded wrapper. Final/development assessments additionally
save every accepted 5 ms body state, so their all-state depth/tilt acceptance
can be independently checked. State this evidence distinction explicitly.

Full retained skills include four short swims, the original 28 successful
60 s windows, three recovery fixtures, the original 148 successful 300 s
windows, six original holds, six withheld holds and two 300 s holds. The old
28/148 keys must remain a subset of passing candidate keys; startup
improvements are allowed. Every fixture executes the **trained actor** with
its actual mode input and decoder. Exact original thresholds and body physics
remain unchanged. No held-out failure permits retraining or selecting another
checkpoint after seeing final results.

The audit must distinguish actual weight updates, finite measured improvement,
qualified primitives and eventual autonomous pond behavior. It must not turn
this 12 s skill test into an infinity claim.
