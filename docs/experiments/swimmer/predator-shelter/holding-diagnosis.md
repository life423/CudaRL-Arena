# Protocol-5 holding failure diagnosis

October 29, 2026. This is a diagnosis of the existing saved learner, not a new
training campaign or a deployed controller. The checkout remains
`codex/autonomous-pond-practice` at
`5ea5d452589c70418836209c00921b84ddcb9dc2`, with the pre-existing dirty protocol-5
and overhang work preserved. Overhang presentation/default-menu work is parked.
The later larger-bevel overhang shape is not validated by these layout-2 tests.

## Existing implementation and evidence

Protocol 5 already divides the actor's final weights and biases by the saved
per-joint gains `[8,4,1024,2]`. Cover means use these physical output units;
food means multiply by the saved gain, and actor backprop applies that same
food gain. Gaussian sampling/log probabilities use the physical means and the
unclipped Gaussian samples; physical torque is then clipped to `[-1,1]`.
The earlier final-layer conditioning fix is present and was not repeated.

The protected `motor-v5-300.log` still records 0/24 five-second near-calm holds,
mean longest safe streak 0.458333 seconds, and calm feeding 11.5/min versus the
inherited 12.5/min. Component-test passes do not establish the missing skill.

## Matched physical comparison

`tools/holding_diagnose.cu` runs the actual shelter `Batch` with layout 2 and
calm water, using identical complete cloned states for every controller.
Autorestart and practice are off. There are four exact stationary cover poses,
24 existing near-calm cases and 16 signed velocity cases. The latter use
forward velocity ±0.003 m/s with yaw ±0.3 rad/s, or vertical velocity ±0.002 m/s
with body-pitch rate ±0.3 rad/s. These are mild disturbances, not proof of
braking from full swimming speed or arbitrary arrival states.

Each trajectory has 200 physical 50 ms ticks. Success requires five consecutive
seconds with sampled whole-body predator exposure ≤0.05 and minimum
bed/solid/surface clearance ≥0.08 model units, including the distal fin.
One model unit is 4 cm; the clearance margin is 3.2 mm. The diagnostic checks
every tick, independently of the production cached exposure and goal-distance
streak counter. No stabilizer, teleport or rescue is deployed.

| Deterministic controller | Exact rest | Near rest | Initial velocity | Mean discounted return: exact / near / velocity |
|---|---:|---:|---:|---|
| Zero torque | 4/4 | 20/24 | 11/16 | 10.67 / 10.18 / 8.52 |
| Calibrated inherited actor | 0/4 | 0/24 | 0/16 | −1.03 / −0.69 / −1.15 |
| Frozen protocol-5 actor | 0/4 | 0/24 | 0/16 | 2.15 / 2.46 / 2.04 |

Zero torque earns greater discounted return than either actor in **all 44
matched cases**. Thus safe stillness is attainable from every exact resting
pose and preferred by the existing return. It is not yet established from
every disturbed state: passive zero torque fails five of the velocity cases.

The unchanged reward is decomposed in the trace and checked against the actual
GPU-returned value. Maximum reconstruction error is 7.16×10⁻⁷. All 61,600
recorded rows are finite; clocks advance exactly without done/truncation or
resets. All 308 variant/case complete starts are byte-identical to their
fixtures. The fenced repeat produces a byte-identical `trace.csv` (SHA-256
`d27bdfbfa6086d0ef0d19d4f9530a6ee9ab3f0a8580f43b9743f48f3b9c396e3`).

### Failure order and bookkeeping

Protocol 5 first loses safety through exposure in 41/44 cases, bed margin in
2/44 and boulder margin in 1/44. Every first safety failure occurs before
negative penetration. Later first penetration is bed in 33/44 and boulder in
11/44. Surface containment is not the first failure in this sample.

For example, nominal pose 0 first loses cover at 0.60 s: exposure 0.09259,
clearance still +0.26181 and no obstacle contact. Bed penetration begins only
at 1.65 s. The fin/contact discrepancy cannot alone explain this earlier loss
of predator cover.

Four near cases already exceed 5% exposure at t=0. Their positive clearances
remain valid; they are reported as initially exposed rather than counted as
departures from shelter. Excluding the first diagnostic tick, as the old probe
does for its approach interval, exactly reproduces all 24 historical protocol-5
longest-hold values and their 0.458333 s mean. Counting that tick gives 0.495833 s.
This is a measurement-window difference, not an improvement or a simulator split.
The zero training counters at update 300 are separately explained by an episode
reset boundary; they do not invalidate the fresh no-reset failure result.

### Confirmed fin/contact discrepancy

Safety proof samples nine axial slices with the full distal fin. Actual bed
contact uses three points and 2 mm tail radius, while proof includes the 6 mm
distal fin envelope. `arch.contacts` counts solid obstacles, not bed contacts.

In protocol-5 `near_20` at 4.40 s, proof head/proximal/fin bed gaps are
`+0.06430, +0.09610, −0.00352` model units. The actual three-point bed-contact
gaps reconstructed from that same pose are `+0.06539, +0.09876, +0.12080`.
The proof sees a penetrating fin while bed-force sampling sees no overlap.
This is a confirmed physical-model/proof mismatch; it is not proof that fixing
it alone would produce learned holding. Thresholds remain unchanged.

### Exploration and saturation

With matched physical Gaussian noise (log standard deviation −1.5), zero-mean
control falls from 35/44 deterministic holds to 2/44. Both neural actors still
achieve 0/44 with common noise, and protocol 5 achieves 0/44 with its saved
exploration. Noise is therefore a contributor, but not the sole explanation:
the deterministic protocol-5 mean already fails from rest.

The saved standard deviations are approximately `[0.2111,0.2229,0.4857,0.2224]`.
Deterministic protocol-5 cover actions clip only 0.0227% of joint-3 values and
none on the other channels in these traces. The old permanently saturated
raw-output pattern is not the current deterministic cover failure.

## Fixed-batch update audit — corrected and repeatable

`tools/holding_update_diagnose.cu` captures one 64-step on-policy rollout on
disposable full-save copies, with separate food/cover gradient contributions
after the same shared food/cover per-minibatch advantage normalization. A first diagnostic run
failed gradient additivity and is retained under `update-audit/INVALID.txt`.
Its actor/std gradient attribution and copied-step effects must not be cited.

That diagnostic used default-stream host uploads immediately consumed by a
nonblocking stream. The tool now uses explicit consumer-stream copies/fences
and mandatory upload, row-gradient and parameter-gradient additivity checks.
This is an instrumentation fix, not a change to production PPO or a confirmed
production-training defect. The corrected v2 audit and v3 reuse of the exact
saved batch pass the upload, row/parameter-gradient additivity and input
immutability guards. All 65,536 old action log probabilities replay bit-for-bit;
means, values and GAE also reproduce exactly. Restoring complete learner state,
including Adam configurations/moments/step counters and environment/random state,
produces bit-identical full 32-minibatch PPO updates on two copies. Adam advances
from 9,600 to 9,632 steps. The fixed-rollout artifact hash is
`8ee3bc669e1d2a01767d5c5e1b14a2ef54ae2d75e15047ffe7786563129b90aa`.

### Required Jacobian and measured conditioning

Let `z_j(theta)` be actor output, and `g_j` the saved actuator gain. Protocol 5
uses physical Gaussian mean `mu_j=z_j` on cover and `mu_j=g_j*z_j` on food.
Consequently `dL/dz_j=dL/dmu_j` on cover and `g_j*dL/dmu_j` on food. These are
required chain-rule factors. Setting both to one without changing the forward
parameterization would give the wrong derivative. Neither production factor was
changed. Log probabilities are for unbounded latent Gaussian actions; the
environment clips sampled torque. Large latent KL is not clipped-torque KL.

The rollout contains 32,768 food and 32,768 cover samples. The audited first
production minibatch contains 4,004 food and 4,188 cover samples. Both task
contributions use the same shared minibatch advantage mean −0.198373/std 0.778913 and the
original 8,192-sample denominator. Splitting does not renormalize each task.

| First minibatch measurement | Result |
|---|---:|
| Food actor-gradient L2 norm | 99.46395 |
| Cover actor-gradient L2 norm | 0.33840 |
| Combined actor-gradient L2 norm | 99.47975 |
| Food/cover actor-gradient cosine | +0.04502 |
| Combined global clipping multiplier | 0.005026 |
| Food/cover critic-gradient cosine | −0.42438 |

Thus food dominates this shared actor clipping norm, but the actor gradients are
nearly orthogonal, not directly opposing. The fresh cover contribution is scaled
by about 1/199 before Adam in the combined step. This is **not** proof of 199×
smaller parameter learning: saved Adam moments/preconditioning also matter.
Over all 32 minibatches clipping multipliers range from 0.000714 to 0.005026.
The large food joint-2 gain is mathematically correct and its latent mean is
already almost always saturated as torque. The complete update changes food
latent-Gaussian KL by 3.692 (clip fraction 0.723), versus cover KL 0.000479
(clip fraction 0.000061); joint-2 raw mean RMS changes by 1.324 while its clipped
mean changes by only 0.0287. This establishes conditioning/saturation concerns,
not a broken likelihood calculation or the behavioral cause by itself.

### One-step controlled effects remain inconclusive

Copied Adam steps preserve saved moments. A zero-current-gradient/moments-only
control separates old optimizer momentum from new loss effects. A separate
actor-balanced one-step counterfactual retains the required Jacobian but weights
the food objective by 0.0034023, with the original std gradient unchanged. It is
an explicitly changed objective, not a derivative correction or deployed fix.

All seven frozen policies (baseline; food-only, cover-only, combined, moments-only,
balanced copied steps; and full 32-minibatch combined PPO) achieve **0/4 nominal
and 0/24 fresh near holds**. Fresh mean longest streak ranges only 0.435–0.450 s.
The full update nudges the mean action toward known safe zero commands at all
four fixed nominal observations, but does not improve actual continuous holding.
Small deterministic feeding samples range 11.5–12.25/min calm and 10.5–11.5/min
in 0.75 cm/s current. These one-step outcomes cannot decide learning success,
capacity or which optimizer mechanism causes the plateau.

## Completed bounded comparison

The protocol in `holding-followup-protocol.md` was registered before execution.
Four complete update-300 learner copies run 16 iterations of 1,024×64 physical
steps. The three learning arms each make 16 standard PPO updates (four epochs,
eight minibatches), **48 distinct PPO updates total**. No-update advances a
separate logical sampler clock while preserving every weight, Adam state and
learner update counter. First rollout payloads match across all arms. Later
rollouts are freshly on-policy for each diverging policy.

Only actor/std loss rows are masked for holding-only or feeding-only. Each
minibatch retains the production shared advantage statistics and 1/8,192 loss
denominator; the critic still trains on both tasks. Thus this isolates removal
of a task's policy/std update contribution, not every possible critic or data
interaction. Initial exploration matches; learned std may subsequently diverge.

Development evaluation is fixed at iterations 0/8/16. Final held-out cases and
fresh food seed are used only after all arms finish. Combined iteration 1
reproduces the validated full update-301 snapshot exactly. All three final
update-316 saves read back and restore exactly, including Adam step 10,112.
The frozen control produces bit-identical development scores at 0/8/16, and
evaluations leave learner worlds, random counters and optimizers unchanged.
Every new rollout passes pre-update likelihood replay checks. Source unchanged.

| Final iteration-16 arm | Nominal holds | Blind near holds | Mean longest near streak | Reference food/min: calm / current | Fresh food/min: calm / current |
|---|---:|---:|---:|---:|---:|
| No update | 0/4 | 0/24 | 0.404 s | 11.50 / 11.50 | 12.375 / 11.50 |
| Holding-only actor/std | 0/4 | 0/24 | 0.571 s | 1.50 / 1.875 | 1.50 / 2.875 |
| Feeding-only actor/std | 0/4 | 0/24 | 0.423 s | 12.25 / 10.625 | 12.25 / 11.50 |
| Current combined | 0/4 | 0/24 | 0.413 s | 11.75 / 10.75 | 12.00 / 11.125 |
| Original inherited reference | 0/4 | 0/24 | 0.585 s | 12.50 / 11.625 | 12.25 / 11.625 |

Current is 0.75 cm/s. Each food cell uses 16 worlds×600 ticks (30 s/world),
eight simulated world-minutes. These are small paired retention screens,
not universal performance claims. No-update, feeding-only and combined exceed
90% of inherited feeding on both seeds/currents; holding-only fails badly.
**Every arm fails the unchanged five-second holding gate.**

The blind near suite contains six fixtures already exposed at initialization:
near 3,5,6,8,9,11. All are physically clear. Results are 0/18 holds from initially
safe near starts and 0/6 recoveries from exposed starts for every learned arm;
do not describe all 24 as departures from safe cover. Passive zero torque holds
18/18 initially safe near fixtures and all four exact poses on this seed.

Holding-only improves partial metrics (blind worst clearance −0.0112 versus
no-update −0.1259), but even its reduced penetration depth remains a failure.
Its actor gradients no longer require global clipping in the last checkpoint
updates; lack of food-driven clipping is therefore not sufficient to solve
holding in this budget. It also catastrophically changes deterministic feeding,
so shared exploration alone cannot explain that retention loss.

### Final noise, reward and velocity traces

The two actors selected in advance (final holding-only and final combined) each
receive the complete 44-case no-reset trace: four nominal, 24 blind near and
16 signed velocity starts. Both score **0/44** under deterministic means,
common physical noise and their own saved standard deviations. Passive zero
holds 4/4 nominal, 18/24 near and 11/16 velocity cases; common-noise zero-mean
control holds only 3/44. One trajectory per noise condition/case is diagnostic,
not a success probability. No learned braking recovery is demonstrated, including
the five mild velocity starts where passive control fails.

Of the 38 initially safe starts, holding-only first fails through exposure in
28 and bed safety margin in 10. Combined first fails through exposure in 35,
boulder margin in two and bed margin in one. All these first margin/exposure
failures precede any negative penetration. Holding-only nominal 0 first loses
cover at 0.80 s without ever penetrating; combined loses it at 0.60 s and only
later penetrates the bed at 1.65 s. The fin/contact discrepancy remains a real
separate defect, not the dominant explanation for these initial failures.

Holding-only improves deterministic return (nominal mean 4.564, velocity 4.530)
over parent update 300 (2.152 and 2.042). Combined reaches 2.488 and 2.338.
Passive zero still earns more return in 44/44 paired cases for both final actors
(nominal/near/velocity means 10.675/10.012/8.523). This answers the deterministic
comparison; it does not establish the global expected stochastic PPO optimum.
Reward reconstruction and observation checks pass (residuals below 7.2×10⁻⁷ and
1.44×10⁻⁶ respectively). Soft reward still grants partial credit outside strict
cover; higher return has not produced the required continuous behavior.

At nominal 0 the holding-only mean changes from
`[.07952,−.25580,.06241,.27729]` to `[.04376,−.20430,.03568,.36572]`.
Several components shrink while the fourth grows; it has not learned the known
safe zero command. Inputs explicitly signal zero goal error, zero linear/angular
velocity, zero joint rates, cover mode and desired heading. The corrected
gradient path is active, but the resulting controller still creates movement.

## Diagnosis and decision

**Confirmed defects/limits:** the initial diagnostic had an upload ordering
fault (corrected and repeat-verified); strict fin clearance and bed-force sampling
disagree; the shared policy couples cover updates to feeding, with measured
retention loss; food dominates shared gradient clipping in the audited batch.
The single four-value exploration vector cannot condition variance on task.
Cached exposure affects reward without being observed, adding reward-state
aliasing; its importance to this failure is not isolated.

**Ruled out as sole explanations:** unavailable resting equilibrium; the old
missing gain correction; broken replay likelihood/required chain rule; incorrect
Adam continuation; missing goal/mode/velocity inputs in these fixtures; stochastic
noise alone; and bed penetration as the first failure in most tested cases.
Removal of food actor/std updates is not sufficient within the registered budget.

**Unresolved:** relative roles of inherited optimization, shared weights and
clipping, mode-independent exploration, partial-credit rewards, and observation
history. Sixteen updates do not establish a capacity limit or that PPO cannot
learn the skill. This fixed-width MLP is not structurally forbidden from producing
zero on cover and nonzero food actions; exact retention of the entire inherited
food function with those same weights is not proven.

The only correction justified and validated so far is the **diagnostic ordering
fix**, not a new learning-system fix. Do not arbitrarily remove required gains,
deploy holding-only, lower safety thresholds, add a brain, or extend this budget.
Before any redesign, a next discriminating experiment should establish whether
the existing conditioned motor can fit known safe cover commands while preserving
its food outputs on frozen matched observations. Success would direct attention
to PPO optimization/credit assignment; failure would still need to separate
optimization from representation. This is a proposed next diagnostic, **not run**.
A task-specific motor output/variance change remains a hypothesis for discussion,
not a proven necessary redesign.

Held-out five-second holding and feeding retention have **not both passed**.
Recovery from the tested mild velocity starts has not passed. Approach/arrival,
autonomous choice of shelter over food, concealment throughout danger and
resumption of feeding remain unproved. An assigned shelter goal does not test
that goal-selection sequence.
No actor has been promoted or opened as a successful predator demonstration.

### Direct answers

| Question | Evidence-backed answer |
|---|---|
| Is safe holding physically achievable here? | Yes at all four exact stationary poses and initially safe near fixtures under passive zero torque. Not established for every arrival/disturbance. |
| Does it earn better return? | Zero torque beats both old and final neural actors in all 44 matched deterministic cases. Expected stochastic optimality is unproved. |
| Can the frozen policy's deterministic mean hold? | No: parent and both final actors fail all nominal/near/velocity five-second cases. |
| Does PPO move toward holding? | Some action/return/short-streak metrics improve; no required behavioral success after the bounded comparison. |
| Do feeding updates interfere? | Food dominates shared clipping; removing it changes cover learning but does not solve holding, and cover-only updates severely damage food retention. A unique mechanism is not established. |

## Evidence and preservation

Compact reviewable results: `holding-controller-results.json`,
`holding-update-results.json` and `holding-followup-results.json`.
Raw evidence is under `build/holding-diagnosis/2026-10-29-protocol5/`, including
the pre-diagnosis dirty-tree patch/untracked snapshot, protected SHA-256
manifest, full traces, per-case JSON and the invalid initial audit.

The source protocol-5 save SHA-256 is
`13dadfe3eb0b76251a7ea93870f9e373e77c97589f5843dae2abf10174e5b7b5`;
the original boulder source remains
`d9a5c0fd07163c6558fb11c30e6db9cd65ca98dd5711f20e631f080ce668d8a6`.
All 51 protected files match the pre-diagnosis manifest. No diagnostic actor is
adopted and no personal checkpoint is an output. Existing overhang work remains
preserved and parked. No commit/push/merge or production learning change was made
for this diagnosis. Focused calibration/physical sampling/finite-difference GPU
tests pass 5/5 with zero skips; they support the math audit, not learned mastery.
