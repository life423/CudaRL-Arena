# Continuous swimming and learned pond turns — November 23, 2026

**User-authorized preregistration; execution has not started.** The current
objective is continuous swimming in the pond with learned turning and no food.
New current compensation and gait-straightening work are deferred. This is a
fresh finite study, not a resumption of a stopped study or reuse of its budget.
Only the root execution owner launches physical/optimizer phases after source
freeze and successful zero-integration preparation.

The creature must decide when and how to turn from actual body/bank sensing.
No food target, waypoint route, scripted heading change, rotated/translated
body, shoreline rescue, bounce, artificial inward force or world reset supplies
that behavior. Godot displays the native body and offers continuous Free swim.
The live mode has no eight-second, sixteen-second or other finite task terminal.
Finite training/evaluation horizons belong to the harness, not the product.

## Preserved learned motor and new steering parameters

Preserve Model 6 geometry, mass, inertia, water forces, contacts, ten nominal
5 ms substeps per 50 ms control, 1 Hz target filter, four actuator outputs and
all +/-10 microN*m torque limits. This study uses zero current. The frozen
41-input network is canonical checkpoint 336, SHA-256
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
The frozen source 7 learned holding checkpoint is
`C54F73B57AB4280D99802A7253C13004F95C754E39E0ACF8F6833377BBB760C3`,
including both saved parameters and gain`0.26211655139923096`.

Old explicit fixed-heading SWIM and stationary HOLD requests preserve their
original branches exactly. HOLD remains `saved_gain*base[j]`; its parameters
are not retrained. Free swim uses that same backbone and four-output motor,
with sixteen new episode-level Gaussian means, eight for each yaw joint.
Pitch outputs remain the inherited network's outputs. There is no new gait
shape gate and no separate food/goal-selection policy in this experiment.

Free-swim observations use a read-only local-forward motor request. Horizontal
lane/destination error is zero; the vertical path error retains the original
fixed world-space depth. Requested heading is the current physical head's
horizontal forward direction, with requested horizontal velocity 8 mm/s along
it. The original body-frame feature units, actual velocities, joint/filter
states and uninterrupted clock remain. Startup feature 34 reaches zero after
40 controls and stays zero; feature 40 is zero in Free swim. This is not an
external route, a desired turn or a body rotation. Old explicit request modes
keep their original observation semantics.

The eight sensed features, in order, are:

1. Constant 1.
2. `1/(1+d_front)`.
3. `1/(1+d_side_positive)-1/(1+d_side_negative)`.
4. Outward bank-normal component along body forward.
5. Outward bank-normal component along body side.
6. `tanh(body_yaw_rate/.5)`.
7. `tanh((.008-body_forward_COM_velocity)/.004)`.
8. `tanh(body_side_COM_velocity/.008)`.

Here `d=clamp(directional_full_body_gap/.072,0,4)`. A directional gap is the
minimum positive translation distance to the active ellipse over all 24 native
body/fin support corners along the sensed horizontal direction. The normal is
the normalized active-ellipse gradient at the actual COM, projected onto
horizontal body forward/side. At horizontal COM radius<=1 um the normal is
exactly zero, avoiding amplification of center-placement roundoff. Horizontal
forward is the physical head+X direction projected ontoXZ and normalized, with
a finite+X convention for a vertical head; side is
`{-forward.z,0,forward.x}`. These conventions never rotate the actual body.
Yaw rate is the world-Y angular velocity. Distances, normal and body-frame
velocities come from shared native geometry/kinematics. Sensors never choose
a desired turn or move the body.
One independent sixteen-dimensional vector `u~Normal(mu,0.20)` is sampled
once per whole training episode and retained for every actual control.
For yaw joints 0 and 2, respectively, use the corresponding eight coefficients:

```
correction = .75*tanh(sum_k(tanh(u[k])*feature[k]))
actual_yaw_latent = backbone_yaw_latent + correction
actual_pitch_latent = backbone_pitch_latent
command[j] = tanh(actual_latent[j])
```

An all-zero vector explicitly returns the original backbone output. The new
means start at zero, and deterministic assessment/inference uses `u=mu`.
Store a separate versioned policy/package with parent hashes, sixteen means,
fixed sigma, feature/mapping contracts and fresh optimizer history. Never
reinterpret old packages or adopt an unqualified new actor.

## Actual pond and valid missed-turn terminal

The shared active native/view ellipse has semiaxes **0.20 m and 0.12 m**. The
visible flat water extends to 0.22/0.14 m; this is not permission for native
body escape. The behavioral inset ellipse has semiaxes **0.195/0.115 m**.
This is a 5 mm reduction of each semiaxis, not an asserted uniform Euclidean
5 mm offset. Full articulated native body/fin bounds are checked, not only
the head or COM. The rendering and native boundary contracts use the same
world coordinates; no smaller hidden rectangular swimming box is substituted.

After each accepted 5 ms substep, test all native body/fin support corners.
The first entry outside the inset is a legitimate missed-turn task terminal,
provided the actual hard active ellipse and every physical/numerical guard
remain valid. Stop the remaining substeps of that control, record its actual
1..10 substeps/duration, emit true episode done once and pad later rows inactive.
This is valid PPO data with a failure penalty, not an invalid batch. Never
rescue, clamp, reflect or reset that body. An actual hard-ellipse escape,
nonfinite state, invalid joint/torque/contact state or numerical failure remains
a global physical/data stop before optimizer eligibility.

This separation is necessary: an untrained creature must be able to experience
a missed turn and learn from it. It must not have to solve the task before the
first PPO step. Qualification requires no missed-turn terminal at all.

## Starts and horizons

Training has sixteen fixed starting cases, alternating original joint pose 0/3
by case id. Position is the actual COM, not a point substituted for it.

- Four centered COM cases, with horizontal headings `k*pi/2`, `k=0..3`.
- Twelve near-bank cases: COM at normalized ellipse radius 0.55 at cardinal
  azimuth `k*pi/2`, `k=0..3`; for each location, heading is the actual outward
  ellipse normal angle plus `(g-1)*pi/3`, `g=0..2`.

The eight withheld starts are four centered cases with diagonal headings
`pi/4+k*pi/2`, and four near-bank cases at radius 0.55 at those diagonal
azimuths, heading along the outward normal plus alternating+/-45 degrees.
No withheld result is used to adjust the selected candidate.

All starts have zero velocity and the registered original depth, physically
validated body/fin clearance and original filter state. Placement is an
explicit episode reset only; no placement occurs within an episode or any
long validation world.

Use **4,096 GPU worlds**,256 per case, with a maximum **320 controls/16 s**
per training episode. An ordinary early boundary terminal is true done with
inactive padding, not a reset or truncation/bootstrap. The harness deadline
also produces the registered whole-episode terminal. In Free swim and frozen
long evaluation the native world remains continuous/alive: the external
harness stops collecting at its deadline without installing a product limit.

## Reward, episode probabilities and PPO

For each accepted control use actual duration `dt=actual_substeps*.005` and
actual body-frame COM velocity. The reward rate is

```
1 - ((v_forward-.008)/.008)^2 - .25*(v_side/.008)^2
  - .05*(depth/.001)^2 - .05*(roll/10)^2 - .01*(pitch/10)^2
```

Multiply this rate by actual`dt`, then subtract
`(.05/8e-6)*positive_joint_work_delta`, where that delta is exactly
`double(post_BodyState.work_float)-double(pre_BodyState.work_float)` from the
recorded states. The double-precision physical energy ledger is unchanged.
Add **-4 once** for an ordinary missed
turn. Episode return is the full accepted reward sum divided by **16**, never
by the survival duration. Use the native work difference, not an unrecorded
mean-squared-torque field. Zero horizontal motion earns approximately zero
motion reward and still pays work/posture costs; spinning in place receives no
travel bonus. No food, route or scripted center-seeking reward is introduced.

Score the original sixteen-dimensional Gaussian choice exactly once per
episode. No tanh-command probability/Jacobian, repeated per-tick score,
critic, GAE or discount bootstrap is introduced. Baseline: leave-one-out mean
of the other episodes in the same initial case, group 256/denominator 255,
independent of action and resulting trajectory. Fit one positive finite global
within-case residual RMS on one independent calibration batch, then freeze it.
All inactive rows have zero credit and no new body/optimizer steps.

Fresh sixteen-mean Adam: LR 0.005, beta 1=0.9, beta 2=0.999, epsilon 0.00001,
global gradient norm cap 0.5. PPO clip 0.2, fixed sigma 0.20, epoch KL guard 0.01,
at most four epochs and four 1,024-episode minibatches per epoch. Maximum
16 Adam steps/update, **64 updates/1,024 Adam steps** in the study. Calibration
seed is `0x4E23B197`, training seed `0xA715C2D3`, indexed by
`episode*16+coefficient`, with counter 0 for calibration and 1..64 for updates.
There is no reseeding, retry, actor selection from reserved results or
unregistered hyperparameter search.
Permutation seed is `0x0316412d + update*4 + epoch`, with four complete
1,024-row minibatches and no omitted last row.

## Separate physics, pointwise policy and closed-loop diagnostics

The stopped gait attempt used independently inferred CPU/CUDA commands in
its trajectory-comparison gate. At control 32 it exceeded the registered
2e-5 rad/s yaw-rate/reference-rate tolerance;960 controls and zero PPO/Adam
were spent. Its failure and exact sources remain closed and preserved.
That closed-loop comparison is not relabeled as a physical-solver failure.

This new gate tests physical agreement with **identical command sequences**
and identical initial states. Keep every existing physical tolerance: position
2e-5 model units, quaternion/joints 2e-5, linear velocity 2e-7 m/s,
angular/joint velocity 2e-5 rad/s, actual normalized torque 2e-5 and six cumulative
energies 1e-12 J +1e-4 relative. Exact legacy/new CPU defined-state parity
remains required for complete ten-substep controls.

- Ten old fixtures (four SWIM/six HOLD),40 controls on legacyCPU/newCPU/newCUDA:
  **1,200 controls**.
- Sixteen Free-swim training starts,40 controls on newCPU/newCUDA with the same
  CPU-generated commands: **1,280 controls**.
- Total fresh gate: **2,480 controls/24,800 substeps**.

Separately check CPU/GPU network and learned decode outputs pointwise on
identical observations/sensors, including nonzero Gaussian rows, identity,
holding passthrough, all four full minibatches/gathers/reductions and last rows,
correct mean-loss finite differences and checkpoint/optimizer round trips.
These are zero-integration/zero-Adam fixtures. Independent closed-loop CPU/GPU
trajectory drift may be reported diagnostically; it is not substituted for or
used to relax the matched-command physical comparison.

## Selection, long frozen validation and retained skills

Baseline includes the original 4 short swims/6 learned 64 s holds/2 x 60 s
swimming reference (10,720 controls), plus at most sixteen 60 s zero-head
Free-swim probes (19,200 controls), stopping each legitimate failed case without
rescue. Then one independent 4,096-world calibration batch (1,310,720 maximum
controls). Require finite correct rows/episode likelihoods/returns, actual
terminal masks/counts and positive residual RMS. No artificial minimum count
of lucky successful random episodes is imposed. Independently audit readiness
and release it by integrity-bound marker before the first Adam step.

After every PPO update, test all sixteen fixed starts for 60 s. A selection
probe passes only with no missed-turn/hard failure, depth<=1 mm and whole-body
tilt<=10 degrees, overall horizontal COM path speed>=6 mm/s, every complete
nonoverlapping 8 s motion window after 4 s>=3 mm/s, and COM spans at least
0.12 m inX and 0.06 m inZ. Report startup separately. At the first 16/16 pass,
freeze the candidate and stop learning before reserved validation. If 64 updates
exhaust first, freeze the final candidate as unqualified; no automatic extension.

Run **eight withheld starts x 30 uninterrupted minutes** after successful
selection, at most 36,000 controls each. Each must remain physically valid and
alive with no missed-turn event/reset, maintain depth<=1 mm/body tilt<=10
degrees, have overall horizontal COM path speed>=6 mm/s and every complete
nonoverlapping 8 s motion window after 4 s>=3 mm/s, traverse COM spans
**X>=0.20 m/Z>=0.12 m**, and accumulate at least **2*pi radians** of meaningful
actual heading change. Unwrap physical body azimuth and maintain an anchor
initialized to its first angle. While `heading-anchor>=pi/4`, add a positive
`pi/4` turn and advance the anchor by that amount; while
`heading-anchor<=-pi/4`, add a negative `pi/4` turn and reduce the anchor by
that amount. Require positive accumulated turns plus absolute negative turns
to reach 2*pi. Report both signs, net unwrapped change, unwrapped range and
raw absolute yaw variation. This 45-degree hysteresis excludes small gait
wiggle while accepting genuine alternating U-turns; it is qualification
measurement only, never action feedback or another controller.
These are finite measurements supporting continuous operation, not proof of
mathematical infinity.

Motion is the horizontal **COM polyline arc length**, summing actual accepted
control-boundary COM displacements, divided by actual elapsed time. It is not
net endpoint distance, fixed-forward displacement or legacy head/root travel.
Heading comes from physical orientation with proper unwrap; joint oscillation
or a renderer rotation cannot count as turning. Coverage gates reject a small
stationary orbit that never uses the pond.

Run all old retained-skill fixtures on the selected/final frozen candidate:
four 8 s swims; poses 0/3 x 60 s with all old 28 passing window keys retained;
three 64 s fixed-depth recoveries at pose 0 with offsets 0/−0.5/+0.5 mm and
every final 8 s offset error strictly below 0.25 mm; poses 0/3 x 300 s swimming
with all old 148 passing window keys retained; six original 64 s holds at
poses 0/3 xoffsets 0/±0.5 mm; six withheld 64 s holds at poses 1/2 x 0/±0.25 mm;
two centered 300 s holds at poses 0/3. Total **46,240 controls**. Old passing
window sets are subsets; genuine startup improvements are permitted. Preserve
all old speed/RMSE/path/lane/depth/tilt/work and whole-body holding criteria.
No new joint-pitch/gait-straightness criterion is added.

After full qualification only, run one bounded diagnostic ablation: retain the
two intercept coefficients `u[0]` and `u[8]`, set the other fourteen
coefficients to zero, and repeat the sixteen starting cases for sixty seconds each
(19,200 controls maximum), comparing with already recorded selected-policy
prefixes. This reports state dependence versus constant bias; it does not
select/tune a new actor or add a qualification hurdle.

## Fixed maximum accounting, storage and adoption

| Phase | Maximum controls |
|---|---:|
|Matched-command gate |2,480|
|Old learned baseline |10,720|
|Sixteen initial Free-swim probes |19,200|
|Independent calibration |1,310,720|
|64 fresh PPO collections |83,886,080|
|64 sixteen-case 60 s selections |1,228,800|
|Eight 30 min withheld worlds |288,000|
|Old retained-skill suite |46,240|
|Constant-intercept coefficient ablation after qualification |19,200|
|**Total** |**86,811,440**|

If the inherited zero head already passes all sixteen baseline sixty-second
cases, including motion and coverage, freeze update zero and skip calibration
and PPO. Run the same one-shot withheld validation and retained-skill suite.
A withheld failure closes the candidate without training or reselection. A
qualified update-zero result demonstrates inherited behavior; it is not a claim
that the new turning coefficients learned anything.

Maximum study substeps **868,114,400**; actual partial-terminal substeps are recorded
and need not equal controlsx 10. Enforce **900 combined native/process active
seconds** across all physical phases. No physical cap or time allowance is
silently extended. First invalid physics/numerics/data/I/O stops affected work;
ordinary task failures remain valid terminal observations.

Compact evidence retains full initial worlds and fixed depth/heading context,
sampled sixteen-dimensional choice/logp/means, per-control active/terminal and
actual substep flags, and 40 float 32 values: PRE filter/torque context 12,
POST BodyState 23 (including work/travel), actual applied latents 4 and reward 1.
Reconstruct all 41 network inputs and all eight sensed features independently
from actual states/geometry. Preserve complete episode returns, terminal
causes, final worlds/tasks, all checkpoints and acceptance traces. Every actual
control is audited; padding is not simulated time or evidence of swimming.

Initially require projected full new evidence **<=20 GiB**, with that projection
plus **8 GiB free working reserve** fitting the actual disk. Before each batch's
allocation/collection/writes, require its full parts/metadata/overhead plus the
8 GiB reserve to fit, and the total written/projected study evidence to remain
within 20 GiB. This does not require 20 GiB to remain unused after writing data.
Keep the native 512 MiB envelope limit using bound split parts when required.
For 4,096 worlds, the physical advance kernel uses 64-thread blocks and 64
blocks to expose work across the device's 48 SMs; initialization/observation
remain 128-thread launches. Actual compiled register/local-memory and resident
block metadata are recorded during zero-integration preparation. This launch
choice changes no state indexing, random stream, equations, PPO math or budget.
Do not delete or overwrite prior data. All parents, approved protocol, binary,
transitive source/library contracts and initial optimizer/means freeze by exact
hashes before physical execution; prior zero-preparation failures are retained.

The offline verifier uses saved data only, with no body or optimizer calls.
Its final proof includes `assessment_sha256` matching the exact native
`training-result.json`, `passed`, `qualified`, final candidate SHA and a flat
`input_sha256` binding every required result/source/policy. App metadata requires
the matching assessment/proof pair and a qualified package. Copy only minimal
portable assessment/proof metadata into product resources after qualification;
failed attempt details remain available separately. Qualified saved inference
is not described as live PPO updates.

After independent qualification only, one separately counted **200-control /
2,000-substep** actual Godot native/render capture starts at the default centre
case and continues for ten physical seconds, beyond the old eight-second stop.
Verify each accepted native state's bounds, depth, tilt and uninterrupted time,
and verify displayed articulated poses exactly match the same native
`get_state` poses (float32 position difference at most 1e-6 model units and
normalized absolute quaternion-dot error at most 1e-6), with policy file
SHA/version/means bound to the qualified
candidate. No UI physics or scripted movement may supply those poses. Record
actual motion diagnostics without claiming this short CPU capture establishes
thirty-minute CPU endurance. No extra reference simulation or independently
inferred CPU/GPU closed-loop trajectory comparison is permitted by this
presentation check. Paused rendering/UI checks have zero physical steps.
The separate app cap is 200 controls / 2,000 substeps; the combined study plus
app maximum is **86,811,640 controls / 868,116,400 substeps**, under the same
combined 900-second active/process cap. The native study cap above is unchanged.
Normal continuous product use is enabled only after the matching qualified
package/proof and these fidelity checks.

No unqualified policy adoption, hidden route, food system, current/gait feature
expansion, merge or deployment is authorized by these numerical provisions.
Preparation and independent zero-integration verification must complete before
the root freezes the entire causal-input closure and launches physical work.
New protocol/checkpoint/trace/phase-release versions bind their exact fields,
all parents and predecessor hashes; they do not reinterpret old saved formats.
