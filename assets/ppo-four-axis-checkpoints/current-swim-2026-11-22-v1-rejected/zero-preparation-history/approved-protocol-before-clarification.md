# Current swimming v1 — preregistered execution protocol, November 22

**Preregistered; execution not started.** The execution owner will preserve this
exact protocol in the unique run directory during preparation, then freeze the
executables and transitive inputs after zero-integration checks and before the
first physical gate. The user's instruction is to make current-capable swimming
work while preserving learned calm-water abilities. Earlier studies and their
unused budgets remain closed. This authorizes one finite attempt, not retries.

## Objective and scope

Measure purposeful **pond-relative swimming through specified steady horizontal
currents**, using the existing four-output body motor and native C++/CUDA PPO.
Success means controlling travel relative to the pond, not accepting advection
with the water as swimming. Retain the verified calm swimmer and learned calm
holding. Current holding, changing/rotating currents, disturbance recovery,
moving-start braking, growth and new limbs are outside this study.

Keep the old live viewer, original checkout, all previous checkpoints/full saves
and evidence. Do not modify the twelve pinned legacy controller-contract sources
to admit current. New current model/motor/task and observation definitions are
explicit versioned experimental contracts. No production adoption, merge,
deployment, extra seed, retry or automatic follow-on study follows from this run.

## Physical model and compatibility

Preserve Model 6 geometry/mass/inertia, both universal joints/four axes,
the +/-10 microN*m internal torque caps, joint limits, the 1 Hz target filter,
50 ms controls and ten 5 ms physical substeps. The imposed flow is a water
condition, never a direct body-position/velocity adjustment. Use water-relative
point velocities in the same distributed drag law. No forward-thrust term,
teleport, moving lane/anchor or numerical rescue is introduced.

The old composite movement model ignores the physical world's current field;
the new path must implement current in that composite model explicitly. Its
zero-current branch delegates the old computation to preserve exact numerical
behavior and old checkpoint compatibility. Keep the existing calm-only motor
guard intact for the original path.

For fixed drag coefficients at a physical substep, let `R` be translational and
rotational resistance, `F_water` the generalized imposed-water load and
`S=sum(c*u_axis^2)` over translational quadrature components. Audit actual
post-solve velocity `v` using water-relative dissipation
`D=v'Rv-2*F_water.v+S` and reservoir-supply power `P=S-F_water.v`.
Then `P-D=F_water.v-v'Rv`, the fluid power on the body. Include `h*P` in external
work and `h*D` in fluid loss, with gravity/buoyancy work retained. Preserve
`h*F_water.v` separately as a diagnostic. Use a numerically stable nonnegative
quadrature form or double precision; do not clamp materially negative losses.
The usual solve-residual, energy-balance, fin/body clearance, torque, nonfinite
and escape rejection remain active. Record first unsafe pre/candidate states.

The existing physical-world limit of 7.5 mm/s is not a measured motor capability.
This study uses a maximum **2.5 mm/s** steady horizontal current and no vertical
component. Flow direction and magnitude remain fixed in world coordinates for
the complete episode.

## Reusable policy and exact probability

Preserve canonical checkpoint 336's frozen 41-input migration, complete inherited
weights and optimizer. Canonical parent SHA-256 is
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
Preserve the qualified calm-holding amplitude and its saved two-parameter state;
its deterministic amplitude is `0.26211655139923096`. Do not reset it to one or
relearn holding inside this current-swimming study.
Its complete source checkpoint is holding update 7, SHA-256
`C54F73B57AB4280D99802A7253C13004F95C754E39E0ACF8F6833377BBB760C3`.

The new policy binary uses magic `0x31575343`, version 1, explicit task 1,
action 3, observation schema 5 (44 recorded values), Model 6 body, 41-input
frozen backbone, sixteen learned mean parameters and canonical parent update
336. Record the current-model/motor v1 source-contract hashes separately; this
is not a reinterpretation of the older calm package. Save complete fresh Adam
state and preserve the full parent body/holding checkpoint bytes.

The new observation record contains the same first 41 values and appends three
actual body-frame current components divided by 0.002 m/s: forward, vertical and
lateral. The original network consumes its unchanged first 41 values. The new
motor correction uses forward `f` and lateral `s` flow features; the vertical
feature is still recorded and versioned. Do not substitute water-relative body
velocity for the old pond-relative velocity observations or alter request frames.

Learn sixteen fresh Gaussian means `mu[k]`, initially all zero. Once at each
independent eight-second episode reset, draw one native seeded latent vector
`u[16] ~ Normal(mu[16], 0.5^2 I)` and keep it for that complete episode. Map
`theta[k]=0.5*tanh(u[k])`. There is no per-control action noise and the variance
is fixed, not learned. For joint `j`, let `b[j]` be the frozen body-network mean:

`z[j] = b[j] + theta[4j]*f + theta[4j+1]*s`
`             + theta[4j+2]*f*b[j] + theta[4j+3]*s*b[j]`.

Pass these four latents through the same physical tanh/filter/servo chain.
This is a flow-conditioned parameter extension of the one body motor, not a new
behavioral network or hand-coded current-to-body force. In exactly zero current,
an explicit branch must return the old swimming computation and saved holding
computation directly. Do not rely on approximate zero residuals or multiply
potentially invalid parameters by zero.

PPO scores the complete stored Gaussian vector **once per episode**. Sum the
sixteen Gaussian log densities into one joint log probability and one policy
ratio. No tanh Jacobian enters a latent-action PPO ratio. The mean gradient is
`d logp / d mu[k] = (u[k]-mu[k])/0.25`. Stored `u`, not an inverse of rounded
`theta`, is authoritative. Deterministic evaluation uses `theta=0.5*tanh(mu)`
with no sampling, preserving the learned correction.

Fresh native Adam updates only the sixteen new means, with its own age-zero
state: learning rate 0.01, existing beta/epsilon settings, global norm cap 0.5,
PPO clip 0.2, no entropy bonus/critic loss, minibatches of 64 complete episodes,
at most four epochs/update and an epoch KL guard of 0.01. Maximum sixteen Adam
steps/update and 512 in the study. The backbone, old optimizer and saved holding
parameters never update. This is restricted policy-parameter learning; success
does not prove full shared-network retention or integrated interactive learning.

## Cases, ground-frame task and acceptance

Each current episode starts from rest with the historical reset/filter semantics
and the 1.2 x 0.8 m long-swim arena. Use eight seconds / 160 controls and the
historical requested-speed task: 8 mm/s along the episode's original horizontal
heading, measured in the fixed pond frame. Keep the original start lane/depth
and task history. There are no intermediate resets, schedule changes or recentering.
The episode ends once at 160 controls: done=1, truncated=0.

The sixteen original current cases are the product of initial poses 0/3,
world-current directions +X/-X/+Z/-Z, and magnitudes 1/2 mm/s. Each 256-world
batch contains exactly sixteen independently sampled complete episodes per case.
Case ordering and seeds must be fixed in the frozen implementation, without
seed search or failed-episode replay.

The case order is pose `{0,3}` outermost, magnitude `{0.001,0.002}` m/s next,
then direction `{+X,-X,+Z,-Z}` innermost. Environment `i` uses case `i%16`, giving
exactly sixteen replications/case. Calibration seed is `0x36b91a57`; training
seed is `0x84e62c13`. Sample coordinate `k` with native
`rng::normal(seed, uint32(batch_index), uint32(environment*16+k))`.
Calibration uses batch index 0; fresh PPO batches use indices 1 through 32.

Use unchanged eight-second swimming acceptance: scored 2–8-second ground-frame
forward speed 7.2–8.8 mm/s, 3D speed-error RMSE <=2.5 mm/s, full-episode lateral
error <=2 mm, absolute depth error <=1 mm, tilt <=10 degrees, final scored path
angle <=5 degrees and positive actuator work/distance <=1 microJ/mm. Retain the
legacy scorer's exact definitions. Heading/yaw is recorded but must not be
misrepresented as an extra historical 5-degree instantaneous head-yaw gate.
Inspect the reset sample and all endpoints for current-suite lane/depth/tilt.
Still-water long-window startup misses remain failures, not retroactive passes.

The one-shot withheld suite is sixteen cases: initial poses 1/2, four unit
diagonal world-current directions (+X+Z, +X-Z, -X+Z, -X-Z), and magnitudes
1.5/2.5 mm/s. These poses/directions/magnitudes never enter training or
calibration. Each case is one uninterrupted 160-control episode. No further
learning is permitted after viewing withheld results.
Its order is pose `{1,2}` outermost, magnitude `{0.0015,0.0025}` m/s next, then
diagonal signs `{(+,+),(+,-),(-,+),(-,-)}` innermost, with world X/Z components
equal to the selected magnitude divided by `sqrt(2)` with the indicated signs.

After original cases, final calm retention and withheld cases all pass, run two
one-shot uninterrupted 64-second / 1,280-control current endurance cases: pose 0
with -X current at 2 mm/s, then pose 3 with +Z current at 2 mm/s. Retain one
start, original heading/lane/depth and continuous task/counter history. Require
the same whole-episode lane <=2 mm, absolute depth <=1 mm and tilt <=10 degrees,
including reset; the full scored 2–64-second interval must meet ground-forward
speed 7.2–8.8 mm/s, 3D speed-error RMSE <=2.5 mm/s, path angle <=5 degrees and
positive actuator work/distance <=1 microJ/mm. Additionally require fifteen
consecutive four-second windows ending at 8,12,...,64 seconds to meet speed
7.2–8.8 mm/s and RMSE <=2.5 mm/s. The first window is 4–8 seconds. Do not invent
a new 2–4-second startup speed gate; the preserved swimmer already has known
startup-window misses. An endurance failure closes/rejects without more learning.

Current-suite full success is sixteen original passes, sixteen withheld passes,
both 64-second endurance passes and calm retention. It establishes only the
specified steady-current swimming cases; do not describe it as current holding,
indefinite endurance or arbitrary-water skill.

## Return, calibration and data integrity

Use the historical swimming reward and original components, with no current
bonus, action imitation, successful-state label or altered speed/lane/posture
scale. Whole-episode return is `R=sum(per-control reward)/8`, undiscounted with
gamma=1. There is no per-tick GAE or bootstrap for a complete sampled episode.
Record all per-control rewards, observations, frozen body means, four commanded
latents and physical outcomes so the return and current response can be checked.

For an episode, its baseline is the leave-one-out return mean of the other
fifteen episodes in the same fixed initial-pose/current case. Never condition
the baseline on sampled `u`, `theta`, success or any resulting trajectory.
Freeze advantages for PPO epochs. Fit one positive global within-case residual
RMS from one independent 256-episode calibration batch and keep it fixed.
Calibration episodes never enter optimizer minibatches.

Before learning, independently verify finite samples/parameters/rewards/returns,
the complete Gaussian-vector likelihood and mean gradients, same-policy ratio
one, finite differences, no-Jacobian latent scoring, leave-one-out exclusion,
case counts, reset/terminal semantics, frozen-backbone/holding identities and
new checkpoint round trips. Independent log-probability error must be <1e-4.
Fix calibration RNG, training RNG, streams and case order before collection.
The global residual RMS must be finite and strictly positive. Record return
distribution diagnostics without inventing a return cap or a minimum number of
behavioral successes as a prerequisite for PPO.

No physics failure, nonfinite data, bad probability, rejected action or invalid
batch reaches an optimizer. A physical/numerical/data/contract fault stops the
whole attempt; a behavioral miss within the valid physical domain is recorded
as a failed case and does not change a threshold or authorize a repeat.

## Phases and stop rules

1. Freeze all sources, executable/libraries, initial controller/head/optimizer,
   policy contracts and protocol, preserving exact old bytes. Perform preparation
   math/configuration/checkpoint checks with zero physical integration.
2. Pass the separately counted current-physics gate below before all baseline,
   calibration and PPO work. Failure closes the attempt with no learning.
3. Run the original calm assessment once: four short swims, six full 64-second
   holds using the saved learned holding amplitude and two 60-second swims,
   10,720 controls. Require 4/4 short, 6/6 holds and the exact prior 28 passing
   long-window keys. Keep both known long startup misses as failures.
4. Run the sixteen original current cases with initial means zero, 2,560 controls.
   If all sixteen already pass, skip calibration/PPO and proceed directly to
   frozen final retention and the one-shot withheld/endurance suites. Do not perform
   unnecessary updates or claim this inherited baseline result as new learning.
5. Otherwise collect one independent 256 x 160-control calibration batch. Close
   calibration and validate its fixed readiness conditions before training.
6. At most 32 fresh PPO updates, each 256 x 160 controls. Save complete raw
   episodes, sixteen-dimensional parameters/optimizer and audit/counter state.
   Check zero-current exact passthrough without physical integration after every
   update. Run all sixteen original current cases after every completed update.
7. Stop PPO at the first full 16/16 current pass. If no full pass occurs by update
   32, run the final 10,720-control calm assessment for the retention report and
   close/reject without spending withheld or current-endurance reserves. This
   does not permit further optimization. If a full pass occurs,
   freeze the candidate, run the full 10,720-control calm retention assessment
   once, then the 2,560-control withheld current suite once. If both pass, run the
   two 64-second current endurance cases once, 2,560 additional controls. A
   retention, withheld or endurance failure rejects/closes; do not resume PPO
   with unused budget.

Record cost/reward changes as diagnostics. A 16/16 pass against a baseline with
fewer passes already demonstrates original-case behavioral improvement; no extra
mean-cost-below-baseline condition is inserted into qualification.

All actual integration attempts and failed substeps count. No automatic retry,
reseed, extra arm, tuning, gate relaxation or budget extension is permitted.
Store exact native poses for candidate inspection, but leave the working viewer
on the qualified calm package until a separately counted faithful preview is
ready. Frozen saved-policy viewing is never reported as ongoing learning.

## Physical gate — approved finite layout

The cap is **3,760 controls / 37,600 physical substeps**, including all
backends and the legacy comparison. Four seconds is only a bounded numerical
and safety gate; it is not swimming/current competence.

| Gate block | Backend executions | Controls |
| --- | --- | ---: |
| Ten zero-flow fixtures (four swim, six calm holds), 40 controls each | Legacy CPU, current CPU, current CUDA | 1,200 |
| Eight actuated-flow fixtures, 80 controls each | Current CPU and CUDA | 1,280 |
| Eight zero-target servo-advection fixtures, 80 controls each | Current CPU and CUDA | 1,280 |
| **Gate cap** | | **3,760** |

The eight driven-current fixtures use initial poses 0/3 crossed with four
cardinal world-current directions at 2.5 mm/s. The eight advection fixtures use
the same pose/current grid with four zero joint-target commands through the
unchanged active servo. They are **not torque-free passive/free-joint trials**.
Require projected COM displacement along the imposed current to be strictly
positive after 80 controls on each backend; this is a sign check, not a swimming
speed/competence gate. The driven fixtures use the preserved deterministic body
motor with zero correction means.
The zero-target servo fixtures initialize every joint angle, held target, filter
rate and generalized body velocity to zero; they keep their prescribed level
initial yaw and body position.

Zero-flow current CPU must exactly match legacy CPU's defined body, command,
world and base-telemetry fields, not struct padding. CPU/CUDA comparisons require
position error <=2e-5 model units, quaternion/joint-angle error <=2e-5, linear
velocity error <=2e-7 m/s and angular/joint-rate error <=2e-5 rad/s at every
compared control. The six cumulative energy-accounting values use absolute
1e-12 J plus relative 1e-4 tolerance. A pose match cannot conceal a failed
energy/force audit. No CUDA skip counts as a pass.

All samples must respect original clearance, torque, joint/filter, finite-state,
work and solve-residual constraints. Retain the original normalized solve
residual <=1e-8, per-substep loss rejection below -1e-14 and cumulative energy
balance tolerance of 2% of positive actuator work plus fluid loss, with 2e-10 J
absolute allowance. Do not relax a physical gate after observing a failure.
Test first unsafe pre/candidate capture and force-law/Galilean properties in
zero-integration fixtures as well.

The pure-assembly zero-integration checks run on CPU and actual CUDA for level
yaw poses 0/3 crossed with four cardinal currents at 2.5 mm/s. Set joint angles,
held targets, filter rates and angular/joint velocities to zero. For a comoving
body require every generalized force residual
`abs(force[i]-sum(R[i][j]*v[j])) <=1e-11` (N for translational rows 0–2, N*m for
rows 3–9) and absolute comoving relative-water dissipation <=1e-16 W. The
Galilean comparison is new `(body at rest, imposed flow)` against legacy
`(linear velocity=-flow, zero imposed flow)`, retaining the same mass,
resistance, inertial/bias definitions and generalized net-force relationship.
These assembly tests perform zero physical substeps and do not consume hidden
integration work.

## Finite maximum accounting

| Component | Maximum controls |
| --- | ---: |
| Physical current/compatibility gate | 3,760 |
| Baseline full calm retention | 10,720 |
| Baseline sixteen current cases | 2,560 |
| Independent calibration (256 x 160) | 40,960 |
| 32 fresh PPO batches (256 x 160) | 1,310,720 |
| 32 original-current assessments (16 x 160) | 81,920 |
| Final full calm retention | 10,720 |
| One-shot withheld current suite (16 x 160) | 2,560 |
| One-shot current endurance (2 x 1,280) | 2,560 |
| **Combined cap** | **1,466,480** |

The combined cap is **14,664,800 physical substeps**, 32 PPO updates,
512 fresh mean-parameter Adam steps, zero critic calls/updates and 900 combined
active native/process seconds. Gate and baseline work consume this same attempt,
not a hidden separate budget. Report actual spend and unspent/closed reserves by
phase. Both native budgets and the external launch watchdog account for prior
phase time; an elapsed-time receipt cannot reset the 900-second allowance.

## Execution interfaces and evidence

Use one unique `RUN` identity. Preparation is
`current_swim_study --prepare RUN PROTOCOL`, followed by
`current_physics_gate --prepare RUN/physics-gate PROTOCOL`,
`current_physics_gate --zero RUN/physics-gate` and
`current_swim_study --preflight RUN`. These preparation/zero steps may not call
the physical stepper or an optimizer. Existing identity guards prevent an
already attempted phase from being restarted.

After freezing the exact source/executable/library/input manifest, execute
`current_physics_gate --run RUN/physics-gate` once. Its actual gate proof is
`RUN/physics-gate/gate-passed.bin`, bound to the gate protocol and same approved
protocol bytes as the learner. Its `run-started/active-seconds.txt` accounts for
gate time. The learner directly validates that proof and receipt; there is no
GUI success flag or manually invented release to bypass a failed physical gate.

Only after gate acceptance, execute `current_swim_study --calibrate RUN` once.
This phase includes calm/current baseline collection and either the baseline-only
reserved-validation path or one independent calibration batch. Verify its saved
evidence before a separate `current_swim_study --train RUN` launch. Training is
allowed only for a valid calibration release with original current passes <16;
the release binds the actual gate, calibration scale and baseline artifacts.

Store the sixteen original-case configurations, exact seeds, policy/optimizer
states, whole-episode raw arrays, ABI/layout descriptors, per-control compact
pre/post physical state and all deterministic CSVs. Independent offline
verification must reconstruct all 44 observations, frozen body-network outputs,
learned coefficient/action mapping, Gaussian likelihoods, complete returns,
leave-one-out advantages, calibration scale and acceptance/counters. Final
qualification must bind the exact saved candidate SHA-256. Preparation,
independent reading and viewing are not PPO updates or behavioral evidence.
