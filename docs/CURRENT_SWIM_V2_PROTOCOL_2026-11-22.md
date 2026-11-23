# Current swimming v2 — approved GPU-sized protocol, November 22

**Approved; execution requires the phase-specific causal-input freezes below.**
On November 22 the user explicitly approved the corrected GPU study: one fresh
32-update maximum and fifteen-minute active execution allowance. Only the root
execution owner may launch physical/PPO phases. This is Option B: repair the isolated current-path
neutral-buoyancy arithmetic, measure useful GPU batch size, then run one bounded
native C++/CUDA PPO study if every gate passes. Pure assembly, configuration,
probability, file-format and build preparation may run without advancing bodies
or applying an optimizer. The complete numerical protocol is fixed before the
physical gate; gate-causal inputs freeze before that gate, and all baseline,
profiling and PPO causal inputs freeze before their first physical collection.

The [v1 result](CURRENT_SWIM_RESULTS_2026-11-22.md) stays closed: 2,634 controls /
26,340 substeps, zero PPO/Adam steps, no behavioral current evaluation and no
qualified current policy. Its unused budget is not reused here. The proposed
v2 allowance is additional and separately versioned. Preserve v1's 105 frozen
inputs, complete failed evidence/archive, all earlier policies/full saves and the
working calm-water viewer.

## Objective and controller

Qualify purposeful pond-relative swimming through specified steady horizontal
currents while retaining the verified calm swimmer and calm holding. Advection
with the water is not swimming success. Keep one reusable four-output body motor;
the goal chooser is excluded. This study does not establish current holding,
braking from motion, changing/rotating currents, disturbance recovery, growth,
new limbs or integrated interaction-driven learning.

Preserve Model 6 geometry, mass/inertia, both universal joints/four axes,
+/-10 microN*m internal torque limits, joint limits, the 1 Hz target filter,
50 ms controls and ten 5 ms substeps. No global propulsion, teleport, scripted
rescue, moving lane/anchor, contact change or relaxed safety threshold is allowed.
Current strength is at most 2.5 mm/s, horizontal and constant in world coordinates
through each episode. The existing 7.5 mm/s physical-world configuration limit is
not a demonstrated skill or a study target.

Canonical checkpoint 336 remains byte-preserved, SHA-256
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
Use its same frozen 41-input migration and inherited optimizer. Preserve the
qualified seven-update calm-holding source checkpoint, SHA-256
`C54F73B57AB4280D99802A7253C13004F95C754E39E0ACF8F6833377BBB760C3`,
and deterministic holding amplitude `0.26211655139923096`.

Keep the v1 policy parameterization: sixteen fresh means `mu[16]`, initially
zero; one sampled `u ~ Normal(mu,0.5^2 I)` per complete eight-second episode;
`theta=0.5*tanh(u)`, held for that episode. The original 41 observations and body
network are unchanged. Record three extra actual body-frame current components
divided by 0.002 m/s. With forward/lateral flow features `f,s` and inherited body
mean `b[j]`, each joint latent is

`z[j]=b[j]+theta[4j]*f+theta[4j+1]*s`
`          +theta[4j+2]*f*b[j]+theta[4j+3]*s*b[j]`.

Use the same physical tanh/filter/servo mapping. Exact-zero flow takes an explicit
legacy passthrough, including the saved holding amplitude. This is restricted
flow-conditioned parameter learning around the same frozen body network, not a
new network per ability or proof of full shared-network retention.

Score the complete stored Gaussian `u[16]` once per episode: one summed log
density and one joint PPO ratio. No tanh Jacobian or repeated per-control score
is inserted. Fixed sigma is 0.5; it is not learned. Gradient for mean `k` is
`(u[k]-mu[k])/0.25`. Deterministic tests use `theta=0.5*tanh(mu)` without sampling.
Native fresh Adam updates only the sixteen means: learning rate 0.01, inherited
beta/epsilon settings, global gradient cap 0.5, PPO clip 0.2, zero entropy/critic
loss, at most four epochs and four minibatches per epoch, with epoch KL guard
0.01. Maximum sixteen Adam steps/update, 32 updates and 512 fresh Adam steps.

## Explicit numerical version 2

`current_model_v2`, `current_motor_v2` and `current_task_v2` are separate from the
preserved v1 and the twelve pinned calm controller sources. In nonzero flow,
round displaced mass `density*volume` and represented body mass individually to
float32 before subtraction, preventing CUDA FMA contraction from introducing
the inherited neutral-buoyancy residual. Mass, volume, gravity, drag coefficients,
servo gains and physical acceptance limits do not change. Zero-flow execution
continues to delegate the legacy computation exactly.

Preserve v1's water-relative drag and direct positive quadrature for dissipation.
Evaluate coefficients at the old relative point velocity and energy at the
post-solve velocity. The relative-water loss is the sum of translational
`c*(v_point-u)^2` components and rotational drag squares; reservoir-supply power
is independently summed as `-c*u_axis*(v_point-u)_axis`. Do not clamp negative
losses or construct external work merely to force an energy-identity pass.

Pure Galilean/comoving assembly checks isolate hydrodynamics with gravity zero
in those assembly fixtures only. Comparing a newly corrected nonzero-flow force
against a legacy zero-flow static gravity roundoff would mix two numerical
contracts. **Physical gate, baseline, profiling, calibration, training and
evaluation worlds retain the original default gravity of 9.81 m/s^2.**

Keep hydrodynamic generalized-force error <=1e-11 (N or N*m by row), comoving
relative-water dissipation/supply <=1e-16 W, exact nonzero-flow mass/inertial
arrays, and resistance difference <=`64*FLT_EPSILON*max_matrix_abs_R`. These
isolate float quadrature roundoff without relaxing the force criterion.
Zero-flow delegation remains exact. A separate default-gravity assembly proof
compares v1/v2 at level poses and preserved v1 failure/pre-failure states on CPU
and CUDA: unchanged mass/resistance/inertial arrays and water load; unchanged
legacy zero-flow branch; corrected neutral nonzero-flow static force exactly
zero. The existing 24-assembly preparation is zero physical/optimizer work and
does not substitute for the new physical gate.

## Physical compatibility gate

Reserve exactly 3,760 controls / 37,600 substeps across every backend:

| Block | Backends | Controls |
| --- | --- | ---: |
| Four calm swim and six calm hold fixtures, 40 controls each | Legacy CPU, v2 CPU, v2 CUDA | 1,200 |
| Poses 0/3 x four cardinal 2.5 mm/s currents, 80 controls each, saved motor | v2 CPU and CUDA | 1,280 |
| Same flow/pose grid, 80 controls each, zero joint-target commands | v2 CPU and CUDA | 1,280 |

The zero-target grid uses active unchanged joint servos, not torque-free passive
motion. Initialize its joint angles, held targets, filter rates and generalized
velocities to zero. Require positive projected COM displacement along the
current after 80 controls on each backend; this is an advection sign check,
not a swimming-speed claim.

At each compared control retain position error <=2e-5 model units,
quaternion/joint-angle error <=2e-5, linear velocity error <=2e-7 m/s and
angular/joint-rate error <=2e-5 rad/s. Six cumulative energy fields use absolute
1e-12 J plus relative 1e-4 tolerance. Keep solve residual <=1e-8, per-substep
loss rejection below -1e-14, cumulative energy error <=
`0.02*(positive_actuator_work+relative_water_loss)+2e-10 J`, all original
clearance/joint/torque/finite/escape gates, and first unsafe pre/candidate capture.
Legacy/v2 CPU zero-flow defined fields remain exact. No CUDA skip is a pass.
Any failure closes v2 before later phases; no retry or automatic threshold change.

## Fixed task, original cases and retained skills

Use the original 1.2 x 0.8 m long-swim arena and reset/filter semantics. A current
episode starts from rest, requests 8 mm/s along its initial horizontal heading
in the fixed pond frame, and lasts 160 controls/eight seconds. No intermediate
reset or reference change is allowed. Its one true terminal is done=1,
truncated=0 at control 160.

The sixteen original cases are pose `{0,3}` outermost, current magnitude
`{0.001,0.002}` m/s next and cardinal direction `{+X,-X,+Z,-Z}` innermost.
Environment `i` uses case `i%16`. Each supported `N` is divisible by sixteen.
Original-case assessment uses one deterministic eight-second episode per case,
not all `N` training worlds. Keep the historical acceptance: scored 2–8-second
ground-forward speed 7.2–8.8 mm/s; 3D speed-error RMSE <=2.5 mm/s; whole-episode
lane <=2 mm, absolute depth <=1 mm, tilt <=10 degrees (including reset);
scored path angle <=5 degrees; positive actuator work/distance <=1 microJ/mm.
There is no invented instantaneous 5-degree head-yaw or 2–4-second startup gate.

Full calm retention is the unchanged 10,720-control suite: four short swims, six
64-second calm holds with the saved learned amplitude and two 60-second swims.
Require 4/4 short, 6/6 hold and the exact same 28 passing long-window keys; the
two known startup-window misses remain failures. Run calm retention before the
original current baseline and at final closure. Exact zero-flow passthrough is
also checked without integration after every update.

## GPU sizing profile and sealed selection

After the physical gate and baseline calm/current suites pass their prerequisites,
if the initial deterministic current policy already passes 16/16, skip profiling,
calibration and PPO. Instead run final calm retention, one-shot withheld current
validation and both current endurance cases; report inherited response, not new
learning. Otherwise perform the following profile **once**:

- Candidate world counts: 2,048, 4,096, 8,192 and 16,384, in that order.
- Exactly three 32-control prefixes per `N`: one warm-up and two timed batches.
- Use the real initial policy: `mu=0`, Gaussian sigma 0.5 and the same bounded
  persistent coefficient mapping and physical equations, not neutral/fake actions.
- Profile seed `0x57c0a199`, stream/counter 0/1/2 for the three batches. Coordinate
  `k` of environment `i` uses index `i*16+k`, preserving common RNG prefixes across
  candidate sizes. No resampling or successful-world filtering is permitted.
- A 32-control prefix is 1.6 seconds of a 160-control task. Its task is still
  running at control 32: it is not a completed eight-second training episode or
  a fabricated true terminal. Record diagnostic reward sums, not full returns.

All twelve prefixes, including the four warm-ups, consume physical budget:
`3*32*(2048+4096+8192+16384)=2,949,120` controls. No profile episode, reward,
baseline, return, advantage or coefficient sample enters calibration or PPO.
Any physical/numerical/data/I/O fault stops the study rather than dropping a
world, selecting a smaller retry or repeating a timing run.

Rank candidates by measured aggregate end-to-end throughput of the two timed
batches: `2*N*32 / (time_batch_1+time_batch_2)`. Timing begins before world/reset
initialization and ends after validation, all recorded transfers, serialization
and completed file writes/close. Record collection/transfer and file-I/O wall
time separately, plus actual CUDA-event time for observation, body-network and
physical-step stages. Do not confuse host elapsed time with kernel time or
claim durable disk synchronization unless explicitly implemented. Exact score
ties choose the smaller `N`. Do not choose by an assumed utilization percentage.

Record actual device name, SM count, compute capability, CUDA/driver versions,
memory capacity/free memory and available clocks/utilization observations. The
coordinator has queried the local NVIDIA GeForce RTX 5070, 48 SMs and approximately
12.79 GB reported global memory. Preserve exact queried bytes and properties in
the run's device/profile evidence rather than relying on these rounded figures
or claiming peak hardware efficiency. Preserve the measured table
even if the fastest `N` is smaller than 16,384.

Seal the selected `N`, profile artifact hash and timing scores before independent
calibration. This seal is immutable for the rest of the attempt. No later
throughput/learning outcome may change `N` or cause another profile. Each PPO
minibatch then contains `N/4` complete episodes: four minibatches/epoch, at most
four epochs and sixteen Adam steps/update, independent of `N`.

## Calibration, PPO and likelihood integrity

Collect one independent complete `N x 160` calibration batch at the sealed size,
using seed `0x36b91a57`, stream 0. Fit one positive global within-case residual
RMS from the sixteen fixed case means. Case group size is `G=N/16`; each episode's
PPO baseline is the leave-one-out mean of its other `G-1` same-case episodes.
Never condition this baseline on sampled coefficients, success or trajectory.
Calibration data are never optimized. Require finite complete records, valid
likelihoods and a finite strictly positive residual RMS, without inventing a
minimum successful-action coverage or arbitrary reward percentile gate.

Use whole-episode `R=sum(original swimming rewards)/8`, gamma=1, no per-tick
GAE/bootstrap, no reward/component change. Training seed is `0x84e62c13`, stream
the one-based update 1 through 32, coordinate index `i*16+k`. Draw once at reset
and retain `u[16]` through all 160 controls. CPU-double/actual-CUDA probability,
gradient/finite-difference, latent transform, grouping, fresh optimizer and
checkpoint checks must pass before physical collection. Old log-probability
reconstruction error must be <1e-4. No invalid collection reaches an optimizer.

After every completed update, save full parameters/optimizer and raw data, check
exact zero-flow passthrough without integration, and assess all sixteen original
current cases (2,560 controls/update). Stop PPO at the first 16/16 pass. Behavioral
misses within valid physics are recorded; hard faults stop the whole attempt.
Cost changes are diagnostics, not an extra qualification condition. If update 32
does not reach 16/16, run final calm retention for the report and close/reject;
withheld and current-endurance reserves remain unspent.

## One-shot withheld and current endurance validation

After the first original 16/16 pass, freeze the candidate, require final calm
retention, then run sixteen withheld eight-second cases once. Order: poses
`{1,2}`, magnitudes `{0.0015,0.0025}` m/s, diagonal signs
`{(+,+),(+,-),(-,+),(-,-)}` in X/Z, with component magnitude divided by sqrt(2).
These cases do not enter profile, calibration or training. Use the same swimming
criteria. Do not resume PPO after seeing withheld results.

Only if all sixteen withheld cases pass, run two uninterrupted 64-second cases:
pose 0 with -X current at 2 mm/s, then pose 3 with +Z current at 2 mm/s. Keep
the original reset, lane, depth, heading and continuous task history. Require
the same whole-episode lane/depth/tilt limits, full scored 2–64-second speed
7.2–8.8 mm/s and RMSE <=2.5 mm/s, path <=5 degrees and work <=1 microJ/mm.
Additionally require all fifteen four-second windows ending at 8,12,...,64
seconds to pass speed 7.2–8.8 mm/s and RMSE <=2.5 mm/s. The first window is
4–8 seconds; the known 2–4-second startup misses are not silently reclassified.

Full qualification requires original 16/16, withheld 16/16, endurance 2/2 and all
calm retention. It is bounded steady-current swimming evidence, not current
holding, indefinite endurance, a fully trainable shared motor or automatic
production adoption. Any final validation failure rejects/closes without more
learning or reuse of its unused allowance.

## Trace storage, immutable limits and disk guard

Full GPU trace payload is 125 floats/control: observations 41, body-frame flow 3,
frozen body means 4, four joint latents 4, reward 1, post metrics 16, compact
PRE state 33 and compact POST state 23. Record full metadata as well: persistent
latent samples/log probabilities, complete return/counters, case configuration,
raw ending world/task state, selected size/profile identity and optimizer state.
Pure offline verification must reconstruct observations, body outputs, bounded
coefficients, commands/filter continuity, likelihoods, returns, leave-one-out
advantages, acceptance, retained-window keys and exact spend.

Do not raise the immutable native 512 MiB envelope/read limit. A batch is a
directory with a bounded `meta.bin` plus eight bounded trace-part files and a
manifest. Parts carry explicit schema 2, field identity and metadata-payload
checksum; the manifest records exact byte counts and native payload checksums
after writes close. The largest part, 41 observations at `N=16384,T=160`, is
410 MiB. No physical record may be discarded to fit a file-size limit. Failed
I/O leaves partial evidence and closes before any optimizer step.

At the maximum size, complete calibration/PPO and profile trace payloads consume
`(33*16384*160+2949120)*125*4 = 44,728,320,000` bytes, **41.66 GiB**. This counts
GPU raw trace payload only, not CPU assessments or headers/metadata. Allow about
another GiB for metadata, raw ending states, manifests, source and CPU evidence;
the full retained study is conservatively about **43 GiB**, not a 41.66 GiB
filesystem guarantee.

Require at least **52 GiB** free on the actual output volume before execution,
and independently require the conservative projected whole-study bytes plus
**8 GiB reserve** to fit. Before each profile or calibration/PPO batch, check
actual `N`, prefix/full horizon, all part/metadata/header overhead and the 8 GiB
working reserve before allocation/collection/writing. Insufficient space stops
before collection; it does not authorize dropping raw data, changing `N`, moving
an unreviewed path or extending the run. Verify start/free-space telemetry and
retain it with the attempt.

## Finite maximum budget

| Component | Maximum controls |
| --- | ---: |
| v2 physical compatibility gate | 3,760 |
| Baseline full calm retention | 10,720 |
| Baseline sixteen original current cases | 2,560 |
| Four sizes x three 32-control profile prefixes | 2,949,120 |
| Independent calibration, max `16384 x 160` | 2,621,440 |
| 32 fresh PPO batches, max `16384 x 160` | 83,886,080 |
| 32 original-current assessments, `16 x 160` | 81,920 |
| Final full calm retention | 10,720 |
| One-shot withheld current suite, `16 x 160` | 2,560 |
| One-shot current endurance, `2 x 1280` | 2,560 |
| **Combined maximum** | **89,571,440** |

Cap **895,714,400 attempted physical substeps**, 32 PPO updates, 512 fresh Adam
steps, zero collection/training critic calls/updates, and **900 combined active
native/process seconds** across gate, baselines, all warm/timed profiling,
calibration, learning and evaluation. Smaller sealed `N` reduces actual work;
it does not grant more updates, more seeds or replacement collections. Include
failed/internal/backend attempts and warm-ups. Do not reset clocks between
phases. Preserve actual phase receipts, counters and all unused closed reserves.

No retries, extra arm, reseed, silent correction, threshold relaxation, profile
extension or budget reuse is automatic. The prior v1 spend remains separately
reported and closed. Build/pure-math/debug preparation executes zero physical
controls and zero Adam steps; it never establishes behavioral success.

## Execution and reporting after final freeze

One runtime-sized native executable owns `--prepare`, `--preflight`, `--baseline`,
`--profile`, `--calibrate` and `--train`. A separately versioned gate executable
owns preparation/pure assembly and its one physical gate. A CLI does not bypass
the required source/input seals merely because this approved protocol exists.
Each phase refuses an existing attempted identity, validates predecessor releases
and preserves hard-fault evidence.

Use two explicit causal-input freezes:

1. **Before the one physical gate**, freeze the approved whole-study numerical
   protocol, gate executable, its transitive physics/motor/NN source and linked
   libraries, parent/checkpoint inputs, initialization/case definitions and gate
   preparation/zero evidence. Gate-causal inputs remain immutable for every
   subsequent phase. The gate can run while new PPO-only runtime-size plumbing
   finishes, because that code cannot affect the gate and all policy/budget/case
   choices are already fixed by this protocol.
2. **Before any baseline/profile/calibration/PPO physical controls**, finish the
   complete runtime-sized baseline/profile/learner executable, run its matching
   zero-integration configuration/math/checkpoint/storage checks, then freeze its
   complete sources, binary, transitive libraries, parent states and initial
   parameter/optimizer inputs together with the unchanged gate seal. No causal
   code is changed after its phase starts. In particular, passing the gate does
   not authorize changing physics, NN semantics, numerical criteria, cases,
   observation/reward/action definitions, profile selection or budgets.

The learner's `--prepare RUN PROTOCOL` may accept an existing study root only
when it contains the sealed v2 physics-gate/input-freeze artifacts and no prior
learner preparation or baseline/profile/calibration/training attempt. It must
verify identical approved protocol bytes, preserve the gate artifacts and refuse
an old v1 root, stale protocol, overwrite or repeated learner identity. This
permits gate-first preparation; it is not a restart of an attempted phase.

Sequence after approval/freeze: gate -> baseline calm/current -> optional bounded
profile and sealed `N` -> independent calibration/readiness -> native PPO with
original-case assessment per update -> final calm -> withheld -> endurance.
If baseline already passes, use the stated direct frozen-validation path.
Independent evidence verification is read-only and binds the exact final saved
policy hash; a truthful failed-arm verification is not qualification.

Update current status and the existing single Godot frontend in the same pass.
Keep saved inference, recordings, actual learning and unstarted plans distinct.
Preserve the qualified calm package until a separately counted faithful preview
of a qualified current policy is ready. No automatic learner adoption, anatomy
change, merge, publication or deployment is included. The user's new finite
approval is recorded above; the matching phase-specific implementation/preflight
reconciliation and causal-input freeze are mandatory for its one-shot execution.
