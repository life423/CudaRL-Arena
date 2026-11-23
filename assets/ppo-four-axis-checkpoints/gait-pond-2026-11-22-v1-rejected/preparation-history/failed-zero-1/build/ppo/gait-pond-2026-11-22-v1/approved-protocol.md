# Balanced gait and pond — November 22, 2026

**Preregistered and explicitly authorized; execution has not started.** The
user explicitly authorized fixing the crooked swimming gait with native GPU
PPO, retaining learned skills and making the pond usable. This is one fresh
finite attempt. The closed current-swimming studies remain closed, unchanged
and unadopted. Their unused budgets do not fund this attempt.

The objective is a capable swimmer with a balanced, moving tail and its existing
holding/recovery skills. Godot presents the actual articulated native state.
There is one reusable four-output body motor, no extra behavior network,
scripted propulsion, pose straightening, food collection script or automatic
navigation. The chooser, growth, limbs and currents are outside this study.

## Preserved inputs and policy

Keep Model 6 geometry, masses, inertia, water forces, contacts, ten 5 ms physical
substeps per 50 ms control, 1 Hz target filtering, observations, physical safety
guards and four +/-10 microN*m torque limits unchanged. Zero current is explicit
throughout. Canonical checkpoint 336 and all saved learned-amplitude parameters
are frozen. The historical files and both current-study archives remain intact.

The root owner must finish zero-integration implementation checks and freeze
the complete causal input closure before physical execution. This is the
approved numerical protocol for one fresh attempt.

Canonical parent SHA-256:
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
Learned-amplitude source 7 SHA-256:
`C54F73B57AB4280D99802A7253C13004F95C754E39E0ACF8F6833377BBB760C3`.
Its deterministic holding gain is `0.26211655139923096`.

The same frozen 41-input backbone supplies four latent outputs `b[j]`. Eight
new Gaussian means start at zero. Once per complete eight-second SWIM episode,
sample `u[0..7] ~ Normal(mu, 0.35)` independently, then retain that vector for
all 160 controls. The swimming head is

```
bias[j] = 0.75 * tanh(u[j])
gain[j] = exp(log(2) * tanh(u[4+j]))
latent[j] = gain[j] * b[j] + bias[j]
command[j] = tanh(latent[j])
```

Thus gains stay positive in `[0.5,2]` and biases in `[-0.75,0.75]`. The initial
zero vector explicitly returns the original backbone values, preserving exact
identity rather than depending on a zero addition's rounding. Deterministic
tests and the viewer use `u=mu`, never a manually selected correction.

The stationary HOLD branch explicitly returns the saved `g*b[j]` using the
unchanged two holding parameters and derived gain; no swimming correction
enters it. This branch is not retrained. Mode selection follows the explicit
request. Retain the same four motor outputs and bounded physical actuators.

Store a separately versioned eight-parameter checkpoint and viewer package with
task/action/observation/model versions, canonical and holding parents, all frozen
source contracts, eight means, fixed sigma, mapping constants, optimizer state,
actual update/Adam counters and integrity checks. Old packages retain their old
reader/interpretation; never reinterpret an old file as the new format.

## Meaning of a balanced moving gait

The saved 336 reference already swims but its late tail is biased. A saved-data
audit over 2–8 seconds found first pitch means about +6.7 to +6.9 degrees,
second pitch means about -26.8 to -27.0 degrees and signed head-to-tail vertical
bend about -21.5 degrees. Yaw standard deviations are about 13.5–13.7 degrees:
that real oscillation is useful movement, not a defect to render away.

The new terminal gait interval is **POST controls 81..160**, exactly the samples
after four seconds through eight seconds. Use arithmetic population means over
these 80 samples. Also report the earlier 2–8 second diagnostics, all four joint
means/standard deviations/ranges, pitch RMS/peaks and full geodesic head-to-tail
kink. The required gait gates are simultaneous:

- Absolute mean of joint 1 pitch <=10 degrees.
- Absolute mean of joint 3 pitch <=10 degrees.
- Absolute mean signed head-to-tail vertical bend <=10 degrees.
- Demeaned angular RMS across four joints >=5 degrees.

For the bend, construct the tail link's local +X axis using the actual joint
kinematics, rotate it into the head frame and compute
`atan2(y,sqrt(x*x+z*z))*180/pi`. This signed vertical bend permits normal yaw
undulation. Do not substitute an instantaneous 10-degree geodesic-kink gate.
The oscillation measure is
`sqrt(sum_j(mean(angle[j]^2)-mean(angle[j])^2)/4)*180/pi`.
The division by four is inside the square root: equivalently
`sqrt(mean_j(var_time(angle[j])))*180/pi`. It measures motion around each
joint's own mean (the exact 4–8 second reference spans 9.967–10.014 degrees). These definitions and
thresholds are fixed before collection and are not relaxed after results.

For 60-second and 300-second swimming, apply these same gait gates separately
to every late four-second window ending 8,12,...,60 or 300 respectively. A
window ending at control `end` uses POST controls `end-79..end`. Do not require
the two historically failed startup-speed windows to become passes; retain
their actual results and the original acceptance rules.

## Task, objective and PPO

Use 8,192 CUDA worlds per batch, fixed without another size search. The previous
frozen GPU profile measured 468,325 controls/second at 8,192 versus 500,294 at
16,384. Reusing that performance evidence spends no new physical controls and
does not reuse old training samples. All new data are fresh.

Case `world%4` selects original poses 0,1,2,3 at 8 mm/s with the original request,
reset/filter state and long-swim arena. There are 2,048 episodes per case.
Every episode lasts 160 controls with true terminal at 8 seconds, no intermediate
reset, target relocation, anchor/time reset or truncation.

The episode return is the original swimming reward sum divided by 8, minus
`0.05*((mean_pitch1/10)^2+(mean_pitch3/10)^2+(mean_bend/10)^2)`, minus 1 if any
original eight-second swimming acceptance gate fails. The three means use only
POST 81..160. Preserve all original per-control rewards and separate terminal
penalty components. Gait quality cannot replace original movement acceptance.

Original simultaneous swimming requirements remain: steady ground speed
7.2–8.8 mm/s, speed RMSE<=2.5 mm/s, whole-episode lane<=2 mm,
depth<=1 mm and tilt<=10 degrees, path angle<=5 degrees and positive joint
work/distance<=0.001 J/m. Physical clearance, finite state, joint and torque
guards remain unchanged at every physical substep.

Score the eight-dimensional Gaussian `u` exactly once against the whole episode
return. Do not score bounded coefficients, four tanh commands or 160 duplicated
per-control choices, and do not introduce a Jacobian or per-control exploration.
Use gamma 1 and no critic. The baseline is the leave-one-out mean return of other
episodes in the same pose: group 2,048, denominator 2,047, independent of action
and trajectory. Fit one positive finite global within-case residual RMS on a
separate 8,192-episode calibration batch and freeze it for the whole attempt.

Fresh eight-mean Adam: learning rate 0.01, beta 1=0.9, beta 2=0.999,
epsilon 0.00001, global gradient norm cap 0.5. PPO clip 0.2; up to four epochs,
four minibatches of 2,048 episodes each per epoch, at most 16 Adam steps/update.
Use the fixed 0.01 epoch KL guard. No variance learning, hyperparameter tuning,
critic warmup, extra arm, reseeding, automatic retry or budget extension.
Calibration seed is `0x67a13c85`, training seed `0xa2d54913`. The counter is
batch index 0 for calibration and 1..64 for PPO; the random index is
`episode*8+coefficient`. Permutation seed is `0x02c64931 + update*4 + epoch`.
These streams are fixed, never resampled after seeing a result.

## Preparation, compatibility and readiness

Freeze the approved complete protocol, executable, transitive native sources,
linked libraries, parents, initial means and fresh optimizer state before the
first physical control. Copy/hash the causal inputs and preserve all earlier
study originals. Preparation uses zero physical controls and zero Adam steps.
The root execution owner alone may run physical phases.

Actual CPU-double/CUDA zero-integration fixtures must check all 8,192 initialized
worlds,41 observations, backbone outputs, mapped nonzero Gaussian choices,
probabilities, gradients/finite differences, all four 2,048-row minibatches and
last rows, reductions, LOO self-exclusion, hold passthrough for varied/extreme
swimming means and exact checkpoint/optimizer round trips. Do not use numerical
fixture data as PPO observations or claim pure fixtures establish swimming.

Fresh compatibility gate: four original swimming starts and six original holding
starts,40 controls each, on legacy CPU, identity-head CPU and identity-head
CUDA. Count all three backends:1,200 controls/12,000 substeps. Require exact
defined legacy/identity CPU state/telemetry parity. CPU/CUDA tolerances remain
position 2e-5 model units, quaternion/joints 2e-5, linear velocity 2e-7 m/s,
angular/joint velocity 2e-5 rad/s and six cumulative energies
1e-12 J +1e-4 relative. Preserve full states, torques and first rejected state.
Any physical/numerical defect stops affected work before optimization.

Baseline: four original short swims, six original 64-second learned holds and
two 60-second swims (10,720 controls). Require the original four short passes,
six hold passes and all 28 historical passing late-window keys; record gait defects.
Canonical recovery and five-minute reference evidence may be reused only as
SHA-bound frozen reference after the zero-code/identity checks. The final
candidate must execute every retained fixture anew.

Frozen canonical reference files under
`assets/ppo-four-axis-checkpoints/update-336-episode64/` are:

- `recovery64-gates.csv`, SHA-256
  `E16A668F6FEC219880890EB34100F8ADBA3C1E57663FC4B632F5F18886899A5E`.
- `long-windows.csv`, SHA-256
  `0262F2CD0E73A464FF45217984F24B2D998346BF10918F917DB8EAE1446B5499`.
- `five-minute-evaluation/windows.csv`, SHA-256
  `21079CA3C3F8EC1F911C86D1A9F8EFAEAB36E5F30638AF9A3256936C1F2E10A6`.

Then one fresh calibration batch,8,192×160=1,310,720 controls. Require finite
rows/returns/likelihoods, correct actions/terminals, valid physical accounting,
LOO and a positive finite residual RMS. Report sampled pass coverage without
inventing a minimum lucky-good-action count or return cap. Independently audit
and close readiness before the first PPO step.

The independent report is `independent-calibration-verification.json`, kept
separate from the final `independent-verification.json`. After a successful
read-only audit, write `calibration-verified.bin` using the ordinary integrity
envelope. Its payload is u32 magic`0x31564747`, u32 version1, u64 study-protocol
checksum, then u64 whole-file FNV checksums of the independent calibration
report, `fixed-scale.bin` and `calibration-ready.bin`, in that order. Native
`--train` requires this exact release before the first Adam step. Do not rewrite
the calibration proof when producing the final report.

## One training attempt and final retained skills

At most 64 updates,8,192×160 fresh controls each. After every update, run all
four original deterministic eight-second cases. Continue only while below the
fixed update/time/physical caps. Stop at the first update passing all four
original movement AND gait gates, then run reserved final validation once.
Do not select a later candidate after seeing reserved results.

The final candidate must retain all existing learned skills:

| Final fixture | Definition | Controls |
|---|---|---:|
|60-second swimming | Poses 0/3,arena`{0.6,0.4}`; retain all 28/30 old passing keys, actual startup results reported; new gait gates in 28 late windows |2,400|
|64-second fixed-depth recovery | Pose 0 SWIM at 8 mm/s, offsets 0/-0.5/+0.5 mm against the original fixed COM-depth target |3,840|
|300-second swimming | Poses 0/3,arena`{3.0,0.4}`; retain all 148/150 old passing keys; new gait gates in 148 late windows |12,000|
|Original 64-second holds | Poses 0/3×depth offsets 0/-0.5/+0.5 mm, original order |7,680|
|Withheld 64-second holds | Poses 1/2×depth offsets 0/-0.25/+0.25 mm, original learned-amplitude order |7,680|
|300-second holds | Poses 0/3 centered,6,000 uninterrupted controls each |12,000|

Recovery follows `tools/ppo_motor.cu::fixed_depth_replay`: initialize pose 0 and
the original forward request before translating the start; never translate its
world-space depth target. Each case must pass the unchanged 64-second forward
task. For both nonzero offsets, every final 160 POST states additionally satisfy
`abs(depth_error)<0.25 mm` (strictly less than half initial error). No scripted
recovery, reset or target drift is allowed.

Long swimming preserves the exact existing window scorer from
`tools/ppo_motor.cu::sustained`: first window measures 2–4 seconds, subsequent
windows four seconds, cumulative lane/depth/tilt and the original window work
and path rules. The retained passing keys are poses 0/3 crossed with end times
8,12,...,60 (28) or 8,12,...,300 (148). Record both actual end 4 startup results. The old passing keysets must be
subsets of the candidate passing keysets; genuine startup improvements may raise
the totals to 30/30 or 150/150. Do not weaken standards, force a historical
miss to remain a miss, or conceal losing any old passing key.

Holding uses the unchanged whole-body scorer: speed<=0.5 mm/s, anchor<=2 mm,
depth<=1 mm, lane<=2 mm, tilt<=10 degrees and final-eight-second forward
excursion<=2 mm, including the reset sample. The anchor, original target depth,
heading and learned holding gain remain fixed throughout each uninterrupted
64-second or 300-second case. Natural joint motion is permitted.

Run the complete 45,600-control final retained-skill suite once on the frozen
candidate even if 64 updates exhaust without all four short movement/gait
passes. These tests establish which old abilities remain; they do not release
more learning, a different candidate or another search. Such an exhausted
candidate remains unqualified even if some old skills pass. Any physical failure, invalid batch,
I/O fault, numerical defect or cap closes affected work with saved first-failure
evidence, actual attempted counters and last-good checkpoint. No automatic retry.

## Fixed budgets and evidence

| Phase | Maximum controls |
|---|---:|
|Compatibility gate |1,200|
|Baseline |10,720|
|Independent calibration |1,310,720|
|64 PPO collections |83,886,080|
|64 four-case assessments |40,960|
|Final retained-skill suite |45,600|
|**Study total** |**85,295,280**|

Maximum study substeps: **852,952,800**. Maximum 64 PPO updates and 1,024 fresh
Adam steps. The root watchdog and native ledgers enforce **900 combined active
process seconds** across all physical/calibration/training/final phases, with
prior-time receipts across phase boundaries. Pure preparation, offline audit and
paused zero-physics UI inspection do not advance physical/optimizer counters.
Unused controls/time close with the attempt and do not authorize another arm.

Every collected row stores PRE 33, POST 23, actual backbone 4, actual joint latent 4
and original per-control reward 1:65 float 32 values/control. The complete request
and PRE state reconstruct all 41 input features independently. Preserve sampled
u 8, old episode logp, all means, per-episode counters, terminal flags/returns,
all separate gait/failure penalties, final full world/task states and optimizer
checkpoints. No sampled-control evidence shortcut is permitted.

At maximum,65 calibration/PPO batches contain 85,196,800 trace controls,
22,151,168,000 bytes (about 20.63 GiB) before metadata/CPU evidence. Require at
least 35 GiB free initially and verify that the conservative 24 GiB projected
total evidence plus 8 GiB working reserve fits. Before every batch allocation
and collection, check that batch's full parts, metadata and overhead plus the
8 GiB reserve fit; check again before its writes if allocation changes the
requirement. A later disk shortage closes work rather than discarding evidence.
Do not delete prior data to proceed. Preserve the native 512 MiB envelope cap;
use split integrity-bound parts and exact manifests. Any write fault stops
before optimizer eligibility and preserves partial counters.

The offline verifier may use CPU/NumPy after native phases close. It reconstructs
all observations, frozen NN outputs, bounded head outputs/tanh commands, target
filter continuity, kinematics, every gait/movement/holding/recovery score,
episode probabilities/returns/LOO/scale, all checkpoint parents/counters and
hashes. At most four bounded workers, approximately one per 4 GiB available RAM.
It must not call a body stepper or optimizer. Bind final result, exact saved
candidate, nullable unrun suites, all evidence and source hashes in the final
report; truthful failed-arm verification is not qualification.

## Pond and faithful preview

The pond exposes honest Rest/Swim requests and Pond/Close views of the actual
native creature. Keep useful actions discoverable, remove obsolete inspector
choices and make saved inference versus actual learning clear. No food feature,
automatic path or background learner is invented for this pass. Paused UI,
layout, camera and accessibility checks use zero physical controls.

Only after qualification and independent verification, allow one separately
counted **80-control/800-substep viewer fidelity capture**. Compare its complete
native state/action trace against the matching first 80 controls of already
recorded final candidate evidence, avoiding an extra reference simulation.
Publish/label this preview counter separately from study counters. Preserve old
packages/readers and the qualified calm package until the new versioned package
and its actual Godot rendering agree. No unqualified actor adoption, source
contract reinterpretation, merge or publication is included in this protocol.

The user's subsequent request to teach goal-directed turning is authorized as
the next separate, finite motor study after this gait outcome. It does not
extend this attempt, change its data or permit scripted rotations, heading
teleports or automatic-navigation claims. Preserve the learned gait, holding
and recovery evidence for that subsequent request-conditioned motor comparison.

The native CLI phases are `--prepare`, `--preflight`, `--gate`, `--baseline`,
`--calibrate`, `--train`, each with the isolated run directory. Preparation also
takes this approved protocol path. Gate, baseline and calibration issue separate
integrity-bound predecessor receipts with cumulative controls, substeps, native
active time and whole-file FNV hashes of their evidence. A phase refuses an
existing started identity, stale protocol or changed predecessor. The watchdog
also accumulates process time across all physical phases.

Native protocol/checkpoint and split-part versions are explicit and independent
of old packages. All eight means start at zero and all optimizer state is fresh.
Preparation and preflight have zero body/optimizer counts. After their successful
checks, the root owner freezes this protocol, exact native executable, transitive
source/library closure, parent files and initial parameter/optimizer inputs as
SHA-bound copies before `--gate`. Freeze bytes remain immutable through closure;
later debug work requires preserving this attempt and a separate authorization.
