# One bounded braking probe — preregistered October 29, 2026

**Completed:** exact budgets spent, research readiness failed; no PPO launched.
Read [MOTOR_BRAKING_RESULTS.md](MOTOR_BRAKING_RESULTS.md). The protocol below
remains the original preregistration, not permission to repeat the run.

The latest user request amends **research readiness**, not final acceptance.
A conventional controller passing all16 cases is no longer required before
research training. Its0/16 result and the original false gate stay unchanged.
One fixed two-state direct-torque search may establish limited readiness for the
already capped128-update motor pilot. No whole-app/controller rewrite, gain
sweep, budget extension or deployment of open-loop commands is authorized.

## Preservation and ordering check

`build/motor-braking/2026-10-29-v1/` snapshots66 current dirty files and hashes222
original save/result artifacts, including all numerical and reference evidence.
The active viewer's process-locked append-only log is explicitly excluded;
no save/checkpoint/dataset/result is excluded. All old inputs remain in place.

The inspected current `command_motor::Batch::set_states` already queues H2D,
counter clearing and `observe_kernel` on the supplied stream, then synchronizes
it. Do not claim an absent race was repaired. The focused regression uses the
actual nonblocking stream, pinned host storage and queued delay. A deliberately
wrong local default-stream control must demonstrate test sensitivity. This
CUDA data-path check is independent of the failed CPU reference study.

## Frozen search settings — before any search execution

- Input: the exact complete existing fixture4 and fixture5 from
  `build/motor-control/2026-10-29/reference-v1/reference-fixtures.bin`.
  Their actual captured entry speeds are .0110311192 and .0282923263m/s.
  Position, orientation, all10 velocities, four joint angles, previous actions,
  original request and metadata are copied; no recapture or velocity injection.
- Physics: unchanged `deep::advance_physics`, .05s control,10×.005s substeps,
  same body/drag/buoyancy/inertia and four±10µN·m torque limits; calm open water.
- Seed: `0x4D072610`. Each candidate is48 direct torque values:12 successive
  .5s constant commands for each of four joints, clipped to[-1,1]. No actuator
  gains, neural weights, feedback rescue or goal retargeting.
- Search:64 candidates×4 rounds×2 starts×120 ticks = **61,440 control steps**.
  Each round includes passive zero and the incumbent;62 fixed-seed samples.
  CEM top8 updates mean/variance without smoothing. Initial mean0,std.15;
  std floor.005. No additional seeds, rounds or repeated gain sweeps.
- Every safe finite sequence ranks ahead of every penetrating sequence.
  Nonfinite dynamics abort the probe and retain evidence. Failed finite
  candidates are recorded and do not become rescues or training successes.

Fixed objective, in SI units: sum over120 post-step observations of
`dt*[.5*(headSpeed²+COMspeed²)/(2*.02²) +.25*RMS216speed²/.02²
 +.10*bodyOmega²/.5² +.10*mean4(jointRate²)/4²
 +.25*(originalHeadingError/10deg)² +.10*(originalGoalError/.072)²
 +.002*mean4(command²)]`, plus five times the last20 post-step observations'
mean of `(headSpeed²+COMspeed²)/(2*.002²)+RMS216speed²/.001²
 +bodyOmega²/.02²+mean4(jointRate²)/.05²`.
No tuning against final motor cases. The immutable machine protocol also saves
input hashes, complete chosen starts, objective and exact numeric settings.

## Separately budgeted frozen replay and research gate

Best and passive each replay from both exact starts for20s: **1,600 additional
control steps maximum**, no optimization during replay. Best runs its six-second
sequence, then exactly zero torque for14s. All actions and complete states are
recorded; no simulated position reset, scripted rescue or deployed controller.

Both starts must satisfy all research conditions:

1. Finite and nonpenetrating through all20s, with the unchanged whole-body
   envelope and physical boundaries. Safety failures remain failures.
2. Head and COM≤2mm/s for11 observations spanning.5s; dwell completion at least
   .5s earlier than paired passive. Cumulative stopping path≤passive and≤72mm.
3. Exactly201 observations spanning10.00–20.00s are quiet. The achieved-position
   anchor is captured once at10.00s, with no recentering or window selection.
   Drift≤3.6mm, original heading≤10°, head/COM≤.5mm/s, max216point speed≤1mm/s,
   body omega≤.02rad/s, each joint rate≤.05rad/s and excursion≤.05rad.

The original stop-request goal remains unchanged. Report original7.2mm goal,
3s braking/72mm and full original settle/hold gates separately. An achieved-stop
anchor establishes only local braking+quietness; it does not pass the original
position goal, generalize to new starts or relax final acceptance. Measurements
are at50ms control samples; this is not a formal continuous-time certificate.

## Conditional learned-motor pilot readiness

Only if this probe passes both starts, plus the independent numerical and actual
command-motor data-path checks, may a **separately named research-readiness gate**
start the existing128×512×64 pilot. Bind the evidence/source/environment hashes.
Never rewrite `physical_reference_passed=false` as true. Old gate formats do not
authorize the amended research route. Final64-case acceptance is unchanged.

Before launch, task protocol2 explicitly uses a true600-step finite-horizon
terminal (`done=1,truncated=0`): time remaining is observed, so it must not
bootstrap an unspecified value at time0. No physics/reward/optimizer change.
For this diagnostic pilot, the first finite clearance breach or invalid dynamic
state rejects the collected rollout **before GAE/Adam**, saves its first failing
states/data and stops the run. Keep the last good complete checkpoint. Do not
turn a breach into an early reset/terminal reward shortcut. Finite rollout,
advantage and gradient checks also precede their optimizer consumption.

The focused scratch path check uses frozen rows/captures, GPU/CPU observations,
inference, latent likelihoods, one-step rewards, terminal/reset semantics,
independent GAE, gathers/normalization and complete restore. No PPO update or
extra learning campaign is spent on readiness. Successful persistence alone is
not proof of correct observations or learning math.

If the bounded probe fails, stop the research route. Report its best behavior,
preserve all failed traces and recommend one concrete motor-interface or stopping
requirement decision. Finite-budget search failure is not uncontrollability.
