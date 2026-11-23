# Frozen motor request-interface comparison v1

The user explicitly authorized this controlled comparison on November 23:

> ok go ahead with the controlled comparison. keep the saved motor unchanged and don't start retraining yet. lets see which interface works best

One finite diagnostic is authorized. Zero PPO/Adam, no changed motor weights,
no physics/torque/filter/body/pond change and no automatic policy adoption.
The original studies remain closed; this allocation is separate.

## Controlled difference and preserved state

Use the exact saved update-336/holding-source-7 41-input backbone already bound
by the frozen-turn probe. Both arms load the identical 41-32-32-4 weights and
decode deterministic means through tanh. No 42-input migration or new network.
Both run model6, four angle targets, the unchanged 1 Hz reference filter,
ten 5 ms substeps per 50 ms control and +/-10 microN*m physical torque caps.

Arm A is lane_following, the original observer. Arm B is heading_only: after
warmup, set only a read-only observer copy's world-X/Z lane origin to actual
COM X/Z before the original inverse rotation. Preserve world depth and all
other features. Do not zero body-frame components after rotation.

At every genuine course-change event, BOTH arms rebase their stored lane X/Z
to their actual current COM, preserve original lane Y and rotate the ORIGINAL
persistent course by the listed absolute offset. Rebase on each event, not
only the first event. The final return to offset zero is a genuine change.
The first zero-program event does not change the stored request or lane at
all; B still activates its heading-only observer. This preserves the exact
original zero-program task while testing whether masking changes its behavior.

Warmup uses the original observer in both arms. Requests never mutate physical
world/filter/rate/torque state, clocks or elapsed controls. Their scheduled
changes are diagnostic inputs, not deployed scripted navigation. No reset,
teleport, rendered rotation, rescue, command substitution or recollection.

## Fixed main grid: 88 trajectories, 44 matched pairs

Four fixtures: (pose0,warm80), (pose3,warm80), (pose1,warm90), (pose2,warm90).
Use original initial worlds without a new translation. Both warmup phases
use real physical controls; no uncounted state capture/replay. Each fixture
has both arms and all eleven scenarios below. Case ID is
((fixture*11 + scenario)*2 + arm), arm0=A, arm1=B.

| Scenario | Events: absolute degrees relative original course | Post-warm controls |
| --- | --- | ---: |
| 0 | zero offset at relative0; no stored-task change | 320 |
| 1/2 | single +5/-5 at relative0 | 320 |
| 3/4 | single +15/-15 at relative0 | 320 |
| 5/6 | +5/-5 increments at relative0,20,40,60,80,100 to +30/-30; hold160 controls after last event | 260 |
| 7/8 | +5/-5 increments at relative0,10,...110 to +60/-60; hold160 | 270 |
| 9/10 | +5/-5 increments at relative0,20,...100 to +30/-30; reverse at relative160,180,...260 back to0; hold80 | 340 |

An event's one-based before_control is warm_controls+1+relative. A scenario's
maximum controls is warm_controls+post-warm controls. Every trajectory runs
until that bound or its first ordinary guard; no early success stopping.
The final hold means continue swimming on the final requested course, not
stationary Rest. Scenarios9/10 explicitly request the original course again.

Before source freeze or any physical run, two independent idealized geometric
calculations found that the initially considered eight-second final return
segment would drive all eight rigid-body ideal8mm/s return paths into the
inset bank. The final protocol instead uses four seconds, retaining the final
four-second metric window; all eight ideal paths then fit. This is a synthetic
kinematic feasibility check, not a physical trial or proof of learned turns.
It reduced the original proposed control cap from36480 to35200 before execution.

## Fresh matched-command gate and pure preparation

Four prerequisite cases: A/B pose0 warm80 zero offset through120 controls;
A/B pose3 warm90 +5 degrees through130 controls. Each executes CPU and CUDA
with the SAME actual command per interval. Compare all recorded pose/velocity,
filtered targets/rates, torque and six energy ledger values under the previous
frozen-probe tolerances. CPU must apply its corresponding request event before
comparison; same-observation NN outputs are checked independently.

Gate allocation is (120+120+130+130)*2=1000 controls/10000 substeps. A physical,
numerical or backend discrepancy stops the whole attempt before main cases.
Only the root execution owner launches it. Native diagnostics have no optimizer.

Before physical entry, pure preparation must verify exact parent binding,
pointwise CPU/CUDA inference, event schedules, zero identity, no request-induced
world mutation, preserved depth/clocks and observer masking including tilt.
Its kernels/inference perform zero physical integration. Freeze causal sources,
binary/static libraries, protocol/cases/layout and parent files; independent
preparation proof and the actual user's new grant bind this freeze. Reused
old approval/proof files are rejected. Check original protected files before
and after; do not overwrite existing evidence or checkpoints.

## Hard finite caps, including failures

| Scope | Controls | Attempted 5 ms substeps |
| --- | ---: | ---: |
| Main grid | 34200 | 342000 |
| CPU plus CUDA gate | 1000 | 10000 |
| TOTAL | 35200 | 352000 |

The warm80 fixture costs4220 controls per arm; warm90 costs4330. Actual partial
and rejected attempts count; report accepted separately. One native physical
process is capped at240 seconds, including loading/inference/evidence I/O.
An external watchdog enforces the same wall limit. Collection stops by230
seconds to reserve10 seconds for evidence closure; the complete native process
and external watchdog still have the240-second cap. At most512 MiB new native
research evidence including input snapshots and preparation/protection records,
with a separate1 GiB free-space reserve. Compilation and independent offline
audits are outside the native physical-process timer and cannot step physics.
No retries, tolerance relaxation, extra grid/arm, optimization or extension.

Keep strict existing hard model/energy/domain guards. At each accepted5ms
state, inset ellipse .195/.115 m, absolute original global COM-depth5mm and
head tilt20 degrees are ordinary terminals, in bank/depth/tilt priority.
Active ellipse .20/.12 m, nonfinite/numerical/solver/energy failures are hard
stops. Preserve the first unsafe PRE and candidate before later state handling.
Main ordinary failures stop only that case. Hard faults stop the entire run.

## Saved evidence and independently computed metrics

Record complete per-control World/ForwardTask, observation41, NN means4,
commands4, actual attempted/accepted counts and termination causes. Save every
accepted5ms body pose/velocity plus six double energy values and all requested/
applied substep torques. Save case/event metadata, backend matched comparisons,
same-observation inference, process timing and exact parent/source identities.
An offline reader reconstructs observations/requests/actions/filter/torques,
energy identities, geometry and movement metrics. It does not replay the force
integrator; do not describe it as an independent dynamics reproduction.

Use actual mass-weighted COM velocity/course, not head direction. For course
progress, compute one-second/20-control boxcar mean horizontal COM velocity,
with positive yaw atan2(-vz,vx); unwrap relative to the final pre-request
one-second course. A contiguous window chain from that baseline must have
speed at least2mm/s to supply turn-progress credit. Invalid gaps yield null
progress and fail that criterion, not invented zero-speed directions.

Descriptively report complete-case final4s speed/course/vector error, all-state
depth/tilt and boundary extrema, post-request arc, minimum8s arc-speed, minimum
two-second net displacement, positive work and torque saturation/margins.
Report first response to half the first nonzero requested offset, if achieved.
Estimate local radius from speed/absolute finite-difference rate of valid
one-second mean COM course only above1degree/s and2mm/s. Include sample counts;
this smoothed conditional statistic is not a minimum feasible turning radius.
Geometry covers every saved5ms state, not a mathematical continuous sweep.

## Predeclared functional comparison and selection

A functional case must complete its horizon with no ordinary/hard terminal,
all accepted5ms depth<=1mm and tilt<=10degrees including warmup. Its final80
POST controls must have mean velocity projected on final request>=6mm/s,
horizontal course error<=5degrees and full3D velocity RMSE versus .008*heading
<=2.5mm/s. Every complete8s rolling post-warm arc-speed window must be>=3mm/s.
Two-second net displacement is a separate diagnostic, not a new pass gate.

For a nonzero single/repeated final course, valid signed actual course change
must reach at least half its final absolute offset. For return scenarios9/10,
the one-second COM course immediately BEFORE the first return (POST controls
warm+141 through warm+160) must have achieved at least15degrees in the first
requested direction. Final course must then be within5degrees of ORIGINAL
persistent course. Never count simply remaining straight as a completed turn.

Rank arms lexicographically by zero functional passes(out of4), repeated plus
return functional passes(out of24), all nonzero functional passes(out of40),
then median final4s forward projection across40 nonzero cases, assigning zero
to incomplete cases. Final-speed differences below1e-9m/s are a tie. This is
a relative preference only; publish both complete distributions and failure
causes. It must not conceal stalled/terminated cases in a surviving-case mean.

An arm provides enough measured evidence to DESIGN a chooser only if it passes
all4 zero, all8 one-second +/-30 repeated and all8 return cases. The faster
half-second +/-60 cases characterize extra capability but are not mandatory.
Even that result does not qualify continuous navigation, arbitrary starts,
current swimming or a trained actor. If neither is adequate, report that fact
regardless of which ranks higher. No motor retraining follows automatically.

## Disclosure, closure and preservation

Update Godot Details with byte-bound independently verified arm measurements
and selection/limitations; preserve the saved actor, camera, zoom and pond.
Any hidden GUI tests must execute zero native controls/substeps/updates.
Do not expose a new learned capability or run physical viewer diagnostics.

Write the final report with actual spend, faults, source hashes, unchanged
parent/protected evidence and zero PPO/Adam. Keep all failed preparation/audit
attempts. Offline parser corrections do not authorize recollection. Stop after
this comparison and disclosure; no chooser training, motor PPO, merge/push,
model change or broader follow-up is included in this authorization.
