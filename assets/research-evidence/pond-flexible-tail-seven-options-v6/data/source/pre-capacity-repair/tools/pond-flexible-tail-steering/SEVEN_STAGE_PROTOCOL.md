# Seven authorized improvements, then stop

November 24, 2026. The user explicitly requested all seven listed options and
then a stop. This finite program implements and measures each option without
overwriting the working motor, registered Godot project, prior checkpoints or
historical evidence. No further experiment follows this program automatically.

## Stages and fixed limits

All learning is native C++/CUDA PPO with genuine 1,024-world GPU batches, the
existing GPU rollout/MLP/GEMM/PPO/Adam infrastructure, no food, scripted steering,
artificial forces or training resets disguised as deployed recovery. Exactly
one reusable motor and one goal chooser remain the behavioral architecture.

1. Rhythm: retain the old score and add main-stroke measurement; expose passive
   yaw angle/rate, and a bounded causal penalty for collapsed passive stroke
   energy (0.10/s times one minus quality). Do not reward tiny ripple counts.
2. Joint stroke/steering: expand the same motor to three Gaussian latent outputs:
   yaw trim, log-amplitude and phase-rate normalization. Scales are
   exp(log(2)*tanh(a)) and exp(log(1.5)*tanh(f)); no physical timestep changes.
3. Network capacity: a 31→16(tanh)→3 correction with zero initial output and a
   frozen linear skip inside the same policy, preserving its initial function.
4. Speed: request 6/8/12 mm/s with observed requested speed. Add 0.50/s actual
   speed quality exp(-((actual-target)/3mm/s)^2), graded by actual movement.
   This stage deliberately permits policy authority during nominal Swim; retain
   physical swimming requirements, not identical old actuator commands. Rest
   remains inactive. Earlier stages retain the exact straight bypass.
5. Rate: request signed 2/4/8 deg/s, observe that request, and integrate only the
   persistent desired heading at physical control boundaries. Actual body motion
   remains joint/water physics. Add 0.75/s measured COM-course-rate quality with
   a 4 deg/s scale. Paired directions, speed practice and state continuity remain.
6. Bank anticipation: train the existing goal chooser's new request schema,
   selecting rate {-8,0,+8 deg/s} and speed {6,8,12 mm/s} every0.8 simulated s.
   These are persistent rate/speed requests, not the old15°/4s interface. Give
   bank-relative normals, actual-course clearance rays and constant-transport
   time estimates, plus visitation history. These observations select no action.
7. Body material: compare an explicit half-yaw-stiffness tail, unchanged pitch
   stiffness/relaxation, mass, geometry, actuators and integration. Elastic force,
   energy and damping all use the same descriptor. This is isolated research,
   never a silent change to old bodies or the live project.

Motor stages1–5 and7 each allow128 NEW updates,64 decisions/world,4096-sample
minibatches, four PPO passes,8,388,608 transitions and536,870,912 maximum substeps.
Per motor process:1200s cooperative/1800s external cap. Save complete state and
immutable exports after each update; check every8. All prior PPO settings stay
as declared in the phase study. Stages1–4 retain balanced15/30 practice; stage5/7
use balanced continuous rate and speed requests. Speed/rate matching supplements
existing COM transport, alignment, correction, depth, tilt and effort terms.
In these exclusively calm-water program tasks, the critic's otherwise-zero
current-x/z slots24/25 carry requested speed offset and signed course rate.
Legacy critic observations are unchanged. At8mm/s and zero rate those features
remain exactly zero, while future request dynamics become observable to value
learning without changing the36-input critic shape.

Chooser stage6 allows64 updates,8 decisions/world,1024-sample minibatches and
four passes:524,288 new macro transitions. Gamma0.994/lambda0.98 reflect0.8s
decisions; actor/critic rates0.0003/0.001, clip0.2, entropy0.01, value0.5,
target/hard KL0.015/0.05. Maximum671,088,640 substeps includes training warmups.
Its1200/1800s process caps match the motor. Freeze the selected motor throughout.

Total main allowance:832 PPO updates and50,855,936 transitions. Diagnostic
optimizer work below is separately counted. Total training wall cap3600s across
the program. No automatic restart, extension, repeated seed search or loosened
criterion. A defect stops affected work; preserve every failed attempt. Repairs
before data collection must be explicit and versioned; no discarded optimizer
updates. Learning shortfalls alone do not abort the remaining valid stages.

## Transfer and verification

Initial parent: phase-mixed-v5 checkpoint88 and its exact actor export, task
e77d7688e076201850f5aa9eb2c1960a1b10a70986bb1b934f57d4a88e6035fa.
Later stages use the strongest safe assessed candidate from the preceding stage,
including its pretraining candidate if learning regresses. Rank first by physical
retention12/12 and all main-stroke15 keys, then total main-stroke turn passes,
then lower mean course error. Keep original legacy scores visible. Record chosen
parent hashes/update/model schema and all unselected/failed assessments.

Preserve body, servo, histories, critic, sampling progress and existing optimizer
coordinates across stage transfer. Newly introduced output coordinates start at
zero moments. The new nonlinear correction starts with fresh optimizer state;
the frozen skip retains the prior policy. Preserve critic age independently.
Check deterministic initial command/mean transfer and complete saved state before
training. Mode, features, material, action dimensions and frozen skip are bound
to the new FTYO3/FTSSv6 schemas; old loaders reject new formats.

For each of six motor stages, run one3-update exact-resume diagnostic:196,608
transitions/12,582,912 maximum substeps,600s cap, CPU/CUDA observation tolerance
1e-6. Chooser gets one3-update diagnostic:24,576 macro transitions and31,457,280
maximum substeps including warmups. Reserve at most one additional explicitly
logged diagnostic attempt per stage ONLY after identifying and correcting a
specific implementation defect. Both attempts consume the doubled diagnostic
allowance; do not discard their optimizer/physics evidence or restart failed main
training automatically. An incomplete diagnostic is a failed attempt. Material qualification
allows64,000 substeps/20s and tests default bit equivalence, CPU/CUDA, mirrored
motion, timestep refinement, energy/momentum and neutral rest before body PPO.

## Fixed assessments and honest interpretation

Every motor stage has a pretraining baseline plus16 scheduled assessments.
Each retains32 fixed15/30/60/90 cases and12 swimming/from-rest cases PER ARM:
88 trajectories,1,100,800 maximum substeps and30s cap. Six-stage total112,281,600
assessment substeps. Existing motion gates are unchanged: COM course<=5deg,
projected and rolling speed>=6mm/s, tilt<=10deg, depth error<=1mm, positive whole-
body certified separation and>=80% matched root/passive stroke frequency/swing.
Report exact bypass where applicable; do not require it in speed/rate stages.

The old passive crossing count conflates sub0.1rad ripples with main strokes.
Preserve its scores, and add schema1 directional-extrema cycles confirmed by
0.1rad prominence, with the SAME80% matched frequency/swing thresholds. Synthetic
tests must reject a missing main stroke while ignoring tiny ripples. This is an
explicit measurement correction, not an undocumented claim of legacy passage.

Final-only supplemental motor assessment per stage:6 speed cases,12 rate cases
(same requests as training, held-out starts), two uninterrupted mirrored sequences
of30/60/90 turns (20s turn+2s onward each), and two20s bank approaches. Maximum
382,400 substeps/30s, all failures recorded. No fresh reference branches are hidden;
label where a matched same-request stroke reference is unavailable. No resets
between successive requests. Stage1–4 unsupported rate results remain diagnostic.

Chooser pretraining plus8 scheduled checks retain12×120s diagnostic starts and
conditional4×300s endurance cases,2,112,000 substeps/30s maximum per assessment.
Preserve original criteria: full horizon, mean speed>=6mm/s, tilt<=10deg,
depth error<=1mm, positive certified gap, at least8/14 visited cells and two turns;
diagnostic union coverage>=20% of the fixed-posture50mm map. Count stalls,
perimeter looping, path, speed, novelty and guard stops separately. All12 must
pass before endurance. Repeat the final frozen chooser against the body variant
once (same2,112,000 cap), even if it remains unqualified, after its physical gate.

The current flexible-tail shoreline certifies contact-free separation and stops
when that conservative certificate fails. It does not model bank contact. Never
call a guard stop a measured collision or claim guaranteed infinite swimming
from any finite test. Depth/current/disturbance recovery remains separate scope.

Full substep ceiling4,241,983,616 includes six motor training+two reserved diagnostics+fixed
and supplemental tests, chooser training+diagnostic+nine assessments, one final
body/chooser assessment and material checks. Record actual partial attempts.
CPU-only diagnostics, compilation, timing and evidence I/O are measured separately.

Publish to Godot only if the exact policy/body passes motor movement/main-stroke,
continuous requests, all pond diagnostics and endurance. Otherwise keep the
protected live project, save all candidates and status honestly. No additional
experiment, unattended training or background process remains after the final
report. Then stop working, as requested.
