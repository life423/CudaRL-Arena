# Bounded motor milestone — October 29, 2026

**Later research-readiness amendment and completed probe:**
[MOTOR_BRAKING_RESULTS.md](MOTOR_BRAKING_RESULTS.md) supersedes the requirement
that a conventional controller pass all16 cases before research. The one
two-capture61,440-step search plus1,600 replay steps failed its alternative gate.
No PPO pilot ran; all earlier results below remain unchanged historical evidence.

**Decision: numerical preflight passed; physical reference did not. No PPO
pilot ran and no experimental controller was adopted.** The command task,
bounded trainer and separate live Godot bench are implemented and compiled.
Their implementation is not evidence of dependable learned movement.

The registered [motor plan](MOTOR_CONTROL_PLAN.md) remains the contract. All
180 original protected saves/results matched their SHA256 hashes afterward.
The working pond, goal chooser, personal saves and parked overhang work remain
unchanged. No commit, push or merge was performed for this milestone.

| Work actually executed | Result | What it establishes |
|---|---|---|
| Independent CPU-double versus GPU saved-actor calculation: 16 frozen rows, 3,108 gradients, six selected finite differences, one fresh Adam/clipping step | Passed every registered comparison | This particular numerical path agrees; not movement or PPO success |
| Passive zero torque, 16 registered open-water trials | 15/16 safe; 2/16 complete passes | Passive recovery succeeds in two pulse cases; does not establish active braking |
| One fixed conventional joint-PD reference, same 16 complete starts | 11/16 safe; 0/16 settled/held/passed | This reference is inadequate; not proof the body is uncontrollable |
| Eight actual-motion braking captures from original actor | 8/8 qualified | Evaluation starts have actual swimming momentum, not invented velocity |
| Isolated new motor preparation | Completed; zero physics steps and zero PPO updates | Fresh diagnostic initialization and immutable protocol are saved |
| Attempt to start with failed preflight gate | Explicit rejection before CUDA probe/trainer/fixtures | The failed reference cannot silently launch the pilot |
| Native Godot comparison and rendered UI smoke | 63 checks passed; Forward+ on RTX 5070 | Matched live clocks, repeatability, pause/restart, failure recovery and layout; not control competence |
| New actor's 128-update pilot and 64 final cases | **Not run** | Learned move→brake→stop→hold remains unproved |
| Strict shelter, predator decision and feeding integration | **Not run for new task** | No new claim of hiding or retained feeding |

## Numerical evidence

`build/motor-control/2026-10-29/numerical-v1/summary.json` records a handwritten
CPU-double forward/tanh/backprop, physical transform, standardized MSE and Adam
reference. It does not use the production transform/loss/backprop helpers as its
only oracle. The fixed selections are the first weight and bias in each layer.
The actor is protocol-5 update300; eight food and eight cover TRAIN rows are
frozen. No source was trained or modified.

CPU/GPU losses are .0608472999937/.0608472973404; gradient norms are
1.728416754/1.728416681. Global clipping is active with scales
.289281916/.289281928. Maximum full-gradient difference is 1.85e-7; selected
GPU/CPU finite-difference discrepancy is 1.43e-8; maximum fresh-Adam parameter
change discrepancy is 3.43e-7. All satisfy the predeclared absolute-plus-relative
tolerances; none was enlarged. The existing food gains, including1024, were
preserved. This does not establish every possible numerical path as correct.

## What the physical reference actually did

`reference-v1/cases.csv` and the full physical/surface traces are preserved under
`build/motor-control/2026-10-29/`. The study spent exactly its registered32-trial
budget: two controllers ×16 cases, each ≤600 control ticks including warmup.
There are four move cases, eight actual-motion brake cases, two90° turns and
two bounded water-pulse recoveries per controller. All require settling followed
by a continuous10s quiet whole-body window. No reset, rescue or safety relaxation
is used. Terminal headings are level; starts and positions still vary in3D.
Gate windows and recorded kinematics are sampled at50ms control ticks while
physics integrates at5ms. These are simulator acceptance checks, not a formal
continuous-time proof of every clearance between recorded control ticks.

The fixed conventional reference met travel timing in2/4 move cases, but neither
settled. It met0/8 braking envelopes and0/2 timed turns; neither recovery held.
25,032/36,324 evaluated torque components saturated (68.9%). For move case0 it
entered the goal band at4.85s while the head still moved15.87mm/s, the tail
84.71mm/s and the fastest joint9.30rad/s. By10s its goal error was69.52mm.
Reaching a target did not mean arriving under control.

Passive braking took4.9–11.3s to meet the low-speed dwell, exceeding the proposed
3s bound. The active reference failed to improve that into a valid stop/hold.
This leaves **active braking strategy and the physical envelope unresolved**.
In particular, the reference assumes .015m/s² in its distance-to-speed rule;
that is a controller design constant, not a measured achievable deceleration.
Reversing the commanded tail-wave phase was not validated as a braking skill.
The old stationary zero-torque shelter result remains valid and does not answer
whether a moving body can arrive there safely.

The executed reference binary is preserved as
`reference-v1/reference-tool-at-run.exe`. After that build, the geometric proof
was strengthened with continuous head-ellipsoid support against tank planes and
an inactive nonfinite-warmup abort guard. The original results/gate are untouched.
`reference-head-proof-review.json` documents an offline double-precision review
of all18,194 recorded float32 states: maximum clearance tightening .624118mm,
zero safety classifications, outcomes or first-penetration ticks changed.
That review performed **no simulator steps** and is not a rerun of the newly
compiled native proof. Executed/current rule fingerprints remain separately
recorded:14459424774619558674/11144086969182494675. Both leave the gate false.

## Prepared replacement boundary, not a promoted swimmer

The isolated task uses persistent requested position, terminal velocity and
heading with physical body/joint state. One59→32→32→4 motor emits direct physical
joint commands; no food/cover gain is silently mixed into the new task. The
original articulated integrator, four±10µN·m actuators and C++/CUDA PPO remain.
Compatible body feature columns9..32 and hidden features transfer; the torque
head, critic and Adam states are fresh. There is no inherited-output imitation
loss and no network per ability.

`pilot-v1/pilot-protocol.json` freezes the cost, all scalers, source identities,
seeds and optimizer settings. The conditional cap remains128 updates×512 worlds
×64 steps=4,194,304 training steps, one seed, no sweep. Its pure-CPU preparation
produced `initial-policy.bin`, **not a trained candidate**. The CUDA learner and
checkpoint machinery compiled but were not exercised by a training run. A future
passed reference must also match current source/environment/rule fingerprints.
Do not edit a gate file to circumvent the failure or reuse blind cases for tuning.

## Live viewing and verification

Open `godot-project/motor_bench/motor_bench.tscn` with Godot's Run Current Scene,
or launch Godot with `--path godot-project res://motor_bench/motor_bench.tscn`.
The separate `motor_bench` extension does native CPU simulation/inference using
the shared physical equations. It leaves the production `cudarl.dll` and saved
pond alone. The first comparison is original saved swimmer versus conventional
reference. Dropdowns also offer calibrated protocol5 and passive zero torque.
The untrained experimental slot stays disabled. This is a developer diagnostic,
not a new consumer/export release or autonomous predator demonstration.

Choose a trial; both panes start from identical complete states. Pause freezes
both clocks; restart/next/controller changes are explicit. Trials stop at their
registered endpoint and do not automatically restart. Failed starts hide stale
meshes and metrics. Negative clearance remains visible as a safety failure.
“Best quiet hold” is the longest past streak, not a claim the current hold lasts.

Rendered1280×720,1024×768 and780×1000 states were inspected. The smoke verifies
source loading, identical-actor determinism, live advancement with zero updates,
absent-candidate rejection, actual-motion braking capture, UI pause/resume,
restart/next/controller change, failed-start recovery and viewport bounds.
These are functional repeats of registered fixtures; they add no new efficacy
conditions and do not revise the32-trial reference results or gate. Screenshots
and logs remain in the evidence directory. No consumer pond/menu was changed.

## Next decision

Do not launch the PPO pilot yet. The smallest useful next investigation is
**active braking/control authority at the existing four-torque interface**:
why the fixed reference's reverse-wave/damping commands saturate and retain
tail motion rather than dissipating momentum. Use its existing traces first.
Discuss one bounded replacement of that diagnostic reference or a justified
physical-envelope study before spending more simulator trials. A better
reference would establish feasibility, not become a deployed rescue.

The numerical result does not justify a numerical-core rewrite; this reference
failure does not justify another brain, changed physics or a whole-app overhaul.
Shared exploration, cached-exposure state aliasing and fin/contact mismatch remain
separate unresolved issues. No new learned actor has earned behavioral acceptance.
