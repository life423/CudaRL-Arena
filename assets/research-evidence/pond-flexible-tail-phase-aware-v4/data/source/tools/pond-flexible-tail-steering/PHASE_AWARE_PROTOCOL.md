# Phase-aware steering — authorized session 4

November 24, 2026. The user chose option one: make steering aware of the tail
beat and swimming speed, while keeping the implementation coherent. Train a
separate native CUDA PPO candidate; preserve all working models and evidence.
This does not authorize full-gait retraining, another chooser run or a body/
physics change. All previous finite sessions are closed.

## Coherent policy extension

Use the existing motor runtime, 31-input linear steering readout, GPU rollout,
MLP/GEMM, PPO and Adam infrastructure. One shared policy module defines the
feature layout, model specification, parameter mask, validation and checksummed
model format for CPU inference, CUDA collection, training and evaluation.
Keep the existing fitted gait and physical yaw-trim decoder unchanged.

Initialize from protected whole-beat motor56, SHA-256
888BA2A571CEB9F90091D147A4CC56C6CECD092B3B6F58D679FBFCC779E98891.
Keep its first four coefficients and original whole-beat feature mapping as a
frozen base. Add zero-initialized phase/speed features in unused readout slots.
The complete initial policy must reproduce every old command, including large
turn requests; do not repeat the previous experiment's automatic gain increase.
The additional full-angle input supplies signed error/90 degrees without the
20-degree information cap. This is an additive learned extension of one motor
policy, not a separate ability network or hand-authored steering rule.

Feature slots are named and versioned. Slots0–3 retain the frozen base; slots
4–12 are trainable: signed error/90, root phase position, root phase velocity,
body-frame lateral COM velocity/8mm/s, error times measured speed/8mm/s,
error times each second-harmonic phase component, and squared error times each
first-harmonic phase component. Root phase is measured from actual joint angle
and velocity using the fitted gait's normalization. Whole-beat COM measurement
is shared with the preceding verified feedback runtime. The feature interactions
let a linear head express timing-dependent steering and speed-dependent strength.
All nine new features are odd under the matching physical left/right reflection.
Slots13–30 and the output intercept stay zero. Existing base coefficients and
their Adam moments must remain bit-identical/zero respectively.

Export phase models as an explicit new FTYO2 format with feature schema and
frozen base binding. Old FTYO1 readers must reject them. Historical modes and
checkpoints remain readable through their declared schema. Full-state session
identity binds the new model kind, observer, source, feature layout and inputs.

## Bounded PPO session

Use 1,024 CUDA worlds, 64 decisions/world at80ms, 65,536 samples/update,
minibatches4,096 and four PPO passes. Preserve actor/critic rates0.0003/0.001,
fixed Gaussian log standard deviation-1.2, gamma0.995, lambda0.97, clip0.20,
entropy0.01, value coefficient0.50, target/hard KL0.015/0.05. The critic remains
a 36→32→1 training auxiliary. Keep reward terms/scales, physical substeps,
torque limits, trim magnitude/slew and13s turn+2s onward episode unchanged.

Practise only balanced left/right30-degree requests from the same training-only
archive and mirrored seed machinery. This is the first phase-aware lesson,
not simultaneous changes to the angle curriculum, reward or episode length.
Preserve motion, servo, applied trim, phase/course history and policy state
through handoffs and across rollouts. Only a scored training terminal starts
a new training episode. No deployment reset or scripted turn is introduced.

Maximum128 updates /8,388,608 sampled decisions /536,870,912 attempted physics
substeps; cooperative training budget1,200 wall seconds, external limit1,800.
Save full actor/critic/Adam/world/history/RNG state after each complete update,
with immutable model exports. Finish an already-started update. Numerical,
native-physics or serialization faults stop the session and preserve failure
state plus the previous complete checkpoint. No silent retry or budget extension.

## Prerequisites and fixed assessment

First run zero-physics checks: frozen gait; all-angle initial mean and actuator
command equivalence; exact inactive swimming/Rest bypass; new model-format
rejection/round-trip; actual phase/speed feature sensitivity; full-angle
distinction; and physical reflection parity. Write the tested initial FTYO2
actor. Verify no copied source/base weight changed. Keep these tests focused on
shared policy behavior rather than duplicating implementations.

Then run one matched baseline with protected whole-beat56 versus the zero-new-
weight phase model. Verify full CUDA resume with three actually executed
updates: one/two continuous, reload one and repeat two. That diagnostic has
196,608 samples,12,582,912 maximum substeps and600 external seconds. Initial/
restored CPU/CUDA feature agreement must remain within1e-6. Observer checks
add zero physical advances. Diagnostic Adam and physics work count separately.

At every eight updates, evaluate the same32 fixed turns per policy: four held-
out phases and both signs of15/30/60/90 degrees. Training/evaluation phases
remain separate. Preserve13s turn,2s onward,2s beat closure and the existing
final-two-second/first-onward-two-second scoring windows. Retain actual COM
course error<=5degrees, projected and rolling speed>=6mm/s, whole-case tilt
<=10degrees, depth error<=1mm, positive certified whole-body clearance and
>=80% matched root/passive complete-beat frequency and swing. Record every
case, actual course change, response/settling time, radius, speed, phase,
learned correction, stroke, posture, clearance and failures.

Run all12 original straight/from-rest cases per policy with exact inactive
bypass. Do not label these a new holding/braking qualification. Each assessment
has at most1,100,800 substeps and30 external seconds; baseline plus16 scheduled
checks total18,713,600 substeps and1,496 trajectories. CPU evaluation can run
against immutable exports while GPU collection continues. No gate is loosened.

The lesson succeeds only if all eight30-degree cases pass in both scored
windows while all eight original15-degree keys and all12 swimming cases pass.
Report60/90 generalization separately; full sharp-motor qualification still
requires all32 fixed turns. Rank retained candidates by30-degree passes, then
lower30-degree error, retaining all scheduled curves and failed cases. Missing
turn mastery is a progress result, not a reason to abort valid training.
Required swimming/native failures request a cooperative stop.

This motor study does not publish to Godot. The working live bundle remains
unchanged until repeated-turn and pond/endurance tests qualify the exact new
model schema/runtime. No subsequent phase/body experiment starts automatically.

Report compilation, GPU physics/inference/PPO, transfers/waits, evidence, CPU
checks and end-to-end time with actual measurement boundaries. Host waits
overlap GPU execution. Preserve source/input identities, all run receipts,
complete checkpoints, and a saved-state audit with zero extra physics.
