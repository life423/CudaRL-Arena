# Whole-beat steering feedback — authorized session 2

November 24, 2026. The user authorized fixing noisy steering feedback, another
GPU PPO session, earlier learned bank avoidance and Godot adoption only after
qualification. All earlier sessions are closed. Preserve their source snapshots,
checkpoints and failure evidence, the working live motor and the fitted stroke.
No food, scripted steering, physical shortcut or criteria relaxation.

## One feedback correction

The opt-in observer measures whole-body COM velocity over one actual root-tail
beat. Root yaw-rate zero crossings are interpolated at the existing substep
rate and confirmed with the existing 0.02 rad/s deadband. A continuous causal
COM window uses the most recent complete beat's measured duration, supporting
the existing 0.4–2.0-second history envelope. Before a complete period is
available, use a clearly labeled one-second displacement from genuine archived
COM history. Never fabricate velocity from the desired heading. Track actual
period availability, age and low-speed validity in saved state and telemetry.

Replace the actor's signed measured-course feature and the corresponding
course/transport/correction reward measurements consistently. Pre/post correction
uses the same window duration to prevent period updates from earning artificial
progress. Keep facing, angular-rate and applied-offset features, decoder cap,
slew, frozen gait phase calculation and all physical equations unchanged.
The default historical observer remains available for closed-study replay.

Initialize a separate candidate from the original update-eight offset, not from
the previous experiment's final candidate. Keep all original files protected.
This isolates the feedback change under matched budgets, starts and seeds.

## Motor run and checks

Use the session-one configuration unchanged: 1,024 native CUDA worlds, 64
80 ms decisions per world, 65,536 samples/update, minibatches of 4,096, four
passes; actor/critic rates 0.0003/0.001, fixed log standard deviation -1.2,
gamma 0.995, lambda 0.97, clip 0.20, entropy 0.01, value coefficient 0.50,
target/hard KL 0.015/0.05. Keep the four-coefficient actor and 36→32→1 critic.
Use balanced ±15/30/45/60° training requests, 13-second turns and two-second
onward swimming, continuous state across that handoff and rollout boundaries.
Reward coefficients and failure scoring are unchanged from session one.

Maximum 128 updates / 8,388,608 transitions / 536,870,912 attempted substeps,
with 1,200 cooperative training wall seconds and a 1,800-second external cap.
Finish an update already started; save complete actor/critic/Adam/world/RNG
state and immutable actor exports every update. Bind observer mode, task and
frozen inputs. No silent retry or budget extension. Numeric/native/serialization
faults stop the study. Scored training clearance/posture terminals remain counted.

Before physical work, run focused native synthetic feedback checks, saved-data
gait equivalence and exact inactive bypass. Then run one matched baseline and
the three-update exact-resume diagnostic (196,608 transitions; at most
12,582,912 substeps). CPU/CUDA observer parity at initial and restored states
must agree within 1e-6 per feature with no additional physical advances.

Evaluate immutable checkpoints every eight updates with the original 24 turn
cases and 12 straight/from-rest cases per arm, maximum 883,200 substeps and
30 external seconds each. Compare old feedback/original offset against the
new feedback/candidate. Retain all existing motion, depth, tilt, clearance,
root/passive swing and frequency thresholds. Score final two turn seconds and
first two onward seconds as before. Report complete passes and motion-only
classification separately; do not waive a rhythm failure. Full sharp steering
still requires all 24 cases, including the eight 60° cases. Every checkpoint
must retain all 12 inactive straight/from-rest cases and exact bypass.

Measure half-second and beat-window feedback noise on the same saved control
traces, actual left/right course change, response/settling time, projected and
rolling speed, effective radius, stroke shape, depth/tilt and minimum clearance.
Synthetic/observer checks do not count as physical navigation qualification.

## Conditional pond stage

Early avoidance can use repeated gentle turns. It need not wait for 60° motor
mastery. A candidate is eligible for this limited stage only if it preserves
all eight original ±15° complete pass keys, all 12 swimming/start cases and
their physical envelopes. Choose the eligible checkpoint with most 30° passes,
then most 60° passes, then lower 60° course error. Retain its mode metadata.

First evaluate that frozen candidate with the preserved best chooser
(sustained-session update 56), persistent heading requests, and identical native
feedback in both motor/evaluator. Use the existing 12 fixed 120-second cases;
require all 12 passes and >=20% pooled coverage of the existing 345-cell map.
Only after those pass run the existing four 300-second endurance cases.
Each invocation retains its 2,112,000-substep/20-second external allocation.

If a motor is eligible but the frozen chooser fails, train one separate
food-free chooser session of at most 64 updates / 524,288 macro decisions,
with 1,024 CUDA worlds, eight decisions/world, persistent headings and the
existing optimizer/reward/task mix. Frozen motor weights do not learn in this
stage. Give it earlier terrain information using fixed forward distances
40/200/400 mm and side distances 150/350 mm. These are observations of native
whole-body clearance, not a commanded turn, path, inner boundary or rescue.
Initialize the new candidate once from the preserved chooser actor; establish
and retain its own complete critic/Adam/world/RNG history thereafter. This is
a new task version, not a claimed resume of an incompatible old session.

The conditional chooser has a 1,200-second cooperative / 1,800-second external
wall budget, maximum 2,516,582,400 attempted substeps (300 controls/decision).
Its separate three-update exact-resume check has at most 117,964,800 substeps
and 600 external seconds. Run the unchanged fixed navigation gate every eight
updates; do not change criteria in response to results. Preserve all failures.
If no motor is eligible, chooser training and autonomous publication remain
unrun, and report that gate explicitly.

## Godot admission and reporting

Publish only after all required motor-retention and pond/endurance gates pass.
Use the same learned motor/chooser, feedback mode, sensing and native physics
as the qualifying tests. Preserve the original live bundle for rollback, bind
new model/observer schemas and hashes, and inspect the actual registered Godot
project through Project Manager. Verify play/pause/restart, camera controls and
autonomous status. Failure leaves the working live motor in place.

Report actual completed updates, stage timings, all diagnostic/physical/optimizer
work, checkpoint selection, feedback-noise reduction, turning and retention,
pond cases/coverage/stalls/clearance stops, and exactly which bundle Godot loads.
Host waiting overlaps GPU time; compilation and CPU evidence are separate costs.
