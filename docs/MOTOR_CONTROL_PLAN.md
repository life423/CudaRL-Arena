# Dependable body motor — registered plan, October 29, 2026

Latest user direction supersedes treating inherited outputs or food rate as the
motor specification. Preserve all old code, saves and results for rollback, but
judge movement against physical behavior. This document registers the proposed
first slice; it does not authorize an unbounded training extension. The completed
bounded implementation/results are in [MOTOR_CONTROL_RESULTS.md](MOTOR_CONTROL_RESULTS.md):
numerical preflight passed, physical reference failed, conditional PPO not run.

## Decision

**Research-readiness amendment:** the latest user request supersedes requiring
the conventional reference to pass all16 cases. [MOTOR_BRAKING_PROBE.md](MOTOR_BRAKING_PROBE.md)
registers one61,440-step two-capture direct-torque probe plus1,600 frozen replay
steps. If it establishes safe braking and quiet holding on both starts and the
command data path passes, it may unlock the original capped pilot through a
separate research gate. The0/16 result, false reference gate, final standards,
128-update budget and nonpromotion rules are unchanged. No larger sweep.

**Execution authorized:** the user asked to do more than one tiny step per
turn. Carry the bounded milestone through its checks, physical harness, one
conditional pilot and live comparison without stopping after helper work.
The existing numerical/physical/budget stop rules still apply. Reference
fixtures are registered before execution with **level terminal yaw headings**;
initial pitch and target depth still vary in3D. Quiet arbitrary tilted holds
are not assumed feasible under the unchanged buoyancy restoring torque.

Recommend an **isolated replacement of the motor training task/command contract**,
with one reusable body controller. Keep goal selection separate. Reuse native
C++/CUDA, PPO/MLP infrastructure, the shared articulated simulator and Godot.
Do not add a neural network per ability or rewrite the application.

| Option | Addresses | Limitation |
|---|---|---|
| Smallest repair: retain current59 motor and add physically verified dynamic cover-state coverage | Rest-only fitting leaves velocity/joint/previous-action feedback unconstrained | Does not establish the missing approach/braking skill; retains mixed food/cover semantics and output conditioning |
| Recommended: isolated commanded-locomotion motor task | Makes move, turn, brake, settle and hold one physical control problem; trials persist after arrival; removes inherited-output imitation as the objective | Requires a bounded new training pilot and honest failure gates; not yet proven to work |

Some goal/velocity/heading inputs already exist. Adding another motivation or
merely adding those fields is not the proposed solution. The material changes
are task semantics, dynamic-state coverage, consistent actuator units and an
acceptance suite that includes what happens after reaching the goal.

## Existing evidence, not a restarted diagnosis

The completed supervised study remains unchanged at
`build/holding-supervised/2026-10-29-v5/run-v1/`:
cover-only holds5/16 unseen starts but loses feeding; joint holds1/16 and retains
feeding in the paired screen. Both hold0/4 familiar nominal starts and fail the
full gates. Component correctness, example fitting and counters are not swimming.

The48 qualified zero-action starts really stayed stationary and safe for ten
seconds. The cover-only fit meets TRAIN prediction tolerances, but its residual
torques are physically significant for the small articulated tail.

Example: cover-only TRAIN nominal3 (trace case19). Values below align command
with the preceding state that produced it, rather than treating post-step state
as the network input.

| State time | Head displacement | Head speed | Largest joint angle | Largest joint rate | Largest previous applied command | Largest next command |
|---|---:|---:|---:|---:|---:|---:|
| 0s | 0mm | 0mm/s | 0rad | 0rad/s | 0 | .00424 |
| .50s | .030mm | .099mm/s | .0163rad | .181rad/s | .01925 | .02751 |
| .95s | .143mm | .758mm/s | .0643rad | 2.017rad/s | .19423 | .14200 |
| 1.50s | .498mm | 1.648mm/s | .203rad | 2.458rad/s | .23936 | .29787 |

Exposure first exceeds.05 at2.00s, while bed/solid clearance is still positive.
The head barely moving does not imply the whole animal is quiet. Initial
prediction error creates joint/rate/previous-action states absent from the
stationary supervised cover rows, and commands subsequently grow/oscillate.
This is observed closed-loop amplification, not a measured unstable eigenmode
or proof that any one input or insufficient capacity caused it.

The earlier action-unit correction is present. The completed fitting study did
not prove the whole numerical stack faulty. The requested tiny independent
real-actor CPU-double loss/selected-parameter finite-difference/fresh-Adam check
has now passed its narrow independent preflight. Do not repeat it as an open-ended
numerical campaign or interpret it as proof of closed-loop motor behavior.

## Motor interface and training boundary

One motor receives a persistent request: position, desired terminal velocity
(zero for this slice), and desired3D heading, together with physical body/joint
state. It outputs the same four internal joint torques, obeying existing torque
limits. The goal persists through arrival and the ten-second hold; nothing is
eaten, respawned or switched into a scripted stopping action.

Use one consistent physical action convention across requests. A new versioned
motor must not inherit food-versus-cover gain semantics accidentally. Preserve
the old adapters/derivatives for the old references; this is not a claim that
their already-correct chain rule needs fixing again.

Keep the current MLP width initially. For the proposed single pilot, transfer
compatible physical-state feature layers from the original accepted motor,
initialize one small physical-unit torque output layer, and use fresh actor
optimizer/critic state. This reuses learned features without making the old
action function or food rate a loss target. Freeze the exact initialization and
seed before the pilot; do not compare many initializations opportunistically.

Train one coherent command-tracking objective over position, terminal velocity,
heading and whole-body/joint motion. Include moving, turning and mildly
disturbed physical starts. Do not repeat stationary-only imitation or bolt a new
motivation onto food pursuit. Register one fixed objective/normalization before
training; do not retune coefficients against the withheld suite.

## Acceptance envelope — provisional requirements, not measured capability

Unchanged body scales: length72mm, mass2.537g, four internal actuators limited
to ±10µN·m; 5ms physical substeps and50ms control ticks. Position errors should
be expressed relative to body length, and quietness must include the tail.

The following proposed values must first be checked against a bounded
same-physics conventional reference and then frozen **before any neural
candidate evaluation**. They are desired control requirements, not guarantees
deduced from torque limits. If the reference cannot support them, stop and
review physical feasibility/reference adequacy with the user; do not quietly
relax them after observing a failed candidate.

| Behavior | Proposed envelope |
|---|---|
| Purposeful travel | Goal .5–1 body length away in3D; halve initial error within4s; reach the position band within12s; report path efficiency and reversals |
| Turning | After a declared heading change up to90°, reach ≤10° heading error within5s; report translation and oscillation |
| Braking | From actual1–3cm/s motion, reach ≤2mm/s for.5s within3s and travel ≤1 body length after the stop request; report signed overshoot/cross-track error |
| Settle | Position error ≤.1 body length=7.2mm and heading error ≤10°; integrated move-stop trials settled by15s |
| Hold | All limits continuously for10s: goal/heading bands above, head/COM speed ≤.5mm/s, whole-body sampled-point speed ≤1mm/s, body angular speed ≤.02rad/s, joint rates ≤.05rad/s, joint peak-to-peak excursion ≤.05rad, settled-position drift ≤3.6mm |
| Recovery | Small native physically generated disturbances; regain the same settling envelope within5s and then hold10s; no position jumps or rescue commands |

An unmoving bent tail can be valid. Do not confuse quiet joint motion with a
requirement that every joint angle be zero. Record head, COM and tail-point
motion; do not accept flailing around an almost-stationary head.

The braking envelope particularly needs a physical reference. A simplified
straight rigid-body drag estimate predicts3cm/s→2mm/s may coast for~13.5s and
~11cm. That is not a new simulation or an active braking bound. Zero torque
holding from rest never established useful braking authority.

All trials require finite states, actual force-driven movement and no physical
penetration/surface breach. After the open-water gate passes, apply unchanged
whole-body clearance≥.08 model units and predator exposure≤.05 for the full
ten seconds in the existing shelter fixtures. Keep the fin/contact mismatch,
shared exploration and cached-exposure reward aliasing tracked separately.

## First bounded implementation slice and budget

1. **Narrow preflight:** independent real saved-actor check on16 frozen rows
   (8 food/8 cover), two selected parameters per layer (one weight/one bias),
   CPU-double physical loss/finite differences and one fresh Adam/clipping
   step against GPU. Independent reference formulas; no training campaign.
   Use the first weight and first bias in each of the three layers, fixed
   before inspecting differences. Double central difference step is
   `1e-5*max(1,abs(parameter))`. Predeclared absolute+relative tolerances:
   physical forward means `2e-5 + 2e-5*abs(reference)`; standardized physical
   loss `1e-7 + 2e-5*abs(reference)`; selected/full gradient components
   `2e-6 + .002*abs(reference)`; fresh Adam parameter changes
   `2e-7 + .002*abs(reference_change)`. Report actual errors and gradient/clip
   norms as well as pass/fail. Do not enlarge tolerances after inspecting a
   disagreement; investigate that local numerical path first.
2. **Physical acceptance harness:** calm open water, persistent non-food
   commands and complete physical traces. At most two reference controllers
   (passive zero and one conventional same-torque diagnostic),16 scenarios
   per reference, at most32 trials of ≤30s each (19,200 physical control ticks).
   A conventional controller is a labelled feasibility reference,
   never a deployed rescue or learned-behavior claim. Failure of one reference
   alone is not proof that the physics is uncontrollable.
3. **Freeze the physically supported envelope and fixtures.** If unresolved,
   stop before neural training. Record all rejected/failed reference trials.
4. **One conditional motor pilot:** maximum128 PPO updates×512 worlds×64
   steps=4,194,304 physical training steps; one initialization/training seed,
   one objective, fresh optimizers, no hyperparameter sweep or automatic
   extension. Final policy is update128, not a selected best on blind results.
   Register16 development cases and64 independent final cases before training.
   Fixed seeds: reference fixtures `0x4D072600`, torque-head initialization
   `0x4D072601`, training worlds/sampling `0x4D072602`, development fixtures
   `0x4D072603`, final withheld fixtures `0x4D072604`. No seed selection after
   outcomes. Freeze the physical fixture-generation rules before using them.
5. **Final64 open-water cases:**24 integrated move→brake→stop→hold,
   8 direction changes,8 braking starts from captured real swimming motion,
   16 initially settled holds,8 physical-disturbance recoveries. Vary position,
   depth, yaw/pitch and complete body/joint/momentum states. Split by complete
   starting conditions/trajectories, never adjacent frames. No mid-trial resets.
   The integrated sequence must include motion generated by the candidate's
   own torques; injected initial velocity alone cannot establish that sequence.
6. Compare original accepted motor and latest calibrated motor as frozen
   references through documented target adapters. Keep their original action
   mappings; expose unsupported command fields rather than silently supplying
   new intelligence. Same complete fixtures, commands, force limits and metrics.
7. Godot live physical A/B viewer with target/heading markers and speed/error
   traces, stable camera and trial identity. Failed behavior remains visible;
   recordings are supplemental. No switch into a fallback during a trial.

No food, predator decision, currents-as-a-new-feature, limbs or full-app rewrite
in this slice. Bounded water/force pulses may be used as explicit disturbance
tests, not as a new current curriculum. No learning pilot starts just because
this plan was written: the reference envelope must be resolved first.

## Gates and stopping rules

Report every case and worst-case behavior, not just a mean or learning curve.
For first-slice acceptance require every final case to meet its registered
motion/settling/ten-second gate with zero safety/nonfinite failures. Partial
success remains partial. No promotion on food rate, counters or component tests.

Stop on numerical-reference disagreement, physically unresolved requirements,
nonfinite dynamics, budget exhaustion or failed final gates. If a pilot fails,
identify whether command mapping, state coverage, action representation,
optimization or physical control authority needs investigation. Do not infer a
whole-app or capacity failure. Discuss any larger boundary change or added run
before implementation.

Keep four claims separate: numerical correctness; fitting example commands;
successful closed-loop movement; autonomous predator avoidance. Only the last
includes deciding to abandon food, reaching shelter, remaining hidden and
resuming feeding. Food collection is a later integration test; its historical
rate is a comparison, not an untouchable acceptance specification.

## Preservation and current status

Rollback snapshot: `build/motor-replan/2026-10-29/working-files/` plus complete
working-tree patch/status. Manifest preserves180 existing save/result artifacts;
the earlier160 protected files still match their hashes. Original files remain
in place. Completed supervised checkpoints, datasets and traces are unchanged.
The independent numerical check and32 registered physical reference trials have
now run. Conventional reference0/16 full passes; passive2/16. The physical
envelope is unresolved, so the gate blocks the compiled conditional pilot.
Only CPU preparation and an explicit false-gate rejection were executed for the
new learner: zero PPO updates, no adoption. The separate live Godot motor bench
is implemented and verified; the visible consumer pond is unchanged. See the
results document before any additional simulator experiment or scope extension.
