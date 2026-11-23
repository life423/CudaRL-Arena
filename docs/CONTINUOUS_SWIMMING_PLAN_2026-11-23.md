# Continuous swimming: combined plan and starting point

Latest update: [v2 is closed before physics](FROZEN_INTERFACE_COMPARISON_V2_RESULTS_2026-11-23.md).
The fixed-input arithmetic repair passes, but nested evidence paths prevented
the one-shot native launch.0/88 interface cases and no corrected-runtime
retention ran. Prepare long-path-safe evidence plus a native final-tree inventory
preflight before fresh authorization. No training or retry is active.

Earlier update: the user subsequently authorized the controlled comparison. Its one
attempt is now [closed at the CPU/CUDA gate](FROZEN_INTERFACE_COMPARISON_RESULTS_2026-11-23.md),
before main interface cases. Saved weights are unchanged; no retraining or
further run is authorized. Address the measured backend agreement obstacle
before resuming the sequence below.

Saved November 23, 2026 after review of the user's three supplied analyses.
This is the recommended sequence, not an executable experiment protocol or
permission to implement/run it. The user explicitly said not to implement yet.
No new simulation, PPO, model change, adoption or publication is authorized.

## Start here: answer the interface question before retraining

The first deliverable is a reviewable, finite diagnostic protocol comparing
the inherited swimmer under its original lane-following observer and the
proposed heading-only observer. Prepare only the prerequisites needed for
that comparison, rather than first completing the full PPO training pipeline.

The question is: **Which, if either, request interface lets the existing
learned motor make repeatable turns that fit this pond?**

The completed probe supplied a persistent new course while retaining lateral
error against a new straight lane, scaled by 2 mm. A curved transition can
increase that error. This is a plausible input/objective conflict, not a
proven explanation or a hard physical lane constraint. Removing familiar
feedback could improve or worsen the swimmer; the new observer has no measured
physical benefit yet. Read the [idea register](IDEA_REGISTER_2026-11-23.md).

### What the first comparison must isolate

- Use identical frozen saved motor weights and matched physical fixtures.
  No learning, extra steering policy, torque change or physics adjustment.
- Arm A: original lane-following observation; retain the persistent requested
  course and explicitly specify the lane-rebase convention.
- Arm B: heading-only observation; remove horizontal world-space lane error
  before body-frame rotation while preserving original depth and the same
  persistent course/terminal velocity. A zero added intent column is permitted
  only with verified same-input inherited inference.
- Include zero-offset controls in BOTH observer modes, isolated small turns,
  repeated left/right requests and return to straight swimming. Choose finite
  representative poses, gait phases and request schedules in the protocol.
  One small-angle success is not evidence of repeated-turn competence.
  Define settling onto the final course separately from requesting a return
  to the original course; do not call both different maneuvers one recovery.
- Requests persist in world coordinates. Increment the previous persistent
  intention, not the oscillating head direction. Do not assume a one-second
  chooser interval or 5-degree-per-second physical turning ability.
- Preserve body state, original vertical reference, elapsed time, joint targets,
  reference rates and any action history across each command change. No reset,
  recentering, scripted body motion or hidden recovery during a trajectory.
- Define all allowed arm differences. Run zero-change controls to distinguish
  observation effects from wrapper/initialization defects. Changed zero-offset
  behavior in Arm B is a result, not excused by zero-column migration.

### Evidence and exit decisions

Record actual whole-body COM course and speed, depth, tilt, response delay,
turn radius, requested/applied torque and all-link clearance through the turn.
Retain commanded course separately. Whole-body clearance must account for
rotation/deformation and filter lag; current directional rays are not already
a swept-turn predictor.

Before execution, freeze the exact fixtures, body/action/observer identity,
CPU/CUDA matched-command checks, independent saved-data audit, unchanged
physical guards, behavioral criteria, source hashes and finite caps on every
attempt/control/substep, runtime and evidence bytes. Include failed attempts.
No prior study's budget or approval can be reused. The numerical allocation
is not chosen by this plan and must not be invented at execution time.

- If either interface provides repeatable turning with useful propulsion,
  depth/posture and a feasible whole-body envelope in this pond: preserve
  those weights and proceed to the chooser design. Select the interface by
  a rule fixed before the comparison if both qualify. Motor PPO may be unnecessary.
- If the comparison is valid but the motor remains inadequate: identify the
  measured shortfall and prepare the bounded shared-motor PPO stage below.
- If a numerical, data-path or guard defect invalidates the comparison: report
  that defect and the actual spend. Do not infer a motor/physics failure or
  automatically repeat, enlarge or retune the study.

## 1. Protect existing learned behavior throughout

Keep original checkpoints, saves and evidence immutable. Work on separate
candidates. Retention is measured on the actual candidate in every mode, with
its real decoder and observations, never a hidden old-policy bypass.

During any motor learning, mix freshly collected current-candidate swim/rest
practice with turning practice. Historical fixed trajectories are not current
on-policy PPO samples. Use focused regression checks during learning, then the
complete unchanged suites before adoption: four short swims, the 28 previously
passing 60-second windows, three recoveries, the 148 previously passing
300-second windows and all 14 hold cases. Preserve known startup misses.

Practice supplies retention gradients; checks detect regression. Neither
guarantees that every training update preserves all behavior. A regressed
candidate must not replace the working version. Any rollback/retry policy
must be specified before a study; no automatic budget extension follows.

## 2. Improve the same motor only if the first comparison requires it

Retain one reusable four-output motor and native C++/CUDA per-control PPO.
Fine-tune inherited weights rather than adding a turn-specific behavioral
network. Train coordinated turns in both directions, repeated turns and a
return to straight travel without resetting actuator or physical state.

Use manageable requests and measured-capability progression within finite
per-stage and total caps. Do not assume a fixed update count teaches a skill,
or require abrupt 90-degree jumps as the first pond capability. Complete the
full training gates and independent auditors only if this stage is needed.

Design the complete live utility to preserve credit for legitimate moving
turns while rewarding improvement toward persistent course/speed and stable
depth/posture. Include effort within consistent bounds/units. Review fast
wrong-direction travel, stalls, oscillation, head-only alignment and early
failure explicitly. The current narrow velocity utility and the reviews'
alternative formulas are proposals requiring reconciliation, not approved
reward changes.

Keep model6, dimensions, water forces, 1 Hz filter and torque limits fixed for
this sequence. Existing recordings show no torque saturation in the failed
probe. Physical ablations remain alternative later investigations, not the
default response to a failed learning allocation.

## 3. Teach one bank-aware chooser above the qualified motor

Initially freeze the qualified motor so navigation learning cannot overwrite
its weights. The learned chooser uses bank geometry, approach velocity,
persistent intent, actual COM course, tracking quality and task context.
Its request slew and decision interval must fit the measured response delay,
turn radius and whole-body clearance, not an assumed 1 Hz interface.

For continuous-SWIM, the chooser selects feasible swimming intentions; it
cannot choose HOLD/zero speed to earn survival credit. Preserve Rest as an
explicit separate request to the same motor. Future shore approach uses
explicit task context; bank aversion is not a permanent ban on shore contact.

The main reward is useful safe physical travel. Use consistent signed
potential shaping only if needed, with matching discount, task state and
terminal handling. Positive-only or warning-gated clearance bonuses can
pay repeated approach/retreat cycles. A predictor changing without real
safety improvement must not masquerade as successful avoidance.

## 4. Define continuing navigation separately from a finite turn lesson

A true finite skill lesson may terminate at its declared horizon. Continuing
pond collection cutoffs are different: bootstrap the final state value while
actual physical failure remains terminal. Specify discount, objective, masks,
value context and continuity together. Do not tie continuing failure penalties
to an arbitrary collection boundary or simply flip the old terminal flag.

Detect useful COM transport over multiple strokes, with rolling speed and
displacement/course diagnostics. Arc length alone can reward oscillation.
Large start-to-finish displacement over the whole run is inappropriate for
valid pond circuits. Coverage/exploration is a separate objective; a mandatory
70 percent coverage threshold is not selected for the immediate swimming goal.

## 5. Validate a frozen candidate, then update the actual Godot app

The proposed progression is 16 initial one-minute cases followed by eight
withheld uninterrupted 30-minute calm-pond runs with varied starts/headings.
Freeze exact criteria and budgets before execution. The reviews' candidate
targets include 6 mm/s mean travel, no complete eight-second post-startup
window below 3 mm/s, 1 mm depth, 10-degree tilt and 5-degree settled course
error for supplied-turn tests; these numbers still require a final protocol.

Report motion, both-direction turning, body clearance and unchanged skill
retention. Test the exact candidate in the native CPU/Godot path as well as
the disclosed CUDA scope. Matched-command agreement alone is not evidence
of long independently closed-loop performance on both backends. Freeze the
candidate before withheld evaluation; no retrospective checkpoint selection.

Load the exact validated artifact into one persistent native world. Godot
must forward new requests without resetting the fixture or altering body
poses. Preserve the accepted pond, close aerial view, manual zoom and readable
GUI, and verify the rendered result. Current Rest/Swim fixture resets do not
establish braking or seamless mode changes. Display honest capability and
learning/inference labels.

Success supports the duration and conditions actually tested, not infinite
reliability. Currents, live adaptation, growth and shore exit remain later
capabilities with their own requirements. No merge or deployment follows
automatically. This saved plan leaves implementation and execution on hold.
