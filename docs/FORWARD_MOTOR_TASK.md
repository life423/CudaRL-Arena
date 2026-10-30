# Forward-only motor preparation, version1

Update: the later user-approved single pilot is now executed and CLOSED.
See `FORWARD_DEMO_PROTOCOL.md` and `FORWARD_DEMO_RESULTS.md`. The preparation
and proposals below describe the state before that execution. The native
adapter now exists; four PPO updates completed, but controlled movement failed.

Implemented `src/motor/forward_task.h` is pure task math and bookkeeping. It
never steps/resets a world, runs inference/learning, selects a physics backend
or changes the Godot scene. Synthetic-state unit checks do not simulate motion.

## Compact initial lesson

One reusable movement motor receives the existing persistent `MotorRequest`.
This task defines its position reference as the **whole-body COM**, in model
units (1unit=4cm), so a head lunge alone cannot count as travel. At explicit
episode start, latch a COM destination2cm ahead, terminal velocity1cm/s along
the initial forward heading, and that heading. No target is moved each control.
The first experiment should use upright horizontal starts in calm water.

The current defaults are a proposed task contract, not executed acceptance:
maximum80 controls/4s; goal-plane crossing with lateral error<=2mm, terminal
velocity-vector error<=5mm/s and heading error<=10degrees for three consecutive
controls. This first lesson asks for controlled forward travel through a target,
not braking or ten-second holding. Those remain later lessons of the SAME motor.

The36 features are body-frame COM goal error (scaled by72mm), desired terminal
velocity (20mm/s), desired heading, world up, COM velocity (20mm/s), root angular
velocity (4rad/s), four joint angles (1rad), four joint rates (12rad/s), four
actual last-applied normalized torques, four held targets (1rad), remaining
finite-horizon fraction, and actuator-interface identity. No food, current,
obstacle, chooser or placeholder feature is present.

Four Gaussian latent actions map through tanh to four normalized commands.
Future PPO stores/replays the LATENT sample and its Gaussian log-probability;
the command, held target and resulting torque are distinct. Latent entropy is
not mislabeled as bounded-command entropy. The helper supports the two existing
interfaces but does not choose one or claim servo superiority. Effort uses the
world's actual mean-squared applied normalized torque.

Reward is normalized reduction in COM goal error, minus50ms-scaled terminal
velocity error, lateral error, heading error and actual torque cost. Evaluation
uses the independent travel/speed/heading criteria, not reward alone. An observed
finite-horizon expiry is terminal, not bootstrap truncation. Duplicate/skipped
world clocks, invalid state, contact or an interval leaving the declared domain
reject the sample; a future collector must reject it before GAE/Adam. There is
no automatic reset or recovery. Task counters/request must join the FUTURE
learner checkpoint; diagnostic physical saves are not silently repurposed.

## Readiness and integration decisions

The live world still uses physics2, ten5ms substeps per50ms control. The corrected
event path is experimental:100us internal intervals and event bisections. Its
scalar and short rolled checks pass, but original-free equivalence, sustained
support and a movement-runtime cost measurement have NOT run. An event-path
motor would need a versioned integration/actuator cadence/save decision; these
pure helpers do not make that decision or add another world owner.

The quickest scoped first experiment can instead retain physics2 **only within
an explicit free-water domain**, >=1cm full-envelope clearance at every native
substep, fixed original fluid and zero current. Contact physics stays enabled;
contact/clearance breach ends collection rather than being hidden or rescued.
This is a new limited-domain readiness decision, not a claim that the failed
physics2 support gate passed. No-contact agreement should compare the SAME
chosen step/control path on CPU/CUDA; event versus legacy paths are not assumed
bit-exact. Existing static J/M checks do not independently prove all fluid/C terms.

The current CUDA wrapper allocates/copies/synchronizes every control. It is fine
for a tiny readiness trace, but no persistent batch/rollout collector or complete
PPO trainer remains in this tree. MLP, Adam, Gaussian losses and GAE are reusable.
Do not restore the old multi-session hierarchy or advertise an untrained policy.

## Small next runnable proposal, not executed

Before a learned demonstration, decide the limited-domain physics/interface
contract. Proposed first choice to review: existing physics2 and fixed-gain
joint-target servo, as a hypothesis, with no claim it wins a torque comparison.
Use four fixed interior states and four50ms controls per state on CPU/CUDA:
**32 controls /320 native5ms advances total**, failures included, no retries.
Freeze exact starts/actions first. Check all substep clearance/finite/torque
bounds, observation/action mapping, actual-effort accounting and backend parity;
report elapsed time. This is diagnostic actuator readiness, not learned swimming.

A learned run then needs one compact native rollout/update adapter and a frozen
protocol. Example smallest pilot ceiling:16 worlds x32 controls x4 PPO updates
=2,048 training controls; four fixed80-control cases before/after=640 evaluation
controls. Together with32 readiness controls: **2,720 controls /27,200 native
substeps**, with no retries/extensions and rejected collection stopped before
GAE/Adam. This is a proposal, not a guarantee four updates produce swimming.
The user-approved demonstration must have its exact finite budget and label
frozen before any of those advances; no physical or optimizer call is made by
this preparation. A saved-controller demonstration cannot exist until weights
are actually trained and evaluated. Godot would display one creature only.
