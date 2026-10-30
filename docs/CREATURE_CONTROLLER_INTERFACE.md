# Two behavioral policies and one physical request interface

Approved October29,2026. This contract guides the replacement creature; it does
not replace the current pond, change its body, or claim any new skill is ready.

## The two learned policies

1. **Goal chooser:** selects what to pursue, where to go and when to hold.
2. **Reusable movement motor:** turns a persistent physical request into joint
   commands. The same motor must handle swimming, turning, braking, holding and
   the physical movement required by shelter/rest. No motor network per ability.

PPO value critics are training auxiliaries. They estimate return; they do not
choose behavior or operate the creature. Each training alternative may have its
own actor, critic, optimizer and exploration parameters. The paired experiment
compares two alternatives for the ONE motor role, not two motors simultaneously
running the creature. The chooser is not instantiated or trained in that study.

## Persistent goal and movement request

Ownership is explicit:

| Owner | Responsibility |
| --- | --- |
| Godot scene/view | Draw native state, accept player actions, camera/UI; no body forces or learner ownership |
| GDExtension `DeepPondLab` | Own the selected native session/playback and route typed interaction/state calls |
| Native session | Own background trainer, visible simulation, checkpoint lease, snapshots and adoption checks |
| Goal chooser | Own persistent intention and requested destination, terminal velocity and heading |
| Movement motor | Sample four actuator actions conditioned on body state and the current request |
| Physical servo, when selected | Convert joint-angle requests into limited internal joint torques every5ms; no extra body stabilization |
| Shared simulator | Apply actual torques, drag, buoyancy and contact; report resulting state and work |
| PPO trainer/checkpoint | Own critics, sampled actions/log probabilities, optimizer/random state and complete continuation state |

The chooser-to-motor contract is the replacement design, not a claim that the
current production food scorer already supports every intention. Motor benches
supply requests directly and keep chooser learning outside their protocol.

The current body has two universal joints with two controlled axes each: four
actuator outputs. A joint-target motor requests four axis angles. Its physical
servo is an actuator interface, not another learned policy. Future body-layout
and controller-schema identifiers must fence changes in actuator count/sensing.
Growing limbs requires explicit compatible policy/observation transfer and new
physical tests; it does not require one motor network per skill. The geometric
destination/velocity/heading request remains reusable. No growth, limbs or
manipulation are implemented in the current motor experiment.

The existing `command_motor::Request` contains:

- `position`: destination in existing model/world coordinates (4cm per unit);
- `velocity`: desired terminal velocity in SI m/s, not a throttle command;
- `heading`: unit forward direction in world coordinates, not a tail animation.

A chooser-owned goal wrapper will retain stable goal identity/type, issue time,
commitment/progress and validity. Identity is bookkeeping, not a food-slot bias
feature. Food, shelter, exploration and hold are typed goals; they are not all
represented by edible or invisible pellets. A valid Hold request latches an
ordinary destination/heading and requests zero terminal velocity.

The motor keeps executing the request after first arrival, including braking
and holding. The chooser explicitly replaces/completes the goal; a disappearing
pellet or scripted zero-torque phase is not the stopping mechanism. A moving
target can update one goal's position/velocity without constantly changing its
identity. Support for nonzero terminal velocity/moving goals must be validated
after the first stationary-command motor experiment; having the field is not
proof of the skill. The first slice uses level terminal headings, not arbitrary
quiet tilted poses under buoyancy torque.

Goal reconsideration cadence/commitment and physical validity masks are designed
mechanics, clearly labelled as such. They must not become an undeclared
`if threat then force shelter` behavior selector. Exposed but physically
accessible food remains a genuine choice. Assigned shelter targets used in motor
lessons are not evidence that the goal chooser learned to abandon food.

## Staged chooser work, after motor acceptance

Reuse the existing shared goal scorer. The food scorer already handles all
active128 pellet slots; the shelter prototype has2food+4cover options and an
eight-action categorical limit. Extending the goal set therefore requires the
masked sampling/loss and critic/schema to support the full candidate pool;
silently considering only a few pellets is not acceptable.

**First feeding and safety:** design a bounded energy reserve, food refill,
basal consumption and conversion of positive measured actuator work into an
appropriate movement cost. `BodyState.work` is mechanical work and
`deep::energy()` is kinetic energy; neither is currently a nutritional reserve.
Food's useful restoration decreases near capacity; rest cannot create food
energy. Freeze accounting and reward units before chooser training.

Observe reserve, body state, current goal/progress, food positions/velocities,
observable danger and valid shelter poses/clearances. Danger/exposure samples
must be coherent or carry explicit age/validity; unknown must not mean safe.
Learn Feed/Shelter/Hold goal selection with useful energy restoration, charged
expenditure, low-reserve consequences and threat exposure, rather than another
unconditional food-count reward. Keep these choices separate from physical
execution and log which part failed in each behavioral trial.

**Later exploration/rest:** provide a saved visitation map with diminishing
novelty credit for repeated areas and changed-environment tests. Movement alone
must not earn novelty. Initially omit fatigue and a separate fatigue-rest task;
Hold/Wait permits energy conservation while basal consumption continues. Add a
distinct rest benefit only when its physiological/gameplay purpose is justified.

Required competing-choice tests include hunger beside food, hunger versus
exposed food/danger, satiety without danger, remaining covered through a threat,
returning afterward and later investigating a changed region. Clone the same
physical location with different internal/danger state and require different
learned choices. These are future behavioral gates, not additions to the motor
comparison's objective or budget. Growth and limbs remain deferred.
