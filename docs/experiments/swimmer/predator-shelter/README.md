# Passing predator shelter pilot — not promoted

The user chose a **passing predator shadow**, superseding the proposed current
surge as the first shelter event. Native threat geometry, two-controller PPO
extensions, a separate resumable research path and a Godot prototype are
implemented locally. **No shelter policy has passed readiness or replaced the
working boulder swimmer.** The normal menu continues opening the boulder pond.

## Protected source

The running boulder pond was closed normally. Its complete source at learner
update **7,856**, applied motor **7,568**, is protected as
`build/predator-shelter/source-boulder.bin`. SHA-256:
`D9A5C0FD07163C6558FB11C30E6DB9CD65CA98DD5711F20E631F080CE668D8A6`.
The original user save matched that exact hash after the trials. A later final
audit observed its ordinary boulder learner advance **7,856→7,872**, with
consistent 1,024-world clocks and changed original 55-input actor parameters.
The writer was not attributed; a separate normal boulder startup is present in
the app log. Preserve this newer full save, **do not restore the older one**.
Its SHA-256 is `F8BFA0F8DB893DF71FB7268B14145C7BDE684D72A25BC3D1361CB3D6F0EAF10D`,
backed up as `build/predator-shelter/personal-boulder-observed.bin`.
No experimental shelter actor replaced the boulder actor.
Research outputs remain outside Git. New task optimizers/worlds have their own
histories; embedding the original full source does not mean its old optimizer
is being stepped on the new objective.

## What was actually tested

The initial frozen-motor diagnostic requested legal cover goals directly in
separate worlds. This is a capability test, **not a learned decision to hide**.
With food replenishment and secondary observations matching the real scene,
18/24 cases reached sampled cover, but **0/24 held protection for five seconds**;
the longest protected streak was 0.20 seconds.

Continuous native PPO then trained the inherited body controller in 1,024
worlds. Half retained feeding; half practiced approaching/holding cover. Four
new inputs describe the hold task and desired heading; initial expansion
55→59 preserves the original food outputs. Output remains four physical joint
commands. The body, forces, food and contact equations did not change.

Both 300-update trials used the same source, seed, population and held-out
cases. Protocol 1 used a binary safe-pose holding bonus. Protocol 2 supplies
smooth visibility/clearance feedback while retaining the strict evaluation.
They have separate full checkpoints; protocol 2 does not silently resume a
protocol-1 optimizer as if the objective were identical.

| Actual held-out result | Protocol 1 | Protocol 2 |
| --- | ---: | ---: |
| Five-second protected holds | 0/24 | 0/24 |
| Cover arrivals | 19/24 | 19/24 |
| Mean sampled exposure | 62.77% | 59.05% |
| Calm food/min, reference 12.500 | 12.375 | 12.250 |
| 0.75 cm/s food/min, reference 11.625 | 11.000 | 11.625 |
| Finite held-out runs | 24/24 | 24/24 |

All 3,108 actor parameters differ from the inherited expansion. Weight changes
and retained food performance are **not** proof of the missing holding skill.
The exposure numbers are development evidence, not confidence intervals or a
claim that one training recipe is conclusively better.

## Diagnosis before more training

A zero-torque physical feasibility test holds all four nominal cover poses for
ten seconds in calm water with positive whole-body clearance. This establishes
that a motionless safe equilibrium exists; it is not deployed creature control.
Passive drift in currents is a separate, harder stationkeeping task.

CPU inference from the same legal poses reveals inherited raw joint commands
around 99–119 for one actuator; the permitted physical range is −1 to 1.
Across the ten-second learned-policy diagnostic, 59–68% of sampled commands
are saturated and raw magnitudes can approach 1,000. Protocol-2 fine-tuning
does not remove this behavior. Clipping can make ordinary Gaussian exploration
ineffective there; this is a strong next investigation, not proof that it is
the only cause. Do not merely increase the training duration or relax hiding
criteria. Review bounded action parameterization/transfer before another
large trial, preserving ordinary swimming behavior through matched tests.

Code inspection also found different tail-fin envelopes in bed contact and
full-body visibility clearance. The policy diagnostic reports bed-envelope
penetration around −0.13 to −0.14 model units with smaller physical bed-contact
penetration. This remains a separate physics issue to investigate. No contact
equations were changed or buried fins counted as hiding.

## Implemented boundaries

- A seeded four-second warning, twenty-second oblique pass and calm gap.
- Actual bird-eye line of sight through the existing sphere/torus solids.
  A distinct fixed-sunlight projection drives the visible shadow.
- 216 sampled surface points across head and both articulated tail links.
  Protected time requires ≤5% visible samples and ≥0.08 model units of full-body
  clearance. This is sampled visibility, not exact continuous occlusion.
- The existing shared goal scorer can expand to food and four cover candidates;
  categorical PPO uses the same math. Food is always present and edible.
  No danger branch chooses a goal or disables collection.
- The pilot explicitly accepts its existing two automatic food slots. It
  rejects incompatible many-pellet saves rather than silently ignoring food;
  the separate public feeding pond still supports all active pellets.
- Body readiness requires 18/24 five-second holds and ≥90% matched original
  calm food performance. Both current candidates fail. A native Godot check
  confirms the failure is reported without starting goal learning or creating
  a misleading approved shelter save.
- Later goal-policy adoption has separate development/fresh gate seeds,
  exposed-at-warning eligibility, sustained physical cover, post-threat food
  return and ≥90% original calm retention. No such learned policy is deployed.
- `pond/predator_pond.tscn` is an isolated prototype, absent from the main menu.
  Its visual fixtures are QA, not a live learned shelter demonstration.

## Reproduce

Verification at this stopping point: **229/229 native tests with GPU required,
zero skips**, portable **2/2**, native Godot readiness rejection **11/11**.
Real Forward+ rendering passes **115/115 fixture checks** at 1280×720,
768×1024, 390×844 and 844×390. The actual readiness error shows recovery
feedback instead of a frozen game; unavailable native controls are disabled
and keyboard menu return remains reachable. Separate static cue fixtures use
a real paused boulder pose, are explicitly labelled **NO LEARNING**, and show
bounded shadow pixels at all four sizes without darkening the entire pond.
Original source/candidate/cove hashes match after the fixtures. This verifies
presentation and rejection, not a learned hiding demonstration or successful
full shelter-session promotion/resume.

Internal UI review (conservative design heuristics, not measured scores):
usability 9, hierarchy 8, consistency 9, feedback 9, responsiveness 8,
accessibility 8, performance 8. Remaining recommendations: reduce disabled
sidebar content in a future player-facing unavailable view; review the complete
working interaction with keyboard/assistive tooling; measure live frame time
and concurrent PPO throughput once a controller actually qualifies. The
current native-app checks are not a web accessibility/performance audit. Do
not redesign the working boulder interface to chase these prototype scores.

Build the developer targets `predator_motor_probe`, `predator_hold_diagnostic`
and `shelter_motor_train`. The latter takes a full boulder source, a **new**
output file, a total update count and optional population (default 1,024).
Reusing a compatible output resumes its actor, critic, exploration, Adam and
all practice worlds. Sources and outputs must differ. Close an actual game
normally before replacing the Godot extension DLL.

Raw logs, immutable inputs and both failed full trials are under the ignored
`build/predator-shelter/` directory. The protocol and this report deliberately
preserve failure evidence. Lily pads, an open-sided stone overhang, roots and
a short hollow log are **discussed habitat options, not added objects**.
