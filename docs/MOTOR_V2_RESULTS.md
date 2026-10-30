# Motor v2: contact gate failed before learning

October 30, 2026. The new study is **stopped at physical readiness**. Neither
movement policy trained or reached final evaluation. The result supports a
focused contact-response decision, not a motor-interface winner or an app
overhaul. The separately authorized Godot audit/cleanup may continue without
resuming these experiments.

The verified starting point was clean branch `codex/autonomous-pond-practice`
at `dbbcf7b0caf2dadbfd14f08523415dea8d524f01`. The new implementation and this
report are uncommitted working-tree changes. The old paired experiment,
six-update partial torque checkpoint, saved controllers and results remain
historical evidence; neither new arm inherits that partial actor.

## Registered scope and versions

Read [MOTOR_V2_PROTOCOL.md](MOTOR_V2_PROTOCOL.md),
[MOTOR_V2_TASKS.md](MOTOR_V2_TASKS.md) and the
[two-policy contract](CREATURE_CONTROLLER_INTERFACE.md).

- Contact physics version: **2**; fingerprint `8169187946518266246`.
- Executed contact protocol fingerprint: `16067306998033709655`.
- New motor head/training/final keys:
  `0x4D082601 / 0x4D082602 / 0x4D082604`.
- Four actuator outputs are the two controlled axes at each of **two universal
  joints**. They are not four separate joints.
- Historical joint-servo gains are retained without tuning. Their earlier
  regulation passes were not rerun or relabelled as new evidence.
- Maximum new motor training: 64 PPO updates per arm, 512 worlds and 64
  controls per update. Actual new motor training: **zero**.

The chooser is excluded. Directly supplied persistent movement requests carry
destination, terminal velocity and heading. PPO critics remain training helpers.
The goal chooser, growing body, limbs and production swimmer are not replaced.

## What was implemented

The isolated rectangular laboratory now has an opt-in contact implementation
using the directional head ellipsoid and elliptical tail/fin cross sections,
including both tail endpoints. Endpoints split each tail link's contact
coefficients to avoid an arbitrary one-ended torque on a level body.

Contact stiffness remains 0.8 N/m per link; approaching-contact damping changes
explicitly to `2*sqrt(0.8*link_mass)`. The legacy non-contact assembly and
semi-implicit integration algebra are reused. Body dimensions, mass, actuator
limits and the production pond's physics path remain unchanged. This is a
versioned change to physical contact forces and trajectories, not a claim of
literally unchanged physics.

Compliant contact from zero through **0.5 mm overlap** is allowed. Larger
penetration, full escape, invalid states and numerical failures are rejected.
The allowance was fixed before running, from static 1% density-deficit
compliance, two 5 ms normal-travel increments at 20 mm/s and numerical margin.
It was not selected from the previous 0.193 mm breach or increased afterward.

The new environment/trainer keeps sampled Gaussian actions distinct from
resulting targets/torques, previous-action columns as actual last-substep
torques, actual-torque/work accounting and complete state/save metadata. It
adds observable water velocity and floor/contact state. Arm-local physical
failures can stop one arm; shared physics, numerical or integrity failures stop
all affected work. These paths are implemented; their required 32-control motor
readiness gate has **not run**.

## Execution results

| Stage | Actual execution | Result |
|---|---|---|
| Contact geometry and classifications | 133 independent/static checks | Passed; includes endpoints, orientations, reproduced old fin discrepancy, contact, excessive penetration, escape and invalid-state checks |
| Independent geometry oracle | CPU-double plane/support comparison | Maximum difference about 9.88 nm |
| New contact dynamics | 2,220 substeps per backend; **4,440 total** of the 113,600 cap | Shared stop after a rolled-body fixture exceeded the fixed overlap limit |
| CPU/CUDA comparisons before stop | 2,400 numeric comparisons | Zero outside the registered tolerance |
| Original recorded fin case | Corrected forces respond to its actual envelope | Overlap stayed within allowance during the recorded 0.2 s prefix; full five-second case not completed |
| Loaded support acceptance window | Requires the registered final one-second window | **Not completed**; the run stopped before that window |
| Final timestep acceptance | Requires completed five-second trajectories | **Not completed**; early failure occurred at all three timesteps |
| Free-water old/new arithmetic comparison | Scheduled after the dynamic cases | **Not run** |
| CPU-only motor preparation | Registered 64 final fixtures and fresh initial-policy files | Completed with zero physical controls and zero learning updates |
| Attempted motor launch with failed gate | Refused at the physical gate | No learner/readiness execution or new motor device probe followed |
| New motor data readiness | Authorized cap 32 controls | **Zero controls executed** |
| Direct-torque / joint-target PPO | Authorized maxima 64 updates each | **0 / 0 updates** |
| Withheld policy evaluation | Registered 64 episodes per arm | **0 / 0 evaluated** |
| Offline diagnosis | Read the saved frames and evaluated geometry/algebra only | Zero replay, integration, inference or optimizer steps |
| New Godot bench status verification | 27 assertions | Passed; scene is `motor_v2_bench/motor_v2_bench.tscn`, no trained actor and no learning updates |

The contact summary contains false aggregate flags for requirements whose
mandatory sequence was interrupted. Those must not be reported as independently
executed failures of the support window or free-water comparison. Likewise,
geometry/parity passes do not establish reliable contact dynamics.

## Precise failure and interpretation

Fixture 5 began with the body rolled 90 degrees, zero actuator commands and a
declared 1% negative-buoyancy test load. The breach was the **head ellipsoid**,
not the distal fin. The fixture's label includes a rolled fin orientation but
does not identify the anatomy that eventually contacts the bed.

| Substep | First excessive head overlap | Trajectory time | CPU/CUDA result |
|---|---:|---:|---|
| 5 ms | 0.588119 mm | 0.185 s | Both fail |
| 2.5 ms | about 0.506610 mm | 0.185 s | Both fail |
| 1.25 ms | 0.513539 mm | 0.18625 s | Both fail |

All recorded values remained finite, and escape was not reported. The overlap
limit remained 0.5 mm. No adjustment, alternate damping, new seed or repeat ran.

The unchanged buoyancy offset supplies about **8.33 µN·m** of initial passive
righting torque. During the 5 ms trajectory, head roll changed from 90 degrees
to about 75.9, 37.9, −14.1 and finally −47.1 degrees. Body roll speed reached
about **18.4 rad/s**. As it righted, the head's vertical extent shrank, floor
contact opened and the body fell; the extent then expanded as the roll
overshot upright and the head recontacted the bed.

At the first 5 ms breach the head center was already rising at about
0.97 mm/s, while the lowest rotating surface point was still moving downward
at about **20.94 mm/s**. Full-body geometry therefore detects a problem that a
head-center-only velocity test would miss. The joints remained near their
initial zero angles until impact; this was not a learned or scripted motor
command failure.

These recorded states and source equations support a coupled angular
recontact/normal-response explanation. The static density-deficit tolerance
derivation did not establish support for this righting event. The similar
failure time at all three substeps does not support timestep reduction alone
as a complete fix. It also does not prove every penalty formulation is
inadequate or that the body is uncontrollable.

The offline tool distinguishes spring algebra at a recorded postpose from the
preceding microstep's actual contact response. Only aggregate spring force was
recorded there; individual damping impulses and the full intervening state
history were not saved. No extra replay was used to fill that evidence gap.

## Decision and remaining claims

Keep the geometry correction and fixed safety criteria as preserved research
work. **Do not start either PPO arm, tune the servo, loosen the allowance or
silently rerun this contact protocol.** No interface recommendation is supported
because neither new policy trained or completed behavioral evaluation.

The smallest next decision is whether to authorize a bounded contact-only
normal-response pilot that accounts for gap, velocity and coupled inertia—for
example a unilateral normal constraint/impulse or implicit-contact formulation.
Keep the existing envelope, body, actuator limits and overlap allowance, and
require focused validation before another motor comparison. That is a local
contact-solver scope decision; it is not a whole-app rewrite or permission to
modify unrelated fluid/body behavior.

The movement goals remain unproved separately: purposeful travel, turning,
actual braking, slow arrival, ten-second unsupported calm-water holding,
unsupported holding against a pond-relative current, and bottom-supported
resting. Floor support must not earn unsupported-hover credit. Current holding,
shelter choice, feeding integration, exploration/rest and growth are not claimed.

The later user request for a whole-Godot audit and cleanup is separate authorized
presentation work. It does not reopen contact testing or motor training. Every
presented scene must identify whether it is live physics or playback and
whether its weights actually update.

## Evidence

All new evidence is under `build/motor-v2/2026-10-30-v1/`:

- `sources/user-request.txt`, baseline commit and `protected-before.json`:
  scope and preservation baseline.
- `contact-execution-manifest.json`: exact executed sources/binary/protocol.
- `contact/protocol.bin` and `protocol.json`: fixed numeric fixtures, coefficients,
  tolerance, timesteps and budget before execution.
- `contact/summary.json`, trace, gate and 192 recorded frames: original stopped
  result, never replaced by a later variant.
- `contact-inspection/`: independent saved-state interpretation and its explicit
  limitations; no additional dynamics.
- `motor-v2-protocol.json`, fresh preparation files and final fixtures: CPU-only
  setup, not trained checkpoints or evaluated policies.
- `global-stop.json` and `status.json`: shared physical-readiness stop.
- `gui-check.log`: no-actor/no-learning status verification.

The preservation baseline records 477 protected files. Final post-work
preservation and rendered presentation review are reported by the root task;
this document does not pre-emptively claim their completion.
