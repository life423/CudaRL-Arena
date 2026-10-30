# Paired motor experiment: regulation passed; training stopped on safety

Experiment `paired-motor/2026-10-29-v1` is **stopped**, not awaiting an automatic
restart. The direct-torque arm completed six updates. Its seventh rollout failed
the unchanged whole-body clearance gate before GAE/Adam. The joint-target arm
has zero updates. Neither interface has passed the motor acceptance suite, and
there is no evidence that one learned interface is better than the other.

The working pond, personal saves, body, physical equations, torque limits and
both existing decision controllers remain unchanged. No policy was adopted,
merged or deployed. The previous 128-update experiment remains separate.

## What actually ran

| Stage | Actual execution | Result / limitation |
|---|---|---|
| New CPU/CUDA experiment and viewer | Strict release builds of regulation, paired trainer and isolated viewer | Compiled; this is not behavioral evidence |
| Joint regulation | Eight fixed cases on each backend; exactly 3,200 controls / 32,000 physical substeps | 8/8 CPU and 8/8 CUDA passed |
| CPU/CUDA parity | 184,000 recorded numeric comparisons | Zero outside the registered tolerance |
| Focused data readiness | Reused all 3,200 frames; no additional integration or optimizer update | Observation, actual-effort, latent likelihood and full-save checks passed |
| Direct-torque PPO | Six completed updates, 196,608 controls, 192 actor/critic/std Adam steps each | Partial checkpoint only |
| Rejected seventh rollout | 32,768 rollout slots; 32,710 actual integrated controls, 58 frozen slots | One finite world crossed the bed with its distal fin; rejected before GAE/Adam |
| Joint-target PPO | Zero updates | Global stop prevented this arm from starting |
| Final fixture preparation | 314 controls of preserved-swimmer warmup, captured once before training | Starts saved; no candidate results or model selection |
| Final 64-case policy evaluation | Zero policy-evaluation controls | Movement, turning, braking, stopping and ten-second holding all remain unproved |
| Godot viewer | Actual six-update torque file, empty untrained joint-target pane | 93 UI assertions passed; three rendered sizes inspected |

Training integration spent **229,318 controls total**, including the rejected
collection, below the authorized 4,194,304 ceiling. The final evaluation budget
spent only the 314 fixture-capture controls, below 115,200. Functional viewer
checks use reference-suite states and are not final efficacy evaluations or
learning updates. No gain sweep, extra seed, training extension or final-case
tuning ran.

## Fixed interface and settings

The preregistration is [PAIRED_MOTOR_PROTOCOL.md](PAIRED_MOTOR_PROTOCOL.md).
Both alternatives use the same 59→32→32→4 actor architecture and persistent
position / terminal-velocity / level-heading request. Compatible inherited
body-feature weights are transferred; output heads and critic/Adam/exploration
initialization are fresh and matched. The chooser is excluded.

The joint-target actor would issue normalized Gaussian target latents every
50 ms. Targets are clipped to the existing ±1 rad limits and held. A physical
PD servo recomputes internal torque every 5 ms with the existing ±10 µN·m cap.
There is no extra rate limiter, whole-body force, gait or rescue. CPU, CUDA and
viewer share the same implementation.

PPO stores/evaluates the raw Gaussian latent, including values outside actuator
bounds; it never evaluates a resulting torque as a target action. Effort is the
mean squared actual normalized torque across all 40 joint/substep commands.
Observations 29–32 remain the previous **actual last-substep torque**, not joint
targets. Held targets, telemetry, complete worlds, seeds, actor/critic/std and
all optimizer state have a new versioned save schema and binding hashes.

The one gain set uses the minimum coupled effective inertia over the eight
registered poses, natural frequency 2 Hz and damping ratio 1:

| Joint | Kp (N·m/rad) | Kd (N·m·s/rad) |
|---|---:|---:|
| 0 | 3.68326687e-6 | 5.86210149e-7 |
| 1 | 3.34482343e-6 | 5.32345155e-7 |
| 2 | 9.10730989e-7 | 1.44947336e-7 |
| 3 | 1.43776970e-6 | 2.28828142e-7 |

Registered head/training/final keys are `0x4D072601 / 0x4D072602 / 0x4D072604`.
PPO uses 512 worlds, 64 controls, gamma .99, GAE .95, ratio clip .2,
entropy .002, value weight .5, four epochs/eight minibatches, initial logstd
−.5 and fresh Adam lr .001, beta .9/.999, epsilon 1e−5, norm clip .5.
The maximum was 64 updates per interface, not the old 128-update protocol.

## What regulation establishes

Every 5 ms sample in the final two seconds met the joint tolerances, not just
the 50 ms command endpoints. Worst final-window joint error across both
backends was **0.0027071 rad** (limit .02), worst rate **0.0275694 rad/s**
(limit .05). Final continuous joint settling began by 3.80 s in every case.
No actuator saturated. The two actual-swimming captures peaked at 1.49 and
3.63 µN·m respectively, under the unchanged 10 µN·m limit.

Those captured bodies still moved roughly 2–3 mm/s during the final window.
The joint test did not request learned translational braking, returning to a
destination or whole-body holding. Passing it establishes a usable low-level
joint regulator, not a successful movement motor.

Readiness used zero additional physical controls. Maximum absolute errors were
1.19e−7 for observations, 1.44e−9 for mean actual effort, 2.98e−8 for reward
and 1.14e−6 for CPU-double replayed raw log probabilities. Replay KL was zero;
209 sampled latent components were outside ±1, explicitly exercising clipping
without replacing the action used in the likelihood. Complete zero-update
snapshots, fresh Adam bytes and matched model initialization restored exactly.

## Why training stopped

One world, ID 247, first failed at 19.500 s of its trajectory: rollout 7,
zero-based control offset 5, physical substep 10. Its distal fin clearance was
**−0.19335 mm at the bed**. All numeric values remained finite. The unchanged
contact model's three axial sphere samples, using a 2 mm tail radius, still had
**+4.61863 mm** minimum clearance. The safety envelope includes the vertically
6 mm distal fin and its ends. Thus the contact proxy did not yet register the
same collision as the strict whole-fin test.

The recorded pre-breach actions are stochastic direct torques, not servo
commands. One component clipped to −10 µN·m three controls before the failure;
the failing control's applied commands were approximately
`[-2.856, 8.814, 3.547, 1.121] µN·m`. In the three preceding recorded states,
joint rates changed substantially (including a distal rate from +5.46 to
−6.85 to +5.55 rad/s). These observations describe this failed rollout; they
do not establish a universal exploration defect, insufficient control authority
or the inadequacy of the network architecture.

The failed world froze, and the remaining collection finished without reset
or rescue. Its 58 later sampled-action slots were **not physically applied**.
The whole rollout was then rejected before GAE/Adam. Unsafe rows do carry an
internal terminal flag, but no data from that rejected batch reaches learning.

The recorded actor Adam counter stayed 192→192. The latest full checkpoint is
byte-identical to the embedded pre-update snapshot; the exported six-update
actor is bit-identical to that snapshot's actor. Abort-time weights and optimizer
moments were not independently serialized, so do not claim an additional
abort-time parameter-hash comparison. The pre-GAE guard and saved counters
support the rejection; the exact preserved last-good/export bytes are separately
verified. No behavior learned from the failed rollout was promoted.

## Decision

**Do not retune the successful servo or infer a motor-interface winner.** The
paired learning comparison was censored by the direct-torque safety stop before
the second arm ran. Counters and improving training reward do not substitute
for the untouched behavioral evaluation.

The narrow next decision is whether to authorize a targeted correction that
makes native contact geometry account for the existing fin envelope before a
new preregistered comparison. That changes contact behavior and is outside this
run's unchanged-physics authorization; it has **not** been implemented. It would
not remove the separate need for learned boundary avoidance, braking and
whole-body holding. A joint-target-only study under unchanged physics is also
still untested; it must be explicitly authorized rather than silently resumed.
No broader rewrite, new behavioral network or weaker safety gate is justified.

The two-policy design is documented in
[CREATURE_CONTROLLER_INTERFACE.md](CREATURE_CONTROLLER_INTERFACE.md). Goal
selection, autonomous threat avoidance, exploration, energy/rest and later
growth remain staged behind motor acceptance. PPO critics remain training
auxiliaries, not extra behavioral controllers.

## Evidence and rollback

All new artifacts are under `build/paired-motor/2026-10-29-v1/`:

- `working-files/`, working-tree patch and `protected-before.json`: pre-edit
  source rollback plus personal/previous experiment fingerprints.
- `execution-manifest.json`, `sources/executed-*.exe`: source, binary and
  preregistration hashes for the executed run.
- `regulation/`: immutable protocol/catalog, 3,200 before/after frames, complete
  trace, parity summary and passed gate.
- `readiness-summary.json`, `readiness-gate.bin`: no-integration data checks.
- `run/torque/latest-training.bin` and `candidate-policy.bin`: actual update 6;
  `rejected-rollout.bin`: failed collection with original pre-update snapshot.
- `run/joint_target/`: update-zero preparation only; no trained candidate file.
- `failure-inspection/`: CPU-only decoding/geometry analysis; zero inference,
  replay, integration or optimizer steps.
- `gui-check-paused/`: inspected 1280×720, 1024×768 and 780×1000 renders.

Full-save SHA256: `8CCAACC4A82D0E7384C9A5033EB08BA3F45D2FE790D2566AC2898263C5E061F5`.
Actor-file SHA256: `A10BF7E7849A57A96801213F77D8E936A013D111F3BA073D3CD3EB393CB1C01C`.
Rejected-rollout SHA256: `DEE019BE87BA9624EA0F7BBF933CBB88611A6E1B86D299E182628A987AFF28D5`.

The final preservation audit found **347/347 protected artifacts unchanged**
and **15/15 executed-manifest hashes unchanged**. The shared physical source,
production launcher/pond, old motor protocol and personal save are unchanged
from the pre-turn snapshot. The only changed pre-existing files are project
memory/build integration and the isolated motor-bench source/tests.

The bench was opened paused on the saved partial torque actor. It performs
frozen live physical inference only when resumed; it does not train or replace
the pond's controller. GUI verification covered native loading, real checkpoint
metadata/units, unavailable-arm handling, deterministic matched references,
pause/resume/restart/controller changes and bounds at the three stated sizes.
Full keyboard/accessibility testing and smaller window sizes remain unverified;
these UI results are not a behavioral acceptance claim.
