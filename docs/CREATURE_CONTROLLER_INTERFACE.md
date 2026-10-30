# Controller request and ownership

There are two intended learned behavioral policies: a goal chooser and one reusable movement motor. PPO critics estimate value during training; they are not additional controllers. The minimal foundation currently runs neither behavioral policy.

A future chooser owns goal identity, intention, validity and commitment. It issues a persistent request containing:

- Destination in the shared world/model coordinate system, with explicit conversion of1unit=4cm.
- Desired terminal velocity in SI m/s, not throttle.
- Desired unit heading in world coordinates.

The motor keeps executing that request through arrival, braking and holding until the chooser explicitly replaces or completes it. A hold request latches destination/heading and zero terminal velocity. Nonzero terminal velocity is an interface provision, not a demonstrated skill. Food, shelter, explore and hold must not all be disguised as pellets.

## Active foundation ownership

| Module | Owns |
| --- | --- |
| `physics/dynamics.h` | Body geometry/mass, articulated dynamics, buoyancy, reduced-order water drag and numerical integration |
| `physics/contact.h` | The same complete body/fin envelope for inspection and explicit contact forces; unresolved support gate remains visible |
| `simulation/PhysicalWorld` | Physical state, water current, actuator interface/held targets, applied torque/work and diagnostic state serialization |
| `CreatureLab` | One world instance and Godot value conversion; no trainer, rewards, hidden stabilization or personal learner saves |
| `godot-project/main.gd` | Diagnostic input, run/pause/single-step, camera, rendering and feedback |
| Reusable numerical primitives | FP32 MLP, backpropagation, Adam, Gaussian action math and PPO/GAE kernels; no active curriculum or whole learner |

The current body has two universal joints with two controlled axes each. Diagnostic direct-torque commands are bounded and converted to ±10µN·m. Joint-target mode requests physical angles; the deterministic servo computes limited internal torques every5ms. It supplies no additional whole-body forces or kinematic angle overwrite.

A future PPO policy must store and replay its sampled neural actions consistently: joint targets are not the resulting torques. Preserve transformation derivatives, probability semantics and exploration state. Effort is measured from actual applied torque/work. Previous-action observations must state whether they represent torque or targets; checkpoint all held targets, physical/random/optimizer state that affects continuation.

The new diagnostic save format is explicitly separate from old learner formats. Reading an incompatible file must fail without resetting or overwriting it. Personal pond saves are not migrated by this foundation.
