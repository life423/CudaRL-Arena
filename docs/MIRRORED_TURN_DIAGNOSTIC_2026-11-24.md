# Mirrored turn diagnostic, 24 November 2026

## Frozen protocol

Run once with preserved motor 336 and its gain, SHA-256
`4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d`.
Use the native product pond, calm water, unchanged 5 ms physics and 50 ms
controls. This diagnostic has zero optimizer steps. Start pose 0 once and
save straight-swimming snapshots after ticks 40, 44, 48 and 52. Each phase
gets both signs of 10° and 15°. Rotate the persistent heading and velocity
with the existing heading-only adapter. For delayed phases, issue the request
at its snapshot by the identical rotation with no clock, servo or body reset.

Use lateral reflection about z=0. Reflect position and linear velocity with
signs (+,+,−), the angular velocity and quaternion vector with (−,−,+).
The native joint order is yaw, pitch, yaw, pitch. Reflect angles, rates,
filtered targets, target-rate history, raw actions and previous servo torques
with (−,+,−,+). Preserve all scalar physical and accounting state. Re-query
the terrain for the reflected body. The physical comparison applies exactly
reflected commands to the reflected swimmer. A third swimmer independently
runs the same frozen network from that reflected snapshot and mirrored request.

Run 360 controls per branch (18 seconds following issuance). Record maximum
reflection error per quantity at every 5 ms substep, and its first breach.
Use the existing native replay allowance of 2e−6 absolute per scalar, in its
native representation, for position, quaternion, joint angles, all ten
velocities, filtered targets and rates, applied normalized and requested SI
torques, work/travel and whole-body COM velocity. Require complete finite
trajectories, strictly positive certified clearance and zero contact spring
work, damping loss and potential energy on the paired branches. The coupled
water/servo/contact solver runs even in free water. Report terrain clearance independently
from motion. The winding pond bank itself is not mirror-symmetric.
Replay the first 20 controls of each matched pair on CUDA with identical
commands, production physics compiler settings and the same reflection check.

Score the independent actor in each direction using the existing final
four-second mean COM velocity, ≤5° course error, ≥6 mm/s projected speed,
≤1 mm depth error and ≤10° tilt. Record the actual angle relative to the
unrotated request, first one-second continuous qualifying window, and front
and rear yaw commands, filtered targets and applied torques at request onset.
These paired starts supplement the original 16-case assessment. No fixed
policy symmetry or attainment of target counts as a physical-gate requirement.

Budget 52 warmup + 16 × 360 × 3 CPU controls and 16 × 20 × 2 CUDA controls,
17,972 total and 179,720 substeps. Execution has a 60-second cap. Stop on invalid physics or nonfinite data.
A reflection gate failure blocks the proposed learning change. Save measured
execution time, retained failure evidence, and every attempted case.
