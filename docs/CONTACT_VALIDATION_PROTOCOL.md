# Fixed-A contact validation registration

Protocol `contact-A-validation-v1`, schema 1. This is a new, bounded validation;
the earlier 2,024-call pilot is closed and cannot donate unused calls. This
protocol does not authorize tuning, another contact law, PPO, live-runtime
replacement, a merge or deployment. Contact B and all previous evidence remain
preserved. Until this validation finishes, A remains experimental.

The numerical fixture registration is generated from
`tools/contact_validation_cases.h`. The one-shot harness writes complete numeric
`protocol.json` and checksummed `protocol.bin` before execution. Frozen source
hashes, executable and preservation manifests accompany that registration.
The binary binding and JSON must match the executable before `--run`; an existing
run marker or results cannot be overwritten. No changes between stages, retries,
budget extensions or automatic adoption of a finer timestep are permitted.

## Unchanged law and independent reference

Validate only the existing version-4 compliant law A from
`src/physics/contact_pilot.h`: each analytic patch uses FP32 `k=320 N/m` and
`c=1.9595918 N·s/m`, promoted consistently where double arithmetic is required.
Keep the body, inertia, buoyancy, water drag, actuator limits, normal impulses,
operator split, solver settings and energy rules unchanged. No parameter search
or further calibration occurs. The live version-2 contact implementation and
personal save contracts remain untouched.

The independent scalar oracle solves continuous free flight and unilateral
Kelvin–Voigt contact analytically in double precision. It does not call the
scalar stepping implementation or advance a second numerical reference world.
It releases contact when compressive force reaches zero during unloading;
release need not coincide with zero geometric compression. Its continuous
critical unilateral rebound is therefore distinct from a bilateral spring
integrated until geometric gap zero. Event times and force histories are
reported descriptively; they do not introduce unregistered acceptance gates.

The new scalar fixtures and oracle both use exact double inputs: mass
`0.003 kg`, inward support load `0.0003 N`, impact gap `25 µm` and impact speed
`0.05 m/s`. This explicitly supersedes the pilot fixture's FP32 storage of those
scalar inputs for this new validation. It is not a migration or rewrite of the
old pilot data. The fixed A coefficients retain their original FP32 values.

## Exact physical budget

Every attempted CPU substep or CUDA kernel substep is charged before dispatch,
including failed steps and rejected proposals. Allocation or upload that fails
before a launch consumes no physical advance; a rejected launch can retain a
conservative charge. Committed states, proposed failing states and charged
attempts are reported separately. All cases start from their complete fixed
initial state, never from another backend's or timestep's output.

| Stage | Fixed work | Maximum advances |
| --- | --- | ---: |
| Scalar accuracy | 3 cases × (50 at 5 ms + 100 at 2.5 ms + 200 at 1.25 ms) × CPU/CUDA | 2,100 |
| Rotating accuracy | 1 saved state × (50 at 5 ms + 100 at 2.5 ms + 200 at 1.25 ms + 400 at 0.625 ms) × CPU/CUDA | 1,500 |
| Original support | 3 starts × 1,000 steps at 5 ms × CPU/CUDA | 6,000 |
| Free-water preservation | 2 cases × 50 steps at 5 ms × CPU/CUDA × A/legacy | 400 |
| Passive energy | 2 cases × 50 steps at 5 ms × CPU/CUDA | 200 |
| Fresh withheld support | 4 starts × 1,000 steps at 5 ms × CPU/CUDA | 8,000 |
| **Total ceiling** | No B calls or unallocated reserve | **18,200** |

Scalar cases are loaded support, inward impact and separating contact. The
rotating start is the actual saved legacy CPU roll post-step 33/pre-step 34,
which already has positive whole-body clearance; it is not translated or
projected. Original support starts are the archived bent, pitched and rolled
fixtures 3, 4 and 5. The free-water and passive fixtures retain the corresponding
two pilot conditions; the legacy contact law is called only for free-water
preservation, never as an extra uncounted trajectory.

All scalar and rotating accuracy resolutions are fixed before execution. A
finite fidelity-tolerance failure still completes this accuracy ladder for
diagnosis. It does not authorize a retry, altered coefficient, new resolution,
longer duration or adoption of a finer step. Any actual safety, energy,
solver, stream, nonfinite or integrity failure stops all further A physical
calls immediately. Any failed 5 ms accuracy prerequisite blocks every longer
support, free-water, passive and withheld stage. Skipped work remains unspent.

## Scalar accuracy gates

Compare each sampled numerical gap and speed at the same physical time with
the independent continuous solution. Compare maximum compression, rebound and
cumulative normal impulse over the registered quarter second with its report.
These are physical error limits fixed before this run, not fits to the observed
coarse-step errors.

| Quantity | Maximum error |
| --- | --- |
| Maximum compression | `max(2 µm, 5% of continuous maximum compression)` |
| Rebound coefficient | Absolute `0.02` |
| Cumulative normal impulse | `max(1e−8 N·s, 5% of continuous impulse)` |
| Time-aligned gap | `10 µm` |
| Time-aligned normal speed | `1 mm/s` |

All three 5 ms cases on both backends must meet these limits. Finer scalar
results diagnose resolution; they cannot substitute for a failed 5 ms result.

## Rotating accuracy and finite-resolution check

Compare complete trajectories at common sample times. The finest 0.625 ms
trajectory is a numerical comparison, not exact physical truth. Both the 5 ms
and 1.25 ms trajectories must agree with it within the following same limits on
each backend. Report all intermediate trends; no forced convergence factor is
required and a single fine comparison is not proof of asymptotic convergence.

| Quantity | Maximum difference |
| --- | --- |
| Head position, physical support endpoints and floor gap | `0.1 mm` |
| Body orientation | Geodesic angle `0.02 rad` |
| Each joint angle | `0.02 rad` |
| Head linear velocity | `2 mm/s` |
| Body angular and joint-rate components | `0.5 rad/s` |
| Time-aligned cumulative floor-normal impulse | `max(1e−8 N·s, 5% of finest final impulse)` |

Also report maximum penetration, total impulse, energy differences and all
proposed failing states. Do not quietly call a smaller penetration or a
converged algebraic solve successful physical contact.

## Five-second support and withheld gates

Original and withheld support conditions are calm, density `990 kg/m³`, zero
actuator commands and unchanged gravity/viscosity. Observe every substep, then
evaluate the final **one second** of each five-second trajectory:

- Maximum speed of every link's center of mass: `0.5 mm/s`.
- Maximum angular speed of every link and absolute relative joint rate:
  `0.05 rad/s`.
- Actual vertical total-body momentum balance:
  `|ΔP_COM − (I_contact + I_gravity/buoyancy + I_drag)|`
  must be at most `1e−8 N·s + 5% |I_gravity/buoyancy|`.
- Sustained support balance:
  `|I_contact + I_gravity/buoyancy|`
  must be at most `1e−8 N·s + 10% |I_gravity/buoyancy|`.

Use actual normal impulses and the actual applied free-stage drag response;
spring-only force is not the net reaction. Full states must remain finite,
normal forces must remain nonadhesive, and every complete body/fin clearance
must stay within the unchanged `0.5 mm` overlap limit throughout. No runtime
projection, reset, prescribed stillness or rescue may supply a passing result.

The four withheld starts are explicit new yaw/pitch/roll and joint-angle
combinations, not neighboring frames from an observed trial. They have zero
initial velocities, no current and zero commands. The header fixes their full
numeric registration before any trajectory is evaluated. Each starts at
`+20 µm` minimum floor clearance using exactly one native envelope query during
fixture construction. This is initial placement, not a runtime correction or
feedback from measured performance. They remain untouched as behavior tests
until the final stage and are not used for tuning after inspection.

## Energy, parity and accounting

Keep the prior physical-energy allowance `max(1e−10 J, 0.001 |E_initial|)`.
Scalar stored energy includes its actual load potential. Whole-body passive or
calm loaded energy includes kinetic energy, the actual A patch springs, joint
soft-stop potential, gravity and buoyancy potential. Physical-M contact kinetic
change/contact work, actual external work, split free-stage work and imposed
current work must retain their distinct meanings. Current-driven free-water
behavior is not presented as a closed passive system.

Retain CPU/CUDA state parity `2e−4 + 0.002 |reference|`, gap/net-force parity
`2e−6 + 0.002 |reference|`, exact safety flags and signed normal-force allowance
`−1e−9 N`. Require actual CUDA execution on an owned nonblocking stream, ordered
uploads/consumers/downloads and serial/input integrity checks. Each backend
evolves from its own previous state. Free-water A/legacy preservation is checked
bitwise within each backend. Record every attempted advance and first failure;
if a candidate rejects a solved proposal, retain that proposal separately from
the unchanged caller state.

Static independent preflight remains capped at **256 geometry queries,
256 kinetic evaluations and 96 matrix/LCP queries**. Continuous-reference queries
are capped at **12,000** and counted separately from physical advances. Fresh
fixture placement is cached once per process: four native geometry queries in
prepare plus four in run, at most eight, included in the static ledger. Repeated
calls to list or serialize those cached fixtures do not place them again.
Post-step observables use already computed trajectory states and do not create
additional reference trajectories or physical advances.

## Decision boundary

A passing result validates only the registered contact-A behavior and its
specified 5 ms resolution within these conditions. It does not establish an
optimal contact law, all hydrodynamics, ten-second unsupported holding, a
learned movement controller or autonomous danger avoidance. A failed accuracy
or support gate produces a bounded diagnosis and the narrow next decision;
it does not trigger another run or automatic tuning. Existing pond behavior,
saved controllers, personal saves and historical artifacts remain preserved.
