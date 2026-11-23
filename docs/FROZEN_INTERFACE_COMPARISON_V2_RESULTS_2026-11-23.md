# Frozen interface comparison v2: blocked before physics

**Conclusion: comparison remains blocked.** The corrected fixed-state
arithmetic passes, but the single authorized launch failed while enumerating
the frozen evidence tree, before simulation. Neither interface was tested.
No motor/chooser training, optimizer update or runtime adoption occurred.

| Measurement | Original lane following | Depth-preserving heading only |
| --- | --- | --- |
| Interface cases attempted | 0/44 | 0/44 |
| Repeated/return turns | Not run | Not run |
| COM speed/course, depth/tilt | Not measured | Not measured |
| Turn radius/response delay/whole-body clearance | Not measured | Not measured |
| Interface qualification | Not established | Not established |

The fresh matched-command prerequisite ran0/4 paired cases. Corrected-runtime
retention ran0/25 CPU and0/25 CUDA fixtures. Consequently retained skills and
behavioral regressions are **unassessed**, not passed. Original verified skill
evidence remains historical evidence for its original runtime. A fixed-state
algebra pass does not establish that the old control107 divergence is repaired
in motion, nor establish autonomous pond navigation.

## What the numerical trace established

Identical saved PRE states and commands from all107 controls of both failed
v1 backends were evaluated at densities975/1000/1025:642 rows per profile.
There was no body integration. Ordered traces cover reference filtering,
link kinematics, buoyancy, COM, mass/resistance/bias/force assembly, discrete
operators, servo demands, solves, torques and prospective velocities.

Implicit CUDA contraction changes the neutral buoyancy subtraction: the
original CPU has exactly zero neutral force while fused CUDA leaves a small
residual. The reference-filter trace also differs, beginning at saved
control2. This is not the whole explanation: even with contraction disabled,
the original PREcontrol1 resistance matrix has six out-of-limit comparisons
across the two sources and three densities.

The supplemental primitive trace locates that remaining difference at cbrtf,
before viscosity coefficients and resistance accumulation. Identical head
radius argument5.1200004236307e-7 produces CPU0.007999999448657036m versus
CUDA0.00800000037997961m. The middle-link result also differs by one float.
Recorded coefficient products and ordered additions reconstruct the earlier
R[2,0]/R[0,2] values exactly on all642 rows. This identifies two arithmetic
sources and their propagation; it does not claim a unique full-trajectory
causal decomposition from fixed-state traces.

The isolated v2 profile uses --fmad=false and GPU lookup of the three exact
original CPU cbrtf bit results for the unchanged fixed geometry. Host checks
recompute/validate every argument/result; unknown GPU arguments are rejected
by the existing finite/solver gates. The original CPU calculation remains
byte-identical across all642 rows. Physical equations, geometry, torque caps,
filter/servo rates, timestep and all acceptance tolerances remain unchanged.
The nonneutral density forces remain nonzero with the correct signs.

The final409596 scalar comparisons pass their fixed unit-specific limits;
filter results and neutral buoyancy agree exactly. Independent algebra,
COM, operator and solve checks pass. The no-FMA-only failed preparation and
its raw traces, failed reader/proof and available causal snapshots remain
preserved. Three early build-description files changed when the supplemental
target was added before their snapshot; their original digests are retained,
but those three earlier byte snapshots are unavailable. The corresponding
original executables, trace source and raw outputs are preserved. The final
causal freeze is complete and independently verified.

## Why no physical case ran

The one-shot native launch exited2 after0.096864 seconds with
`recursive_directory_iterator::operator++: The system cannot find the path specified.`
It failed before creating probe-started, loading the NN, launching a kernel
or advancing physics. No native result.json or physical trajectory was created;
the separate launch-stop proof must not be mistaken for one.

I nested saved audit snapshots inside the new evidence snapshot tree without
checking the final tree with the native file enumerator. Paths reach356
characters;33 files exceed the traditional Windows limit. The executable
manifest lacks longPathAware even though LongPathsEnabled is1. This evidence
supports a path-length failure during enumeration; stderr does not identify
the exact offending entry. Windows requires the application opt-in as well
as the system setting. [Microsoft path-length documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).

The failed launch is preserved and the one-process allocation is closed.
No shortening, tolerance change, retry or unused-budget transfer followed.

| Actual work | Count |
| --- | ---: |
| Fixed-input algebra/drag evaluations per backend | 2568 |
| Explicit CUDA kernels for those four evaluations | 4 |
| Zero-integration fixture-preparation explicit CUDA kernels | 531 |
| Preparation NN forward calls, separate from unmeasured library kernels | 3 |
| Canonical host geometry validations across corrected/preparation/launch processes | 3 |
| Physical launches attempted | 1 |
| Fresh prerequisite physical controls/substeps | 0/0 |
| Retention physical controls/substeps | 0/0 |
| Interface physical controls/substeps | 0/0 |
| Motor/chooser PPO updates and optimizer steps | 0 |

## Bindings and next step

Run: build/ppo/frozen-interface-comparison-2026-11-23-v2. The protocol remains
frozen at its original finite caps; this report does not reopen them.

- Arithmetic verification SHA256:9b37ac0b4e45e258245964a0e32c0d6f6f9e533231324a2c1d09d185632ad052.
- Full197-file causal freeze SHA256:262fee70793710c1c591f5af7fd83ea6b3ed9188821bdf5f8dbf6f90c75f5ddf.
- Preparation proof SHA256:1e8fe71e03f9f3d5f250c72822f9c69a27d82b322f5edfc9e10250de1b069547.
- Main executable SHA256:732de230f539438567ba6168144ff4ecaf0d70f3ce79899225dafc28a9c00cae.
- Independent launch-stop proof SHA256:aa45fa23cd4a1f02483269624c42301cb3d636b24ab75749a53d2664df9fa742.
- Saved checkpoint SHA256:810dbd593ce39475cdd6e95b709bef3fdb34d50c9fcbbb2c345c7a8882a0bef0.

All10457 protected originals (63006197911bytes) passed the post-run SHA256
check. The receipt binds the original inventory, arithmetic verification,
actual process receipt and independent launch-stop proof. Original v1 sources,
failed data and saved policies remain unchanged.

Next: prepare a long-path-safe evidence layout/executable and a zero-physics
native inventory preflight over the final sealed tree, then obtain fresh
authorization for the unchanged gate/retention/88-case attempt. Do not train
either policy. Godot must show this launch stop,0/88 and no winner, while
preserving the existing saved creature.
