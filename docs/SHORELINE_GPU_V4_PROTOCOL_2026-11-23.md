# GPU qualification for the corrected bank-clearance calculation

The physical gap is body height minus terrain height. Runtime 4 uses the
gradient of this gap for the support point, the Taylor bound and the downhill
search direction. The contact solver already used the upward terrain normal.
The old runtime's fixed-state comparison and timing results are preserved in
`build/pw29`. They do not qualify the corrected search.

The CPU qualification for this version is specified in
[the runtime 4 protocol](SHORELINE_RUNTIME_V4_PROTOCOL_2026-11-23.md).
The CUDA checker, the candidate motor and the CPU reference use the same
scalar world step. The checker schedules the eight independent worlds in
separate blocks. Inside one control, an accepted post-step geometry query
also supplies the identical pre-step geometry for the next substep. The first
substep still checks the control's incoming state. Every new candidate body
is inspected. Body dimensions, water forces, joints, servos and torque limits
remain those of the saved motor. The CUDA translation unit targets sm_120 in
Release with implicit FMA disabled for shared physics.

## Allocation and stopping criteria

Before dynamics, run the fixed-fixture algebra comparison and two timed
eight-body clearance queries. Those three kernels advance zero physical
controls. Seal the shared sources, executable, original policy, this protocol,
static result and completed CPU qualification before the first dynamic step.

One dynamic attempt compares the four straight saved-motor runs and four
bank-impact runs under the original fixture construction. Each backend must
complete 128 continued controls per case, each with ten 5 ms substeps.
Maximum work is 1,024 CPU and 1,024 CUDA world-controls, 20,480 substeps total,
180 seconds and 512 MiB of evidence. A kernel already in flight can finish
after the wall deadline and its work must be recorded. No optimizer is linked
to the checker.

Use identical commands and initially identical states. Each backend carries
its own body, servo and filter state after the initial copy. Compare every
substep under the fixed tolerances (2e−5 for positions in model units,
quaternion components, angles, angular rates and normalized motor values,
2e−7 m/s for linear velocity and 1e−12 J plus 1e−4 relative for energy).
Preserve complete raw traces, all eight case outcomes and the first numerical
or physical failure. Stop on that failure or exhaustion of the finite budget.
Report the CUDA physics event time separately from host reference, synchronous
transfer and evidence-writing time. These timings describe validation.

Only a completed CPU and CUDA pass bound to runtime 4 and query arithmetic 2
admits the [food-free PPO motor lesson](POND_TURNING_PPO_PROTOCOL_2026-11-23.md).
Its 32-update maximum, eight-update retention gates and preserved parent remain
unchanged. Even a successful turning lesson must be evaluated separately for
continuous autonomous navigation.
