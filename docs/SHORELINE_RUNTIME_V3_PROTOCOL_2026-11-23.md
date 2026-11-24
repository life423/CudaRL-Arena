# Shoreline runtime v3 correction check — November23,2026

Preparation, zero v3 dynamic execution. The user's latest instruction explicitly
prioritizes finishing validation and starting bounded candidate PPO under their
existing authorization. V1 and v2 remain CLOSED with their source, binaries,
counts and failed criteria intact. V2 retained all20reference cases and passed
8bank impacts, then missed the same0.25mm half-step position gate at0.250733749mm
(control153); it executed62553controls/627060substeps in166.045503s, zero CUDA
dynamics/optimizer and zero outward-swimming cases. Do not waive that failure.

## Correction: integrate actual contact duration

V2's delayed first-touch dashpot convention still leaves a timestep-dependent
impact impulse. V3 integrates the SAME spring potential and approach damper over
the frozen linearized gap path g1=g0+h r. It introduces no force constant,
geometry change, additional physical substep, pose correction or route.

For U(g)=0.5k min(g,0)², use average elastic multiplier
Fe=−(U(g1)−U(g0))/(g1−g0), with its continuous limit at zero displacement.
This makes h Fe r=−ΔU exactly along the linear gap path. Multiply the approaching
dashpot by the actual fraction alpha of that path inside compression. At first
touch alpha starts continuously at zero; there is no full-step early activation
or old-gap delay. Elastic multipliers remain compressive. Physical damping loss
is h alpha c r², with no implicit elastic tangent counted as dissipation.

Solve this continuous monotone contact law with double-precision consistent
linearization: at most32Newton systems and8backtracks per physical step.
Require the TRUE nonlinear force/momentum residual≤the existing1e−8, not merely
a solved linearization. Fail on nonfinite, line-search or convergence failure.
No-contact returns the exact original native water/servo result. All physical
joint/torque/filter/rate/acceleration limits,5ms cadence,0.5mm overlap,10µm
geometry uncertainty,131072query cap and v2total-energy checks remain unchanged.

Pure native C++ preparation independently tests450scalar velocity fixtures,
closed-form/no-contact references, true momentum residual and spring-work/
potential identity, with no world integration. Preserve all failed pure attempts.

## Fresh finite phases, unchanged criteria

Use every case, criterion, accounting rule and adoption condition in
`SHORELINE_RUNTIME_V2_PROTOCOL_2026-11-23.md`, under NEW v3 identities/outputs:

| Phase | Controls cap | Attempted substeps cap | Native wall cap |
| --- | --- | --- | --- |
| CPU36-case retention/bank/half-step/outward |68000|688000|900s|
| CPU/CUDA8matched128-control cases |2048total|20480total|180s|
| Combined |70048|708480|1080s|

Evidence≤2GiB, optimizer0. Same half-step limits: COM position0.25mm, velocity
0.25mm/s, orientation0.5degree. Same original28long-window keys and all14holds.
Four outward cases run only after prerequisites pass; count every skipped case
if they do not. Preserve continuous body/filter state and all first failures.
No unrecorded pilot, criterion change, automatic retry or budget reuse.

The actual original CUDA tolerances and fixed-body radius profile remain exactly
as specified in v2. Diagnostic CUDA compilation may use `--Ofast-compile=max`
to avoid the observed multi-minute compiler optimization cost, while retaining
`--fmad=false` and precise arithmetic. This disables compiler optimization passes;
it is NOT fast math or measured GPU throughput. Generic optimized native MLP/PPO
libraries are unchanged. Profile/flags are bound before any CUDA dynamics.

Seal each phase's executable/source/inputs after its static checks. CPU may run
while the independent CUDA diagnostic compiles, with shared physics source
frozen. No CUDA dynamics until the CPU result passes. No PPO collection or
optimizer until both necessary physics phases pass. A separate candidate may
then learn turning under the explicitly stated32-update checkpoint/evaluation
budget; successful turning is NOT an admission requirement for training it.
Native C++/CUDA remains the simulation, inference and PPO architecture.
