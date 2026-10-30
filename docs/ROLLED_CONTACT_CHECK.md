# Original rolled-pose check: fixed registration

User approved one quarter-second check on CPU and CUDA, not another contact
comparison or training run. This check has its own shared ceiling of **6,000
attempted entries/trial advances**, including internal steps, event bisections,
failed attempts and both backends. No retries, tuning or budget reuse.

- Exact initial state: archived primary case5 in `tools/contact_probe_cases.h`,
  originally named `rolled_fin_floor_support`. This is the original90-degree
  rolled pose at rest, not the later cropped recontact state. Copy all23 FP32
  body fields without translation, normalization, projection or new capture.
- Fluid remains density990kg/m3, viscosity0.001Pa*s, gravity9.81m/s2. Current
  and all four actuator commands are zero. Body/inertia/joints/water forces,
  torque limits and the current event/midpoint implementation remain unchanged.
- One trajectory per backend:50 outer steps of5ms, total0.25s each. CPU first,
  then CUDA on a nonblocking stream. Maximum internal duration100us and24 event
  bisections are unchanged. The shared counter charges before each evaluation;
  an exclusive start-directory marker prevents a second invocation.
- Keep contact patches exactly320N/m and the existing FP32 damping coefficient
  promoted to double (1.9595917463302612N*s/m). Safety rejects any full-envelope
  gap below-0.0005m, escape or nonfinite state. Impulses must be nonnegative.
- Keep the current raw physical momentum certificate plus analytically derived
  half-ULP storage allowance. Scores must be<=1; no multiplier or relaxed gate.
  The existing two corruption checks per outer call remain zero-advance algebra
  (at most200 checks for the complete paired trajectories).
- Keep current actual-pose mechanical energy checks at every accepted internal
  segment and the common initial-energy ceiling: initial+max(1e-10J,
  0.001*abs(initial)). This fully submerged calm/no-actuator fixture includes
  gravity/buoyancy potential, physical kinetic, patch springs and joint stops.
- Match CPU/CUDA defined physical fields with2e-4+2e-3*abs(CPU), gaps and mean
  floor force with2e-6+2e-3*abs(CPU); safety/contact flags match exactly. Record
  outer states, minimum internal gap, energy maxima, certificates, counts and
  first rejected proposal. Any failure stops this sole run, with remaining work
  unrun. A device/integrity failure with unknown internal counts is inconclusive,
  not permission to retry.

`--prepare-rolled` writes the immutable numeric protocol without advancing a
state. Before `--rolled`, freeze its executable, source snapshot/hashes, numeric
protocol and this registration under a new ignored
`build/contact-rolled/2026-10-30-v1/`. The executable checks its numeric binding
before creating the exclusive start marker. Preserve every closed study and the
existing numerical repairs; source changes after freezing invalidate execution.

Success establishes only this bounded original-start regression. Sustained
support, bent/pitched cases, free-water equivalence, fresh conditions, learned
movement and runtime/save adoption remain unproved. No other fixture, PPO,
merge, deployment or follow-up is authorized.

## Sole run result: passed, closed

Frozen uncommitted source/executable/protocol hashes are in
`build/contact-rolled/2026-10-30-v1/source-manifest.json`; the92-file source copy
includes the already-validated numerical repairs. The strict CUDA tool build
passed without changing the physical solver. Numeric preparation advanced no
state. The sole `--rolled` invocation completed both50-step trajectories.

| Metric | Recorded result |
| --- | --- |
| Total charged entries/trial advances |5,448 /6,000 |
| CPU /CUDA charges |2,724 /2,724 |
| Outer calls /accepted internal segments |100 /5,012 |
| Events /charged event bisections |14 /336 |
| Corruption controls rejected |200 /200; zero physical advances |
| Maximum full-envelope penetration |31.8885um (limit500um) |
| Maximum raw /storage-bounded momentum score |.178467 /.817585 (limit1) |
| Maximum normalized CPU/CUDA comparison score |.00374249 (limit1) |
| Maximum energy increase above initial |1.94273e-11J, within frozen allowance |
| Retries /additional fixtures |0 /0 |

All executed states passed finite, geometry, nonadhesion, solver, energy,
momentum and matched-backend checks. No actuator command was supplied. Source
hashes were unchanged through execution. All350 files hashed in the four closed
study directories remained unchanged; original checkout remained clean.

This study is CLOSED. Its552 unused advances cannot be reused automatically.
The final interval's minimum clearance was positive (~0.573mm); this is not
evidence of settled bottom support. Longer support, bent/pitched cases,
free-water equivalence, fresh acceptance, a learned motor and runtime adoption
remain unproved. No tuning, retry, training, save migration or deployment ran.
