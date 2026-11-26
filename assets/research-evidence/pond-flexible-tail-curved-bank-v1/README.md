# Curved-path bank feedback: closed, not adopted

November 25, 2026. User authorized curved-path bank feedback, a separate GPU PPO
chooser session, frozen motor and conditional Godot adoption without mouse input.
The feedback implementation passes its numerical and preservation gates. The
trained chooser does **not** qualify for safe close-shore swimming. Godot stays
on appended-coverage-v3 chooser16 and rate-response motor88.

## Implementation

All 35 inherited observations remain unchanged. Four appended inputs describe
24-second predicted curved-path clearance, minimum predicted gap, measured COM
course rate and forecast validity. Actor/critic input columns and moments start
at zero; inherited distant-feature Adam ages survive checkpoint expansion, while
new curved-feature ages start at zero. Model6/schema5 and full session format3
bind the observation contract and serialize both age groups.

The forecast sweeps the current whole-body capsule envelope along a constant-
curvature path inferred from whole-beat COM velocities. Half-second midpoint
queries inflate the envelope to cover between-sample rigid motion. This is an
advisory kinematic forecast: it does not certify future tail deformation,
acceleration, changed requests or dynamics. Actual physical guards remain in
force. Optional query inflation defaults to zero in actual physics. No body,
water, actuator, torque, filter or integration equation changes.

For this opt-in study, the old straight-line 12-second reward cost is replaced
with up to -0.10 per second for predicted curved clearance loss within 24 seconds.
A +0.02/s term requires actual speed >=6mm/s, positive certified gap <=20mm and
a valid clear 24-second forecast. Existing visitation, transport, stall and
native guard costs remain. No prescribed route, action override or extra policy
is installed. The old straight-line sensory fields remain among the inherited
35 inputs; their learned weights are available for PPO to adjust.

## Executed budget and timing

One native Release C++/CUDA session: **1,024 batched worlds, 128 new updates,
1,048,576 macro transitions**, 1,024-row minibatches and four passes. It completed
4,096 new actor Adam calls and 4,096 critic calls. GPU rollout storage, MLP/GEMM,
PPO and Adam stay native. No Python ran. CPU evaluates independent native cases.

| Measured work | Time |
|---|---:|
| Main trainer, native wall | 144.868s |
| Main process, including launch | 145.157s |
| GPU physics/task collection | 134.828s |
| GPU observation/inference | 0.989s |
| GPU PPO/optimization | 1.656s |
| Evidence writing in trainer | 4.471s |
| Transfer/wait measurement | 135.927s |
| Sum of 17 fixed assessment process intervals | 27.026s |
| Sum of five close-shore assessment intervals | 6.775s |

Main throughput is 7,238 macro transitions/s. Transfer/wait overlaps GPU work and
is **not additive**. Evaluations overlap training; sums are not end-to-end job
time. Compilation is separate and recorded in configure/compile logs. One
three-update exact-resume diagnostic took 4.333s externally and collected 24,576
additional transitions. These are distinct from the main 128-update allowance.

`data/work-ledger.json` counts all physical attempts: 671,164,070 main,
15,731,566 resume diagnostic, 21,695,773 fixed evaluations and 4,352,900 close-shore
evaluations = **712,944,309 attempted / 712,943,989 accepted substeps**. Main
collection records 225 clearance guard stops. They are scored training failures,
not numerical faults or hidden live rescues. All attempts stay inside the finite
[protocol](data/source/tools/pond-flexible-tail-chooser/CURVED_BANK_PROTOCOL.md).
Zero new long-coverage or Godot parity physics ran because no contender qualified.

## Results

The close-shore suite uses eight moving starts at 15mm certified whole-body gap:
two phases, east/north banks and both tangential directions, each up to 180s.
A physical pass also requires meaningful swimming/exploration. Qualification
requires 8/8, at least four cases spending 15s at positive gap <=20mm and maximum
stall <=8.14s. Adoption additionally requires mean near time >=baseline+5s and
the unchanged four 30-minute individual coverage requirements. This certificate
is submerged-terrain clearance, not distance to the visible water lip.

| Checkpoint | Original short tests | Five-minute tests | Close-shore passes | Cases with >=15s near shore | Mean near time |
|---|---:|---:|---:|---:|---:|
| Preserved parent / expanded0 | 12/12 | 4/4 | 6/8 | 0/8 | 5.96s |
| 64 | 12/12 | 4/4 | 6/8 | 0/8 | 5.94s |
| 72 | 12/12 | 4/4 | 6/8 | 0/8 | 5.94s |
| 80 | 12/12 | 4/4 | 6/8 | 0/8 | 5.92s |
| Final128 | 2/12 | Not eligible | Not eligible | — | — |

Every scheduled eight-update fixed check is in `data/comparison.csv`, including
all regressions. Close-shore testing ran only for fixed-qualified checkpoints;
the parent suite and expanded0 suite verify the initial comparison separately.
All five suites fail the same two starts: phase8/north/negative tangent at
4.33375s and phase9/north/positive tangent at 3.88875s. A lost clearance
certificate is a guard failure, not proof of actual terrain contact. No reset
occurs within an evaluation. Six surviving cases are not proof of close-shore
mastery; none meets the 15s near-time threshold. Aggregate/union coverage is not
used as evidence of individual coverage improvement.

The motor is never optimized. Resting uses the unchanged saved controller;
**no fresh standalone rest physical suite** ran in this study. Old motor
retention evidence is preserved, while new chooser evaluations measure the
original navigation/from-rest fixtures and endurance. Do not call unchanged
weights a fresh measured rest result.

## Preservation and correctness evidence

- Full35-input parent checkpoint compaction/roundtrip is byte exact, including
  actor, critic, worlds, optimizer moments and histories.
- Initial 35-to-39 GPU actor/critic output differences are exactly zero.
- All35 inherited observations remain bit identical per backend. New curved
  CPU/CUDA fields differ by at most 1.013278961e-6 normalized against the declared
  1e-4 limit; inherited distant fields retain their existing 1e-6 gate. The old
  bank-input9 discrepancy is recorded separately, not silently repaired.
- Static geometry distinguishes one safe curved path from an unsafe straight
  extrapolation without changing the body. Three actual updates prove exact
  full-state resume, including both feature-age groups.
- Before training, all16 original evaluation traces and the complete case table
  match the installed parent's previous archive **byte for byte**. The two
  baseline close-shore case tables also match. Comparing saved files executes
  zero extra physics or optimizer work (`data/unchanged-reference.json`).
- Final package verification passes all11 bound files. Motor88 SHA256 remains
  `01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E`.
  No Godot file, DLL or active model changes; no mouse/keyboard takeover occurred.

## Evidence and stopping point

`data/source` is the exact executed source snapshot. Bindings, all129 portable
actors, all17 fixed evaluations, all five close-shore suites, curves, receipts,
three-update diagnostic and full checkpoints0/64/72/80/128 are archived.
`data/full-local-checkpoint-inventory.json` binds all129 full checkpoints retained
in the ignored local run directory; it does not pretend they are all committed.
`data/manifest.json` hashes the archived files. Originals remain untouched at
`build/flexible-tail/curved-bank-v1`.

**Conclusion: no qualified close-bank improvement; no Godot adoption.** The next
focused proposal is to capture the two failing close-shore states and verify
whether safe speed/turn sequences exist, then use those verified starts in a
targeted chooser curriculum. Existing training did not add those starts. Lack of
exposure is a hypothesis, not a proven cause. Do not expand this closed session
or launch another experiment implicitly.
