# Frozen requested-course probe results — November 23, 2026

Status: the one authorized native run is closed and its independent saved-data
audit passed. All 5,782 controls, 57,800 accepted 5 ms states, four prerequisite
fixtures and 22 probe cases were audited. No policy was trained, selected or
adopted. The held continuous-pond V2 study was not executed.

## What the existing motor can and cannot do

The preserved motor responds to small persistent course changes. It does not
yet execute large turns while maintaining competent swimming. This is narrower
than saying that it cannot turn at all, or that pointing its head establishes
a successful turn.

Both exact zero-offset baselines complete 16 s with final-4-s COM velocity
7.92–7.94 mm/s, course error at most 0.61 degrees, depth error below 0.44 mm and
head tilt below 9.36 degrees. The zero-offset request and lane origin were
unchanged. The four ±5-degree cases retain 7.75–7.83 mm/s and final course
error of 2.93–4.21 degrees. Their lane excursions are not uniformly within the
old 2 mm straight-swimming standard: the positive-offset cases reach
2.72–2.98 mm.

Larger offsets expose a coupled movement problem. The ±15-degree cases cover
the requested overall direction but slow and keep changing course. At
±30 degrees, eventual velocity-course alignment coexists with roughly one
quarter of the original speed and excessive depth/posture/lane error. All
±60/±90-degree cases stop at the registered 20-degree tilt guard. None stops
at a bank or the 5 mm depth guard, so shoreline contact does not explain this
probe's failures.

| Requested offset | Cases reaching 16 s | Final-4-s COM speed | Final-4-s velocity-course error | Observed limitation |
| --- | ---: | ---: | ---: | --- |
| 0 degrees | 2/2 | 7.92–7.94 mm/s | 0.01–0.60 degrees | Stable reference; head still oscillates with gait. |
| ±5 degrees | 4/4 | 7.75–7.83 mm/s | 2.93–4.21 degrees absolute | Positive offsets exceed 2 mm lane reference. |
| ±15 degrees | 4/4 | 5.97–6.54 mm/s | 9.64–10.69 degrees absolute | Lane reaches 5.43–7.03 mm; positive offsets reach 10.53 degrees tilt. |
| ±30 degrees | 4/4 | 2.20–2.34 mm/s | 4.62–6.90 degrees absolute | Depth reaches 1.19–2.11 mm, tilt 11.99–17.47 degrees, lane 14.22–15.21 mm. |
| +60 degrees | 0/2 | Not available | Not available | Tilt termination at 10.075/10.205 s total, 6.075/6.205 s after request. |
| -60 degrees | 0/2 | Not available | Not available | Tilt termination at 5.050/5.000 s total, 1.050/1.000 s after request. |
| ±90 degrees | 0/4 | Not available | Not available | Tilt termination 0.385–0.945 s after request. |

The two +60-degree cases illustrate why head direction is insufficient: each
briefly has 20 consecutive control endpoints with head error at most
10 degrees, yet its mean COM velocity course over the observed turn remains
about 49–50 degrees away from the requested course. Conversely, the unchanged
straight baselines have roughly 10-degree head-error RMS from normal gait,
while their COM velocity course remains accurate. The report keeps head,
velocity and displacement measurements separate.

At ±15 degrees the accumulated turn displacement error is under one degree,
but late velocity-course error is roughly ten degrees. An episode-average
direction score alone would hide this sustained-tracking defect. Early guard
terminations have null final-4-s fields; a short prefix is not a completed turn.

## Exact scope and accounting

The [frozen protocol](FROZEN_REQUEST_TURN_PROBE_2026-11-23.md) used original
poses 0 and 3 at 0, ±5, ±15, ±30, ±60 and ±90 degrees. Each case began with
80 real controls (4 s) under the original persistent course. At control 81,
nonzero requests rotated that original course once and rebased only the lane
origin's horizontal coordinates. Original depth, world/body state, clocks,
filter state and all weights continued without reset. Requests remained fixed
after the switch. Positive offsets use the simulator's +Y convention (+X
toward -Z).

- Four matched-command CPU/CUDA prerequisite fixtures passed: 320 controls.
- All 22 probe cases were attempted: 5,462 controls; 14 reached their full
  horizon and eight ended at the ordinary tilt guard.
- Total: **5,782 control attempts, 57,800 attempted and accepted 5 ms substeps**.
  Terminal partial controls account for 20 fewer substeps than full controls.
- Whole-process watchdog receipt: **13.7060028 s**, below 120 s. Native close
  report records 13.6170838 s through its final accounting point.
- Native recorded 53,322,276 evidence bytes before its close report, below the
  registered 128 MiB limit including frozen inputs.
- Maximum matched-command position difference: 1.66893e-6 model units, under
  the unchanged 2e-5 tolerance. Maximum same-observation inference difference:
  5.96046e-7, under 2e-5. Other registered pose, velocity, motor and energy
  comparisons passed natively and were independently confirmed from saved data.
- **Zero PPO updates, zero optimizer steps, zero hard physical faults, no
  runtime adoption.**

This is finite 16-second requested-course evidence, not continuous pond
swimming, bank avoidance, a 30-minute validation, or a new learned skill.
The original straight and holding policies are unchanged. Prior rejected
studies and their evidence remain preserved.

## Narrow consequence for the next PPO design

The next learning problem is sustained turning with propulsion and depth/
posture stability, together with deciding to turn early enough before a bank.
It is not merely minimizing instantaneous head-heading error. The learner
needs credit for actual COM travel along the requested course, useful speed,
bounded depth/tilt/lane error and continued survival. A heading-only reward can
prefer a nearly stopped or badly leaning creature, as the ±30/+60 results show.

Preserve the useful requested-course and lane/depth feedback rather than
replacing it with the current head direction. Use the existing small-angle
response as a retained starting capability. Any subsequent PPO protocol must
address the measured angle-dependent loss of speed and posture, and must avoid
the cheap-failure objective identified in the held V2 design audit. This report
does not authorize another probe, a training run, a larger controller or an
automatic policy replacement.

Small incremental course requests are a plausible direction because one
5-degree request works substantially better than one 30–90-degree jump. This
probe did not test repeated small requests, choosing when to issue them, or
rounding a bank. It does not establish that a sequence of small turns will
remain stable, and it supplies no scripted navigation or automatic request
sequence to the app.

## Evidence and verification

Local run: `build/ppo/frozen-request-turn-2026-11-23-v1`.
Native result: `result.json`; per-case measurements: `probe-started/cases.csv`.
Full control/5 ms records: `probe-started/controls.csv`, `substeps.csv`,
`states.bin` and `same-observation.csv`. Process receipt: `process-execution.json`.

The 87-input causal freeze has manifest SHA-256
`75f2bef0bc09144b72abe7c5607b5cb002ddc9561358191eff769f794c178b10`.
Independent zero-control preparation proof SHA-256:
`285e1d5cbcc978c6fd090ea9033a85e07d61b0db1ba71150e48601dd91a251d1`.
Independent full-result proof SHA-256:
`06bf4c4dd44b597a9bdbae8f93e3708b5cf25e8ecc3bf7745973e2ea5ef3acfb`.
Executed offline reader SHA-256:
`f62d771e8df630287268c4557dc898554f1f295ce04924684b5e1da24339489a`.

The first offline audit attempt stopped on a one-row NumPy batch-shape
`IndexError`; only that reader shape handling was corrected. The second
saved-data audit passed. No native arithmetic, tolerance, frozen source or
physical run was changed or repeated. Both reader versions and the failed
audit note were preserved.

Exact native result SHA-256:
`3be6d18ac4857f449946408dee7db521a88a54d5271a9ff14b8f6c85569972f2`.
Native per-case CSV SHA-256:
`f219ca1c48b01164c29931b5cfd2242a34f2dc7a4a315b91ccf442385d38d001`.
Process receipt SHA-256:
`ff458f01971973a592ef03ae46df32c8e6c48ffb95703869eb8daf89dd19c58a`.
