# Sustained exploration exercise: completed, holding candidate rejected

The November 22 exercise completed its exact 32-update allowance. Persistent
small-action exploration produced actual quiet training segments, and native
PPO changed all four trainable coefficients. Deterministic holding still passed
**0/6 cases at every assessment**. All four short swims and the same 28 long
windows remained, with swimming traces byte-identical to the baseline. The
holding candidate is rejected; no further training or adoption follows.

## Policy and interpretation

The [frozen protocol](HOLD_EXPLORATION_PROTOCOL_2026-11-22.md) kept checkpoint
336's MLP and inherited optimizer unchanged. Four fresh request-conditioned
output coefficients, initialized to one, were the only parameters updated by
native CUDA PPO/Adam. They apply to stationary requests; swimming uses the
original frozen computation. This is restricted parameter learning, not evidence
that a fully trainable shared network retained swimming without forgetting.

Holding collection sampled a positive log-uniform amplitude within balanced
strata once per true one-/three-second segment, scaling all four latent means
and noise together. Correct conditional densities include amplitude, AR history,
the previous mean and the scale Jacobian. No teacher, zero-command injection,
posture clamp, altered physics or recovery controller was supplied.
**Every deterministic assessment used amplitude one and no sampling noise.**
The initial deterministic actor remained the failing 336 holding actor.

Model 6 body/water equations, two universal joints/four axes, torque caps,
physical filter, rewards and acceptance gates stayed fixed. The four-parameter
Adam used learning rate 0.0025. Goal choice, moving-start braking, anchor
recovery, current holding and an interactive learning worker were outside scope.

## Coverage before learning

The original assessment reproduced 4/4 short swims, the same 28/30 long windows
and 0/6 holds. One independent 32x320 calibration batch then measured:

| Pre-learning measurement | Result |
| --- | ---: |
| Complete physically quiet holding segments | 102/128 |
| Pose 0, one-second / three-second segments | 26 / 25 |
| Pose 3, one-second / three-second segments | 26 / 25 |
| Held-out rows at speed <=0.5 mm/s | 2,223/2,560 = 86.8359% |
| Holding share of normalized residual energy | 62.7336% |
| All-four-command rows within +/-0.03 | 969/5,120 |
| Longest such command sequence | 60 controls / 3 seconds |
| Fit residual RMS, holding / swimming | 0.00785643 / 0.16359446 |

Each quiet segment satisfied speed, anchor, depth, lane and tilt limits at
every actual post-action boundary, including its final endpoint. Both poses
and durations were covered. These attenuated exploratory segments are not
deterministic 64-second holding passes.

Independent verification reconstructs all 1,280 fit-only calibration means,
four rows per cell, task scales, segmented returns and actual scaled densities.
Calibration was reported before the separate PPO launch. The first fresh PPO
batch repeated coverage and signal gates before its first update. Calibration
data were not used for PPO. Per-duration/bin residual offsets are preserved
diagnostics, not additional acceptance thresholds.

## Deterministic results

All 32 assessments retained 4/4 short swims and the exact 28 original long-window
keys, with short and long CSV traces byte-identical to the baseline. The two
historical startup-window misses remain failures; this is not complete formal
long-swimming acceptance.

| Six holding cases | Original coefficients | Final update 368 |
| --- | ---: | ---: |
| Full 64-second passes | 0/6 | 0/6 |
| Final-eight-second mean 3D speed | 8.5365–8.5652 mm/s | 2.4485–2.5892 mm/s |
| Final-eight-second forward excursion | 67.6417–67.8662 mm | 19.3023–20.4295 mm |
| Worst whole-case anchor error | 522.523 mm | 199.610 mm |

The final actor first exceeds 0.5 mm/s at 0.40 or 0.45 seconds in all six cases.
Speed, anchor and excursion fail every case. Depth fails all six, with maxima
1.2072–1.3489 mm against the unchanged 1 mm gate. Lane and tilt pass all six;
worst values are 1.0767 mm and 6.4872 degrees. Slower travel is not holding.

Final learned coefficients:
`[0.68625230, 0.87050003, 0.57416266, 0.66437876]`, from initial `[1,1,1,1]`.
Their optimizer changes are real but do not establish acceptance. The final
stochastic batch contained 111/128 quiet segments, 1,291 command rows within
+/-0.03 and a 60-control quiet command sequence. Its sampled amplitudes differ
from calibration; those counts alone are not a matched improvement measure.

## Exact accounting and verification

| Consumed work | Controls |
| --- | ---: |
| Original assessment | 10,720 |
| Independent calibration | 10,240 |
| 32 fresh PPO batches | 327,680 |
| 32 deterministic assessments | 343,040 |
| **Total** | **691,680** |

Actual substeps: **6,916,800**; PPO updates: **32**; fresh four-parameter Adam
steps: **2,560**; collection/training critic calls and steps: **0**. No retry,
extra seed, extension, physical/numerical fault, timeout, runtime adoption or
automatic follow-on occurred. The attempt is closed.

Native accounting records 2.6523265 prior active seconds plus 81.4125608 training
seconds: **84.0648873 combined**. Independent process logs record 2.7489732
calibration seconds plus 81.4705543 training seconds: **84.2195275 combined**,
within 120 seconds. The report pause was outside the separate phase processes.

`tools/verify_hold_exploration.py` passes all 32 updates and 33 gain/Adam
checkpoints: the backbone is exact, fresh optimizer ages/moments are valid,
and densities, returns, coverage, physical gates and counters agree. Maximum
errors: likelihood 6.432e-6, frozen CPU/GPU backbone output 1.001e-6, return
recurrence 6.316e-7, independent gain-score finite difference 2.833e-9.
All 406 verified inputs retained their SHA-256 hashes.

The coordinator preserved all 5,256 prior artifact files and all 76 frozen
execution inputs during the run. Later authorized GUI/build edits are separate;
executed source snapshots remain available. Two offline parser corrections were
needed: a NumPy Boolean duration index and inclusion of reset depth in hold-peak
metrics. Reset depth is independently reconstructed from saved initial poses and
float32 mass/COM geometry. No native result, gate, tolerance or study input
changed; neither correction required physics or optimizer work.

## Evidence and inspection

Full frozen sources/executable, all 33 gain states, 33 sampled batches including
calibration, physical traces and exact body/joint poses remain in
`build/ppo/hold-exploration-2026-11-22-v1`. Compact protocol, initial/final gain
states, calibration/results, updates, independent verification, summaries and
provenance are in
`assets/ppo-four-axis-checkpoints/hold-exploration-2026-11-22-rejected`.

Separately authorized GUI work provides recorded exact-pose inspection and an
opt-in live frozen research candidate using `res://research_data/latest-policy.bin`.
Those modes are not background learning or production acceptance of the failed
holding actor. GUI build, layout and interaction evidence is reported separately;
the physical study does not claim rendered-UI verification.

Later preview-package verification used 80 separate frozen controls / 800
substeps, split between holding and swimming, with no optimizer work. It checks
package/playback agreement, not a new holding assessment or an extension of
this closed study. Its detailed native and rendered-UI evidence is separate.

Sustained exploratory coverage is demonstrated in the registered short segments.
Useful deterministic holding remains unlearned. No claim of physical impossibility,
a guaranteed next fix, general motor competence or indefinite performance follows.
No further experiment is authorized by this completed result.
