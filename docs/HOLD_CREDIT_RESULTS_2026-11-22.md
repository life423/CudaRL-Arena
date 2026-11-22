# Holding credit-conditioning exercise: stopped, candidate rejected

The November 22 authorized exercise passed calibration, performed two native
CUDA PPO updates and stopped at the frozen swimming-retention gate. Holding
remained 0/6. The candidate is rejected; canonical checkpoint 336 and the paused
viewer remain unchanged. No retry, extension or further exercise ran.

## What changed and why

The [saved-data audit](HOLD_SAVED_AUDIT_2026-11-22.md) found that the previous
duration-pair comparison pooled returns across different terminal horizons.
This exercise used one copied 336 actor on the same B20/60 curriculum, replacing
the pooled baseline with task/pose/duration/local-age means and recalibrated,
fixed per-task residual RMS. Both statistics were fit on independent calibration
worlds only. This jointly changes centering and task weighting; the result
cannot isolate their causal contributions.

Actor initialization, all four learned outputs, exploration distribution,
observations, physical Model 6, 1 Hz action filter, torque caps, reward,
learning rate and PPO settings were retained. No neutral commands, teacher,
scripted assistance or alternative controller were supplied.

The [protocol](HOLD_CREDIT_PROTOCOL_2026-11-22.md) was frozen before physical
collection. The original parent checksum, native executable, source, libraries
and inputs were bound by a 74-file SHA-256 manifest. Separate calibration and
training invocations had one-attempt guards and native/process watchdogs.

## Calibration before learning

One original baseline assessment reproduced 4/4 short swims, 28/30 long windows
and 0/6 holds. The independent 32x320 batch used 10,240 controls, no optimizer,
and the same seeded mixed-duration curriculum as the earlier B arm.

| Fixed calibration gate | Measured |
|---|---:|
| Held-out holding samples at speed <=0.5 mm/s | 483/2,560 = 18.8672% |
| Holding share of normalized squared residuals | 58.5999% |
| One-second duration offset share of residual energy | 13.2245% |
| Three-second duration offset share | 0.0010905% |
| Holding fit-only residual RMS | 0.00725472 |
| Swimming fit-only residual RMS | 0.16359446 |
| Maximum independent segmented-return error | 3.05e-7 |
| Maximum recorded-action log-probability rescore error | 0 |

All frozen gates passed, including each duration's <=25% offset-energy cap.
The result was reported in the current chat before launching PPO. No new
acknowledgment was required by this protocol; the user's existing authorization
covered the gated training phase. Old runs' reporting receipts were not reused.

## Actual learning and measured behavior

| Assessment | Original 336 | After update 337 | After update 338 |
|---|---:|---:|---:|
| Accepted short swims | 4/4 | 4/4 | 0/4 |
| Long windows passed | 28/30 | 28/30 | 28/30 |
| Same 28 original long-window keys retained | Yes | Yes | Yes |
| Full 64-second holds | 0/6 | 0/6 | 0/6 |
| Cumulative new actor Adam steps | 0 | 80 | 160 |

Update 338's short steady speeds are 6.96094–7.01518 mm/s, below the unchanged
7.2 mm/s minimum. This triggered the mandatory stop even though all previously
passing long windows remained. The two original startup-window failures remain;
28/30 is not complete formal long-swimming acceptance.

The final holding actor still travels. First speed breaches occur at 0.35 s
in five cases and 0.40 s in one. Final endpoint speeds are 7.86828–7.94750 mm/s;
final-eight-second mean 3D speeds are 7.88132–7.89154 mm/s and forward
excursions are 62.4472–62.5309 mm, above the 2 mm
limit. Worst anchor error reaches 482.6418 mm. Depth, lane and tilt stay within
their hold limits, but speed, anchor and excursion fail every case. Moving more
slowly than the inherited swimmer is not stationary holding.

Both fresh holding batches again contained zero all-four-axis commands within
+/-0.03 and zero within +/-0.10; no sustained near-neutral sequence occurred.
Near-rest occupancy was 18.1641% and 18.9648%, largely supplied by short rest
starts. Calibration signal magnitude and reset occupancy did not demonstrate
discovery of an action sequence that holds. The attempt does not resolve the
exploration deficiency identified in the saved-data audit.

## Actual budget and verification

| Consumed component | Controls |
|---|---:|
| Baseline assessment | 10,720 |
| Independent calibration | 10,240 |
| Two fresh training batches | 20,480 |
| Two post-update assessments | 21,440 |
| **Total** | **62,880** |

Actual physical substeps: **628,800**. Actual PPO updates: **2**. Actor Adam
steps: **160**. Critic calls/steps during physical collection and training:
**0**. The zero-step preparation separately compared preserved old/new CPU
critic outputs for migration parity; this was not a training critic.

Calibration took 2.6554 process seconds; training took 4.8633 process seconds,
7.5187 combined. Native accounting records 2.5522 prior active seconds and
4.8095 training seconds (7.3616 combined, including calibration finalization).
Both phases exited normally with the behavioral stop recorded; no physical or
numerical fault, timeout, retry, automatic continuation or adoption occurred.
The unused allowance is closed.

The rebuilt focused Gaussian/AR, PPO/GAE and checkpoint suites passed 25/25 with
zero skips and actual CUDA execution. Zero-step migration observations and
old/new actor outputs matched exactly; synthetic mixed-boundary, terminal and
fit-world exclusion checks passed. Independent post-run verification passed:
36 input hashes, three binary batch envelopes and three actor/Adam checkpoint
round trips; disjoint calibration statistics, returns and action probabilities;
all 18 baseline/post-update holding traces, 90 long-window records, short-swim
gates, first breaches and physical counters. It advanced no physics and applied
no optimizer steps. [Verifier](../tools/verify_hold_credit.py) and its machine
results remain with the source/run evidence. The ordinary native `ppo_motor`
target also rebuilt successfully with the optional failure hook disabled.

All 74 frozen input hashes remained unchanged across execution. A preservation
manifest checked 5,168 pre-existing files: 5,167 stayed byte-identical. The sole
intentional source change adds an opt-in CPU failure-evidence hook to
`tools/ppo_motor.cu`; its exact pre-edit bytes and SHA-256 are preserved. Every
old checkpoint and study artifact in the manifest remains unchanged. The hook
changes no equations or acceptance limits and leaves its absent-macro call path
unchanged. It captures first unsafe pre-state/candidate evidence during short
and long CPU evaluations if needed.

## Evidence and next decision

Full source/executable snapshots, parent and migrated inputs, baseline,
calibration, three checksummed recorded batches, initial/update-337/update-338
actor/Adam checkpoints, physical traces and closed results remain under
`build/ppo/hold-credit-336to344-2026-11-22-v1`. Compact evidence is retained in
`assets/ppo-four-axis-checkpoints/hold-credit-2026-11-22-rejected`.

This attempt does not establish that the pooled baseline was the sole cause,
that a particular exploration change will succeed, or that holding is physically
impossible. It does show that this corrected conditioning/normalization package
did not produce holding while retaining swimming under the registered budget.
The next decision should address sustained action-sequence coverage and goal
discrimination using the preserved data, before proposing any further bounded
learning experiment. No additional training is authorized by this result.
