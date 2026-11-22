# Saved holding-data audit — November 22, 2026

The latest failed holding runs have two measurable limitations: exploration almost
never visits the small-command region already demonstrated to hold in still
water, and the paired-duration experiment's pooled baseline adds a large
duration-dependent offset to its actor signal. Neither observation alone proves
why learning failed. A broken reward ranking or inevitable shared-policy gradient
conflict is not supported by the reviewed evidence.

This audit reads existing data and performs network inference, independent
arithmetic, and unapplied analytical gradients. It integrates **zero physics
controls**, applies **zero optimizer steps**, and changes no checkpoint or prior
evidence. Python/NumPy are offline analysis tools, not the native PPO runtime.

## Latest physical and learning evidence

The November 21 results supersede the README's older November 4 summary:

- The still-water feasibility diagnostic passed all six original 64-second cases
  using zero normalized joint-angle targets through the original filter and
  servos. Peak total COM speed was 0.002214 mm/s. A small amplitude-0.03 periodic
  stroke also passed six cases. Neither command sequence is a learned controller,
  disturbance recovery, current holding, or permission to install a fallback.
- The reward audit ranks neutral at the original anchor above drifting and then
  settling far away, and above the failed learned swimmer. Neutral already has
  better reward by 350 ms. Position continues to matter after motion stops.
- The critic-free reset-20 experiment completed eight actor updates, but passed
  zero of six holds and regressed to zero of four short swims and 11/30 long
  windows. It was rejected.
- The most recent paired-duration comparison stopped after three paired updates
  at checkpoint 339. A20 retained 4/4 short swims and the original 28 long windows;
  B20/60 lost two short swims and triggered the registered shared stop. Both
  passed 0/6 holds and first exceeded 0.5 mm/s at 0.35 seconds in every hold case.
  There were 240 actor steps per arm, no critic calls or steps, 156,960 total
  controls, and no adoption. The unused budget is closed.

The original checkpoint 336 remains the preserved swimming baseline. Its two
known startup-window misses remain failures: retaining 28/30 windows is not full
formal long-swim acceptance.

## Actual sampled four-axis commands

`tools/audit_hold_saved.py` independently parses both calibration batches and all
six PPO batches in `build/ppo/rest-hold-duration-pair-2026-11-21-v1`. It audits
the **actual sampled** four latent actions decoded through `tanh`, not only
deterministic network means. These are normalized joint-angle targets before
the original physical filter; they are not torques or inferred trajectories.

Across the 30,720 PPO holding rows:

| All four absolute normalized commands at or below | Rows | Longest contiguous run |
| --- | ---: | ---: |
| 0.03 | 0 | 0 |
| 0.05 | 0 | 0 |
| 0.10 | 1 | 1 control / 0.05 seconds |
| 0.20 | 20 | 2 controls / 0.10 seconds |

Runs never cross actual episode resets. Holding command RMS remains about
0.630–0.640, whereas the demonstrated quiet alternatives use zero commands or
a 0.03 maximum amplitude. These thresholds are descriptive comparisons, not
new acceptance gates. Correlated 50 ms rows are not independent trials.

The exact saved conditional AR-Gaussian distribution assigns only about 0.00385
expected rows to the all-four-within-0.03 region across those visited holding
states. At true reset states its probability is approximately
7.36e-8–1.09e-7 per conditional draw. At the 0.10 threshold the expected count is
about 0.590, versus one observed row. This arithmetic conditions on recorded
states/history; it is **not** a probability of discovering a future successful
sequence. A policy could also hold with other command patterns. Nevertheless,
the completed trials supplied essentially no sustained examples near either
demonstrated quiet solution.

Per-axis command/return covariance and conditional innovation/residual covariance
are saved in the machine output. Holding command magnitude versus
arm-specific residual return has weak negative correlation: approximately
-0.224 to -0.253 in A's PPO batches and -0.105 to -0.108 in B's. Lower command
magnitude is associated with better return in these samples, but covariance
does not establish a causal action prescription or a successful sequence.

## Pooled baseline offsets

The original paired protocol intentionally fits one table across both arms,
indexed by task, pose and absolute slot. At many identical slots A has a
one-second terminal horizon while B has three seconds remaining. Their returns
therefore have different conditional means even with the same initial actor.

| Holding metric, calibration/three PPO batches | A20 | B20/60 |
| --- | ---: | ---: |
| Held-out pooled residual mean | +0.0228…+0.0230 | -0.0253…-0.0218 |
| Fraction of held-out residual energy from its overall mean offset | 65.9%–66.8% | 59.7%–61.8% |
| Positive actor-signal rows in PPO batches | 90.0%–91.3% | 14.8%–17.7% |
| Held-out residual signs changed by arm-specific fit-only means | 37.7%–44.2% | 28.6%–44.4% |

The pooled hold RMS is 0.0286923, compared with the original one-second
arm-specific fit residual RMS of approximately 0.00125827. A roughly balanced
share of squared actor signals does not establish balanced **action-discriminating**
information; much of that energy can come from a horizon-dependent constant.

This does not make the prior on-policy baseline mathematically invalid. An
action-independent baseline does not bias the exact expected score-function
gradient. It can increase finite-sample variance, change sampled gradients,
and interact with clipping and Adam. The counterfactual comparisons here fit
only the original independent calibration fit worlds; held-out worlds never
enter their means or scales. No modified baseline was trained or physically
evaluated by this audit.

## Independent actor-gradient reconstruction

`tools/audit_hold_gradients.py` separately reconstructs the initial, unclipped,
full-batch negative-advantage log-probability gradient at every collection
checkpoint. It includes the previous-observation network derivative required by
the conditional autoregressive likelihood. It does not replay PPO minibatches,
clipping, Adam, or physics.

For the last batch, collected at 338 before the failing assessment at 339:

| Gradient metric | A actual pooled | A arm/slot means | B actual pooled | B arm/slot means |
| --- | ---: | ---: | ---: | ---: |
| Holding/swimming cosine | -0.23891 | -0.03296 | +0.16853 | +0.01703 |
| Holding/swimming norm ratio | 1.71979 | 0.26399 | 1.86109 | 0.89123 |
| Total-gradient cosine to original pooled gradient | 1 | 0.36870 | 1 | 0.72635 |

The failed longer-duration arm does **not** exhibit strong opposed holding and
swimming gradients in this reconstruction. The actual saved parameter delta at
338 gives a negative local derivative for both sampled task losses under all
audited schemes. That does not establish physical improvement, but it rules out
claiming a simple measured gradient-conflict cause from these batches. Baseline
choice materially changes the finite-sample direction, particularly in A.

Independent likelihood rescoring has maximum residual 1.3965e-5, below the
existing 1e-4 bound. Six directional finite differences agree with analytical
gradients to at most 2.8e-12 absolute error.

Centering and normalization have distinct measured effects. In B's final batch,
conditioning means on actual duration/local age while retaining the old pooled
RMS changes the holding/swimming gradient norm ratio from 1.86109 to 0.98935.
Recomputing the task RMS on the duration-conditioned fit residuals changes it
again to 3.91284, with total-gradient cosine 0.61483 to the original. That is a
coupled credit-conditioning and task-weighting change, not a pure centering
causal isolation. All such numbers describe saved-data gradients, not a newly
trained or physically assessed actor.

## Selected next bounded exercise

The user explicitly authorized another exercise in this chat. The separately
frozen [holding credit protocol](HOLD_CREDIT_PROTOCOL_2026-11-22.md) tests one
new copy of checkpoint 336 on the B20/60 curriculum, with calibration-only
baseline means indexed by actual task, pose, duration and local age and a fixed
task RMS recomputed from those conditioned residuals. It allows at most eight
updates, requires the original swimming-retention gates after every update,
and caps the entire attempt at 188,640 controls. It supplies no runtime adoption.

This is a narrow test of the **corrected credit-conditioning package**, including
its coupled normalization change. It is not a clean test of baseline centering
alone and is not a promised holding fix. Reward, physics, actor initialization,
four learned outputs and exploration are unchanged. In particular, this exercise
does **not** address the measured near-neutral coverage deficiency; failure to
discover quiet behavior remains plausible. Describe both uncertainties before
interpreting a result, and do not turn an unsuccessful result into an automatic
exploration sweep or budget extension.

The protocol's calibration gates include per-duration residual-bias accounting
in addition to a balanced squared-signal share. A trained hold must still pass
all six original 64-second cases plus retained swimming. No neutral clamp,
scripted assistance, second skill network or relaxed physical threshold is
introduced. Braking from a moving start, current holding and disturbance
recovery remain separate unvalidated skills even if hold-from-rest succeeds.

## Reproducibility and preservation

From the worktree root:

```powershell
python tools/audit_hold_saved.py
python tools/audit_hold_gradients.py
```

The first command writes
`build/audit-hold-saved-2026-11-22/audit.json`; it checks 25 input SHA-256 hashes
before/after, checkpoint envelopes and payload checksums, independent segmented
returns, CSV/binary rewards/terminals/history, pooled table reconstruction and
conditional AR likelihood. The gradient output is
`build/audit-hold-saved-2026-11-22/gradients/gradient-audit.json`, with 17
checksum-verified binary inputs and 25 SHA-256 inputs, native-return recurrence
agreement within 6.32e-7, and finite-difference checks. All audited prior inputs
were unchanged. Machine outputs contain per-batch metrics and exact manifests.
No new successful controller, policy installation, push, or merge resulted.
