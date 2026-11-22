# Holding exploration design — November 22, 2026

This is an offline design and saved-data probability audit for the user's newly
authorized exploration/holding exercise. It performs no physics integration or
optimizer update, and changes no runtime or old evidence. The separately frozen
execution protocol owns the actual settings, finite budget and stop conditions.

## Evidence and choice

The preceding [credit-conditioning exercise](HOLD_CREDIT_RESULTS_2026-11-22.md)
stopped after two PPO updates: 0/6 holds, 0/4 short swims, with the original 28
long windows retained. It supplied no sampled all-four-axis command within
either +/-0.03 or +/-0.10. Correcting the baseline and residual normalization did
not resolve holding. The earlier saved-data audit establishes essentially absent
sustained quiet-action coverage. The physical neutral and small-stroke diagnostic
shows that quiet holding is possible in these still-water reset cases, without
establishing learned holding or recovery from motion/disturbance.

The selected design changes exploration rather than initializing a neutral actor:

1. Keep checkpoint 336's MLP frozen. Introduce four trainable request-conditioned
   output gains, all initialized to **one**, not zero. All four outputs remain
   independent. This is one motor with four extra scalar actor parameters, not
   another skill network, a goal chooser or a conventional holding controller.
2. In holding training, draw a positive stochastic amplitude once per actual
   holding segment and retain it across that segment. The draw scales the full
   sampled latent Gaussian action, including mean and noise. It never injects
   a zero command, supplies an action label, clamps posture or changes physics.
3. Use deterministic evaluation with amplitude exactly one and noise disabled.
   A low-amplitude stochastic trajectory cannot itself qualify as learned
   holding. The initialized deterministic actor remains the failing 336 actor;
   only PPO changes to the four learned gains can alter its holding mean.
4. Nonzero swimming requests keep their original means and exploration exactly.
   The frozen MLP and request-conditioned gain mask preserve swimming for any
   identical observations, rather than hoping rehearsal prevents interference.
   Actual swimming assessments still catch implementation or state-path errors.

This is a deliberately restricted acquisition exercise. Four amplitude parameters
cannot supply a general controller for anchor recovery, turning, braking or
currents. A success would establish learned suppression of an inherited gait
for these rest requests, not a finished reusable locomotion system.

## Why merely increasing Gaussian standard deviation is insufficient

`tools/audit_hold_exploration.py` reads the last failed B20/60 collection checkpoint
and batch from the completed duration comparison. At its true rest starts, mean
latent outputs average approximately `[0.049, 0.268, 0.467, -1.100]`.

The probability that **all four** sampled normalized commands lie within
`+/-0.03` at those fixed saved reset observations is:

| Gaussian standard deviation | Mean probability |
| --- | ---: |
| Original `exp(-1) = 0.367879` | 7.50e-8 |
| 0.50 | 2.63e-7 |
| 0.75 | 2.72e-7 |
| 1.00 | 1.54e-7 |
| 1.50 | 4.64e-8 |
| 2.00 | 1.70e-8 |
| 3.00 | 3.73e-9 |

This is a fixed finite analytic comparison, not a hyperparameter search or a
physical trial. Increasing sigma alone spreads probability away from zero in
four dimensions as well as toward it. Raising AR correlation alone would retain
an initially rare near-zero sample longer but would not improve the stationary
probability of reaching it. Long-lived additive bias can cancel one phase of a
gait but need not cancel its changing four-axis means. Setting a holding gain
to zero before learning would install the known behavior, and is not selected.

## Persistent scale-mixture exploration

For a true holding segment, draw `g` log-uniformly in `[0.01, 1]`; keep it fixed
until that segment's real terminal/reset. Swimming uses `g=1`. The distribution
has positive support only: there is no neutral action atom or forced quiet
episode. A low draw makes an entire sampled four-axis sequence small, rather
than requiring four independent cancellation events at every control.

Analytically, `P(g <= 0.03) = log(3)/log(100) = 23.8561%`, and
`E[g^2] = (1 - 0.01^2)/(2*log(100)) = 0.108563`.

At fixed recorded observations and correspondingly rescaled AR history,
257-point one-dimensional quadrature gives about 13.70% instantaneous
all-four-within-0.03 probability, versus the original extremely small mass.
This is a frozen-observation counterfactual, **not** a future physical probability.
Changed commands change subsequent observations.

An additional algebraic check rescales the already recorded latent commands,
without replaying the policy or integrating physics:

| Constant algebraic gain | Quiet rows / 5,120 | Entire recorded 20/60-control segments quiet / 128 |
| --- | ---: | ---: |
| 0.01 | 5,095 | 110 |
| 0.03 | 540 | 0 |
| 0.10 | 12 | 0 |
| 1.00 | 0 | 0 |

This supports testing persistent amplitude exploration; it does not demonstrate
that any new actor trajectory will hold. The pre-learning physical calibration
must measure actual sequence coverage and low-speed occupancy under the sampler.
No previous diagnostic or algebraic rescaling can substitute for that gate.

## Actor and exact probability contract

Let `b_t = frozen_MLP336(x_t)`, `a_j` be four learned scalar gains initialized
to one, and `h_t` the explicit request-conditioned mask (one for the studied
stationary request, zero for the studied 8 mm/s swimming request):

```
m_t,j = [1 + h_t*(a_j - 1)] * b_t,j
z_t = g*m_t + rho*(z_(t-1) - g*m_(t-1))
      + g*sigma*sqrt(1-rho^2)*epsilon_t       when history exists
z_0 = g*m_0 + g*sigma*epsilon_0              at a true reset
command_t,j = tanh(z_t,j)
```

`sigma=exp(-1)` and `rho=0.8` retain their original meanings. The physical
target filter, servo and torque caps are unchanged. Four `z` values are the
actual sampled latent actions applied to the four physical axes.

For history flag `H`, the exact conditional Gaussian mean and variance are:

```
c_t = g*(m_t - rho*H*m_(t-1)) + rho*H*z_(t-1)
V_t = g^2*sigma^2*(1-rho^2*H)
log pi(z_t | g, history) = log N(z_t; c_t, V_t*I)
                       = log pi_base(z_t/g | rescaled history) - 4*log(g)
```

The same positive amplitude is retained across history; a new amplitude occurs
only when action history is truly cleared. The independent amplitude sampling
law does not depend on actor parameters, so its factor cancels from the PPO
old/new ratio. Condition on the **recorded actual** amplitude when scoring.
Do not pretend the transformed action still has the original unscaled Gaussian
density. The tanh action convention remains the original latent-action PPO
convention; no new post-tanh probability approximation is introduced.

For parameter `a_j`, with frozen MLP outputs at the current and previous actual
observations:

```
d log pi / d a_j = (z_t,j-c_t,j)/V_t
                  * g*(h_t*b_t,j-rho*H*h_(t-1)*b_(t-1),j)
```

The previous-mean term is required. For swimming it vanishes exactly. Apply the
ordinary clipped-PPO loss multiplier to this score, retain the recorded old
log probability, and update only the four new gain parameters with their own
explicit fresh Adam state. The old MLP parameters and optimizer evidence remain
unchanged. Finite-difference checks must verify both reset and history cases.

All amplitudes, four sampled latents, old probabilities, reset identities and
four gains must be serialized explicitly. A new checksummed study schema must
bind the frozen backbone, mask and sampling laws; it must not impersonate an old
40/41-input checkpoint or silently load into the viewer.

## Calibration strata and credit

Exogenous amplitude strongly changes returns. A baseline indexed only by
duration could repeat the previous study's constant-offset problem. Use four
equal-log-width bins with edges:

```
0.01, 0.0316227766, 0.1, 0.316227766, 1
```

Draw continuously log-uniformly within the selected bin. For each holding world
and its eight alternating 20/60-control segments in a 320-control batch, choose:

```
bin = (floor(world/8) + floor(segment_index/2)) % 4
```

This balances each duration, initial pose and bin in both original fit and
held-out world splits. Every pose/duration/bin has four whole fit episodes and
four held-out episodes. Marginal bins are balanced, while the within-bin draw
is random and independent of actor parameters. Its density is
`q(g | bin) = 1/[g*log(sqrt(10))]` within the bin and also cancels in the ratio.

Use frozen calibration fit-world means by task, pose, actual duration/local age
and amplitude bin. There are 640 swimming cells and 640 holding cells, each with
four fit rows. Recompute fixed per-task residual RMS on fit worlds only. Hold
out whole worlds; do not fit on assessment outcomes or adjust statistics during
training. Report per-bin/duration bias energy, signal energy and actual
sustained-command coverage; the execution protocol must freeze their gates.
Continuous within-bin amplitude variation remains a source of residual variance.

Because these four parameters have a different scale from MLP weights, their
learning rate must be explicit. The historical MLP rate could move a unit gain
too little in the small update budget and should not be reused without an
explicit scale rationale. Any new rate,
gain parameterization, KL/gradient clip, update cap and calibration/assessment
count must be frozen before physics, with no retry or automatic retuning.

## Acceptance and limits

The exercise should use one copied baseline, one independent pre-learning
calibration and one fixed maximum update allowance. Every deterministic
assessment must use `g=1`, no exploration noise, the same learned gain file,
the original six 64-second hold gates, four short swims and original 28 long
window keys. Save initial/all intermediate parameters and all actual updates.
Reject numerical defects, inadequate calibration or lost swimming according to
the frozen stop rules. No viewer installation follows automatically.

If holding succeeds, report the before/after deterministic distinction and the
four learned gains. If it fails, close the allowance. Quiet stochastic samples,
small gains, improved reward, exact swimming preservation or a passing numerical
test alone do not establish learned stationary holding. Neither result answers
moving-start braking, position recovery, current rejection or general motor
competence.

Reproducibility: `python tools/audit_hold_exploration.py` writes
`build/audit-hold-exploration-2026-11-22/analysis.json` with saved-state outputs,
probability arithmetic, counterfactual sequence counts and unchanged input
SHA-256 hashes. This design audit performed zero physics and optimizer steps.
