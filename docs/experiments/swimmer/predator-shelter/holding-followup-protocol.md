# Registered bounded follow-up to the holding diagnosis

Registered October 29, 2026 **before execution**. The user requested a controlled
follow-up after the one-update comparison remained behaviorally inconclusive.
This is not authorization for an open-ended sweep or a production change.

## Budget and controls

- Four arms: no update, holding-only body-policy loss, feeding-only body-policy
  loss, and the current combined loss.
- Same complete protocol-5 update-300 checkpoint, actor, critic, saved Gaussian
  log standard deviation, all Adam configurations/moments/counters, environment
  states, selector, context and counter-based random seed at initialization.
- **Exactly 16 iterations per arm**, 1,024 worlds ×64 steps per iteration.
  The three learning arms each use the unchanged four epochs/eight minibatches:
  at most **48 distinct PPO updates total**, with no adaptive extension.
- First rollouts must match. Later rollouts are freshly collected on-policy
  using each arm's current means and standard deviations. Policies/world states
  may legitimately diverge. No repeated off-policy training on stale data.
- Preserve the current mixed population, reward, physics, inputs, architecture,
  action-unit Jacobian, optimizer settings, GAE and shared advantage statistics.
  Mask only actor/standard-deviation gradient rows for the excluded task *after*
  the same shared normalization, retaining the original minibatch denominator.
  The three learning arms retain the same all-task critic-update rule.
- Exploration settings and initial standard deviation match; learned standard
  deviation may diverge as a consequence of each arm's policy objective.
- No-update freezes all parameters and optimizer state. Its separate logical
  sampler clock follows the same registered counter keys; do not falsify a
  resumable checkpoint by advancing update_id without its Adam steps.
- Counter-based sampler/permutation keys use the source update plus iteration.
  Record actual task counts, clipping, KL and gradient diagnostics. No source or
  personal checkpoint is an output, and no diagnostic actor is adopted.

## Registered evaluations

Frozen evaluations happen at iterations 0, 8 and 16. Development near-holding
seed is `0x48D1E117`. The final held-out near-holding seed is `0x48D1AB37` and
must not select an intermediate checkpoint. Final checkpoint is iteration 16.

Use the four exact nominal resting poses and 24 near starts. Keep five-second
whole-body safety unchanged: exposure ≤0.05 and clearance ≥0.08 model units,
including the fin, on every tick. Report initially exposed fixtures separately.
Record deterministic means and stochastic behavior separately; neither substitutes
for the other. Before seeing the follow-up results, the additional trace arms
are fixed as final holding-only and final combined. Run `holding_diagnose` on
both with the declared blind near seed, four nominal poses and 16 signed-velocity
starts. Its seven registered conditions compare zero/calibrated inherited/final
means, matched physical Gaussian noise at log standard deviation −1.5, and the
final actor's own saved standard deviations. Use the existing tool's fixed
counter-based random keys and one trajectory per condition/case. These are
matched diagnostic noise samples, not a stochastic reliability estimate or a
new training budget. The final checkpoints are selected by iteration count,
not their evaluation score.

Feeding retention uses matched deterministic calm and 0.75 cm/s trials, 16 worlds
×600 physical steps, on both reference seed `0x571EF001` and final fresh seed
`0x571E2027`. Include the original inherited controller and frozen protocol-5
baseline. Report counts/rates and the limits of these small samples.

A *promising diagnostic screen* requires 4/4 nominal holds, at least18/24 final
near holds and at least90% of the original inherited food rate in both currents
and both feeding seeds, with finite states. This does not mean the product is
fixed or broadly reliable. Arrival with momentum/braking and autonomous
food→shelter→hold→feeding during a predator pass remain subsequent gates.

## Hypotheses and decision rules

| Hypothesis | Supporting result | Result weakening it / limit |
|---|---|---|
| Food policy updates obstruct holding learning | Holding-only improves held-out holding, combined loses or lacks that improvement | Comparable improvement in combined weakens that account; a win does not distinguish clipping, shared weights, critic or shared exploration as the sole mechanism |
| Exploration prevents reliable expression | Deterministic holding succeeds but matched stochastic actions fail | Failure of deterministic means means exploration is not the sole cause |
| Inherited optimization/conditioning obstructs learning | Holding-only cannot move toward safe behavior, but a matched independently initialized policy with the same architecture later learns a genuine improvement | Holding-only success weakens this; failure after16 updates alone cannot establish this cause |
| Architecture lacks capacity or adequate observations | Requires further evidence that separates missing information/representation from optimization difficulty | Success of a same-architecture independently initialized policy weakens a capacity limit; no such independent-policy trial is authorized by this protocol |
| Reward/proof mismatch dominates | Return improves while strict continuous safety does not | Safe zero torque already earns more return in44/44 matched cases; this rules out a simple global preference for the observed failed behavior, not every local reward issue |
| Existing controller is sufficient for the bounded lesson | Held-out holding improves with retained feeding | Resting near shelter still does not prove safe arrival, goal choice or the full predator sequence |

Do not initialize another controller merely to favor a redesign. If an independent
diagnostic policy becomes necessary, match environment/settings/budget and compare
its pre-training baseline too: a near-zero random output that already holds at
rest is not evidence that it learned more effectively.

## One-step balance result remains a diagnostic counterfactual

The earlier actor-task-balanced copied step kept the required chain-rule gains
intact but changed the food objective weight. It removed the measured shared
gradient-norm clipping in that batch, yet still achieved0/28 holds. It is not a
chain-rule correction, a proven fix or a production training change. The four-arm
follow-up above does not include another balance, reward or architecture variant.
