# Frozen feeding-policy held-out comparison

Eight held-out seeds; 16 worlds per seed; 120 seconds per actor. Each subgroup contains 32 matched pairs. Policies use frozen mean actions with no learning or deployment.

| Group | Before food/min | After food/min | Paired change | Change % | Seed-cluster 95% interval | Food-rate evidence | Wins/ties/losses |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| retention | 18.234 | 18.531 | +0.297 | 1.628 | [-0.156, +0.734] | inconclusive | 16/4/12 |
| calm | 14.531 | 14.531 | 0.000 | 0.000 | [-0.078, +0.063] | inconclusive | 4/25/3 |
| steady | 14.156 | 13.984 | -0.172 | -1.214 | [-0.516, +0.078] | inconclusive | 4/22/6 |
| fluctuating | 14.391 | 14.359 | -0.031 | -0.217 | [-0.125, +0.063] | inconclusive | 2/27/3 |
| overall | 15.328 | 15.352 | +0.023 | 0.153 | [-0.074, +0.129] | inconclusive | 26/78/24 |

## Reliability and timing

| Group / policy | Available-food gap median / max (s) | Confirmed immobility cases / events | Terminal immobile | Invalid cases | First-food censored / observed | Observed first-food median (s) | Mean accepted creations | Mean cap time (s) |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| retention / before | 5.575 / 9.550 | 0 / 0 | 0 | 0 | 0 / 32 | 2.325 | 39.469 | 0.000 |
| retention / after | 5.275 / 9.800 | 0 / 0 | 0 | 0 | 0 / 32 | 2.375 | 40.063 | 0.000 |
| calm / before | 6.150 / 11.000 | 0 / 0 | 0 | 0 | 0 / 32 | 2.450 | 30.031 | 0.000 |
| calm / after | 6.625 / 12.550 | 0 / 0 | 0 | 0 | 0 / 32 | 2.425 | 30.031 | 0.000 |
| steady / before | 6.650 / 18.400 | 0 / 0 | 0 | 0 | 0 / 32 | 2.775 | 29.438 | 0.000 |
| steady / after | 6.525 / 14.450 | 0 / 0 | 0 | 0 | 0 / 32 | 2.825 | 29.438 | 0.000 |
| fluctuating / before | 6.925 / 11.550 | 0 / 0 | 0 | 0 | 0 / 32 | 2.800 | 29.813 | 0.000 |
| fluctuating / after | 7.175 / 14.000 | 0 / 0 | 0 | 0 | 0 / 32 | 2.875 | 29.813 | 0.000 |
| overall / before | 6.275 / 18.400 | 0 / 0 | 0 | 0 | 0 / 128 | 2.575 | 32.188 | 0.000 |
| overall / after | 6.325 / 14.450 | 0 / 0 | 0 | 0 | 0 / 128 | 2.600 | 32.336 | 0.000 |

## Prospective interpretation

Overall progress inconclusive; do not claim a general food-rate improvement.

These are provisional decision thresholds declared before the held-out run, not universal guarantees or simultaneous subgroup confidence claims.

- Any after-policy confirmed food-present five-second immobility or invalid case is a reliability gap to address before anatomy.
- A subgroup mean food-rate loss exceeding 5% is a material regression.
- A subgroup lower confidence bound above minus 5% of its before mean provides stronger bounded retention support.
- An overall food-rate improvement claim requires a positive overall lower confidence bound and no material subgroup mean loss.

Material subgroup mean regressions: none.
After-policy reliability gap: none observed in this bounded suite.

| Group | Mean loss exceeds 5% | Retention evidence against −5% margin |
| --- | --- | --- |
| retention | no | bounded retention supported |
| calm | no | bounded retention supported |
| steady | no | bounded retention supported |
| fluctuating | no | bounded retention supported |

## Interpretation limits

- This is progress between saved validated actors, not causal evidence that the new curriculum beats equally long old-curriculum training.
- Aggregate improvement does not establish improvement in every current subgroup. Confidence intervals are descriptive; no multiple-comparison adjustment is applied.
- Eight seed clusters give limited uncertainty resolution; an interval containing zero is inconclusive, not proof of equivalence.
- First-food latency summaries exclude censored cases; compare censoring counts alongside latency.
- Timed food can create legitimate empty-pond waiting. Available-food gaps/immobility are reported separately from raw food gaps.
- Matched nominal arrivals can be suppressed differently at the occupancy cap. Accepted food creations include initial pellets.

Provenance and all duration/censoring/cap metrics are preserved in summary.json. Analysis does not train, alter policies, choose another candidate or deploy a winner.
