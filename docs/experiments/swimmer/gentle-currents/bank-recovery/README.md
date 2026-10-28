# Learned food-selector bank recovery — October 27, 2026

Native PPO training improved recovery on **86 untouched bank-contact starts**: the new selector reaches food within 20 seconds in **83/86 cases**, versus **62/86** for the previous progress-aware pilot and **40/86** for the original deployed selector. All foods remain in valid underwater positions. The swimming controller, creature body, physical equations and PPO coefficients are unchanged. No arms, additional decision network or deployed rescue rule was added.

The selected candidate is saved separately; the original Godot pond policy remains active. Rare gaps remain in ordinary long swims, so this is a demonstrated improvement in bank recovery, not a claim of stall-free swimming or robustness to all water conditions.

## Protocol and data separation

The [protocol](PROTOCOL.md) was fixed before collection/training. Authentic states came from the original frozen controller's physical trajectories, with at least 20 seconds since food, low ground speed and contact/near-contact with the sloped pond bed. At most one state is retained per source world. All food must satisfy the unchanged `deep_pond::water` constraints, including clearance above the bottom. No food is placed on land or moved to force recovery.

The initial 8,192-world/five-minute collection was too small. The documented **pre-training revision** expanded to 32,768 worlds per seed and ten minutes, preserving the qualification rules and five source seeds. It produced 95 training, 51 development and 86 holdout states from disjoint source-world identities. The holdout outcomes were not used for training or selection. [Pool manifest](pool-manifest.json), [initial collection audit](collection-initial/), and exact [datasets](data/) are retained.

The hard states are predominantly from stronger currents: training contains three calm, 24 at 0.5 cm/s and 68 at 1 cm/s; holdouts contain one calm, 18 at 0.5 cm/s and 67 at 1 cm/s. These are deliberately selected difficult states, not the natural frequency of failure or an evenly balanced flow survey.

## Training and persistence

- Two copies start from the existing seed-71/109 update-128 full saves. Existing selector actor, critic and Adam are retained; the original full saves remain unchanged.
- Expected episode-start mixture: 50% sampled recovery states, 25% ordinary starts and 25% independently folded starts. Recovery starts retain physical state, actual food/current/target and progress history. Episode time/count and RNG identity are reset; inherited time-since-food is capped at ten seconds to permit a recovery attempt before the unchanged training failure rule. This is training reset-time sampling, not an inference-time teleport or target override.
- Ordinary-start currents remain 25% calm, otherwise uniform speed 0–1 cm/s and random horizontal direction. Replayed starts retain their measured source current.
- The existing 34-feature food scorer/40-float critic observations, categorical PPO, GAE, LR 0.0003, gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01, value coefficient 0.5, 32-decision horizon, four epochs, eight minibatches and gradient cap stay unchanged. Food reward and training-only failure penalty stay unchanged. The 33-input/four-output swimming actor stays frozen.
- Each seed completes **256 additional updates at 8,192 worlds**, stopping at cumulative update 384. Full state is saved separately under `build/pond-bank-study/seed-71/` and `seed-109/`. This phase performs 2,684,354,560 control-level environment steps in total; each uses the original ten 5-ms physical substeps. No open-ended run is left active.
- Full checkpoint version 2 embeds the recovery-start pool along with actor, critic, Adam and current worlds. Legacy version-1 checkpoints remain readable. [Both real update-384 saves restore byte-identically](resume-verification.json) with zero additional optimizer updates. The selected actor below is from update 320; latest resumable training state is update 384.

The two [learning logs](learning-seed-71.csv) / [second-seed log](learning-seed-109.csv) retain normal-layout development scores. The fresh-directory import path initially omitted the CSV header; the raw local logs were preserved and the archived tables received their missing header after training stopped. The CLI now writes headers for new imported-run logs. This formatting correction does not alter any weights, training state or measured rows.

## Selection before holdout access

Ten saved candidates (phase start and each 64 additional updates per seed) are evaluated on the 51 development bank cases, normal development collection, and the labeled known-stall regression. Gates require normal calm/current collection at least 98% of original, recovery of the known fixture within 20 seconds, and at least ten percentage points more development recovery than the original. Rank 20-second recovery first, bank collection second, then earlier phase update/lower seed. [Selection table](selection.csv) and [reproduction script](select-development.ps1).

Winner: **seed 71, cumulative update 320 (192 additional updates)**, [selected policy](selected-policy.bin), SHA-256 `4186F47471EDE2D0034EB566EFA1EC959B03D0A015E686FFA5F4262FCEB4A563`. It recovers 50/51 development cases within 20 seconds, with 14.1961 foods in the following minute. The original reaches 19/51, previous pilot 40/51. The known fixture's first food is within 14 seconds; that fixture remains development evidence, never a training example or fresh holdout. [Selected record](selected.json).

Both training seeds learn development recovery: each rises to 50/51 by its final checkpoint. Fresh tests below concern the single selected seed-71 actor and do not establish variance across independently selected training seeds.

## Untouched bank results

Both selectors are compared through the same bank evaluator, without resets, target overrides or reward-based termination. The prior target, food positions, body state, progress history and current are identical for each matched start.

| Policy | Food within 20 s | Food within 60 s | Mean food in next 60 s | >20 s continuous low speed |
| --- | ---: | ---: | ---: | ---: |
| Original deployed selector | 40/86 (46.5%) | 43/86 | 6.419 | 44 cases |
| Previous progress-aware pilot | 62/86 (72.1%) | 65/86 | 10.221 | 23 cases |
| New bank-trained selector | **83/86 (96.5%)** | **84/86** | **13.756** | **3 cases** |

Compared with the original, 43 previously unsuccessful cases now recover within 20 seconds and no previous success is lost. Compared with the prior pilot, 21 additional cases recover and no previous success is lost. These are paired gains of 50.0 and 24.4 percentage points on this challenge set. Three cases miss the 20-second target; two still have no food by 60 seconds. Raw cases and summaries are in the six `holdout-*` directories; [aggregate](aggregate.json).

This verifies learned target-choice recovery under the retained swimming controller. It does not prove every target is reachable from every contact pose, or that the motor can never require adaptation. The source bank recordings and the fresh evaluator use different batch shapes; the evaluator shares physical equations and target-promotion logic, and comparisons here use identical execution paths across policies. The original recorded regression remains separately checked.

## Fresh ordinary swimming and limits

The selected actor is frozen before evaluating two new layout seeds over all eight directions at each speed and calm (2,176 one-minute worlds per policy). Means across the directions:

| Water | Original food/min | Previous pilot | New candidate |
| --- | ---: | ---: | ---: |
| Calm | 17.7500 | 17.6953 | 17.7578 |
| 0.5 cm/s | 16.9443 | 17.0430 | 17.0342 |
| 1.0 cm/s | 14.3262 | 14.4482 | 14.5107 |

At 1 cm/s the candidate improves average collection about **1.29% over the original**. Its layout-clustered paired difference is +0.1846 food/min, approximate 95% interval [+0.0644, +0.3047]. Its +0.0625 difference over the previous pilot has interval [−0.0457, +0.1707], so a collection advantage over that pilot is not established. Calm results are essentially unchanged in this sample. Intervals are conditional on these policies/layouts, cluster repeated directions within each layout, and are not corrected for multiple tests or training-seed selection.

Among the 1,024 one-minute 1 cm/s trials, >20-second food gaps occur in five original, five pilot and three candidate worlds. None occur at 0.5 cm/s or calm. The maximum candidate gap is 24.9 seconds. This supports some short-trial recovery improvement, not absence of stalls.

Fresh five-minute probes (64 worlds each, one new seed) remain **mixed**:

| Condition | Original / pilot / candidate food/min | Original / pilot / candidate worlds with >20 s food gap |
| --- | --- | --- |
| Calm | 17.8219 / 17.9125 / 17.8656 | 0 / 0 / 0 |
| +X, 1 cm/s | 14.7719 / 14.7406 / 14.8313 | 0 / 2 / 1 |
| −X/−Z, 1 cm/s | 14.1656 / 14.3563 / 14.3156 | 1 / 1 / 1 |

The candidate has two long-probe gap cases versus one original and three pilot cases; its worst gap is 25.2 seconds. This small sample does not establish lower long-run stall incidence. Repeated original-reference results match exactly between the new/pilot comparisons. All raw normal cases and summaries are in `fresh-minute`, `fresh-long`, `prior-fresh-minute` and `prior-fresh-long`; [analysis implementation](../../../../../tools/analyze_bank_study.cjs) is at repository `tools/analyze_bank_study.cjs`.

**Promotion decision:** preserve the candidate separately and keep the original Godot policy active. Strong challenge-set recovery does not erase the unresolved long-run cases. There is no automatic model installation, live training UI, new creature anatomy or claim of all-current swimming in this change. The new failure identities are recorded for subsequent development; any further candidate selection needs new untouched evaluation data.

## Verification and commands

All **164 native tests pass, zero skipped**: [regression](regression.log), [curriculum checks](curriculum-tests.log). Tests cover frozen motor, unchanged underwater food, rejected invalid recovery food, full-state save/restore and exact next-update continuation. All seven protected policies/full saves compare unchanged: [hash record](preserved-inputs.json). `deep_physics.h` and the deployed bundle match the preceding commit. No production scene/UI changes required rendered testing. GPU optimization remains parked in the saved backlog.

Example new phase using the archived valid pools (choose a fresh output directory):

```powershell
cmake --build --preset release --target pond_bank_study pond_progress_train pond_current_eval cudarl_tests --parallel 6
build/Release/pond_progress_train.exe --baseline docs/experiments/swimmer/gentle-currents/frozen-policy.bin --import-save build/pond-progress-study/seed-71/training.bin --bank-pool docs/experiments/swimmer/gentle-currents/bank-recovery/data/0xBAA71001/pool.bin --bank-pool docs/experiments/swimmer/gentle-currents/bank-recovery/data/0xBAA71002/pool.bin --output build/pond-bank-repeat/seed-71 --updates 256 --eval-every 64
```

To resume a saved new phase, omit `--import-save`; the full save carries its curriculum. Updates are **additional**; `--updates 0` performs evaluation/save without learning and does not mean unlimited. Use a copy/new output directory for zero-update verification. The read-only profiler now accepts checkpoint version 2 for a future authorized performance pass; no new optimization benchmark was run here.

The current two-controller plan is retained: goal/food choice and swimming/body control. A future body controller may be expanded/fine-tuned for limbs, with corresponding physical changes, but no limbs or new abilities are implemented in this step. This phase covers recovery in steady uniform horizontal currents through 1 cm/s. Changing directions, spatially varying water and broader motor adaptation remain future bounded stages.

Full new learner states stay local under ignored `build/pond-bank-study/seed-*`; datasets, selected weights and small evidence are archived here. Training/evaluation helpers are stopped. These changes and the prior fixed-target diagnostic remain uncommitted on `codex/portable-pond`; prior checkpoint `2af8c6d` remains intact. No merge or push was performed.
