# Six-order food-input bias test — October 26, 2026

The user approved a controlled test of input-order bias before changing training. This study evaluates a frozen copy of the live saved-best controller at **update 14,125**, training seed 1. Its stored GPU comparison score is 13.609375 food/minute. The exact [checkpoint](checkpoint.json) and its SHA-256 [manifest](manifest.json) are retained. No optimizer updates were performed, and neither the live policy nor its saved-best file was overwritten.

## Controlled intervention

The three physical food identities are called A, B and C for this report. `BAC` means that the neural network receives B's coordinates first, A's second and C's third. Only the first six observation entries are permuted. Joint angles, joint rates and body velocity remain identical. Each order stays fixed for the entire episode.

Crucially, the physical food arrays, placement counters and RNG streams are never reordered. Initial poses, food locations, world seeds, physics, rewards and collection/respawn rules match across the six versions of a case. Once their decisions differ, subsequent trajectories and food placements can diverge naturally because collection and body-clearance rejection depend on those decisions. The test does not claim identical future food placements after different actions.

Two suites use the same frozen actor and deterministic mean actions:

- **Comparison:** the original 64 cases, seed `0x51A7E123`.
- **Fresh layouts:** 256 additional cases, seed `0x9E11B1A5`, separate from the fixed comparison and training worlds. These have now been inspected as development evidence.

Each case runs for 1,200 control steps (60 simulated seconds), counting every collection, including zero-collection results. There are **1,920 one-minute episodes** plus eight repeatability controls. Replays use the same C++ CPU policy inference and shared articulated physics as the visible tadpole. This avoids interfering with the running GPU learner. This is not a throughput benchmark.

## Results

For the 256 fresh cases per order:

| Input order | Least-collected physical food | Mean food/min | Paired difference from ABC | Approx. 95% interval for difference |
| --- | --- | ---: | ---: | --- |
| ABC | A | 13.4141 | 0 | Reference |
| ACB | A | 13.3477 | -0.0664 | [-0.1634, +0.0306] |
| BAC | B | 13.3711 | -0.0430 | [-0.2275, +0.1415] |
| BCA | B | 13.3945 | -0.0195 | [-0.1970, +0.1579] |
| CAB | C | 13.3438 | -0.0703 | [-0.2470, +0.1064] |
| CBA | C | 13.3945 | -0.0195 | [-0.1979, +0.1589] |

Across those 1,536 fresh-layout episodes, collections by neural input entry were:

| Input entry | Food collected | Share |
| --- | ---: | ---: |
| First | 163 | **0.79%** |
| Second | 10,142 | 49.36% |
| Third | 10,243 | 49.85% |

The first input supplied the **first food collected in only 2 of 1,536 episodes**. In every one of the 256 fresh physical starting scenes, at least one alternative input order changed which physical food was collected first. All variants collected some food, so that statistic is not inflated by changing failure into success.

On the exact same unstepped physical starts, the five alternative orders changed at least one clipped actuator command by more than 0.1 in **210–242 of 256 cases** (82.0–94.5%, depending on order). Actual torque commands are clipped to [-1, 1], so the diagnostic measures commands the physical model receives rather than differences hidden by saturation. Raw and clipped actions are retained separately.

## Interpretation

**This frozen controller has a strong arbitrary input-order bias.** The neglected physical food follows whichever piece is placed in the first neural input entry. Its neglect is not explained solely by that physical food's position. The network might still use that entry's coordinates indirectly; the evidence concerns collection behavior, not a claim that those weights are zero.

**The test did not detect a clear mean collection-rate advantage from changing input order.** Means span only 13.3438–13.4141 food/minute, and every paired interval includes zero. The intervals use paired case differences with a normal approximation and are unadjusted for multiple comparisons. This is not a formal equivalence test, nor evidence that the strategy is optimal. A controller can change which food it neglects while retaining a similar average rate.

The original 64-case CPU identity replay scored 13.46875 food/minute versus the saved GPU evaluation's 13.609375. Tiny CPU/GPU inference/integration differences can change collection/respawn events over a long trajectory. All six versions in this study use the same CPU arithmetic, so the input-order comparison is paired within one implementation; these absolute rates should not be substituted directly into the live GPU graph.

This is one trained snapshot from one training seed. The 320 distinct physical starts do not represent 320 independent training runs. The test supports the user's observation of neglected opportunities, but it does not diagnose every visible near miss or quantify an optimal food/minute ceiling.

## Checks and artifacts

- 13,440 body-observation scalar comparisons passed unchanged.
- Every reordered observation retained all three original food-coordinate pairs exactly once.
- Eight identity replays matched collection count, first physical choice and final position exactly.
- The actor's parameters were unchanged after the study; the frozen checkpoint's SHA-256 still matches the manifest.
- The live Godot process stayed open and responsive throughout.

Raw files: [per-case outcomes](cases.csv), [summary and paired intervals](summary.csv), [one-step action probes](action-probes.csv), and [invariant checks](invariants.txt).

Reproduce with the frozen checkpoint, from the repository root:

```powershell
cmake --build --preset release --target food_order_bias
build/Release/food_order_bias.exe docs/experiments/swimmer/food-order-bias/checkpoint.json build/food-order-bias-reproduce 256
```

The appropriate next experiment is **shuffled-input training**, retaining PPO, architecture, physics and food rules while comparing held-out food/minute and slot dependence under equal training budgets. More even slot use alone is not the objective. No shuffled training, new network or algorithm change was implemented as part of this test.
