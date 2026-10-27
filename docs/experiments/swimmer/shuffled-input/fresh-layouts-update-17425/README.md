# Frozen-policy comparison on fresh layouts — October 26, 2026

The user requested testing the shuffled best against the original on fresh food layouts, while leaving the existing Godot training run unchanged.

## Checkpoints and protocol

Both actors were frozen **before** testing these seeds. The original is update **29,125**, selected at 13.828125 food/min on the existing comparison suite. The shuffled actor is update **17,425**, selected at 15.34375 food/min. The live learner improved between inspecting its previous best and copying the file; all results here use the copied update-17,425 actor, without replacing it when later live records appear. Both actors and SHA-256 hashes are stored alongside this report in `original.json`, `shuffled.json` and `manifest.json`.

The primary test comprises **512 one-minute worlds**: 256 each from new seeds `0xC73B2A91` and `0x4F092D67`. A separate sustained test comprises **64 five-minute worlds**, seed `0xB5D843E9`. These seeds differ from the training seeds, the fixed comparison suite, and both previously used holdout seeds. Evaluation uses canonical food-input ordering, deterministic mean actions, the existing 13→32→32→2 CPU MLP and unchanged inertial physics/food rules. Neither policy learns during evaluation.

Each pair starts with identical body state and food positions. Each animal then chooses its own actions; collection order and body clearance can make subsequent respawn sequences diverge under the same seeded placement rules. The test does not force identical future food positions or move food to favor a policy. Five-minute trials continue the same physical world beyond the 60-second training episode limit, matching the visible pond's continuous playback.

The evaluator runs as a single CPU process, with no GPU environment or optimizer. Godot's native training worker continued advancing throughout. No pause, reset, rate change, checkpoint promotion, physics change or app restart was performed.

## Results

| Test | Original food/min | Shuffled food/min | Paired improvement | Approximate 95% interval for improvement |
| --- | ---: | ---: | ---: | --- |
| Fresh A, 256 × 60 s | 13.637 | 15.102 | +1.465 | +1.259 to +1.671 |
| Fresh B, 256 × 60 s | 13.621 | 15.102 | +1.480 | +1.243 to +1.718 |
| Combined, 512 × 60 s | **13.629** | **15.102** | **+1.473 (+10.81%)** | **+1.316 to +1.630** |
| Sustained, 64 × 300 s | **14.016** | **15.628** | **+1.613 (+11.50%)** | **+1.477 to +1.748** |

The shuffled actor wins **395**, ties **70**, and loses **47** of the one-minute pairs. It wins **all 64** five-minute pairs. Confidence intervals use the sample variance of paired food/minute differences and a normal approximation (`mean ± 1.96 × standard_error`); they describe variation across these sampled worlds for these frozen actors, not uncertainty across independently trained policies.

This supports better average food collection by this shuffled checkpoint on new layouts. It does not establish a globally optimal route policy or a general advantage across training seeds. The earlier equal-budget 1,000-update pilot was mixed across seeds; the present comparison answers which of these two saved actors performs better, at their recorded training budgets.

## Remaining weaknesses

The gain is not universal. One shuffled one-minute trial collected zero food, versus none for the original. Median time to first food is **3.40 s shuffled versus 3.30 s original**; medians exclude no-food trials, whose counts are reported separately. The 95th percentile of each one-minute trial's longest observed food-free interval is **9.60 s shuffled versus 7.50 s original**. The same metric over five-minute trials is **9.60 versus 7.60 s**. Intervals include initial and end-of-trial waits and are not a pooled distribution of completed inter-meal times.

The shuffled actor collects more on average but still has occasional hesitations/failures. Replaying the losing and zero-food layouts is a concrete next diagnostic before changing PPO parameters or reward weights. `weak-layouts.csv` records the losing primary cases, sorted by food deficit and then longest wait. Training settings remain unchanged.

## Reproduction and validation

```powershell
cmake --build --preset release --target paired_pond_eval
build/Release/paired_pond_eval.exe docs/experiments/swimmer/shuffled-input/fresh-layouts-update-17425/original.json docs/experiments/swimmer/shuffled-input/fresh-layouts-update-17425/shuffled.json build/replayed-fresh-layouts 256 64
```

`cases.csv` records each paired result, seed, world index, horizon, first collection and longest food-free interval; `summary.csv` records aggregate outcomes. `invariants.txt` confirms **24 exact identity replays**, **six exact matches to the visible `ForagingPond` implementation** (both actors at world zero of all three blocks), unchanged actor weights and canonical observations. File hashes matched the pre-test manifest after evaluation. An independent PowerShell calculation from the 512 raw primary cases reproduced both means and the 10.8054% gain.

A separate evaluator control ran the original actor against itself on both fresh seeds and the sustained seed (two worlds per seed). Every pair tied exactly, with zero difference and `[0,0]` confidence intervals. Output is retained in `build/pond-learning-checks/paired-identity-control/`. This task only added the CPU evaluator/CMake target and report artifacts; it did not rebuild the running Godot extension or change the learner.

## Keeping the current run training

The running **Godot game window** owns the native C++ background training thread and its 1,024 CUDA worlds. Keep the game process open, learning unpaused and the PC awake. Maximum speed removes update pacing. Watching the saved candidate best changes visible inference only; training continues. Minimizing does not stop that native worker. The Godot editor/project manager by itself does not run this training session.

Closing the game joins the trainer and preserves the best playback actor. Current persistence does **not** save the critic, optimizer moments, exploration variance or rollout/environment state for resuming full training. Reopening starts a new learner while retaining the best actor for playback/comparison. There is no separate continuously running headless trainer backing this Godot session.
