# Shuffled food-input training — October 26, 2026

Follow-up: the live learner's frozen update-17,425 actor subsequently outperformed the original update-29,125 actor on **512 new one-minute worlds** (15.10 versus 13.63 food/min) and **64 five-minute worlds** (15.63 versus 14.02). See [the paired fresh-layout report](fresh-layouts-update-17425/README.md) for exact checkpoints, confidence intervals, occasional failure cases and uninterrupted-training details. This later comparison does not replace the equal-budget pilot below.

The user approved a single intervention: randomize the order of the three food-coordinate pairs during training, retain PPO and the existing physical model, compare with today's saved best and provide a Godot view.

## Frozen reference and watchable experiment

`manifest.json` identifies the exact starting reference: update **29,125**, seed 1, **13.828125 food/min** on the existing 64 fixed one-minute comparison ponds. Its SHA-256 is recorded; the frozen playback copy is `godot-project/pond/policies/shuffle_reference.json`. This is the strongest saved policy at the start of this task, newer than the update-14,125 checkpoint in the earlier six-order bias experiment.

Run the Godot project and choose **Tadpole Pond → Compare shuffled**, or run `res://pond/shuffle_compare.tscn`. The left pond uses only the frozen CPU actor, with no background trainer. The right pond starts fresh with 1,024 CUDA environments, continuously applies the latest actor and saves its strongest evaluated actor separately at `user://pond-policies/shuffle-v1/best.json`. **Watch candidate best** switches the right animal to that saved actor while training continues; **Watch latest learner** switches back. Switching preserves the animal's body, momentum, food and clock. Pause learning pauses optimizer updates while both visible animals continue swimming. Maximum speed removes training pacing; disabling it selects 10 updates/second.

Both worlds start with seed 1 and the same food/physics rules. Their actions and subsequent food sequences naturally diverge. The main graph compares canonical evaluation scores, not the two individual live food counts: the original best is a constant dashed line, and the amber line follows the fresh learner's latest evaluations. The right numeric best is retained separately. A later update can score worse than an earlier one. The frozen reference has a large training head start; this view alone cannot establish whether augmentation is superior at equal training budgets.

On narrow windows, the Saved best / Shuffled learner tabs select the visible pond while both CPU simulations continue. Short windows scroll. The normal Watch it learn action still starts the original canonical-input experiment and retains its separate checkpoint.

## Precisely what changed

`SwimmerConfig::shuffle_food_inputs` defaults to false. For the three-food task, a counter-based CUDA kernel picks one of six pair permutations independently for each environment/control tick using a separate RNG key. It modifies only the first six observation scalars. Seven body observations, physical food IDs, placement counters, action sampling RNG and environment states are untouched.

The permuted observation is copied into the rollout tape **before actor and critic forward passes**. All four PPO optimization epochs use that same tape alongside its original actions and log probabilities. Minibatches never reshuffle food pairs: doing so would compare likelihoods of different observations. Last-state and time-limit critic bootstraps use the next tick's augmentation. Canonical evaluation and visible inference retain the original stable ordering. This encourages learned order robustness; it does not make the MLP mathematically permutation invariant.

PPO clipping, entropy, GAE, optimizer, rollout/minibatch sizes, network architecture, reward, motor commands, inertial dynamics, food placement and horizon remain unchanged. This is native C++/CUDA training; no Python/PyTorch dependency was introduced. The new optional checkpoint `training_variant` field describes how an actor was trained; older canonical checkpoints remain compatible.

## Controlled pilot

Command (from repository root):

```powershell
build/Release/shuffle_experiment.exe docs/experiments/swimmer/shuffled-input godot-project/pond/policies/shuffle_reference.json 1000 2
```

Each fresh variant receives **1,000 updates = 65,536,000 environment steps**, with seeds 1 and 2. Selection uses strict improvements on the existing comparison suite every 25 updates. The selected actor is then evaluated on 64 separate one-minute ponds seeded `0x9E11B1A5`. The reference is also evaluated on those same separate ponds. Neither actor-only playback checkpoint is used to warm-start an optimizer: saved files lack critic/Adam state. Raw curves and scores are in `learning.csv` and `scores.csv`; JSON files contain each selected actor.

| Variant | Seed | Selected update | Comparison food/min | Separate 64-pond food/min |
| --- | ---: | ---: | ---: | ---: |
| Canonical, fresh | 1 | 650 | 12.09 | 12.00 |
| Shuffled, fresh | 1 | 775 | 7.92 | 7.61 |
| Canonical, fresh | 2 | 925 | 11.39 | 11.31 |
| Shuffled, fresh | 2 | 850 | 11.58 | 11.75 |
| Frozen existing best | 1 | 29,125 | 13.83 | 13.52 |

The result is mixed across seeds and worse on average for shuffling at this budget. Two seeds are insufficient for a reliable general ranking, and neither fresh shuffled actor beats the frozen reference. The live viewer is a separate, indefinitely continuing seed-1 run and may subsequently exceed these pilot checkpoints. Do not substitute its later score into this equal-budget table.

## Repeating the six-order diagnostic

`food_order_bias` replays all six fixed input orders for each checkpoint, without learning, on 64 comparison and 256 separate physical worlds. Outputs in `bias-reference/`, `bias-seed-1/` and `bias-seed-2/` include per-world results, aggregate slot counts, action probes and invariants. This diagnostic uses CPU mean-action inference, matching visible playback; small CPU/CUDA trajectory differences mean its scores should be compared within this replay method.

| Actor | Collections attributed to input slots 1 / 2 / 3 | Mean food/min across six orders, 256 worlds | Worlds whose first physical choice changes with order |
| --- | --- | ---: | ---: |
| Frozen reference | 0.61% / 49.68% / 49.71% | 13.64 | 256 / 256 |
| Shuffled seed 1 | 38.01% / 37.66% / 24.34% | 8.18 | 225 / 256 |
| Shuffled seed 2 | 30.23% / 34.32% / 35.45% | 11.70 | 192 / 256 |

Shuffled training substantially reduces the observed neglect of the first input slot. It does **not** eliminate order sensitivity, guarantee shortest paths or establish a food-rate improvement. Eight identity repeats and 13,440 body-observation scalar checks pass for each actor; neither weights nor physical food placement rules change during replay. A stronger order-independent architecture or different learning intervention remains a separate decision, not part of this change.

## Validation and remaining limits

All **136 native tests pass, zero skipped**. The new native GPU test checks all six permutations, pair integrity, unchanged body observations, deterministic augmentation, retained rollout observations, recomputed behavior log probabilities and finite time-limit bootstrap values after crossing the real episode horizon. The existing CPU/CUDA physics parity and evaluation-isolation checks remain in place.

The Godot comparison smoke test passes **92 checks**, including independent playback/training worlds, 1,024 shuffled environments, training-variant metadata, frozen-file isolation, measured curve values, pause/resume, best/latest switching without teleporting, camera bounds for each animal and all food at four window sizes, menu round trips and stopping the worker on exit. The menu regression passes **102 checks** and checkpoint failure handling passes **14 checks**. The normal live-pond regression passes **187 checks**. Its first run under simultaneous full-suite/live-viewer GPU contention missed update-count deadlines; rerunning after the full native suite completed passed. The test now polls update publication with bounded waits rather than assuming a new update will appear within 150 ms. Screenshots inspected at 1280×720, 768×1024, 390×844 and 844×390; initial render-target allocation errors were corrected by rendering each embedded pond directly into its owning viewport. Short-landscape scrolling and the narrow-window chart legend remain usability limitations rather than evidence of mobile-device validation.

Internal UI review: hierarchy 9, consistency 9, feedback 9; usability 8 (best/latest button describes the next action and needs careful reading), accessibility 8 (keyboard focus, text labels, dashed baseline and 44 px targets checked; native assistive technology untested), responsiveness 8 (short landscape requires scrolling; physical devices untested), performance 8 (two visible worlds and augmentation are functional, but comparative frame-time overhead is unprofiled). Priorities are clearer persistent best/latest view state, assistive/device checks and measured frame times, without expanding gameplay scope. The shuffled policy has not proven better or reached an established optimum.
