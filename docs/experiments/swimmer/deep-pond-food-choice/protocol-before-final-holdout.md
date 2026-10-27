# Deep pond food selection

The user approved running food-choice experiments headlessly, then loading the strongest validated controller into the existing deep pond. Keep the swimming skill, PPO and shared native physics; no further cosmetic or limb work is included.

## Design

The saved best motor network (update 64,350) remains frozen. It maps 33 inputs through two 32-unit tanh layers to four tail torques. Only food-coordinate inputs are reordered to make a selected physical food the primary target. Physical food IDs/positions, collection radius, drag, masses, constraints, contacts and 5 ms physics substeps are unchanged.

The learned selector uses categorical PPO to choose one of three physical food slots every 20 control steps (one simulated second). It receives actual food collections as rewards, without nearest-food labels, scripted steering or direct propulsion. The motor network continues operating every 50 ms. A collection during the commitment period moves only the consumed food under the existing sampler; the next high-level choice occurs at the normal decision boundary.

The initial shared selector improved one-minute collection, but one of 64 five-minute trials exposed a folded-tail stall. Branching that exact state showed that selecting a different food let the unchanged motor escape; the original selector and nearest-food controller both remained stuck from that state. This is diagnostic evidence, not a deployed rescue rule.

A separate recovery phase transfers the learned selector weights, keeps the motor frozen, and restarts only the experimental selector optimizer/worlds. Training uses five-minute episodes with 50% normal resting joints, 25% near-folded joint configurations and 25% broad resting joint configurations. Foods are sampled after the pose is chosen so placement constraints remain valid. At decision boundaries, no collection for at least 20 seconds plus current speed below 0.5 cm/s produces a -5 terminal penalty and a new **training** episode (no truncation bootstrap). Time limits retain normal truncation bootstrapping. Inference never resets, teleports, forces a target or applies this penalty. These are task/curriculum changes, not fluid or rigid-body changes.

A first 36-input / three-output selector converged to a fixed-slot shortcut. The replacement applies one shared `32 → 32 → 32 → 1` scoring network to each food; its three scores are categorical logits. Per-food inputs contain relative XYZ, 24 body/motor observations, previous-target membership, distance to the food, min/max distance from that food to the others and distance to the nearest other food. Symmetric summaries and shared weights remove slot identity from the policy. This is a learned scoring rule, not a programmed nearest-food decision. An asymmetric critic estimates values from all 36 raw observations, but does not choose actions.

All six food permutations must preserve each physical food's score when the previous-target membership is permuted consistently. Exact score ties use stable slot order; this tie-breaking exception does not grant a general learned slot preference.

PPO: 1,024 worlds, 32 decisions per rollout, four epochs, eight minibatches, learning rate 0.0003, gamma 0.99 and lambda 0.95 **per one-second decision**, clipping 0.2, entropy coefficient 0.01, value coefficient 0.5 and gradient norm cap 0.5. These are selector settings, not changes to the motor's training or physics. Rollout, categorical sampling, GAE, gradients, shared scoring and Adam run in native C++/CUDA. No Python or ML framework is used.

## Evaluation protocol

Compare the exact same saved motor weights under four target-choice modes:

1. Original canonical food ordering.
2. Programmed nearest-food selection every 50 ms.
3. Programmed nearest-food selection every one second (the learned selector's cadence).
4. Learned PPO selector every one second.

Initial layouts/environment IDs match within each trial. Subsequent food placement depends on collection history and the existing body/other-food avoidance rules, so divergent controllers do not necessarily see identical later spawn positions. Evaluate all three physical collection slots, food/minute, speed below 0.5 cm/s, time after at least ten seconds without collection, and target changes before collecting the previous target. Switching after eating the previous target is recorded separately as an ordinary retarget.

Train shared selectors with seeds 11 and 29 for 800 updates each. Choose the checkpoint with the highest score on the fixed 64 × 60-second selection suite (seed `0x6A91B23D`); break ties in favour of seed 11. Development seed 384901731 is diagnostic, not final holdout. Final evaluation uses two new 256-world one-minute suites (seeds 2839471051 and 917362409) and 64 uninterrupted five-minute swims (seed 3481762059). Choose the candidate **before** opening the final holdouts. Extended evaluation disables episode resets; the simulation timestep and training's 60-second episode boundaries remain unchanged.

Those first holdouts evaluated the pre-recovery actor and exposed the stall. They are now discovery/regression evidence, **not fresh holdouts for the recovery phase**. The recovery run uses seed 47 and 800 updates, initialized from seed 11's selected actor with a fresh selector optimizer. Choose the highest fixed-suite score among checkpoints that collect food within 20 seconds from the recorded stall fixture. Then freeze the selected bundle before evaluating two new 256-world one-minute suites (seeds 1098742313 and 3220576417), and two new 64-world uninterrupted five-minute suites (seeds 1130573917 and 2501948063). The prior long seed remains a separate regression suite. The fixture gates checkpoint selection but never enters gradients; mixed training starts use independently sampled layouts/joint poses.

Paired approximate 95% intervals use the standard error of per-world food/minute differences. They describe these sampled layouts, not all environments or training seeds. Two training runs and finite holdouts cannot establish global optimality. Instrumented evaluation copies observations/states to the CPU for detailed reporting, so its wall time is not a training-throughput benchmark.

## Tools and persistence

```powershell
cmake --build --preset release --target pond_choice_eval pond_selector_train

.\build\Release\pond_choice_eval.exe --checkpoint SOURCE_COPY --output REPORT_DIR --seed 384901731 --trials 64

.\build\Release\pond_selector_train.exe --source SOURCE_COPY --checkpoint SELECTOR_SAVE --policy BEST_POLICY --updates 800 --seed 11

.\build\Release\pond_choice_eval.exe --checkpoint SOURCE_COPY --selector BEST_POLICY --output REPORT_DIR --seed 2839471051 --trials 256

.\build\Release\pond_choice_eval.exe --checkpoint SOURCE_COPY --selector BEST_POLICY --output LONG_REPORT_DIR --seed 3481762059 --trials 64 --seconds 300
```

The trainer resumes its own selector save and runs the requested number of **additional** updates. Ctrl+C saves. Full selector checkpoints contain actor/critic, Adam moments/counters, frozen motor weights, world states, RNG counters, selected targets, update count and best actor. Updates autosave every 25 steps; learning metrics append to a sibling CSV. Exported controller bundles contain the best selector and the exact frozen motor together. Checksums and separate format versions reject incompatible/truncated data. The user's original pond checkpoint is never used as the selector's output path.

`--motor-policy BUNDLE` transfers both saved networks instead of requiring the source pond save. With no full selector save, the selector optimizer/worlds start fresh and update numbers start at zero; with a matching full save, all training state resumes and its configuration is read automatically. `--updates 0` runs until Ctrl+C. `--recovery` enables the bounded recovery curriculum above. `--fixture STALL_FILE` gates best-policy selection on actual neural recovery from the recorded state. Candidate exports are separate from the packaged validated demo, so additional training cannot silently replace it.

The deep pond can load a controller bundle for frozen inference. It must identify that mode honestly: no background learning, no active training controls, and a saved validation point rather than a fabricated live learning curve. Removing/clearing its optional controller path restores the original full-resume training route.

## Validation

Native tests cover food promotion, physical state and motor-input preservation, CPU/CUDA physical agreement, all six permutation orders, CPU/CUDA score agreement, unchanged motor weights and bit-identical full resume across an episode timeout. Separate Godot tests exercise actual saved-controller loading, multiple food slots/depths, checkpoint preservation, clear playback status, responsive layout and Back to demos.
