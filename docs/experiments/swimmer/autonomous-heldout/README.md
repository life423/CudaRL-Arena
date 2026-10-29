# Frozen before/after comparison: no clear collection-rate improvement

The [predeclared protocol](protocol-before-run.md) compares saved validated motor **4,688** (learner age4,751, before autonomous practice) with saved validated motor **9,216** (learner age9,218, after additional practice). These are frozen snapshots taken at study start, not whatever actor the continuing live learner adopts afterward. Both use the identical inherited food selector. No training, policy installation, live save write or learner interruption is performed by the evaluation tool.

## Result

Across **256 two-minute trials**, 32 paired cases per subgroup, there is **no clear overall collection-rate improvement**. The overall mean changes 15.328→15.352food/minute (+0.153%); the paired seed-cluster95% interval is −0.074to+0.129food/minute and includes no change. Every subgroup's interval also includes no change. No subgroup mean loss exceeds the predeclared5% margin, and the individual intervals support bounded retention against that margin; these are not simultaneous/multiplicity-adjusted guarantees.

| Task | Before food/min | After food/min | Change |
| --- | ---: | ---: | ---: |
| Earlier calm/static task | 18.234 | 18.531 | +1.63% |
| Calm sinking food | 14.531 | 14.531 | 0.00% |
| Steady current | 14.156 | 13.984 | −1.21% |
| Fluctuating current | 14.391 | 14.359 | −0.22% |

Both actors have **zero confirmed five-second food-present/no-collection immobility events**, zero invalid body/food observations and zero time at the16-pellet practice cap in this suite. This does not establish arbitrary-water robustness or settle previously retained difficult bank starts. All128trials/actor collect at least one food. See [full timing/reliability summaries](summary.md) and [raw cases](cases.csv).

An exploratory food-supply diagnostic finds the three sinking-food groups collected **2,757/2,857createdpellets (96.50%) before**, versus **2,744/2,857 (96.04%) after**. This includes initial and late-arriving pellets, some of which may remain in transit at the cutoff. It suggests the mild arrival schedule leaves limited room for food-count improvement; it does not prove optimality or explain every residual miss. The slight overall gain is driven by the old static-food task and must not be presented as a demonstrated gain in dropped-food interception.

## Verification and reproducibility

- Checkpoint framing/checksum, actor dimensions/finite weights and common selector/motor provenance are checked. Full private snapshots remain in ignored `build/autonomous-heldout/20261029043654502`; compact frozen actor exports and their final SHA-256 values are included here. Neither original snapshot changes during evaluation.
- Identical-policy inference on a separate development seed is bit-identical. Frozen actor GPU weights are checked afterward. Per-case accounting validates `foods + final_food_count == created_foods`.
- A separate passive-monitor positive-control test passes25assertions across six synthetic worlds: stationary feeding targets trigger a five-second event; empty resting, movement and repeated collection do not; invalid food/body states are detected. This calibrates the zero-event findings.
- A one-tick completed-wait measurement was corrected after review. Exactly the same frozen actors/seeds were rerun into a new folder; all food, stall, validity, creation and first-food outcomes remain identical. Only the corrected output is reported here. The initial output remains preserved under ignored build output. The runner refuses to overwrite existing evidence.
- Node analysis validates all256rows/pairs/groups/censoring and uses10,000deterministic paired seed-cluster bootstrap draws. Synthetic positive/negative/inconclusive and malformed-input checks passed. No result-based actor selection or training is performed.
- CUDA tool builds with the project's strict warnings. This is a diagnostic addition; the living app's physics, networks, PPO, checkpoint format and UI are unchanged by this comparison. The prior185/185GPU-required runtime suite remains separate evidence, not a rerun claim.

To reproduce with the private snapshots:

```powershell
cmake --build --preset release --target feeding_heldout_eval
./build/Release/feeding_heldout_eval.exe --monitor-self-check
./build/Release/feeding_heldout_eval.exe <before-full-save> <after-full-save> <fresh-output-folder>
node tools/analyze_feeding_heldout.cjs <fresh-output-folder>
```

The three sinking-food groups share nominal time-indexed arrival requests. The retention group deliberately keeps the older collection-triggered food placement task, whose realized refills depend on each actor's behavior. Both start from matched seeds and rules; identical realized food histories are not claimed.

## Next direction

The user subsequently proposed a **larger pond with meaningful obstacles**, including a rock with an opening, and clarified that the idea is not limited to rocks. This is now the next design discussion, replacing a default plan to optimize mild open-water feeding indefinitely. Obstacle geometry must affect the native CPU/CUDA collision model and be observable by the neural controller; visible Godot meshes alone are insufficient. Preserve the two-controller architecture, PPO and learned swimming. A larger habitat, obstacle contacts/perception, route choices and later growth are proposed, not implemented by this report. The test does not stop or modify the local swimmer. At the final process check that window is no longer present; only the project manager remains, with an updated learner save and an empty game error log. No evaluator is left running.
