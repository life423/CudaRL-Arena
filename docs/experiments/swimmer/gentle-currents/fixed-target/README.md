# Fixed-target swimming in currents — October 27, 2026

The frozen deployed motor reached its selected target in **all 4,608 learned-controller trials**, including folded starts and steady currents up to 1 cm/s. Every arrival occurred within **30.35 seconds**. Matched zero-torque controls reached **140/4,608 targets (3.04%)** by both the one-minute and five-minute endpoints. These results support retaining the swimming motor while investigating target choice/recovery; they do not establish arbitrary-water robustness or eliminate the known bank trap.

## What the experiment isolates

The [protocol](PROTOCOL.md) was saved before execution. There are 256 fresh layouts (two seeds × 128 environments), each repeated across two joint-pose classes, nine flow conditions and two control modes: 9,216 total cases. Repeated directions/poses on the same layout are matched conditions, not independent samples. No food-selector inference or training is executed.

The target is physical slot `environment % 3`, promoted using the existing 33-input motor interface and held until collection. The other foods remain in the observation and follow normal collection/respawn rules. This retains the familiar input distribution; it is a fixed-goal test within the three-food pond, not an assertion that the motor is independent of the other food inputs. There were only four incidental non-target collections across the learned trials.

All conditions use the original body, torque limits, drag, contacts and 5-ms physical substeps. Flow is constant in world coordinates, aligned relative to the initial horizontal target bearing. No moving guidance, altered propulsion, teleports, episode resets or target-switch rescue is used. A completed trial freezes at first selected-target collection, so it cannot falsely accumulate post-success stall time.

## Arrival results

Every learned condition has 256/256 arrivals by 60 seconds. Median arrival times at the stronger, 1 cm/s flow:

| Condition | Normal start | Folded start |
| --- | ---: | ---: |
| Calm | 3.10 s | 3.40 s |
| Upstream | 3.85 s | 4.075 s |
| Downstream | 2.80 s | 3.10 s |
| Cross-left | 3.45 s | 3.95 s |
| Cross-right | 3.35 s | 3.60 s |

The slower upstream and faster downstream times are compatible with the imposed-flow task. Arrival alone does not distinguish propulsion from advection, hence the passive control. Passive arrivals occur only in downstream conditions; there are no passive arrivals in calm, upstream or cross-current groups. Even downstream, passive success is only 11.7–15.6% across pose/strength groups, versus 100% for the learned motor.

The five-minute endpoint adds no learned arrivals because all were already successful by 30.35 seconds. It is **not five minutes of continuous learned foraging**: trials stop at first arrival. Unsuccessful passive cases continue to the longer horizon and no additional passive arrivals occur. Continuous multi-food behavior remains a separate test.

## Recovery is still imperfect

Ten learned trials have at least a ten-second span without lowering their best-so-far distance by 0.05 model units; one has a 13.5-second continuous span below 0.5 cm/s ground speed. All recover before the one-minute deadline. The longest no-progress span is 27.3 seconds, in the 30.35-second arrival case (seed 4051620097, environment 23, normal pose, 0.5 cm/s downstream). The longest low-speed span occurs for seed 4051620098, environment 52, folded pose, 0.5 cm/s cross-left; arrival is 23.15 seconds.

These descriptive definitions do not by themselves prove a mechanical stall: turning, drift, contact and an inefficient route can all delay best-distance improvement. They show why 100% eventual arrival is not equivalent to effortless or optimal swimming.

## Interpretation and next decision

The motor can reach fixed targets from these sampled central starts, across all tested current bearings and both pose classes. Immediate motor retraining or changing actuator strength is not justified by a demonstrated persistent failure here. The next proposed AI experiment should concentrate on learned food-choice/recovery near difficult pond states, retaining the motor and physical equations.

The [earlier recorded bank diagnosis](../diagnosis/README.md) remains relevant: holding the original target traps the settled animal, while an alternative target releases it. Those bank states are not represented by merely folding joints at central starting positions. This is a coupled motor/target/contact limitation, and the present test does **not** prove the selector is the only possible cause. The prior progress-aware candidate fixes one development fixture but does not reliably reduce fresh stalls; it remains experimental and undeployed.

A bounded follow-up could compare/train a copy of the progress-aware selector on diverse independently collected slow-progress/bank states, mixed with normal calm/current episodes, then require better fresh long-run recovery without sacrificing collection. That curriculum, new training budget and promotion gate are proposals; no additional training or controller replacement was started by this diagnostic. Stronger, changing or turbulent flows and water-to-land locomotion remain later work. Uniform imposed flow is the existing reduced-order drag model, not CFD.

The earlier bank recording was sensitive to its numerical execution path. This tool shares equations, target-promotion and network weights, but uses its own larger batched evaluator. Its fresh-case results supplement the retained original recording; they do not replace the bank regression or claim bit-identical reproduction of it.

## Evidence and validation

- [Primary per-case data](cases-60s.csv), [primary summaries](summary-60s.csv), [five-minute data](cases-300s.csv), [five-minute summaries](summary-300s.csv), [aggregate](aggregate.json).
- [Execution log](main.log), [frozen-weight verification](verification.json), [all seven protected policy/full-save hashes unchanged](preserved-inputs.json).
- [Focused diagnostic checks](tests.log) validate paired starts, current bearings/speed, exact motor observation promotion, delegation to shared physical stepping, passive zero torque, incidental collection and selected-only stopping.
- [Full native suite](regression.log): **163/163 pass, zero skipped**, GPU required. No shared runtime behavior, Godot UI or scene was changed, so no new rendered verification was necessary.
- Only read-only frozen inference was used. Initial/final complete diagnostic states are local under ignored `build/pond-fixed-target-study/main/initial.bin` and `final.bin`; they are diagnostic snapshots, not training checkpoints. Source policies, original training progress and both earlier candidate full saves are unchanged. No helper is intended to keep running.

## Run again

```powershell
cmake --build --preset release --target pond_fixed_target_eval cudarl_tests --parallel 6
$env:CUDARL_REQUIRE_GPU='1'
build/Release/cudarl_tests.exe 'fixed target'
build/Release/pond_fixed_target_eval.exe --policy docs/experiments/swimmer/gentle-currents/frozen-policy.bin --output build/pond-fixed-target-repeat --trials 128 --seconds 300
```

The output directory must be new, protecting previous evidence. Seeds may be supplied with repeatable `--seed` arguments. GPU inference remains native C++/CUDA without Python/PyTorch. This experiment is uncommitted on `codex/portable-pond`; the preceding validated checkpoint is local commit `2af8c6d`. No merge or push was performed.
