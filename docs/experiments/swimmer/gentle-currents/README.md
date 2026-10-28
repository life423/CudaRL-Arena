# Frozen-policy gentle-current test — October 27, 2026

The user requested measuring the trained tadpole under gentle currents before further training, while preserving the original controller for comparison. This is a headless evaluation, not an adaptation/training run. The Godot scene remains in calm water.

## Frozen baseline

`frozen-policy.bin` is a byte-for-byte copy of the packaged controller, SHA256 `D9730ED4FB2A05904E09ACAB45C43BE47FF7B80B768AA5379C463A11B8D9611D`. It contains the selector at recovery phase update 250 and motor at update 64,350. The original packaged file remains in place. Record hashes of the full local training saves before/after; the evaluator never opens those saves. No new controller will overwrite this baseline.

## Protocol fixed before the sweep

- Calm water and 0.5/1.0 cm/s currents in eight horizontal directions: ±X, ±Z and four diagonals, normalized to equal speed. These strengths are modest relative to the several-cm/s swimming observed in the current demonstration; they are not a biological validation threshold.
- Two new layout seeds `0xC071A101` and `0xC071A102`, 64 environments each, 60 seconds per environment: 128 worlds per condition and 2,176 trials across 17 conditions.
- Same initial resting bodies, orientations and food layouts for each seed/environment across conditions. Later respawns may diverge because they depend on collection history and the body's position.
- Frozen neural weights, existing observations, greedy food selection every 20 control steps and motor output every 50 ms. No current sensor is added. Both networks and shared 5-ms physical substeps run on CUDA. No exploration, gradients, episode resets or deployed recovery intervention.
- Primary measure: mean food/minute and paired difference against calm water, with approximate per-condition 95% intervals. These are not adjusted for testing multiple directions and do not represent multiple training seeds.
- Additional measures: world-relative and water-relative speeds, time below 0.5 cm/s in each frame, time after ten seconds without food, maximum food gap, worlds with gaps exceeding 20 seconds, and zero-collection worlds. Passive downstream drift must not be called learned swimming.
- After the sweep, select the lowest-collection direction at each strength, then run these two directions plus calm water on new seed `0xC071A201`, 64 uninterrupted five-minute worlds each. This diagnoses sustained failures in selected conditions, not all possible currents.

## Physical change and limits

The current is an imposed uniform world-space water velocity in SI m/s. At each distributed drag sample, drag depends on link velocity **relative to water**, and the semi-implicit equation applies the matching fluid-velocity force term. It does not directly translate the body or increase its actuator strength. The zero-current force path, contact geometry, buoyancy, masses, viscosity, food sampler and serialized world structures remain unchanged. Food targets stay stationary until collected to isolate swimming/control robustness.

This remains the existing reduced-order drag model. A uniform imposed flow in a bounded basin is a diagnostic disturbance, not CFD or a mass-conserving closed-pond circulation simulation. It excludes turbulence, waves, vortices and food drift. No claim about real-animal hydrodynamics follows from the result.

New native checks cover passive advection, drag invariance when body/water receive the same velocity boost, explicit-zero/default equivalence and CPU/CUDA agreement with current. Existing calm-water regression checks must pass before interpreting results.

## Reproduce

```powershell
cmake --build --preset release --target pond_current_eval cudarl_tests --parallel 6
.\build\Release\cudarl_tests.exe "pond currents"
.\build\Release\pond_current_eval.exe --policy docs/experiments/swimmer/gentle-currents/frozen-policy.bin --output build/pond-current-study/sweep
```

The evaluator writes per-world `cases.csv`, aggregate `summary.csv`, and `protocol.txt`; it verifies that device weights and the input policy payload stay unchanged.

## Results

The 2,176-trial sweep completed with finite physical states and unchanged network weights on RTX 5070. The table averages the eight directions equally at each nonzero strength; each direction has 128 matched one-minute worlds.

| Current | Food/minute | Change from calm | Range across directions |
|---|---:|---:|---:|
| Calm | 17.46875 | — | — |
| 0.5 cm/s | 16.93652 | −3.05% | 16.66406–17.11719 |
| 1.0 cm/s | 14.24121 | −18.48% | 13.67969–14.73438 |

Every trial collected some food. There were no food gaps exceeding 20 seconds among the 1,024 trials at 0.5 cm/s; three of 1,024 trials at 1.0 cm/s exceeded that gap. This does not establish absence of stalls over longer runs. All eight direction-wise collection differences at each strength were negative; per-direction paired intervals are in [the sweep summary](sweep/summary.csv), not an aggregate corrected significance test.

Both strengths were weakest in direction −X/−Z. Fresh five-minute confirmation (64 new worlds per condition) measured:

| Condition | Food/minute | Paired difference vs calm, approximate 95% interval | Worlds with food gap >20 s |
|---|---:|---:|---:|
| Calm | 17.78125 | — | 0/64 |
| 0.5 cm/s, −X/−Z | 17.00000 | −0.78125 [−0.96701, −0.59549] | 0/64 |
| 1.0 cm/s, −X/−Z | 13.85938 | −3.92188 [−4.35451, −3.48924] | 2/64 |

The important failure is **seed 3228672513, environment 24, 1.0 cm/s toward −X/−Z**, velocity `(−0.007071067579, 0, −0.007071067579)` m/s. It collected only eight foods in five minutes, had a **265.6-second food gap**, and spent 252.65 seconds below 0.5 cm/s ground speed. Its low water-relative-speed time was only 0.1 seconds: it was largely stationary in the pond while water moved past it. This is a prolonged control/physical stall, not evidence of successful station keeping. Environment 31 had a separate 22.9-second gap while still collecting 73 foods. Preserve these cases when assessing future candidates.

**Interpretation:** the saved controller already tolerates these 0.5 cm/s uniform disturbances fairly well, with a small measured collection penalty. At 1.0 cm/s there is a substantial average loss and an exposed long stall. That supplies a concrete robustness problem before any new training. The tests do not identify whether the failure comes primarily from motor control, target selection, contact with the basin or their interaction. Diagnose that case before choosing which network to adapt; any future candidate must retain calm-water performance and be evaluated on additional untouched current layouts/directions.

Raw records: [one-minute cases](sweep/cases.csv), [five-minute cases](confirmation/cases.csv), [confirmation summary](confirmation/summary.csv), [sweep protocol](sweep/protocol.txt), and [preservation verification](verification.json). Both full local training-save hashes and the original packaged-policy hash remained unchanged. There were **zero optimizer updates to the evaluated controller**. The independent seed-7 calm GPU reference still collects 18 foods, matching its pre-current result.

The strong-current confirmation was repeated independently; all calm/strong per-world CSV rows matched exactly, including the prolonged stall. Full native regression passed **157/157 tests with GPU required and no skips**. The portable CPU reference also passed, including the pre-current 80-step trace and 18-food minute. These are physics/controller checks; no Godot rendering or UI was changed by this experiment.

Reproduce the five-minute confirmation:

```powershell
.\build\Release\pond_current_eval.exe --policy docs/experiments/swimmer/gentle-currents/frozen-policy.bin --output build/pond-current-study/confirm-copy --seed 0xC071A201 --trials 64 --seconds 300 --conditions calm,cm05_nx_nz,cm10_nx_nz
```
