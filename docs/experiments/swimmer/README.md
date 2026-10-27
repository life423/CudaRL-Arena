# Swimmer learnability experiment — October 26, 2026

The user authorized a small experiment: can an initially untrained swimmer learn to reach food quickly enough to watch? They subsequently requested a visible viewer, controlled body comparisons, more energetic movement, and a classic tadpole as the first recognizable body. The baby eel was subsequently implemented and compared below; the proto-amphibian remains proposed. After viewing both, the user chose the tadpole. The intended experience should be 3D, but this experiment remains planar.

## Baby eel follow-up

The baby eel uses the shared three-link solver with three equal 0.6-unit slender segments, two independently controlled joints, and no broad membrane. Its width scales resistance at all segment quadrature points. Both bodies have total length 1.8, 16 rad/s motors, the same policy architecture, food positions and training budget. No traveling-wave gait is prescribed to the policy.

Both bodies were rerun for 300 updates across seeds 1–3. The eel reached 638/768 comparison goals (83.1%), versus 622/768 (81.0%) for the tadpole. The mean of seed-level successful-trip medians was 3.88 seconds versus 3.13. Eel head travel per equivalent distal-joint stroke was 0.636 versus 0.930; simulated work/travel was 5.06 versus 8.33. Rearward-target alignment coverage was 52.2% versus 47.5%, with median first-alignment times averaging 2.5 versus 0.6 seconds among aligned attempts. This limited comparison does not establish general superiority. Raw follow-up rows are [eel.csv](eel.csv) and [eel-tadpole-reference.csv](eel-tadpole-reference.csv).

Open `res://experiments/swimmer_compare.tscn` in Godot and run that scene, or launch:

```powershell
C:/Users/aiand/Godot/Godot.exe --path godot-project res://experiments/swimmer_compare.tscn
```

It loads the saved seed-1/update-300 tadpole from `build/swimmer-eel-comparison/body-6` and trains a fresh eel with 1,024 environments. Both panels display recorded policy evaluations using identical food placements and shared playback times; live training is explicitly distinguished from recorded evaluation. Once training ends, both policies are seed 1/update 300. Show untrained compares both update-zero policies. Selecting a pond toggles the paired-start overview; keyboard arrows choose six matched starts. A fresh checkout needs the tadpole reproduction command below, using output directory `build/swimmer-eel-comparison/body-6`.

The body selector includes the baby eel as ID 7. Six native tests pass across all eight body variants. Comparison renders pass at desktop/tablet/phone window dimensions, including matched food/seed/body checks and mouse/touch/keyboard handlers; physical touch hardware remains untested.

## What exists

- A planar three-link swimmer with two independently controlled tail joints.
- Shared C++/CUDA resistive-water physics for the training environments and CPU previews. There is no external forward-thrust or steering action; policy outputs prescribe internal joint velocities.
- PPO with the project's own Gaussian policy kernels, GAE, MLP, gradients and Adam. No Python, PyTorch, cuBLAS or external ML framework is used by this tool.
- 1,024 unseen CUDA swimmers pool experience into one policy. Six independent CPU previews use the latest policy while training runs.
- A standalone Godot experiment scene with fresh training, untrained comparison, body selection, and pond enlargement through mouse, touch or keyboard. This remains separate from the two-demo startup menu.

The classic tadpole has a tapered head whose sampled width changes head resistance, plus a broad distal membrane whose normal resistance is increased from 2 to 4 relative to tangential resistance 1. The renderer uses the corresponding tapered outline and membrane. A regression test verifies that changing membrane resistance changes motion and simulated work under identical joint actions. These are explicit simplified drag assumptions, not coefficients derived from CFD or a calibrated biological animal.

## Controlled results

Each body was trained from random weights for 300 updates, with seeds 1–3, 1,024 environments, 64 rollout steps, four epochs, eight minibatches, a 9→32→32→2 actor and a constant 0.001 learning rate. Food starts 0.75–1.5 simulation-length units away at random bearings; success means the head center reaches radius 0.18 within 40 simulated seconds. Observations include exact relative food position, joint angles/rates and body velocity. This tests control with known food location, not learned vision or blind foraging.

The comparison uses the same 256 starting conditions per training seed, independent of training resets. The 128-condition validation set is separate. Both sets have been inspected during development, so they are comparison sets held out from training, not an untouched final generalization benchmark.

| Measure | Original, 4 rad/s motors | Original, 16 rad/s motors | Classic tadpole, 16 rad/s motors |
|---|---:|---:|---:|
| Food-reaching success, across 3 × 256 attempts | 76.7% | 82.8% | 81.0% |
| Mean successful trip, seconds | 13.94 | 4.32 | 3.61 |
| Mean of per-seed successful-trip medians, seconds | 13.50 | 3.82 | 3.13 |
| Head travel per equivalent tail stroke | 0.684 | 0.701 | 0.930 |
| Simulated mechanical work per head travel unit | 2.346 | 8.172 | 8.330 |
| Initially rearward targets: aligned within 15° | 41.8% | 49.7% | 47.5% |
| Mean of per-seed median first-alignment times, seconds | 4.33 | 1.47 | 0.60 |
| Mean attempt time, including failures at the 40 s limit | 20.02 | 10.45 | 10.53 |

Except for the pooled success count, table entries average seed-level statistics. Median times include successes only; timeout-inclusive attempt time is included to avoid hiding failed attempts. Turning measures first alignment of heading with the current food bearing for starts where food was over 90° behind the swimmer, sampled every 0.2 seconds. Coverage and time must be interpreted together.

An equivalent tail stroke is accumulated absolute distal-joint travel divided by four times the joint limit, i.e. one full-range outward-and-return cycle. Travel is head path length, not net forward displacement. Work integrates the model's hydrodynamic dissipation; work/travel is in relative simulation units, **not GPU energy, electricity consumption, or calibrated biological joules**. It is a diagnostic; no stamina mechanic or hardware energy budget has been added.

At update zero, deterministic policies reached zero of 256 test goals for every reported seed. The sampled untrained policy reached zero in all original/fast-original trials; the classic tadpole reached one of 256 in one of three seeds. Successful behavior therefore does not depend on feeding a reference gait into training. The sinusoidal gait in the tool is only a separate propulsion sanity check.

Earlier exploratory variants included a small head/long tail and broad-fin versions with 4 or 8 rad/s motors. The longer-tail design was slower, around 19 seconds per successful trip. More energetic strokes were the clearest improvement. Faster motors use extra integration substeps, keeping the maximum joint-angle advance per physics substep unchanged.

Raw rows, including every recorded validation checkpoint and untrained baseline, are in [body-0.csv](body-0.csv), [body-5.csv](body-5.csv) and [body-6.csv](body-6.csv). The comparison is small and about one in five classic-tadpole attempts still fails. The validation target of two consecutive evaluations at 80% is not established as achieved. This experiment does not establish swimming-to-land transfer, limbs, biological evolution, arbitrary body generalization or 3D locomotion.

## Watch and reproduce

From the repository root:

```powershell
cmake --build --preset release --target swimmer_experiment cudarl_tests
build/Release/cudarl_tests.exe swimmer:
C:/Users/aiand/Godot/Godot.exe --path godot-project res://experiments/swimmer.tscn
```

The viewer defaults to the classic tadpole and starts a fresh 300-update run. Its explicit 50 ms/update viewing delay is not a benchmark result. Select a pond to enlarge it; select again to return. Keyboard focus on the ponds supports Enter and Left/Right. Show untrained replays the actual update-zero policy evaluation. After training, the viewer clearly labels recorded evaluations and loops each completed attempt with a one-second hold; failures retain their full recorded timeout. Replayed successes do not increment evaluation statistics.

Run an unthrottled comparison, sequentially on one GPU:

```powershell
build/Release/swimmer_experiment.exe build/swimmer-reproduction/original 300 3 1024 0 0
build/Release/swimmer_experiment.exe build/swimmer-reproduction/fast-original 300 3 1024 0 5
build/Release/swimmer_experiment.exe build/swimmer-reproduction/classic 300 3 1024 0 6
```

Arguments are output directory, updates, seed count, environment count, viewing delay in ms, and body ID. IDs 0–6 correspond to the viewer's choices. Each output directory receives CSV results, configuration, actor parameters and recorded trajectories. Viewer runs use unique directories under ignored `build/swimmer-live/`.

`training_seconds` sums complete synchronous PPO update wall times. `wall_seconds` begins before trainer construction and includes evaluations/readback; it excludes the preceding CUDA probe and physics sanity checks. Measurements were made on Windows with an RTX 5070; an existing viewer was open during the body comparisons. These times are not isolated GPU phase timings or a new optimization benchmark. The new swimmer tool has not had a dedicated GPU optimization pass.

## Verification and review

Release builds pass strict warnings. Six focused native tests pass: passive bodies, rotation/translation invariance, reciprocal-stroke behavior, success versus timeout semantics, real membrane effects, and CPU/GPU agreement including terminal observations across all seven body variants. The Godot render test passes at 1280×720, 768×1024 and 390×844, with recorded data, target-size/overflow checks, untrained comparison, pond enlargement and mouse/touch/keyboard handlers. The visible live run produced native previews and results without reported Godot errors. Physical touch hardware and assistive technology were not tested.

Internal prototype review scores are conservative heuristics, not measured user outcomes: usability 8, hierarchy 8, accessibility 7, consistency 8, responsiveness 8, feedback 7, performance 7. Prioritized next improvements are clearer food-collection/reset feedback, assistive-technology testing, stronger visual distinction between live preview and replay, and measured frame/startup costs. Existing frame renders fit representative sizes; actual device testing remains separate. The interface is an experiment viewer, not a finished portfolio game.
