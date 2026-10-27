# Three-food deep pond — October 26, 2026

Latest AI update: **Deep water now plays a validated PPO food selector with the retained swimming network.** Experiments run headlessly; the normal pond identifies saved-controller inference and hides training controls. Collection improved from 15.36 to 17.56 food/minute over 512 fresh one-minute trials, with separate five-minute/recovery checks. See [food-choice results](experiments/swimmer/deep-pond-food-choice/README.md) and [training/resume commands](pond_food_selection.md). Original motor-training saves and the earlier live training route remain preserved.

Latest presentation update: [the natural setting](pond_presentation.md) replaces the exposed bowl/cutaway with continuous ground, clustered shoreline detail and a local occlusion aid. Overview is stationary and higher; Closer view tracks the interpolated world position with a fixed horizon/zoom. The learned swimmer and native physics are preserved.

Presentation update, October 27: [the revised water/camera pass](pond_presentation.md) replaces the ghostly slope with a defined cutaway and water volume. After the user rejected an underwater close-up, the pond now starts with a stationary overview and offers a stable, wider Closer view. Native training and physics remain unchanged.

The user approved bringing the learned three-dimensional swimmer into the established pond, with three food pieces at different depths and no legs. This is implemented in `godot-project/pond/deep_pond.tscn`. The menu's **Deep water** button opens this pond. The one-food tank and shallow three-food pond remain available in the source with their original saves intact.

## Watch and continue training

Choose **Tadpole Pond → Deep water** from the launcher to watch the packaged trained selector. This view runs inference without background learning. Its validation point describes 64 separate fixed one-minute trials, not the visible tadpole's recent rate. Back to demos and camera controls remain available. The menu's Training guide opens separate headless selector training, which resumes its full local save and writes candidates for later evaluation. See [the current workflow](pond_food_selection.md).

**Show untrained policy** switches both swimming and food-choice networks to deterministic random initial weights, using the same orthogonal initialization/output gain as the PPO actors. It runs fixed-policy inference without training or added random-action exploration. Untick it to restore the exact saved trained weights. Each switch restarts the same seed-7 pond layout, resting body state, food counters and display interpolation, so neither controller inherits the other's momentum or progress. This is one repeatable untrained example, not an average over random seeds.

The sidebar labels **Untrained · random weights** and reports the actual CPU/CUDA backend. The bottom plot becomes **Trained controller reference**: its saved evaluation still belongs to the trained policy and does not describe the untrained swimmer. No model file, optimizer/world checkpoint, food sampler or physical equation is changed. The control appears in saved-controller playback; the preserved motor-training route is unchanged.

Comparison verification: 191 rendered checks passed on each of CPU and CUDA, including Space-key toggling, matching initial layouts, identical trained trajectories after restoring, repeatable random trajectories, interpolation reset and model/training-save hashes. Native initialization/physics checks and the existing 374 Godot checks passed. Screenshots were inspected at desktop, tablet and both phone orientations. Internal interface review: usability, hierarchy, consistency and feedback 9/10; accessibility and small-screen data visibility 8/10. Remaining priorities are physical assistive/touch verification and checking access to scrolled statistics on real small screens when that platform work resumes; no redesign is required for this toggle.

The remaining live-training details in this section describe the preserved original motor-adaptation route. Clearing `controller_policy_path` in the deep pond scene restores that route: the first opening transfers the saved 3D swimmer; later openings resume `deep-pond-v1`. It shows best/latest policies while 1,024 worlds train, with pause, speed and save controls. Its headless runner remains `deep_swimmer_train --pond`; it is separate from the new food-selector trainer.

The overview keeps all three foods visible. **Follow swimmer** moves in closer. Each food shows its depth; the sidebar also lists all three depths. The foreground bank fades as a visual cutaway to reveal the water column. This does not remove its physical slope. Terrain, food markers and physics use the same analytic basin profile.

Training autosaves every 30 seconds, on normal app/scene exit, or through **Save training now**. Save failures are displayed. The graph starts a new evaluation history for the new task while retaining the inherited update number. One-food and three-food scores describe different tasks and must not be spliced together or compared as direct regressions.

The **original motor-training** runner is:

```powershell
.\build\Release\deep_swimmer_train.exe --pond
```

Return to the menu before launching headless so the scene releases its save. To resume watching, press **Ctrl+C** in the console, wait for **Saved full training**, then open **Deep water**. Only one trainer may own the save. Closing the Godot menu does not intentionally stop the separate console. Abrupt console closure/crashes can lose progress since the last autosave. Headless speed comes from executing fixed physics steps faster, not enlarging the timestep.

New save:

```text
%APPDATA%\Godot\app_userdata\CudaRL Arena\pond-policies\deep-pond-v1\training.bin
```

Source swimming save, preserved:

```text
%APPDATA%\Godot\app_userdata\CudaRL Arena\pond-policies\deep-v1\training.bin
```

A backup of the user's final one-food save is also under `deep-v1/before-three-food-pond/`. The actual transferred controller was at update **8,675**. Live training files are outside Git and are not backed up by a commit/push.

## What was retained

The actor grows from **27 → 32 → 32 → 4** to **33 → 32 → 32 → 4**. Food zero's relative XYZ stays in the same columns; six columns are added for the other foods, and the original body inputs shift six columns. New columns start with zero weights. Existing input weights, all hidden/output weights and biases are copied. Before training, the expanded actor produces the same motor commands for the same body/first-food observation, regardless of the new food inputs. Native tests check this numerical invariance.

The critic is expanded the same way. Actor/critic Adam moments are mapped to the new parameter layout; new moment columns are zero. Adam step counters, exploration standard deviations/optimizer and cumulative update counter are retained. This is a transfer of learned control and optimization state, not a random restart. New terrain/food rules require freshly initialized background worlds and a newly measured evaluation baseline. Old worlds, history and best policy remain in the source checkpoint. The new full save subsequently resumes its own worlds, optimizer and graph exactly.

The source is read under an exclusive lease. Missing/incompatible sources produce an error rather than silently replacing progress. Once the new save exists, it resumes directly and does not repeat the import. Checkpoint tags distinguish the shallow pond, one-food tank and deeper three-food task.

## Physics and food

The swimmer keeps the existing free 3D articulated dynamics, four internal yaw/pitch tail-motor torques, neutral submerged buoyancy, gravity, distributed viscous/quadratic drag and ten 5 ms substeps per 50 ms controller step. CPU showcase and CUDA training use the same code. There is no automatic thrust, steering, gait, rescue or teleportation.

The pond has radii 4.6 × 3.5 model units, surface Y=0 and a 4-unit-deep centre. One unit is 4 cm. With normalized radial distance `r`, the floor is `-4 + 4.5 * min(r, 1)^6`. Sampled body contacts respond to the floor's normal with spring/damper forces. The existing swimming-only surface containment remains; this step does not implement surface breaching, crawling or land locomotion. The flat outer decorative bank is above water. Decorative rocks/reeds do not add hidden obstacles.

This remains a physically motivated reduced fluid model, not CFD or calibrated animal biomechanics. The new basin changes boundary contacts; it does not change the swimmer's free-water force equations. A regression test checks identical trajectories away from the bank.

All three stable food positions are observable by the neural controller. There is no programmed selected food or nearest-target controller. Food spawns across the usable water volume at depths 2.6–13.4 cm, at least 1.6 cm above the local floor. Pieces are at least 8 cm apart in 3D and 2 cm apart vertically. Spawn points avoid the creature. Only an eaten piece moves; its replacement is at least 10 cm from its previous position. Random rejection sampling has a deterministic constrained fallback. Collection uses 3D distance from the head.

Reward uses progress toward a smooth potential over all three foods, collection rewards and the prior small action-change/joint-stop costs. Progress is computed before food replacement so spawning cannot earn progress. Native PPO, GAE, minibatches, epochs and motor physics are retained. Episodes time out at 60 seconds with value bootstrapping; collection does not terminate them.

## Evidence and limitations

- **147/147 native tests passed.** New tests exercise 2,800 placements/replacements, three distinct depths, consumed-only replacement, 3D collection, floor response, unchanged free-water dynamics, exact actor/optimizer transfer, unchanged source save, CPU/CUDA agreement and exact resume across an episode timeout.
- **67 Godot checks passed** for actual transfer from update 8,675, continued learning, collection by the inherited controller, best/latest viewing, pause, save/resume, three rendered foods, menu routing and the headless command. Follow/overview rendering and controls were checked at 1280×720, 768×1024, 390×844 and 844×390; food visibility in overview was checked. Tablet portrait uses controls above the pond to avoid a narrow clipped view.
- **103 menu regression checks passed** across the existing demonstrations, keyboard navigation and responsive layouts.
- A separate pilot transferred the earlier update-600 swimmer. With stable food inputs, the fixed-suite baseline was 8.921875 food/minute and the best reached 11.53125 by update 1,575. At update 1,600, fresh seed 3292201617 measured 0 untrained / 11.46875 latest / 11.453125 best food/minute across 64 one-minute trials; the trained actors collected food in every trial. These are different layouts/spacing from the tank, so do not compare them directly to its ~19–20 food/minute.
- **Food-choice bias remains.** In that early holdout, all collections were from the first food slot. The expanded policy receives all three positions and can learn from them, but it has not demonstrated efficient selection across all three. Do not present increased collection speed as proof of successful routing or unbiased food choice.
- A rejected per-control-step shuffle pilot fell from ~8.9 to 2.625 food/minute over 1,800 additional updates. Randomly changing the inherited controller's first food every 50 ms disrupted stable approaches. The live/headless pond uses stable input ordering during adaptation. Optional shuffle kernels remain for controlled experiments, not the default. Best-controller viewing protects the demonstration from showing a weaker latest policy as if it were the strongest.
- Pilot logs, test outputs and screenshots: `build/pond-learning-checks/deep-pond-*`. Isolated pilot saves: `build/deep-pond-pilot/` (rejected shuffle trial) and `build/deep-pond-stable-pilot/` (stable trial). Neither is the user's live save.

## Presentation review and next decisions

Internal review heuristics, not measured quality scores: hierarchy/consistency 9, feedback 9, responsive usability 8, accessibility verification 8, visual polish 8. Phone/tablet users scroll the controls to reach secondary training actions; test a physical touchscreen and assistive tooling before claiming full accessibility. World depth labels remain secondary; the sidebar provides readable equivalents. Future visual work should improve water depth cues and the cutaway's finish without obscuring the animal or spending the next step on decorative effects.

The highest-priority learning follow-up is controlled food-choice/retention evaluation and a transfer-aware way to reduce the observed slot preference without disrupting swimming. No legs, anatomy growth, shoreline crossing or walking is included. The longer-term requirement remains one growing creature retaining swimming while learning additional locomotion skills.
