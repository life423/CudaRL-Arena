# Deep-water swimming and accelerated training

Update: the menu's Deep water action now opens the [three-food deep pond](deep_pond.md), retaining this one-food swimmer through actor/critic/optimizer transfer. The one-food tank described below is preserved at `res://pond/deep_water.tscn`. Use `deep_swimmer_train.exe --pond` for the current pond; omitting `--pond` still selects this archived one-food task.

Implemented October 26, 2026. This is the next controlled experiment before legs: one food target at varying depths, continuous native PPO, and one visible swimmer driven by the latest published neural controller. The shallow three-food pond and its saved policies remain separate. The launcher replaces **Compare shuffled** with **Deep water**; the old comparison scene remains available in the source.

## Watch and use the menu

Run the Godot project normally and choose **Tadpole Pond → Deep water**. The scene starts a new 3D controller when no compatible save exists, otherwise resumes saved training. After a two-second introduction, 1,024 background swimmers learn continuously. The foreground runs at real-time speed with mean policy actions; background worlds use exploration. It is a demonstration of their shared controller, not a replay or one of the CUDA worlds rendered directly.

- **Pause learning** pauses policy updates. The visible swimmer continues demonstrating its current controller.
- **Whole tank view / Follow swimmer** switches between an overview and a view framing the creature and its food. Depth markings and the food's line to the floor make height visible.
- **Maximum speed** runs background updates without pacing. Turning it off limits training to five updates/second; physics timesteps do not change.
- **Save training now** requests a full save at the next update boundary, including while paused.
- **Back to demos** or Escape saves and stops the trainer before returning to the menu.
- **Training guide**, on the menu, explains these controls and provides the headless launcher and a copyable PowerShell command.

The graph shows food/minute from 64 fixed 60-second evaluation worlds every 25 updates. Orange is the latest evaluated actor; green is the running best. Evaluation has no exploration or gradients. A rising graph is measured improvement on this suite, not proof of optimality or monotonic future progress. Neither learner has a completion cap.

## Headless: accelerate without rendering

1. Return to the menu so Godot releases its training save.
2. Select **Training guide → Open headless training console**. An interactive PowerShell console opens the native executable. The menu may remain open or the Godot app may be closed.
3. Leave the console running. It prints update, current/best evaluated food/minute, steps/second, simulation speed per world and last saved update.
4. To watch again, press **Ctrl+C** in that console and wait for **Saved full training**. Then choose **Deep water** in Godot.

Only one process may own a save. Starting a second trainer fails rather than allowing conflicting writes. Ctrl+C finishes the in-flight update and saves. Closing the console window abruptly, a crash or a power loss can lose progress after the last 30-second autosave.

From this repository in PowerShell:

```powershell
.\build\Release\deep_swimmer_train.exe
```

It uses the same default population (1,024), training seed (7) and save as Godot. An optional finite budget means **additional** updates after resuming:

```powershell
.\build\Release\deep_swimmer_train.exe --updates 1000
.\build\Release\deep_swimmer_train.exe --help
```

For an isolated experiment, use a different file. Population and training seed must match when restoring that file:

```powershell
.\build\Release\deep_swimmer_train.exe --envs 1024 --seed 7 --checkpoint build\my-deep-run\training.bin --updates 600
.\build\Release\deep_swimmer_train.exe --checkpoint build\my-deep-run\training.bin --evaluate-seed 3292201617
```

The evaluation-only command compares untrained, latest and best actors on fresh layouts and reports food collected in the upper/lower halves and on ascending/descending approaches. It does not write evaluation results back into the training checkpoint.

“51× time per world,” for example, means each of the parallel worlds averages 51 simulated seconds per wall second. It does not mean the foreground runs 51× faster or that changing the numerical timestep speeds up training. This is measured throughput, not GPU utilization. Hardware, rendering and competing processes affect it.

## Saves and compatibility

Default Windows file:

```text
%APPDATA%\Godot\app_userdata\CudaRL Arena\pond-policies\deep-v1\training.bin
```

The full checkpoint includes actor, critic, exploration log-standard-deviations, all Adam states, CUDA world states/counters and RNG reconstruction inputs, evaluation best and learning history. Saves use the existing checksummed atomic replacement and exclusive lease. The foreground creature resets its pose when reopening; training resumes from the saved background state. This is a separate task/format marker from the shallow pond; old 13-input policies cannot operate the new body. Git commits do not back up live saves.

## What the physics actually models

The swimmer is three articulated rigid links: a floating head/body and two tail links. The free base has six degrees of freedom, and each universal tail joint has two rotational degrees of freedom, for ten generalized coordinates. Four bounded internal torques control yaw/pitch at the two tail joints. There is no commanded forward velocity, direct thrust, target-following force or canned swimming cycle.

The shared CPU/CUDA header builds the full configuration-dependent mass matrix from 3D linear/angular Jacobians, link inertias, centrifugal/Coriolis bias and gyroscopic terms. Each 50 ms control interval uses ten 5 ms substeps. Velocity-dependent resistance is integrated implicitly with a Cholesky solve. Quaternion integration permits yaw, pitch and roll; translation is free along X/Y/Z, with Godot Y up.

Mass comes from explicit fitted volumes at 1 model unit = 4 cm. Water density is 1,000 kg/m³; tissue density is also 1,000 kg/m³, giving neutral buoyancy when submerged. Gravity and buoyancy act separately; partial submersion reduces buoyancy and fluid resistance. The head's centre of buoyancy sits 0.4 mm above its mass centre, providing a physical restoring moment. Distributed viscous and quadratic anisotropic drag include the broad distal membrane. Geometry changes forces and inertia as well as appearance.

The 36 × 28 × 16 cm volume has soft spring/damper boundary contacts sampled along the body, including a containment boundary at the surface. There is no hard position clamp. This first experiment is a contained swimming tank, **not a surface-breaching or land-contact model**. Decorative details do not add obstacles. The open front/right view is a visual cutaway of these boundaries.

This is a physically motivated reduced-order fluid model, not CFD, calibrated animal biomechanics or a fluid-volume solver. It omits wakes acting back on the swimmer, added fluid mass and detailed lift. [MuJoCo's fluid-force documentation](https://mujoco.readthedocs.io/en/stable/computation/fluid.html) provides background on phenomenological drag models and implicit integration; MuJoCo is not a dependency.

## Neural controller and task

The native tanh actor is **27 → 32 → 32 → 4**; the critic has the same hidden layers and one output. Inputs are food position relative to the body, local linear/angular velocities, local world-up direction, four joint angles/rates, world position for boundary awareness and previous actions. The agent receives coordinates and physical state, not camera images.

PPO retains the existing update implementation, 64-step rollouts, four epochs, eight minibatches, clipped objective, GAE and Adam. Only task dimensions and the physical environment are extended. The shallow trainer's action count and checkpoint representation are retained.

One food remains fixed until eaten. Target sampling spans depths 3.2–12.8 cm, requires at least 1.8 cm of vertical separation when placed and samples horizontal directions. Targets are initially 4.8–10.4 cm from the creature. Replacements must also be at least 5.2 cm from the previous food; spawning does not move the creature or earn progress reward. The learning objective rewards physical progress and collection with small action-change/joint-stop costs. Episodes time out at 60 seconds with correct value bootstrapping; food collection itself is not terminal. The visible swimmer continues beyond the episode horizon.

## Validation and current limits

- Native checks cover neutral rest, passive damping, internal motor momentum in vacuum, genuine vertical/lateral response, depth sampling, 3D food collection, floor recovery, CPU/CUDA agreement, finite PPO updates, exact optimizer continuation and rejection of shallow/deep checkpoint mixing.
- Godot smoke checks exercise real background learning, pause, manual save, exclusive ownership and resume. Follow/overview renders and reachable controls are checked at 1280×720, 768×1024, 390×844 and 844×390; the guide is checked at desktop/tablet/phone dimensions. These are rendered viewport/keyboard tests, not physical touchscreen or screen-reader certification.
- A separate seed-7 pilot reached 18.828125 food/minute as its best fixed-suite result by update 575. At update 600, fresh seed 3292201617 measured **0 untrained / 18.921875 latest / 19.0 saved best** food/minute across 64 new one-minute trials. The best collected 595 foods in the upper half and 621 in the lower half; 611 ascending and 605 descending approaches. All 64 trials collected food. This is one training seed and one holdout suite, not broad robustness or optimality evidence.
- The initial isolated 600-update pilot ran around 1.05M aggregate steps/second (~51× time per background world) on this RTX 5070. It is a short throughput observation, not a controlled rendering comparison or kernel optimization study.
- Logs/screenshots are in `build/pond-learning-checks/deep-*`; the separate pilot checkpoint is `build/deep-pilot/training.bin`. Neither is the user's live save.

Next decisions remain separate: wider-layout/seed robustness, physical validation beyond these tests, food choice with multiple targets, shoreline contact and eventually the approved two-legged juvenile. No limb growth or walking has been implemented here.
