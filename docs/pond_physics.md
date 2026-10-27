# Pond physics and live learning — October 26, 2026

The user requires physically grounded movement and objected to calling a fixed 300-update budget “training complete” when swimming still looked awkward. They subsequently requested continuous live learning: the pond has no update cap. No food-reaching score, update count or elapsed time is presented as proof of graceful or biologically accurate swimming.

## Current model

The live pond uses swimmer body ID 8 in both the 1,024 CUDA training environments and the visible C++ CPU simulation. Historical bodies 0–7 retain the earlier overdamped experiment for comparison. Old actor weights are not used to initialize live training.

Body 8 is a planar three-link articulated rigid body with five generalized degrees of freedom: two translations, heading and two relative joint angles. Mass and rotational inertia come from fitted head/tail volumes. A 0.04 m/model-unit scale makes the 1.8-unit animal 7.2 cm long. Tissue density is assumed equal to freshwater density, with fixed depth representing a neutrally buoyant horizontal slice. It does not simulate free diving, gravity/buoyancy transients, surface crossings or land contact.

The network commands bounded internal joint torques, not joint velocities or external thrust. The mass matrix includes translational/rotational coupling and the Coriolis terms from the articulated link geometry. Distributed viscous and quadratic drag oppose motion relative to stationary water. Projected areas include the distal membrane; membrane geometry changes drag and mass. Soft joint stops and passive motor resistance prevent unlimited joint rotation. Force integration uses ten 5 ms substeps per 50 ms control step and an implicit positive resistance matrix.

These are physically motivated reduced-order approximations, not CFD, measured tadpole hydrodynamics or a calibrated muscle model. Scale, actuator torque, fitted geometry and drag coefficients are explicit assumptions. The distinction between viscous and quadratic resistance is consistent with standard fluid-force approximations described in [MuJoCo’s fluid-force documentation](https://mujoco.readthedocs.io/en/3.1.1/computation/fluid.html); MuJoCo is a reference only and is not a dependency. All simulation/training code remains the project's C++/CUDA.

The earlier zero-inertia model directly solved force balance at each instant, so a motionless tail could not coast. Body 8 carries momentum; changing speed now requires physical forces. No position multiplier, automatic forward thrust or prerecorded swimming gait has been added.

## Learning and display

The 3D pond starts a genuinely random policy, shows update zero for two seconds, then learns from 1,024 parallel CUDA environments. The existing PPO session publishes complete actor snapshots; the visible animal applies new weights without resetting body state or food. A 10-updates/second viewing throttle makes the run watchable. Learning continues past 300 updates until the scene closes, retaining the same policy and optimizer state. A zero session update budget means unlimited; positive budgets still end normally for other demos and offline experiments. Start over stops/joins that session and starts fresh weights and a fresh pond. Closing the scene also stops/joins training. Continuous training does not guarantee monotonic improvement.

The current authorized task has three simultaneous food pieces and a fresh 13-input policy that sees all three relative positions and physical body state. Only the eaten piece is replaced, without moving the animal or other food. Placement spans a 3.9×2.8 ellipse, keeps pieces 2.2 units apart, and relocates a consumed piece at least 2.6 units into a different quarter of the pond. A recovery-training pass uses full-range resting angles, an outward-at-stop penalty and three-food entropy regularization to address learned joint-stop holds. These change training, not the articulated equations. There is no selected target controller, food-moving shortcut, scripted recovery or automatic rescue. See [the three-food objective, measured results and limitations](three_food_learning.md). Inputs are coordinates, not learned visual perception.

The 3D mesh follows the simulated head and joint angles, with render interpolation and a continuous skin surface. Eyes/lighting are presentation. The scene is a 3D rendering of horizontal physical movement, not a full volumetric fluid simulation.

## Statistics and learning graph

The user approved a racing-style sidebar and bottom graph. The sidebar separates the visible animal's food count, rolling-minute collection count, instantaneous swimming speed and time since collection from the 1,024 background environments, total training steps and steps per wall-clock second. Speed uses the model's 0.04 m/unit scale, in cm/s. Throughput includes pacing and evaluation; it is not hardware utilization. Counters and timing are real measurements.

Pause learning preserves policy and optimizer state while the displayed animal keeps swimming. Resume, a 1–30 updates/s slider and Maximum speed change the existing session without resetting it. Start over explicitly starts fresh learning and clears the graph. Desktop uses a left sidebar; portrait windows use two metric columns above the pond with expandable speed controls. Short landscape windows retain a scrollable sidebar. The pond renders in its own viewport, so panels cannot cover it.

The graph now measures mean food per minute over 64 separate, fixed 60-second three-food ponds. It records an update-zero baseline and reevaluates every 25 updates with seed `0x51A7E123`, the same body and mean policy actions. These CUDA trials feed no experience into PPO and never move the visible animal. The line connects actual measurements without smoothing, and the axis scales numerically rather than implying percentage mastery. Full protocol and development evidence are in the three-food document. Offline experiments keep evaluation disabled by default.

## Evidence

The current pass completed 138 Godot learning checks, nine native foraging tests, all 13 swimmer tests, five PPO demo tests and 101 menu checks. The archived saved-policy pond still passes 100 checks. Four rendered sizes (1280×720, 768×1024, 390×844 and 844×390) were inspected. Checks cover CPU/CUDA parity, distant region-changing replacement, folded-pose recovery with a frozen regression policy, graph rate arithmetic, rolling-minute counts, update 350+, keyboard controls, menu navigation and clean shutdown. The previous one-food percentage metric is historical. See the three-food document for current benchmark and long-run limits.

Six focused inertia tests pass:

- An unpowered resting body remains at rest.
- Uniform translation persists in vacuum with conserved kinetic energy.
- Unpowered water coasting slows and matches the analytic linear-plus-quadratic drag equation within 1% after one second; sampled kinetic energy decreases.
- Internal joint torques do not supply external linear momentum; reducing timestep reduces residual integration error. Angular-momentum error is bounded in the test.
- Water resistance dissipates energy and membrane geometry changes mass and resistance.
- CUDA integration matches CPU trajectories across 32 environments × 200 control steps within the specified 0.003-model-unit tolerance, with matching episode boundaries.

In the initial body-8 learning study, each of three seeds trained for 300 updates with 1,024 environments and the same 9→32→32→2 policy. Each reached 256/256 comparison targets; mean successful-trip times were approximately 2.04–2.05 simulated seconds. The separate validation set reached 128/128. The sets are held out from training but have been inspected during development, so these are comparison results, not an untouched final test or proof of universal competence.

A separate CPU continuous-feeding probe ran the seed-1 policy for 300 simulated seconds on ten pond seeds. Counts were 185–193 collections, with longest waits 2.8–3.5 seconds. This establishes a useful bounded regression, not lifelong reliability or a visual-quality certification.

Raw study rows are [inertial-study.csv](experiments/swimmer/inertial-study.csv). The live integration test checks random initialization, exactly 1,024 background environments, paused update zero, optimizer progress, application of learned policies to the displayed animal, no policy-change teleport, food collection, fresh restart and scene cleanup. Desktop/tablet/phone window renders pass. Physical touch hardware, assistive technology and biological validation remain untested.

```powershell
cmake --build --preset release --target cudarl cudarl_tests swimmer_experiment pond_probe
build/Release/cudarl_tests.exe "swimmer inertia:"
build/Release/swimmer_experiment.exe build/swimmer-inertial/reproduce 300 3 1024 0 8
build/Release/pond_probe.exe build/swimmer-inertial/reproduce/policy-seed-1.json
C:/Users/aiand/Godot/Godot.exe --path godot-project res://pond/pond.tscn
```

The preceding shared-trainer refactor was checked before the physics change: all actor parameters after 300 updates matched the original overdamped seed-1 experiment exactly. This is a refactor check, not a claim that old weights remain suitable for the new physics.

## Next quality gate

Food success alone does not establish effortless locomotion. The next evaluation should examine sustained forward swimming, speed fluctuations, turning and joint/torque smoothness under the same physical model, including unfamiliar starts. The pond remains a prototype. Current internal review heuristics: usability 8 (short landscape windows require sidebar scrolling), hierarchy 9 (pond, metrics and chart have distinct regions), accessibility 7 (keyboard and target sizes checked; assistive technology and physical touch untested), consistency 9 (shared spacing, palette and controls), responsiveness 8 (four inspected sizes, smaller windows remain untested), feedback 8 (live metrics and clear states; collection feedback could be stronger), performance 8 (background evaluation and incremental UI history; frame-time contention remains unprofiled). Prioritized recommendations are native assistive/device testing, shorter-window usability checks, stronger collection feedback, and measured rendering/training contention. These are recommendations, not approval to expand the current one-food scope.
