# CudaRL-Arena · Tadpole Pond

A physically simulated creature that learns through our own C++/CUDA PPO system, with Godot providing the world, player interaction and inspection tools. No Python or PyTorch is used for training or runtime.

The pond tadpole is the first stage. The longer-term direction is dependable swimming → measured adaptation during player interaction → physical growth and underwater limbs → crawling and leaving the pond → walking and eventual manipulation. Those later stages are not implemented.

[Download website](https://cudarl.drewclark.io) · [Master plan](docs/CREATURE_PLAN.md) · [Current local checkpoint](docs/SESSION_CHECKPOINT.md)

## Current status

The existing pond supports live physical swimming, sinking food, currents, local PPO practice and saved progress. CUDA is preferred after a real device/kernel probe; the portable feeding session also supports CPU learning. This inherited swimmer remains in place while its replacement is evaluated.

**Dependable replacement movement is not established.** The latest isolated motor comparison stopped at its contact-readiness gate before either new policy trained. Geometry checks and CPU/CUDA agreement passed for the executed portion, but a rolled-head support case exceeded the registered overlap allowance. No replacement, further training or relaxed gate follows automatically from cleanup. See [the contact report](docs/MOTOR_V2_RESULTS.md).

The current cleanup removes retired demos and repeated orchestration while preserving native physics, learning, saved controllers and reproducibility. CartPole and Racing have been retired from the active source. Historical reports remain evidence about their original commits, not instructions for running the current app.

## Behavioral architecture

Keep two learned behavioral policies:

- **Goal chooser:** selects a persistent intention such as feeding, seeking safety, exploring or holding/resting.
- **Reusable movement motor:** executes a destination, desired terminal velocity and heading. One motor must support turning, braking, arrival and holding; there is no motor network per ability.

PPO critics estimate value during training; they are not additional behavioral controllers. The replacement motor experiment receives movement requests directly, without a chooser. The existing food scorer does not yet implement the full future intention set.

A joint-target motor requests four angles for the current body's two universal joints, each with two controlled axes. A deterministic physical servo converts those targets into limited joint torques every 5 ms. It does not provide extra whole-body forces or rescue actions. Direct torque remains the registered comparison arm. Servo regulation alone does not establish swimming.

See [request units, ownership and future body changes](docs/CREATURE_CONTROLLER_INTERFACE.md).

## Execution paths

| Layer | Responsibility and source |
| --- | --- |
| Godot launcher | Selects the saved boulder learner locally when CUDA and its source save are available; otherwise opens portable feeding. `godot-project/launcher/` |
| Pond presentation | UI, camera, geometry and native state display. `godot-project/pond/` |
| GDExtension | `DeepPondLab` owns the selected native session/playback. `src/godot/deep_pond_lab.*` |
| Native sessions | Visible simulation, background practice, checkpoint leases, comparison and adoption. `src/demos/feeding*`, `arch_session.cu`, `overhang_session.cu` |
| Physics/environments | Shared body, hydrodynamics, contact and task state. `src/envs/swimmer/` |
| Learning | Actors, critics, PPO math, action probabilities and optimizer continuation. `src/rl/`, `src/nn/` |
| Checkpoints | Versioned/checksummed complete saves. `src/core/checkpoint.*` and task-specific serializers |

Godot does not perform neural training or secretly propel the creature. Player changes are passed to native code. Learning updates, validated adoption and reactions from frozen weights must be labelled separately.

The two motor benches currently inspect different stopped protocols: the original command/paired-motor study and the later contact-v2 study. They are development tools, not two behavioral motors deployed in the creature. Their callers, schemas and evidence must be reconciled before retiring a path.

## Windows development

Requires Visual Studio 2022 C++ tools, CMake 3.24+, a compatible CUDA toolkit for the developer CUDA build, and a matching Godot 4.4 `godot-cpp` release library under `external/godot-cpp/bin/`. The default CUDA architecture is `120`; override `CMAKE_CUDA_ARCHITECTURES` for another GPU.

```powershell
cmake --preset windows
cmake --build --preset release
ctest --preset release
```

Open `godot-project/project.godot` in Godot and run its main scene. Native DLLs cannot be replaced while an app/editor process is holding them. A separate output directory can be selected with `CUDARL_ADDON_BIN` for isolated verification.

The application name/save namespace remains **CudaRL Arena** to preserve personal progress. Do not delete or repurpose `user://pond-policies/` during cleanup. Consumer downloads are installer-first and do not require Godot or the CUDA toolkit. [Windows packaging and release instructions](docs/windows_release.md).

The portable CPU build is separate:

```powershell
cmake --preset portable
cmake --build --preset portable
ctest --preset portable
```

## Focused validation and optional research tools

Default native checks cover numerical kernels, PPO/action probabilities, session publication, checkpoint integrity, contact response and CPU/CUDA current agreement. GPU checks explicitly skip on systems without a device; set `CUDARL_REQUIRE_GPU=1` when a missing GPU must fail.

Longer simulator/training regressions and research executables are opt-in:

```powershell
cmake --preset windows -DCUDARL_BUILD_EXTENDED_TESTS=ON
cmake --preset windows -DCUDARL_BUILD_RESEARCH_TOOLS=ON
cmake --preset windows -DCUDARL_BUILD_BENCH=ON
```

These switches enable building tools; they do not authorize repeating a stopped experiment. `nn_bench` additionally uses cuBLAS as a reference. Keep registered research budgets and withheld evaluation cases intact.

Fast Godot checks load resources or use fake sessions, without starting a creature learner:

```powershell
Godot_console.exe --headless --path godot-project --script ../tests/godot/project_resource_audit.gd
Godot_console.exe --headless --path godot-project --script ../tests/godot/pond_lifecycle_smoke.gd
Godot_console.exe --headless --path godot-project --script ../tests/godot/habitat_mode_config_smoke.gd
```

Behavioral acceptance remains separate: purposeful movement, turning, real braking, slow controlled arrival and ten-second position/heading holding. Unsupported calm-water holding, pond-relative holding against a specified validated current and supported bottom rest are different tests. Active joint movement is allowed. Food collection, fitting error, component checks and update counters do not prove these skills.
