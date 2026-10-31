# CudaRL creature motor

One native physically simulated creature, one frozen learned motor and one Godot
scene. The current actor learned stroke strength/frequency for straight calm-water
swimming around8mm/s. The supplied oscillator drives only torque-limited internal
joints. No artificial body thrust or live training runs in the scene.

## Open the demo

```powershell
./tools/view-live.ps1
# With the already-built extension:
./tools/view-live.ps1 -SkipBuild
```

The scene starts paused, has Start/Pause/Reset and speed displays, and pauses after
four seconds per request. The selected actor currently permits only its verified
8mm/s request. The new supervised candidate passed constant-speed evaluation in the isolated native lesson; it has not replaced the live asset. Slowdown/switches remain unpassed.
[Current state and results](docs/CURRENT_MOVEMENT.md).

## Build and focused checks

Requires CMake3.24+, Visual Studio2022 C++ tools, CUDA toolkit for learning, and
matching Godot C++ bindings. Tested Godot4.7.2 with existing4.4 bindings. Set
CUDARL_GODOT_CPP_ROOT/LIBRARY for another dependency location. CPU-only preset
avoids CUDA and Godot dependencies.

```powershell
cmake --preset windows
cmake --build --preset release
./build/Release/foundation_tests.exe speed_request
./build/Release/foundation_tests.exe straight_
./build/Release/foundation_tests.exe stroke_motor
```

Training is opt-in and never part of CTest. The single compiled runner is
`tools/run-straight.ps1`; invoking it creates a new physical/training trial and
requires an agreed budget. The latest approved trial is closed; no automatic retry.

## Ownership

- src/motor/state.h: shared physical/domain math and observation indices.
- stroke_motor.h: one physical/filter/oscillator state and capped native stepping.
- straight_task.h: shared speed request, observation, reward and acceptance.
- frozen_controller: strict portable actor loading and frozen inference.
- src/godot/frozen_swimmer: conversion only; live/ displays native poses.
- src/nn and src/rl: shared C++/CUDA numerical learning primitives.
- tools/stroke_motor_demo.cu: one trainer/evaluator, all requested speeds.

Model6 dimensions, forces and actuator limits remain unchanged in the speed work.
Contact, current, turning, braking and holding remain unvalidated. See the
[creature direction](docs/CREATURE_PLAN.md) and [checkpoint](docs/SESSION_CHECKPOINT.md).

Superseded arrival trainer, contact-study implementations, diagnostic Godot scene,
recorded viewer and their unused entry points have been removed from active source.
Their source is preserved in Git commitc0c0314 and earlier commits. Ignored study
artifacts, personal data and the original checkout have not been removed.

The committed steady-speed milestone, checkpoint and curated training inputs are
in [assets/steady-speed-v1](assets/steady-speed-v1/README.md). It passed only the
constant-from-rest lesson; the live Godot baseline is unchanged.
