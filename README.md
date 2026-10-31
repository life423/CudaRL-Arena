# CudaRL creature motor

A native C++/CUDA physically simulated creature with one Godot viewer. The motor
learns stroke strength/frequency; an explicit oscillator drives torque-limited
internal joints. Godot displays native poses and never supplies propulsion.

## Run and current limits

```powershell
./tools/view-live.ps1             # build the extension and open paused
./tools/view-live.ps1 -SkipBuild  # use the already-built extension
```

Start/Pause/Reset control the single native world. The live scene still uses the
previous8mm/s frozen actor and auto-pauses after4seconds. It does not train.

The separately saved supervised checkpoint passes10/10four-second constant
requests from rest in two reset poses, including withheld5/7mm/s. Requested
4/5/6/7/8 produced4.350/5.318/6.259/7.115/7.843mm/s. It is not auto-loaded into
Godot. Slowdown/switches remain unpassed: the8->4teacher test scored5.234mm/s
against a3.6..4.4limit. Turning, holding, current and contact remain unvalidated;
head yaw can reach28.84degrees. This is not a generally validated movement motor.

- [Checkpoint identity and scope](assets/steady-speed-v1/checkpoint-provenance.json)
- [Frozen evaluation metrics](assets/steady-speed-v1/evaluation.json)
- [Accepted-example provenance](assets/steady-speed-v1/example-provenance.json)
- [Teacher gates, including the failed slowdown](assets/steady-speed-v1/teacher-gates.json)

The same folder contains checkpoint.bin,960accepted observation/action examples
and the baseline reference. The native research checkpoint contains actor+Adam,
not a resumable physical world; only Windows x64/little-endian is verified.
No failed switch examples are used for fitting. All experiment budgets are closed;
unused steps are not permission for another run. Full saved data/evidence remains
under ignored build/; historical source is available in Git.

## Build and focused checks

Requires CMake3.24+, Visual Studio2022 C++ tools, CUDA for learning and matching
Godot C++ bindings. Tested Godot4.7.2 with existing4.4 bindings. Configure
CUDARL_GODOT_CPP_ROOT/LIBRARY for another dependency location. The cpu preset
builds without CUDA or Godot dependencies.

```powershell
cmake --preset windows
cmake --build --preset release
./build/Release/foundation_tests.exe demonstration
./build/Release/foundation_tests.exe speed_request
```

The opt-in zero-step viewer check is:
`Godot_console.exe --headless --path live --script ../tests/godot/viewer_controls.gd`
(use the installed Godot console executable). It disables automatic processing.
The single learning runner tools/run-straight.ps1 consumes the curated examples;
invoking it starts a NEW bounded fitting/evaluation trial and requires approval.
Training is not part of CTest. Do not launch it during structural/document cleanup.

## Ownership and direction

live/main.gd directly extends built-in Node3D; main.tscn is one plain scene.
FrozenSwimmer is one RefCounted bridge composed with FrozenController, which owns
one world, task and actor. Shared motor code lives in src/nn/motor; physical equations
in src/physics; numerical learning primitives in src/nn, src/rl and src/cuda.
UI nesting is node composition, not custom script/scene inheritance. Keep one viewer
and one shared motor path rather than adding wrappers or a network per ability.

The intended system is one learned goal chooser and one reusable motor receiving
persistent destination, desired terminal velocity in SI m/s and unit heading;
world/model conversion is1unit=4cm. Only the current speed lesson is implemented.
Future work is competent swimming/adaptation, then growth/limbs, leaving water,
walking and manipulation. Preserve model6 body/water/torque limits unless an
explicitly authorized model change is made; no artificial body push or rescue.

Follow [AGENTS.md](AGENTS.md) for worktree/data preservation and execution scope.
Third-party legal notices remain in docs/third-party/. The original checkout,
personal saves, checkpoints and non-Markdown evidence must not be purged.
