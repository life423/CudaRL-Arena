# CudaRL creature motor

A native C++/CUDA physically simulated creature with one Godot viewer. The motor
learns stroke strength/frequency; an explicit oscillator drives torque-limited
internal joints. Godot displays native poses and never supplies propulsion.

## Run and current limits

```powershell
./tools/view-live.ps1             # build the extension and open paused
./tools/view-live.ps1 -SkipBuild  # use the already-built extension
```

Press Start once, then change the4�8mm/s request while it swims. Requests preserve
body velocity, oscillator/filter state and running status. A paused request stays
paused; Start resumes. Reset is explicit. The same frozen signed-cadence network
controls both forward strokes and braking; there is no training in Godot.

Continuous viewing remains on by default, with visible resets near the finite
pond edge (20mm clearance). The viewer uses1.2x0.8m horizontal bounds and0.16m
depth; original benchmarks retain0.36x0.28m bounds. Uncheck Continuous for a
four-second segment/auto-pause.
Safety failures stop execution. Longer viewing is not indefinite-stability proof.
Full stopping, heading hold, currents, contact and arbitrary transition sequences
remain unvalidated. The evaluated transitions were4->8->4 with separate settling.
An added4->8->4->8 sequence failed its last upshift speed/RMSE gates despite
no edge reset; enlarging the pond does not repair that policy limitation.

Accepted baseline: signed-transitions-v1/update-600, SHA256
18346C676CF3E12283C51129C1224B3CEDEAC8C2FA22BDC1C1E826A6EFE5BE72.
Frozen inference passed10constant cases,2held-out warmups,4transitions and4holds.
The original automatic selection rejected a5.53%increase in modeled actuator
work/metre at8mm/s; the user explicitly accepted that tradeoff. The original
rejection remains unchanged in the saved evidence.

- [Explicit acceptance and limitations](assets/signed-speed-v1/acceptance.json)
- [Original selection result](assets/signed-speed-v1/original-selection-result.json)
- [Frozen neural evaluation](assets/signed-speed-v1/frozen-evaluation.csv)
- [Live actor identity and source contract](live/assets/controller-provenance.json)

The exact research checkpoint and protocol are pinned in assets/signed-speed-v1.
The portable actor exports only those same2498parameters, with an explicit action
mapping and source hashes; it contains no optimizer or resumable physical world.
The prior forward actor, portable export and accepted examples remain under
assets/steady-speed-v1, with Git checkpoint e484667 available for rollback. No
old/new-controller toggle or alternate viewer is active.

All prior training/physical budgets are closed. Full local evidence remains in
ignored build/; Git does not preserve ignored files. New runs need explicit scope.

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
The single learning runner tools/run-straight.ps1 prepares a frozen trial without
stepping or fitting. Its explicit -Run switch starts the separately authorized
bounded collection, fitting and frozen evaluation; it never adopts a candidate.
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
