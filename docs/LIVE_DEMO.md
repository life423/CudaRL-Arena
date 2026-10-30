# Live frozen swimmer

From this worktree, run `./tools/view-live.ps1`; use `-SkipBuild` with the
already-built DLL. Alternatively open `live/project.godot` in Godot. The tested
engine is Godot4.7.2, with the existing Godot4.4 C++ bindings. The launcher builds
only `frozen_swimmer`; Visual Studio C++ tools and matching godot-cpp are needed.
The frozen demo itself uses CPU native physics/inference, not CUDA training.

The scene starts paused. Start advances the real model6 native world with the
saved actor; Pause stops native time, Reset restores the initial condition.
It automatically pauses after4seconds. The network controls stroke strength
and frequency; the supplied oscillator remains part of the motor. There is no
training, CSV playback, added body thrust, Godot rigid-body physics or rescue.

The portable10,736-byte `live/assets/frozen-controller.bin` is committed with
provenance. It contains only the selected actor and a strict version/source
contract. The loader rejects corruption, incompatible source and legacy saves;
it never substitutes random weights. It was exported from update25 of the
steady-window reward2 trial (source checkpoint SHA256
`31C9982778F32A5E0F421CDBA88AF4431ED27431A355C66D9E290D60EDC97092`).
That trial's frozen evaluations passed4/4, averaging7.958mm/s steady speed.

## Verification performed

Registered in `build/live-demo/2026-10-30-v1/protocol.json`: maximum9,600 native
substeps, no PPO. Native parity used3,200steps across four4second conditions,
matched320 saved rows exactly and checked corrupt-load rejection/pause/reset.
The real Godot extension used another3,200steps, passed4/4 and matched320rows
within the registered tolerance (maximum normalized difference0.000241643).
A rendered720x1080 live run used800steps in4.427wall seconds, captured its own
viewport at2.5seconds, then passed and auto-paused at4seconds. Total7,200steps;
the2,400 reserve remains unused. No retries. Strict native build and Godot
script parsing passed. The screenshot is `build/live-demo/2026-10-30-v1/live-preview.png`.
Godot reported unavailable shader caching in the sandbox; rendering succeeded.

This is short straight swimming in calm interior water. Head oscillation remains
about26.6degrees. Turning, braking, arrival, holding, current and floor contact
are not validated by this demo. Model6 uses approximate distributed water drag,
not CFD; a known small dry COM integration drift remains documented.

The old `godot-project/project.godot` remains the separate physics2 diagnostic.
`./tools/view-movement.ps1` remains recorded playback. Neither is replaced by
the live scene, and no old personal save or original checkout is migrated.
