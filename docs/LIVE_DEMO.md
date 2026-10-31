# Live frozen swimmer

Run `./tools/view-live.ps1` (`-SkipBuild` for the built DLL), or open
`live/project.godot`. This is the sole active Godot scene. It starts paused and
uses native model6 physics/CPU inference with the selected C++/CUDA-trained actor.
Start/Pause/Reset and speed readout are present; each request auto-pauses after4s.
The oscillator is intentional; the network supplies strength/frequency. No training,
CSV playback, Godot body solver or fabricated propulsion runs here.

The actor currently supports only8mm/s. A speed request uses the same task setter
as native training/evaluation, but the UI cannot select an unvalidated range.
Earlier PPO variable-speed candidates failed. The supervised constant-speed
checkpoint now passes10/10but remains isolated; it is not the live asset. The source-bound,
checksummed portable asset keeps the original passing2498actor parameters; schema2
loads actor-only into fresh state and rejects legacy/mismatched/corrupt inputs.
It does not resume a research world or optimizer.

Tested native build and Godot4.7.2 with existing4.4 C++ bindings. The earlier live
c0c0314 demo passed native/Godot4/4 parity and an actual720x1080 rendered4s run.
After cleanup, the unchanged actor matched160saved8mm/s rows exactly inside the
bounded trial, CPU/CUDA readiness passed, and13zero-integration task/request/loader
checks passed. Latest source parses in Godot. Rendered360x640 paused layout passed,
visually inspected, with zero physical advances; no extra live episode ran.

Short straight calm-water swimming only: turning/braking/arrival/holding/current
and contact remain unvalidated. Head yaw about26.6deg and known small dry COM
integration drift remain limitations. Git commitc0c0314 restores the old diagnostic
and replay if needed; those redundant active entry points have been removed.
