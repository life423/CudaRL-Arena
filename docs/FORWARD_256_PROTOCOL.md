# Approved unchanged-task 256-update trial

Explicit user approval received2026-10-30 at21:33:17: run ONE256-update trial.
Start fresh from the SAME initial weights/seeds and four starts as the closed
four-update pilot; do not resume its trained weights. The sole source change
relative to its frozen `tools/forward_demo.cu` is the update count and matching
budget declarations/display. Task, physics2, gains/limits, Gaussian variance,
PPO/GAE/Adam settings, episode resets, safety and success criteria stay unchanged.
The original `FORWARD_DEMO_PROTOCOL.md` describes these unchanged details.

New output identity `build/forward-demo/2026-10-30-256-v1/`. Freeze source,
executable and numeric registration before its sole execution. Budget includes
32 readiness +131072 training (256*16*32) +640 before/after evaluation controls:
**131744 controls /1317440 native5ms slots**, failures included. Maximum1024
Adam steps per network. No retry, extension, retuning, added case or adoption.
Existing stop conditions apply to the whole run. Previous pilot and all closed
physical evidence remain immutable. Native execution estimate20-30s; record
actual setup and native times separately. Report success, COM displacement,
joint stroke behavior and safety even if learning fails. No live Godot claim,
checkpoint overwrite, runtime migration, commit/push/merge/deployment follows.
