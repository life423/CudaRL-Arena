# Flexible-tail Godot extension

From the repository root, build and stage the native extension for the
`flexible-tail-preview` project. Close its running pond window before staging.

```powershell
cmake -S tools/pond-flexible-tail-live -B build/flexible-tail/live-native -A x64
cmake --build build/flexible-tail/live-native --config Release --target stage_flexible_tail_live
```

The build uses matching Godot 4.4 C++ bindings and their prebuilt Windows x64
Release library. A linked worktree can use the original checkout's `external`
directory. Configure `CUDARL_GODOT_CPP_ROOT` and `CUDARL_GODOT_CPP_LIBRARY` for
another dependency location.

The stage target copies the built DLL into the project's extension directory.
Godot Project Manager can then run **Tadpole Pond | Flexible Tail**. The pond
opens paused with **Explore pond**, continuous zoom and Details in a left column.
Restart, the original motor and saved recordings are under Details.
The default native path uses the qualified15-action coverage chooser80 and
frozen motor88 in live_data. The deployment manifest selects its separately
named file; previous chooser88 is preserved. Motor actions are held80ms and chooser decisions0.8s,
matching the evaluator independently of rendering. The viewer performs saved
policy inference and physical simulation; PPO is a separate native GPU process.
The original manual motor remains available in Details. The deployment manifest
records exact model/library hashes and the finite qualification scope.
