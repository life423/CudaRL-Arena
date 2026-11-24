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
opens paused with Play, left/right 15° requests, zoom, and saved recordings.
