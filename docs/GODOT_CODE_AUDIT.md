# Godot resource and lifecycle audit

October 30, 2026. This audit concerns presentation, resource loading and native
API/lifecycle boundaries. It does not authorize another motor/contact experiment,
change training, replace the production swimmer or migrate personal saves.

## Initial findings from source inspection

The project initially contains 37 GDScript files. The longest pond inheritance
chain has seven script-to-script edges, eight scripts total:

`overhang/holding/predator → boulder → cove → arch → feeding → latest_agent → deep_pond → deep_water`.

The three leaf alternatives each inherit the full chain independently. `_ready`
overrides execute through most layers. Geometry, native lab creation, feeding
controls, training/playback mode and camera/UI updates are coupled through this
chain. In particular, `arch_pond._ready` creates its native lab before delegating
to the base, because geometry must be available before world construction;
`deep_water._ready` otherwise builds the UI/world before creating/starting a lab.
This ordering is a concrete maintenance risk, not proof of a current failure.

Native classes depend on the installed extension build. `DeepPondLab` is shared
with the CPU release. `CudaDevice`, `MazeArena`, `NetworkTrainer`, `PpoLab`,
`RacingLab`, `TadpoleLab` and `DeepWaterLab` are registered only by the CUDA build.
The isolated motor viewers have their own optional native modules. Several
retained developer scripts name CUDA-only classes directly; a CPU-only export
does not thereby promise that every legacy developer scene is usable.

The screenshot's unresolved parent/native class must be checked in a fresh,
correctly initialized project process. A standalone script-only check that
reports missing `DemoLaunch` does not reproduce the real autoload context.
Conversely, a stale editor cache must not be declared fixed merely because one
fresh process loads a resource.

## No-simulation resource check

`tests/godot/project_resource_audit.gd` runs as a SceneTree script with
`--path godot-project`. It waits for the configured `DemoLaunch` autoload, then
loads every project `.gd`, `.tscn`, `.tres` and `.gdshader` resource without
instantiating audited scripts/scenes/native classes. It reports compilation,
base-script chains, `_ready` definition depth, static preload/inheritance cycles,
missing dependencies and native class/method availability. `--include-tests`
also loads test scripts without executing their bodies. `--audit-output=...`
writes a new JSON report; it refuses to overwrite earlier evidence.

Root executes the harness and records fresh before/after logs. At initial
authoring, no execution or clean-parser claim is made by this document.
Resource-load success does not test actual scene startup, backend availability,
GPU compute, shader rendering, checkpoint leases, event handling or UI layout.

## Verification boundaries and test gaps

- Existing menu/pond smoke tests can launch real training and write test saves.
  They are not substitutes for a no-simulation project-wide parse check and
  must not be run indiscriminately during the stopped motor study.
- Fresh loading and inheritance tests need the real autoload context; otherwise
  a reported singleton error can conceal the dependency actually under review.
- Missing-native-class startup should be tested by an explicit factory resolver
  override, rather than unloading the user's editor DLL or pretending that a
  class still registered in `ClassDB` is absent.
- Shared startup/shutdown tests should establish one lab per scene, build-world
  geometry availability before rendering, harmless stop-on-exit after failed
  startup, and clear error states without touching personal training saves.
- CPU release tests must exercise the shared feeding route, explicit backend
  choice and genuine fallback. Preserved CUDA-only developer scenes should be
  clearly unavailable when their module is absent; no CPU learning claim follows
  from loading their GDScript alone.
- Frozen viewers must keep update counters fixed. Learning scenes need separate
  deliberate tests of real updates, save ownership and shutdown. This audit
  does not extend any research budget or prove movement acceptance.

## Status

Source inventory and audit harness authored. Fresh before/after resource and
focused lifecycle results are pending root execution and coordinated cleanup.

## Native C++ structure review

This section records source findings, not a request to rewrite the numerical
core. Approximate current file sizes: `deep_pond_lab.cpp` 615 lines,
`swimmer_ppo.h` 386, `arch_session.cu` 331, `shelter_session.cu` 330,
`holding_session.cu` 251, `ppo_session.h` 180 and `feeding_session.cu` 169.
File length alone is not the defect; responsibility and ownership overlap are.

| Boundary | Concrete coupling | Confined improvement |
| --- | --- | --- |
| `DeepPondLab` startup methods | Arch, cove, boulder, predator and holding repeat stop, backend validation, device probe, path conversion, session factory and exception handling. | Extract one private CUDA-session startup helper while preserving every public binding, error message, factory and backend rule. |
| `deep_pond_lab.h` | Owns mutually exclusive legacy `TrainingSession`, feeding-session, playback and CPU-actor state plus mode flags; includes CUDA/PPO implementation headers in the Godot bridge. | First document the active-session invariant; later consider an opaque implementation object after stable lifecycle tests. No checkpoint contains a serialized `DeepPondLab` object. |
| `get_state` / `get_stats` | Bridge computes geometry, converts body/link/food data to Godot values, understands layout IDs and threat fields, and creates presentation status dictionaries. | Move pure geometry and dictionary conversion helpers to focused implementation files without changing returned keys or numeric values. |
| `FeedingSession` / `FeedingStats` | Feeding, habitat training, shelter, holding practice and frozen overhang inference share a learning-named interface. Stats carry stage-specific fields; overhang legitimately has no-op save/drop/stop methods. | Add an explicit mode/capability description through an adapter before changing consumers. Preserve the existing interface and save formats. |
| Custom session workers | CPU feeding, CUDA feeding, arch, shelter and holding each own a lease, worker, mutex, pause/save flags, policy publication and autosave loop. Their adoption/retention gates differ deliberately. | Share only proven lifecycle primitives after tests; do not combine training objectives, candidate gates or raw checkpoint layouts. |
| `BasicSwimmerTrainer` | Shared Gaussian PPO update code also owns legacy experiment settings, shape-based `if constexpr` dispatch, checkpoint magic, history and policy transfer. Input count is sometimes used as task identity. | Keep the verified update math/order unchanged. A future separation of task/checkpoint adapters from the updater needs exact continuation tests, not a broad cleanup in this pass. |

Two existing status/error details deserve contract tests before alteration:

- `DeepPondLab::get_error()` returns the bridge error or the legacy
  `TrainingSession` error, but not `FeedingSession::stats().error`. Asynchronous
  feeding/habitat errors are exposed separately as `learning_error` in stats.
  Consumers that check only `get_error()` do not receive the same error channel.
- The feeding-session dictionary sets `playback_mode=true` and
  `training_ready=true` even when the session owns concurrent PPO; actual
  readiness/activity are separate `learning_ready` / `learning_active` fields.
  This is ambiguous metadata, not evidence that learning is absent. Existing
  consumers must be audited before compatibility keys change.

The existing `PondPlayback` abstraction is a useful boundary already: explicit
CPU/CUDA preference, version/size-checked optional module APIs and continuation
of the same physical frame when automatic fallback is allowed. Explicit CUDA
requests reject unsupported operations instead of silently switching. The
generic `TrainingSession` also already provides idempotent stop/join and policy
publication. These should be retained rather than replaced during UI cleanup.

## Native factory dependency contract

Pond callers expect `{"instance": RefCounted-or-null, "error": String}`.
At this review the shared factory constructs a generic `Object`; the pond
wrapper casts it to `RefCounted`, so that expected ownership contract should
be checked explicitly. Missing classes and missing methods are separate failures.
Check class/method availability before construction; accepting a class name
does not establish backend or task support. A CPU `DeepPondLab` registers habitat
start methods but returns an explicit CUDA-required error for those tasks.

| Consumer layer | Required methods used by that layer |
| --- | --- |
| Common live view | `get_error`, `get_state`, `get_stats`, `step`, `stop` |
| Common learning controls | `set_paused`, `set_updates_per_second`, `save_training_now`, `get_evaluation_history`, plus that mode's start method |
| Deep pond comparison | `set_watch_best`, `set_untrained`; `watch_controller` only for its saved-reference route |
| Latest saved reference/current | `watch_progress_controller`, `set_steady_current` |
| Interactive feeding | `start_feeding_learning`, `drop_food` |
| Habitat views | Their declared `start_*_learning` or `watch_overhang`, matching geometry getter, and the terrain/flow sampling methods actually used by that mode |

Inactive/hidden controls must not create accidental API requirements. Conversely,
an always-visible control must not call an unchecked method after startup.
Optional features should be declared per mode instead of adding every legacy
method to every factory request. Fault-injection tests should verify missing
class, stale class missing one required method and unsupported backend without
unloading an extension in the user's editor.
