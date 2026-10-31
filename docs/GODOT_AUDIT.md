# Current Godot audit and focused cleanup

Base HEAD e84db5fda7d2224416508344071a1c6af80eaad4, worktree clean when inspected.
User expanded the read-only request to cleanup as findings were confirmed. No
new checkpoint integration, training, physical trial, commit or remote operation.

## Structure verified from current source

| File | Role and evidence |
|---|---|
| live/project.godot:5 | One main scene, res://main.tscn. No autoload section. |
| live/main.tscn:2 | Attaches main.gd to a plain Node3D root. No inherited/instanced scene. |
| live/main.gd:1 | The only production GDScript; extends built-in Node3D directly. |
| live/addons/frozen_swimmer/frozen_swimmer.gdextension:2 | frozen_swimmer_init entry; one Windows extension. |
| src/godot/register_live_types.cpp:5 | Registers one custom Godot type, FrozenSwimmer. |
| src/godot/frozen_swimmer.h:8 | Extends engine RefCounted directly; no custom base-class ladder. |
| src/godot/frozen_swimmer.cpp:10 | Owns a FrozenController through a small private implementation; composition. |
| src/motor/frozen_controller.h:8 | Plain C++ class, owning one World, task and frozen actor. No Godot dependency. |
| tools/view-live.ps1:4 | Launches only live/. Builds only frozen_swimmer when requested. |
| tests/godot/viewer_controls.gd:1 | New opt-in test driver extends engine SceneTree, not the production script. |

The application has one .gd, one .tscn, one project and one .gdextension. The new
test is the sole additional first-party .gd; it loads the same existing scene.
There is no current eight-layer custom inheritance hierarchy. CanvasLayer,
MarginContainer, VBox/HBox, meshes and link Node3Ds are ordinary node composition.
The C++ bridge's private implementation is ownership/composition, not inheritance.

Root script -> FrozenSwimmer bridge -> FrozenController -> shared motor stepping.
The bridge loads res://assets/frozen-controller.bin once (cpp:15); native loading
checks schema, shape, finite parameters, checksum and compiled source contract
(frozen_controller.cpp:10). Reset reuses the same actor. There is no per-speed
policy loader, policy inference in GDScript, Godot rigid-body solver or duplicated
Godot water/force implementation. Mesh poses come from native state, and dimensions
from native geometry. Rendering primitives and decorative eyes remain presentation.

The live asset is still the previous8mm/s actor. The separately committed learned
constant-speed checkpoint is not auto-loaded; this audit did not add that feature.
Speed requests delegate to the shared native setter; sampling and physical advances
remain in frozen_controller.cpp:36 and stroke_motor.h:26. The native controller owns
pause/finish guards; Godot owns only the render-frame accumulator and view snapshot.

## Confirmed cleanup performed

- Removed the persistent GDScript running flag, which shadowed native paused state.
  Start/Pause/Reset, buttons and frame scheduling now read the native snapshot.
  UI failures pause the native controller and disable Start/Pause until recovery.
- Reuse the snapshot already read after a control interval rather than fetching
  state before stepping and again while drawing (main.gd:155,184). Cached render
  data is not a second authoritative world.
- Centralized repeated startup/resize camera settings in _layout_camera (line49).
- Fixed a source-confirmed screenshot failure path: the paused capture launched
  an asynchronous save without awaiting/checking it, then could write success.
  It now awaits completion and returns on failure (line199).
- Fixed silent result-file failures: opening/writing/flushing errors now produce
  explicit failure and nonzero exit, including the capture callers (line213).
- Rejected speed input is redrawn from the accepted native value, avoiding a
  misleading displayed request. The validated live speed range is unchanged.
- Removed one unused SDK include in frozen_swimmer.cpp. No new wrapper classes,
  managers, scenes, policy-loading paths or physical-state owners were introduced.

No active legacy project/replay/CreatureLab references were found in the current
viewer, bridge, launch helper or CMake. Historical docs and ignored experiment/
editor caches are not active application inheritance and were not purged.

## Checks actually performed

Godot4.7.2 script parse and strict Release frozen_swimmer build passed. Eight
opt-in headless control checks passed: load paused, Start/Pause agreement, Reset,
UI failure pause/recovery and zero physical advances. The test disables automatic
node processing before tree entry; its only manual process call has delta0.

Real paused renders at360x640 and1100x760 were inspected. Start/Pause/Reset action
targets remain44px high; the speed field remains disabled at the live8mm/s range.
Injected screenshot-save failure and result-file failure both exited3 with error
evidence, not success. All five UI/control cases recorded0physical substeps.
Expected injected errors appear in logs; unavailable sandbox shader caching did
not prevent rendering. Evidence: build/godot-audit/2026-10-31/, including ui-v1/.

Hashes verified unchanged for the live actor/provenance and all11protected native
controller-contract sources. Native bridge change is only an unused include.
No personal save or original-checkout source was modified. Physical4second
autopause/trajectory parity was NOT rerun; it remains prior evidence, not a fresh
claim. No training or optimizer call occurred.

## Remaining maintenance risks, not newly observed runtime failures

- main.gd owns UI/mesh/capture code in one modest script. Keep cohesive functions;
  no need for a new base-script hierarchy or another viewer.
- Dictionary fields and dynamic bridge calls are string contracts; current
  accesses match native output and the exercised path loaded successfully.
- The view still knows50ms,4s and verification-budget constants. A future change
  to native lesson timing should expose that metadata once, rather than add another
  set of timing constants or a wrapper controller.
- The native setter can change/reset a request while running. The current8-only
  control cannot exercise variable-speed switching. Before integrating the new
  constant-from-rest checkpoint, enforce that lesson's reset/start restriction;
  do not imply the old slowdown gate passed. No such integration was done here.

Coverage: all current first-party Godot project/script/scene/extension files,
all three native Godot bridge files, frozen controller, relevant shared request/
stepping interfaces, launch/build references and live asset metadata. Dependencies
were checked at the integration boundary, not audited line-by-line. Ignored build
snapshots/caches, retired historical trees, unrelated trainer internals and vendor
engine/bindings internals were excluded. Manual keyboard navigation and a running
physical episode were not executed. The cleanup was prepared from e84db5f; the
user subsequently authorized its local commit and feature-branch push only.
No main-branch merge or deployment is included.
