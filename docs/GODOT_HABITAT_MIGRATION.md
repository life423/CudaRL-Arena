# Godot habitat view migration — October 30, 2026

This is a reviewable, opt-in script migration, not a physics/controller change.
The four original habitat scripts and their scene routes remain preserved until
the new implementation passes configuration, rendering and private-session
regressions. Do not report a fresh-load parse issue as reproduced if it was only
seen in an existing editor session. Root-owned native factory and lifecycle
changes address explicit extension readiness independently of this migration.

## Current and candidate dependency paths

Historical path:

`deep_water → deep_pond → latest_agent → feeding_pond → arch → cove → boulder`

The ledge, holding and predator prototypes branch from that historical path.
Native availability, geometry preparation, current controls and automatic pellet
rendering are therefore coupled to unrelated earlier learning/viewing stages.
Every historical `_ready()` calls inherited lifecycle code and then polls again.

Candidate path:

`deep_water → deep_pond → habitat_pond_view → modes/{arch,cove,boulder,overhang}`

The first two levels provide shared view/camera/frame infrastructure. The habitat
view composes explicit session configuration, native geometry rendering and
current controls. It does not inherit feeding or frozen-viewer session startup.
The new mode entrypoints are ordinary scripts, not active route replacements.

## Method migration map

| Original responsibility | Candidate owner | Behavior preserved |
|---|---|---|
| `arch._ready()` constructs a native lab before the inherited world builder | Root shared lifecycle calls `_create_lab()` once, then `habitat_pond_view._prepare_lab()` | Native geometry is obtained before terrain/solids are built; no earlier session is started |
| Four `_start_lab()` overrides | `habitat_mode.start_session()` with explicit method/source/destination keys | Exact native start method, population, backend and save identities |
| Four `_load_habitat_geometry()` variants | Mode geometry method + `habitat_render.outline()` | Native arch/cove/boulder/ledge geometry and closed-outline removal |
| `latest_agent._build_ui()` current widgets | `pond_current_controls.build()` | Same named `WaterControls`, `GentleCurrent`, `CurrentStrength`; two tested strengths and signals |
| Current toggles/strength handling | Habitat view `_set_current()` / `_select_strength()` | Same native `set_steady_current(speed, true)` call and no world reset |
| `latest_agent._physics_process()` flow phase | Habitat view flow advancement | Visual phase advances only by actual native elapsed simulation time |
| `feeding._build_world()` and `_apply_pose()` automatic pellet display | Habitat view pooled rendering | Up to128 current native pellet positions; no fake pellets, food replacement or click-driven body movement |
| Cove terrain/water/shore construction | `habitat_render.terrain()/water_mesh()` plus shared native shore adapter | Native outline vertices and terrain heights/normals, same mesh sampling and camera-fit points |
| Arch stone material and torus | `habitat_render.stone_material()/solids()` | Same material parameters, torus dimensions and native Z-axis orientation |
| Boulder sphere | `habitat_render.solids()` | Same native sphere center/radius and layout metadata |
| Overhanging ledge | Existing `overhang_rock.gd` called by the render helper | Existing exact slab/opening geometry; no change to the native solid |
| Boulder/ledge visit-relative graph | Habitat view `_update_session_curve()` | Subtract inherited lifetime totals; preserve native counters and clocks |
| Overhang pause/readiness/frozen status | Habitat view explicit `frozen` mode | Live simulation can pause; no PPO, checkpoint writing or old-policy fallback |

## Save and route compatibility

The candidate keeps the existing exported checkpoint properties. Only the
selected mode's configured source/destination pair is opened:

| Mode | Native start | Source property | Destination property |
|---|---|---|---|
| Arch | `start_arch_learning` | `source_feeding_checkpoint` | `arch_checkpoint` |
| Cove | `start_cove_learning` | `source_arch_checkpoint` | `cove_checkpoint` |
| Boulder | `start_boulder_learning` | `source_cove_checkpoint` | `boulder_checkpoint` |
| Ledge | `watch_overhang` | `source_boulder_checkpoint` | None: source is read only |

All defaults retain their `user://pond-policies/.../training.bin` identities.
Existing `.tscn` paths, names and script `.uid` files remain stable. Prototype
compatibility controls retain the established names; automatic-food habitats
keep manual/auto-feed actions hidden. Separate predator/holding entrypoints are
not silently routed into this candidate migration.

## Checks and promotion boundary

First use `tests/godot/habitat_mode_config_smoke.gd`: it creates no native lab,
starts no learner, writes no checkpoint and advances no physical simulation.
It checks script inheritance/configuration, exported defaults, current-widget
signals and unchanged geometric rendering primitives.

Then compare actual rendered phone/tablet/desktop layouts and the existing
private-checkpoint habitat regressions before root changes any scene route.
Use copied inputs and unique private destinations for tests; never open personal
saves for validation. Verify native pose/food/current continuity, exact geometry,
pause behavior, mode-specific save identities and frozen-versus-learning labels.
The ledge remains live fixed-weight inference; learning captions belong only to
sessions that actually run PPO. A UI load/build alone is not behavior validation.

Automatic approval review rejected a command that would have replaced the four
original habitat scripts wholesale, citing risk to ongoing overhang behavior and
insufficiently narrow authorization. That command performed no writes. This
candidate migration keeps those originals intact and leaves route activation to
the root after concrete checks; it does not bypass the rejected overwrite.
