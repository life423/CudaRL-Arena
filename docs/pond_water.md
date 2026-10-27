# Water presentation, first pass — October 26, 2026

The user approved incremental water work. This first pass adds motion-driven ripples and a subtle body wake, with a visible review before further water work. It preserves PPO, the native neural networks, food rules and authoritative physics.

## Delivered

The visible tadpole's measured head displacement emits trailing arcs. Actual tail-tip travel accompanied by joint movement emits smaller rings. A coasting body can leave a wake without inventing tail strokes; stationary anatomy emits nothing. Disturbances expand and fade over 1.35 simulated seconds, reset on restarting the visible trial, and are masked to the water ellipse.

`pond_ripples.gd` owns a fixed 32-slot MultiMesh pool per visible pond. Only the displayed animals get this layer; the 1,024 background CUDA worlds remain their original native simulation. Each instance stores a physical source position/bearing, birth time and motion-derived intensity. The shader draws local surface-normal/color bands on small transparent geometry, rather than evaluating a long history at every pixel of the whole pond. Expired instances collapse to zero-area geometry. The rendered clock follows the same interpolated simulation time as the animal.

This is a visual surface approximation sourced from submerged motion. It is not calibrated hydrodynamics or a coupled fluid solver, and it applies no propulsion/buoyancy/contact forces. It does not select a food target or change what the network observes. Existing base water and caustic shaders retain their earlier behavior. Depth coloring, refraction and reflections are separate later passes.

## Validation

The water-specific Godot test passes **89 checks** for rest, coasting versus joint activity, fade-out, bounded storage, valid source locations, effect disabling/reset, render/layout behavior at four sizes, all food visible in overview, and exact native-state equivalence to a separate CPU controller through collections. The 320-step real-controller trace generated 76 body wakes and 106 tail disturbances; 16 remained active at capture, with four food collections. An initial test failure was a fixture timing mismatch: the rendered scene had advanced before the reference. Resetting both to the same start before replay yields exact state agreement.

The camera regression passes **440 checks**, live learning **187**, and persistence **34**. Actual on/off renders were inspected in follow and overview at 1280×720, 768×1024, 390×844 and 844×390. The tadpole and gold food remain readable. The first opacity setting was too faint; the final pass increases the local crest/trough contrast while keeping the head-facing wake masked and the layer translucent.

## Matched performance probe

The opt-in `tests/godot/pond_water_bench.gd` resumes the existing shuffled run, continues its training and saves on normal exit. It uses one trainer, two visible ponds, follow cameras, 1280×720, maximum training speed and a **60 FPS cap**. Four six-second samples run **off / on / on / off**, with 1.5 seconds to settle after switching. The previous game was closed normally first; its full learner was copied into `user://pond-policies/water-upgrade-backup-20261026/` for safety. The visible app resumed after the probe.

| Disturbances | Combined training steps/s | Mean frame interval | Highest per-window p95 interval |
| --- | ---: | ---: | ---: |
| Off | 2,669,047 | 16.664 ms | 18.032 ms |
| On | 2,680,451 | 16.661 ms | 17.053 ms |

Both enabled samples had 16–17 active disturbances in each pond; disabled samples had zero. These short windows show no apparent slowdown at this cap. The small positive throughput difference is measurement variation, not evidence that ripples accelerate PPO. Continuing learning, periodic evaluation/autosaving and OS work remain noise sources. These are wall-clock training throughput and rendered frame intervals, not GPU utilization, kernel timings or uncapped frame capacity. [Raw measurements](experiments/graphics/pond-water-20261026.json) are retained.

Internal UI review: hierarchy/consistency/feedback 9 (existing controls and palette retained), usability 8 (short-landscape scrolling), accessibility 8 (readability and existing keyboard controls checked; physical touch/assistive technology untested), responsiveness 8 (four windows), performance 8 (bounded layer and short matched probe; long-run profiling pending). Priorities are review in actual motion, longer performance samples if needed, and choosing the next creature-development experiment. No extra HUD or effects controls were added.

## Next decisions

There were four proposed water passes: **(1) wake/ripples, (2) depth/color, (3) refraction, (4) reflections**. Only the first is delivered. The other three are optional visual work, not prerequisites for a legged creature. The user's latest priority is planning a tadpole that develops legs and climbs out of water using the project's own neural networks trained with PPO. Preserve the current swimmer/checkpoints and scope that as a separate anatomy/physics/training experiment; do not expand water polish automatically or replace learned control with animation.
