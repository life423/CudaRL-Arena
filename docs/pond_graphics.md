# Pond graphics roadmap — October 26, 2026

October 27 update: the deeper pond now has a [water-filled cutaway and revised camera presentation](pond_presentation.md). The user rejected the first underwater close camera. Preserve the stationary overview default and optional stable, wider three-quarter Closer view; do not reintroduce pose-driven zoom or heading-based camera offsets.

The user approved the first manageable graphics step: a closer camera presentation, with the whole-pond view still available. They explicitly requested preserving the remaining graphics ideas. This is the roadmap, not blanket authorization to implement all effects at once.

| Order | Step | Intended result | Status |
| --- | --- | --- | --- |
| 1 | Better camera | Calm close tracking, a useful three-quarter angle, complete tadpole framing, accessible whole-pond overview | Implemented and rendered/interaction checked |
| 2 | Water | Depth-dependent color, refraction, designed reflections and movement-driven ripples | First wake/ripple pass delivered; other water passes remain optional |
| 3 | Tadpole model | More organic anatomy, translucent fins and clearer movement detail | Queued; preserve the real simulated pose and avoid a scripted swimming gait |
| 4 | Pond detail | Irregular shoreline, underwater vegetation, improved rock/ground materials | Queued |
| 5 | Lighting | Better environmental/underwater illumination, caustics and softer shadows | Queued |
| 6 | Environmental animation | Restrained suspended particles, swaying vegetation and disturbed sediment | Queued |

The user's supplied reference also proposed pairing camera follow with swimming ripples as a first upgrade. The active task remains the explicitly approved camera step; water is recorded as the next independent step, so shader/physics scope does not expand silently.

Subsequent approval: the user requested incremental water work. The first [motion-driven wake/ripple pass](pond_water.md) is implemented and measured. Three proposed water passes remain (depth/color, refraction, reflections); they do not block the larger goal. The user has now emphasized planning the legged body and water-to-land learning with the project's native neural networks and PPO, rather than continuing graphics polish indefinitely.

## Visual direction and constraints

Aim for a polished, stylized miniature ecosystem that remains compelling to spectate. The creature's learned physical behavior is central. Presentation should make that behavior legible without claiming the network has selected a particular food target. A camera framing choice is not a policy action.

Forward+ already provides shadows, ambient occlusion, tone mapping and shader support in this project. The current water shader is a lightly animated transparent surface; caustics on the sand are a procedural visual pattern. Later water work must deliberately design refraction/reflection/depth handling. Enabling screen-space reflections alone is not a complete transparent-water solution. Such visual effects do not establish simulated fluid dynamics, buoyancy, surface crossings or land contact.

Preserve the existing color/material identity, clear food visibility in overview, keyboard navigation and comfortable controls. Prefer a steady camera bearing over automatic orbiting, shake or zoom pulses. Keep a stationary overview available for those who prefer less camera movement. Actual OS-level reduced-motion integration and physical touchscreen behavior remain verification limits.

CUDA training and Godot rendering share the GPU. Measure frame intervals and training steps per wall-clock second under matched settings before/after meaningful graphics changes, reporting competing GPU work and evaluation overhead. Do not equate rendered FPS with GPU utilization or assume more effects improve the result. Expensive effects and high-resolution assets require a demonstrated visual benefit.

This roadmap does not authorize changing PPO, rewards, actuation, physical dynamics, food rules or the water-to-land lifecycle. Continue in small visible steps and retain working demos and full training persistence.

## Camera step delivered

Both pond scenes start in **Follow tadpole** view: a perspective camera with a fixed world-facing bearing, a 42-degree vertical field of view and damped translation toward the rendered body's center. It fits the head, articulated tail and membrane, including folded poses and narrow windows. Follow lag is bounded to keep a fast-moving body in view. Turning never spins the camera, and bending never triggers zoom pulses. This camera uses the interpolated physical pose; it does not select a food target, change the policy or drive the creature.

The **Pond overview** choice restores the stationary orthographic whole-pond view and shows all three food pieces. Explicit mode changes switch promptly, without an animated orbit. The comparison has one selector controlling both panes; on narrow windows it shares the existing pond-selection row. The normal pond puts the selector beside the title in portrait and below navigation on desktop. Keyboard selection and 44 px controls are retained. This step offers two framing modes; free orbit and continuous manual zoom have not been added.

The existing game was closed normally to save its full learner before restarting the viewer. A matching camera-mode performance probe resumed that same run, continued training and saved on exit; the visible app then resumed again. A safety copy of the full pre-camera learner is retained at `user://pond-policies/camera-upgrade-backup-20261026/training-before-camera.training`. UI camera preferences are session-local; the learned model/optimizer state remains persistent.

## Verification and performance

- The dedicated camera smoke test passes **440 checks**: keyboard popup selection, portrait/landscape selector reachability, whole-body bounds at 24 extreme-position/heading/joint combinations per pane/window size, stationary overview, fixed camera bearing, all food visible in overview, materially larger close framing, and unchanged native body/food/training/policy-selection state while switching views.
- The comparison regression passes **93 checks**, normal learning scene **187**, training-resume scene **34**, and archived one-food scene **100**. Overview visibility assertions explicitly select overview; follow mode is separately checked for complete anatomy. The first synthetic popup test bypassed Godot's window input route; it was corrected to send physical key events through `Input.parse_input_event`, and keyboard selection now passes.
- Actual renders were inspected at **1280×720, 768×1024, 390×844 and 844×390**, in both views. Test screenshots use fixed render-only anatomy probes for repeatable visual review; the live viewer follows its actual physical trajectory. Short-landscape windows retain scrolling.

The opt-in `tests/godot/pond_camera_bench.gd` measures a single resumed trainer at 1280×720, maximum training speed, a **60 FPS cap**, and two six-second windows per view in **overview / follow / follow / overview** order. There was no second CudaRL trainer running during this measurement. Results are in [the raw performance record](experiments/graphics/pond-camera-20261026.json).

| View | Training steps/s, combined windows | Mean frame interval | Highest per-window p95 frame interval |
| --- | ---: | ---: | ---: |
| Overview | approximately 2.68 million | approximately 16.66 ms | 17.021 ms |
| Follow | approximately 2.68 million | approximately 16.66 ms | 17.021 ms |

The short sample showed no meaningful throughput difference at this cap. These are wall-clock training throughput and rendered frame intervals, not GPU utilization, uncapped rendering capacity or GPU kernel timings. Continuing learning/evaluations and normal OS work remain noise sources; broader performance claims need longer profiling. No water/model/habitat/lighting shader change was included.

Internal UI review: hierarchy 9, consistency 9 and feedback 9; usability 8 (short-landscape scrolling), accessibility 8 (keyboard and readable popup states checked; OS reduced-motion/native assistive technology/physical touch untested), responsiveness 8 (four windows; physical devices untested), performance 8 (matched short sample only). Priorities are longer GPU/frame profiling, OS preference/device checks, and the separately scoped water step. Avoid expanding controls or redesigning the established lab to chase scores.
