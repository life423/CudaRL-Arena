# Deep pond presentation — October 27, 2026

## Bottom readability pass and cosmetic stopping point

The user approved sand/silt/gravel bed detail, gentle sunlight patterns and a restrained cast shadow, then requested stronger separation between the tadpole and the bottom. The floor now uses warmer muted sand and dark neutral silt, sparse embedded gravel texture and slower localized caustics. Embedded stones are material detail, not new collision objects. The native basin and swimmer physics are unchanged.

The deeper-water creature keeps its teal/cream/gold identity with more distinct dorsal colour, a warmer pale underside and a restrained rim light. Its tail uses its radial UV to distinguish top/bottom: the old material incorrectly treated its world-space swimming depth as an underside. This material correction does not change geometry, animation, body scale or actions. The shallow pond's separate skin material is unchanged.

A higher sunlight angle and reduced shadow opacity give a readable floor shadow cast by the real model. The landscape receives creature/rock shadows without casting its own coarse self-shadows. Terrain normals are calculated from the existing surface profile for smoother illumination. The gravel filter was simplified to a derivative-free texture. Some faceting remains visible on the floor; further visual tuning is deferred under the user's stopping instruction.

The presentation suite now passes **138 checks**, including actual image differences for visible bed detail and the creature's cast shadow, and a perspective projection check confirming that a same-sized creature appears smaller farther down. Existing multi-depth visibility, interpolation/camera stability and paired native-state checks remain. Renders/logs are under `build/pond-learning-checks/pond-bed*` and `presentation-*`.

**User priority: stop cosmetic iteration after this pass and return to AI work.** Further naturalism, model, vegetation, water or lighting polish is deferred unless explicitly requested. Next work should evaluate current food-choice bias/route efficiency and preserve learned swimming, with later mixed-skill retention and water-to-land locomotion still the long-term goal. Do not turn remaining aesthetic recommendations into an automatic redesign loop.

## Current natural setting — supersedes the exposed cutaway

The user found the cutaway still looked like a bowl. The pond is now embedded in continuous ground, with a mottled grass/mud shoreline, uneven dry-bank rock groups and instanced vegetation. The exposed soil cross-section, glass-like side section, floating base and bright elliptical outline have been removed from the active scene. The old section/soil shader files remain as historical resources, not active meshes.

The overview remains stationary, with a higher angle to see deeper into the water without exposing the entire bank. The optional Closer view retains its previous three-quarter bearing and fixed zoom. Tracking now uses the same interpolated native world position used to render the head, a dead zone and a 0.4-second exponential smoothing time constant. It does not use tail, body heading, roll or pitch. The camera and creature are sibling nodes under the scene world. The existing manual interpolation of native C++ snapshots is reused; a second engine interpolation layer was not added.

Where terrain, a rock or vegetation lies between the camera and the creature, a small screen-space dithered visibility window reveals it. The mask follows the interpolated body centre and affects rendering only. Ground behind/below the creature remains solid. Projection is calculated from the final camera position; opaque materials with discard avoid sorting a transparent landscape against the water. Some fine stippling remains visible at the edge of this visibility aid. It is a deliberate viewing aid, not an actual hole or new swimming force.

The underwater mesh still samples the same analytic physical basin. Only dry exterior decoration/ground relief is altered; no traversable shoreline, new underwater obstacles, PPO changes, model changes or training reset are included. The water footprint remains elliptical. A truly irregular physical shoreline would need matching native terrain changes and separate validation rather than a misleading decorative collision surface.

The current presentation test passes **135 checks**, including camera/creature transform independence, interpolation endpoints/midpoint, camera stability, four viewport sizes, actual visible-versus-hidden pixel comparisons at 0.8/2.0/3.2 model-unit depths and unchanged native state. The 960-step paired-controller trace remains identical with and without the presentation. Static depth probes are render fixtures, not evidence of newly learned behavior. Test logs are `build/pond-learning-checks/natural-pond.*`; depth renders are `natural-depth-*.png`. The existing training save is preserved and backed up before refresh.

Current internal review heuristics: hierarchy/feedback/consistency 9; naturalism 8, accessibility verification 8. Priorities are human feedback on this setting, less visible stippling/foreground occlusion and more organic *physically matched* shoreline geometry in a separate step. Physical devices/assistive tooling remain untested. The following sections describe the earlier cutaway iteration and are historical where superseded above.

The user requested a pond that visibly contains water while keeping three-dimensional swimming readable. The first underwater close camera was rejected as unpleasant to watch. The final choice starts in a stationary **Pond overview**, with an optional **Closer view** at a comfortable three-quarter angle. The underlying native simulation, neural network, PPO, food rules and checkpoint format are unchanged.

## Delivered appearance

- Opaque terrain replaces the old ghostly foreground slope. A precise foreground render cut exposes the water column; visible soil cross-sections give the retained shoreline thickness. The physical slope is still present.
- The exposed water boundary follows the existing basin geometry. A thin waterline and a lightly reflective surface make the volume legible. Depth-buffer reconstruction supplies a restrained optical depth tint; subtle animated normals provide surface movement. These are presentation effects, not new forces or simulated fluid volume.
- A static low-contrast sky supplies reflections. Direct specular hotspots on the water are disabled because they obscured the creature/food; restrained environment reflections remain. Directional lighting and shadow opacity were reduced, and deeper ground is darker with less stretched caustic contrast.
- Depth labels and food-to-floor guides are shown in overview. The sidebar retains readable depth values in either view; the closer view hides the floating guides to reduce clutter.

The top surface uses the current viewport's depth buffer with Godot 4.4's reverse-Z reconstruction, following [the official depth documentation](https://docs.godotengine.org/en/4.4/tutorials/shaders/advanced_postprocessing.html). Reflection, alpha and normal handling use [Godot's spatial shader interface](https://docs.godotengine.org/en/4.4/tutorials/shaders/shader_reference/spatial_shader.html). This pass does not implement screen-space scene reflections, detailed refraction, CFD, anatomy changes or new environmental obstacles.

## Camera decision to preserve

**Default: stationary overview.** It fits static pond bounds on viewport resize and does not follow the animal, react to tail pose or zoom when food respawns. All three foods remain visible.

**Optional: Closer view.** It retains the same elevated world bearing and a fixed 7.2-unit viewing distance, adjusted only for narrow viewport aspect. The animal can move within a small dead zone before a softly damped pan follows it. Body heading, roll, articulation and food choice do not control camera rotation or zoom. One common focus sets both the eye and target, avoiding the earlier mismatched camera/look-at lag. Explicit view changes switch framing directly without a cinematic orbit.

Do not restore the rejected default underwater/eye-level close-up or pose-dependent fitting. The user wants comfortable spectating with enough pond context to see movement through a volume. The creature's actual roll remains visible; the camera does not rotate to compensate for its learned motion.

## Verification

`tests/godot/deep_pond_presentation.gd` passes **126 checks**. It restores two isolated copies of the same trained policy and compares native states through a 960-step trace while rendering and moving the camera. The visual scene and native reference remain identical. It checks body framing, food visibility in overview, stable camera transforms when the creature turns, a stationary overview when the creature moves, reachable controls and no horizontal overflow.

Actual follow/overview renders were inspected at **1280×720, 768×1024, 390×844 and 844×390**. Screenshots and logs are under `build/pond-learning-checks/presentation-*` and `pond-presentation.*`. This is desktop viewport/keyboard verification; physical touchscreen, screen-reader and OS reduced-motion integration remain unverified. The stationary default offers a low-motion way to watch.

An exploratory 60-FPS-capped water-layer off/on/on/off probe is in `build/pond-learning-checks/pond-presentation-bench.json`. A separate user-started viewer was discovered running during that sample, and its camera was the earlier rejected close-up. Therefore it is **not** a controlled before/after throughput result for the final presentation. Its frame intervals stayed around 16.66 ms, but no GPU utilization or training performance guarantee is claimed. A future controlled benchmark should use the final camera, one trainer and longer windows.

The active learner was closed normally and backed up before refresh. Safety copies are under the pond save's `before-water-presentation/` and `before-steady-camera/` folders. Refresh resumes the existing full training state; no optimizer/model reset is involved.

Internal review heuristics: hierarchy/feedback/consistency 9; visual polish 8, accessibility verification 8 and performance evidence 7. Prioritized follow-ups are a physical-device accessibility pass, controlled rendering/training profiling, then more natural shoreline/material detail. These are recommendations, not authorization to broaden the current step. Food-choice bias remains a separate learning issue.
