# Tadpole Pond entry menu

Latest interaction step: on a healthy CUDA developer build, **Open pond** now opens [sinking-food interaction with concurrent PPO](live_feeding.md). Click/tap water to feed; native particles sink and drift, and the swimming actor learns in background practice worlds. Its initial reference is brief. There is no three-pellet replacement rule; all active candidates are scored with a reported 128-pellet resource guard. The menu falls back to the frozen reference when CUDA is unavailable. **Experiments → Trained reference** preserves the earlier CPU/CUDA saved-weight pond and its untrained comparison. Help distinguishes the new learner's save/resume from the preserved original headless selector console. Modal navigation is deferred until input dispatch finishes to avoid detached-window key events.

The user requested a pond-focused product on October 28. The startup menu now presents **Tadpole Pond** with Open pond, Experiments and Help & training. Its preview is captured from the current natural deep-pond scene. The charcoal/orange identity is preserved, with a single larger preview and responsive actions. CartPole/Racing are removed from the user-facing menu; their scenes, policies, training, artwork and developer routes remain intact. Navigation Lab remains deferred. The working title is Tadpole Pond; automatic evolution/growth is not yet delivered.

For developer-only CartPole/Racing routes, saved playback still loads bundled actor-only JSON weights into the C++ CPU showcase without constructing a trainer or probing CUDA. Fresh training still initializes a CUDA session paused at update zero. These retained capabilities have no launch controls in the pond menu.

**Experiments → Trained reference** preserves the earlier saved swimmer in live physical simulation: selector update 320, swimming update 64,350, healthy CUDA or CPU fallback. Its Current control uses steady 0.5/0.75 cm/s suspended-food advection; [reference controls](pond_current_controls.md) and [model evidence](experiments/swimmer/drifting-food/README.md). Its untrained comparison deliberately resets the same layout, and its saved weights do not update. The new default feeding pond instead has sinking particles and concurrent motor adaptation as described above. Both show actual session collection and stop their sessions when leaving. Turning-flow trials remain research-only.

Live-current follow-up: the existing switch is now **Current**, with a **0.5/0.75 cm/s** strength selector beside it. Both apply steady flow to the swimmer, food and visual transport, without a session reset or policy update. Turning flow remains research-only. See [controls and verification](pond_current_controls.md). Native extension build, dedicated 41 CPU/45 CUDA control assertions, 51 CUDA original-current checks and 141 CUDA navigation checks pass. All four established viewport sizes were rendered and inspected. Escape input is marked handled before changing scene to avoid accessing a detached viewport. The application config name remains unchanged to preserve user-data/save paths. Physical growth and local rock disturbances remain future work.

**Experiments**, available from the launcher and pond, contains the original saved swimmer, shallow-water live PPO training and the newest live swimmer. Available local current-stall and recovery recordings have explicit **Play recording** actions. Recorded diagnostics are supplementary; the ambiguous Watch current test shortcut is removed from live controls. Optional recordings are omitted when their data is absent. The Help & training dialog explains live inference, explicit training, headless saving/resume and preserved checkpoints. The portable prototype retains its single supported original-pond route.

The source project requires the built GDExtension DLL. Training requires a supported NVIDIA GPU/driver. CPU playback has been tested on the RTX 5070 Windows machine, not on hardware without an NVIDIA driver. A distributable Godot export, including JSON export filters, has not been validated.

## Reproduce the saved policies

From the repository root after configuring CMake:

```powershell
cmake --build --preset release --target export_demo_policies
build/Release/export_demo_policies.exe godot-project/launcher/policies
```

This uses the project's C++/CUDA PPO trainers: seed 1, 20 CartPole updates, and 200 racing updates with the 1,200-update annealing schedule. The exporter validates 12 CartPoles for 500 steps and eight racing agents for 1,200 steps before writing their policies. No Python or ML framework is involved.

## UI cleanup — October 28, 2026

The user reported being sent to recordings when expecting live behavior, then requested a cleanup of the menu and whole GUI. This pass addresses navigation and the shared pond interface, while preserving CartPole/Racing functionality and the established visual identity. Shared dialogs use the charcoal palette, visible focus, a dimmed background and 44-pixel Close/Done actions. Technical metadata is progressively disclosed. Short landscape windows give more space to the creature and use a compact graph with fewer axis ticks. Existing model weights, physics and training saves are preserved.

The rendered navigation suite covers the main live route, actual current control, session-rate calculations, optional recorded routes, help/dialog bounds, keyboard opening/closing, Escape staying in the current pond, detail expansion, session cleanup and unchanged saved weights. Captures are under ignored `build/pond-ui-checks/` at 1280×720, 768×1024, 390×844 and 844×390. The Windows screen-control helper failed before any input because its launcher encounters the unrelated locked D: drive; verification uses Godot's own rendered Windows runs and synthetic Godot input, not physical touch or assistive-device testing.

Recorded checks: final rendered CUDA navigation 141/141; CPU headless navigation 121/121; current-control checks 47/47 on each backend; expanded demo-menu checks 107/107; original selector playback 27/27, untrained comparison 191/191, current recording 32/32, recovery recording 57/57 and latest live agent 28/28. Protected policy/full-save references and the selected latest candidate match their saved hashes. These UI checks do not replace the preceding native GPU-required suite or claim new training performance. The application is opened to the newest live agent with gentle current enabled for user review.

Internal review heuristics: usability, hierarchy, consistency, responsiveness and feedback 9/10; accessibility and performance verification 8/10. Prioritized follow-ups are real assistive/touch input checks before platform expansion, then measured frame-time/startup performance when background practice is connected. Secondary pond controls scroll on small screens. These are bounded review judgments, not measured quality or a reason for another cosmetic redesign.

## Historical validation — October 26, 2026

- Current three-demo menu: `tests/godot/demo_menu_smoke.gd` passes 101 checks with zero failures, covering all five actions, keyboard focus and activation, CartPole/Racing playback and learning, automatic pond training, Back/Escape cleanup, and rejection of unsupported pond playback. Vulkan renders at 1280×720, 768×1024, 390×844 and 844×390 were inspected. All three rows fit the desktop; on narrow screens the pond action is reachable through the existing scroll container. See [pond validation](three_food_learning.md) for physical/learning tests.
- Pre-merge regression: `ctest --preset release --output-on-failure` passed all **134 native cases**, including CUDA tests with none skipped, the 1,200-update traffic-learning regression, retained-reference execution checks and three-seed original racing learning. The running pond stayed open during the suite; this run is correctness evidence, not an isolated performance benchmark.
- The evidence below records the original two-demo menu pass; its behavior remains covered by the expanded smoke test.
- Release extension, native test executable, and policy exporter built successfully.
- `cudarl_tests.exe ppo_demo:`: 5/5 passed, including paused initialization, background training, and trained CartPole playback.
- `cudarl_tests.exe racing:`: 9/9 passed, including CPU/GPU environment agreement.
- `tests/godot/demo_menu_smoke.gd`: 71 checks, zero failures. Covers four menu choices, keyboard focus order and activation, Escape, fresh update-zero starts, actual training progress, saved playback without training, returning to the menu, session cleanup, and invalid policy rejection.
- The smoke test loads the actual JSON files through Godot: all 12 CartPoles survive 500 steps; racing completes at least eight clean laps with zero contacts/wall crashes over 60 simulated seconds.
- `tests/godot/demo_menu_render.gd`: Vulkan renders at 1280×720, 768×1024, and 390×844 for the menu and all four demo modes. Images are saved under ignored `build/menu-checks/`. Menu actions have targets at least 44 pixels high, fit horizontally, and the final phone action can be reached by scrolling.
- Visual inspection caught and fixed the compact CartPole arena collapsing to zero height. The regression check now verifies its height and separation from controls. The watch panel also has enough space for its saved-policy label.

These are Windows window-size checks with synthetic input, not tests on physical phones or tablets. No Linux work was performed. The full numerical training suite and long-duration performance benchmarks were not rerun for this menu change.

## Design review

Scores are conservative internal heuristics, not measured user outcomes.

| Dimension | Menu | Demo views | Evidence / next recommendation |
|---|---:|---:|---|
| Usability | 9 | 8 | Five clear menu actions; compact training controls require scrolling. If compact training becomes a priority, put Start learning before advanced CartPole settings. |
| Hierarchy | 9 | 8 | Menu titles/actions are distinct. CartPole's older training panel still gives technical settings substantial prominence; simplify only in a separately agreed pass. |
| Accessibility | 8 | 8 | Focus order, key activation, Escape and menu target sizes pass. Test assistive technology and text enlargement before claiming full accessibility compliance. |
| Consistency | 9 | 8 | Launcher uses one palette and component style; CartPole retains its older controls. Harmonize these only if requested. |
| Responsiveness | 9 | 8 | Menu passes all three sizes; demo panels scroll. Validate real touch hardware if that becomes a deployment target. |
| Feedback | 9 | 8 | Menu has loading/disabled/error states; backend errors are shown. A dedicated no-driver machine test remains useful for training failure presentation. |
| Performance | 8 | 8 | Playback avoids trainer creation; no new overall frame-time or startup benchmark was collected. Measure those before making responsiveness claims. |
