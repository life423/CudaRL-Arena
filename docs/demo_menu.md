# CartPole and Racing startup menu

The approved scope is two demonstrations: CartPole and Racing. Navigation Lab is deferred. The menu follows the supplied charcoal/orange reference. Its preview illustrations come from atlas regions of the user's supplied concept screenshot; they are conceptual artwork, not screenshots of newly implemented 3D scenes. The actual demonstrations retain their existing 2D rendering.

Watch trained loads bundled actor-only JSON weights into the C++ CPU showcase. It does not construct a trainer or probe CUDA. Train from scratch initializes a new CUDA training session, paused at update zero until Start learning is pressed. Back to demos and Escape stop the session before returning. Saved actors are playback snapshots, not resumable optimizer checkpoints.

The source project requires the built GDExtension DLL. Training requires a supported NVIDIA GPU/driver. CPU playback has been tested on the RTX 5070 Windows machine, not on hardware without an NVIDIA driver. A distributable Godot export, including JSON export filters, has not been validated.

## Reproduce the saved policies

From the repository root after configuring CMake:

```powershell
cmake --build --preset release --target export_demo_policies
build/Release/export_demo_policies.exe godot-project/launcher/policies
```

This uses the project's C++/CUDA PPO trainers: seed 1, 20 CartPole updates, and 200 racing updates with the 1,200-update annealing schedule. The exporter validates 12 CartPoles for 500 steps and eight racing agents for 1,200 steps before writing their policies. No Python or ML framework is involved.

## Validation — October 26, 2026

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
| Usability | 9 | 8 | Four clear menu actions; compact training controls require scrolling. If compact training becomes a priority, put Start learning before advanced CartPole settings. |
| Hierarchy | 9 | 8 | Menu titles/actions are distinct. CartPole's older training panel still gives technical settings substantial prominence; simplify only in a separately agreed pass. |
| Accessibility | 8 | 8 | Focus order, key activation, Escape and menu target sizes pass. Test assistive technology and text enlargement before claiming full accessibility compliance. |
| Consistency | 9 | 8 | Launcher uses one palette and component style; CartPole retains its older controls. Harmonize these only if requested. |
| Responsiveness | 9 | 8 | Menu passes all three sizes; demo panels scroll. Validate real touch hardware if that becomes a deployment target. |
| Feedback | 9 | 8 | Menu has loading/disabled/error states; backend errors are shown. A dedicated no-driver machine test remains useful for training failure presentation. |
| Performance | 8 | 8 | Playback avoids trainer creation; no new overall frame-time or startup benchmark was collected. Measure those before making responsiveness claims. |
