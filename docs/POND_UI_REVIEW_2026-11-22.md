# Tadpole Pond frontend — November 22, 2026

The user explicitly requested rebuilding the Godot interface from scratch and
cleaning up its menu. The project now has **one frontend**:
`live/main.tscn` → `live/main.gd`, with `live/pond_world.gd` presenting the exact
native body geometry and transforms. The application name/save namespace is
unchanged; the window title is Tadpole Pond.

The primary view is the physical creature in a quiet teal water environment.
There is a small status line, Play/Pause, Restart and a Hold test/Swim test choice.
Three compact measurements can be hidden. The grid, research headline, charts,
comparison-button row and metrics sidebar have been removed. Results, method,
validation scope and source information belong in a keyboard-accessible Details
drawer. There are no automatic interface animations or automatic playback.

Hold test and Swim test each start from rest. Changing the selection explicitly
restarts the physical state; it does not demonstrate learned braking or a
continuous movement-to-holding transition. Completion is labeled Test complete.
Measurements distinguish speed, distance from the holding target (or forward
swimming distance), and absolute depth error. The fixed-world target is never
recentered to conceal drift. Its label stays below the body and away from the eyes.

## Current saved policy

The frontend explicitly loads
`res://research_data/learned-amplitude-v1.bin`, the verified seven-update native
PPO candidate. It never silently falls back to the older rejected four-gain
package. The visible creature uses frozen saved weights in live native physics;
this interface does not run an optimizer or perform learning updates.

Details accurately reports 6/6 original and 6/6 withheld 64-second calm-water
holds, plus two 300-second holds. All four short swims are retained. The long
swim result remains **28/30 windows, with all 28 previously passing windows
retained**; the two known startup misses are not hidden. Two distribution
parameters were learned while inherited swimming-network weights stayed frozen.
Braking, currents, disturbances and general holding competence remain separate.

`tools/export_pond_view.py` packages small evidence metadata into
`live/research_data/viewer.json`. Package SHA-256, native package version, policy
method and lesson update count must match the policy actually loaded before
assessment claims are displayed. Missing or mismatched metadata produces no
success claim. This permits evidence/model updates without rebuilding the GUI
for every routine check. Missing policy files disable playback and clearly
report the problem rather than substituting a different actor.

## Verification

Actual Windows Godot 4.4.1 rendering was inspected at **1280×800, 390×844,
768×1024 and 844×390**. The viewport sizes are recorded in the reports. The
rendered `tests/godot/pond_ui_smoke.gd` checks pass **31 assertions per size**.
A missing-policy case passes 25 assertions and a metadata/package mismatch case
passes 32: **181 assertions in total**.

Checks include 44px control targets, viewport bounds, primary-button text
contrast in focus/hover/pressed states, mouse Play and keyboard Pause, synthetic
touch Restart, from-rest task resets, Details focus isolation, Escape/return
focus, drawer bounds, optional measurements and error handling. The UI process
loop is explicitly disabled before Play/Pause checks, so native start/pause can
be tested without any call to its physical step function. Every report records
**zero native physical substeps**, including after resets. No extra training or
physical evaluation was performed by this frontend work.

The coordinator separately verified the V2 native preview with its approved
80-control/800-substep comparison against saved physical poses. Those controls
are not part of the UI assertions or added to the closed learning study.

Artifacts are under `build/gui-remake-2026-11-22/`:

- `before/manifest.json` and 15 copied files preserve pre-remake source, packages,
  metadata, old UI tests and unique earlier interface evidence with matching hashes.
- `*-smoke.json` contain the four-size, missing-policy and mismatch assertions.
- `render/desktop.png`, `phone.png`, `tablet.png` and `landscape.png` show the new
  frontend paused with the actual saved policy loaded.
- `*-smoke-details.png` show the disclosure drawer.

The superseded chart/review frontend, paused frontend test and their dedicated
recording exporter/data were removed after a dependency search and preservation.
The native HoldLabPreview/FrozenSwimmer APIs and independent native fidelity test
remain intact. The obsolete Hold Lab app and its Project Manager were closed
normally to switch the registered project; unrelated apps were left alone. Test windows
exit deliberately; normal startup has no auto-close path. Final normal app
launch and Godot Project Manager registration were coordinated separately.

The Project Manager entry now opens this worktree's `live` project instead of
`build/godot-hold-replay-2026-11-21`. The older project and its evidence remain
preserved. A **Tadpole Pond Research** desktop shortcut opens the correct project;
the installed production shortcut is unchanged. The final interactive session
(verification PID 12532) was started normally, without test/capture arguments,
with a clean startup log and remained open. The user confirmed the new GUI looked
good. The user may play or pause it directly; this is separate from automated
physical-test accounting.

Future automated UI checks must not appear as short-lived user-facing app
windows. Keep those checks hidden or clearly coordinated, and launch the normal
interactive app separately after QA so it stays open for the user.

## Conservative design review

These are review heuristics, not measured quality certifications.

| Dimension | Score | Evidence or next action |
|---|---:|---|
| Usability | 9 | One creature view and three core controls; results are disclosed on demand. |
| Hierarchy | 9 | The creature is the focus; research detail no longer competes with play. |
| Accessibility | 8 | Keyboard, visible focus, control sizing, measured primary contrast and synthetic touch pass. Before broader distribution, check Narrator, enlarged system text and real touch hardware, then correct observed defects. |
| Consistency | 9 | One frontend, one pond palette, common controls and clear reset semantics. |
| Responsiveness | 9 | Four rendered sizes and bounds checks pass, including short landscape. |
| Feedback | 9 | Play/Pause, restart, completion, native errors and metadata rejection are explicit. |
| Performance | 8 | No UI learner, replay pipeline or automatic visual animation. Measure frame-time percentiles on reference hardware before making performance claims. |

Phone/tablet images are resized Windows viewports, not a mobile release.
Screen-reader support, 200% text scaling, hardware touch and frame pacing remain
unverified. No WCAG or performance certification is claimed. The recommendations
do not authorize further redesign or new research runs.
