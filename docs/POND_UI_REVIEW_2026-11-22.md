# Tadpole Pond frontend — November 22, 2026

## November 23 close aerial camera and header contrast

The user rejected the earlier full-pond framing as still too distant. The app
now starts above the creature with roughly 230–260 pixels of full-body span on
desktop/tablet/phone and about 160 in short landscape. Actual measured native
head-center to tail-center spans are 175, 157, 175 and 108 pixels at the four
viewports below. The aerial camera follows actual head/tail positions; banks may
extend beyond the frame. Scroll, pinch or + / - adjusts zoom; the existing lower
angle is labeled Side view. Details blocks camera inputs.

Title and status now sit on an opaque #FDFBF5 surface. Their measured contrast
ratios are **11.16:1** (#243E37 title) and **5.53:1** (#566B61 status), independent
of grass/water behind the header. Header height adapts to rendered font metrics,
and the phone camera control starts below it without overlap.

All **140 assertions** (35 at each of 1280x800, 390x844, 768x1024 and 844x390)
pass with zero native controls/substeps. Actual four-size renders were inspected,
including text containment, contrast, full-body camera bounds, modal focus,
keyboard/wheel/pinch zoom and unchanged native poses. Initial QA exposed a phone
header overlap and an undersized short-landscape creature; both were corrected
before this final verification. Artifacts and pre-change source are in
`build/close-aerial-contrast-2026-11-23/verified/` and `before/` respectively.
Hardware pinch/touch, Narrator and 200% system text remain unverified; measured
header contrast is not a whole-app accessibility certification.
The normal app was reopened as PID 34260, inspected with the close aerial
camera and readable header visible, and left paused with clean startup logs.

## Branch publication verification — November 23

The complete current source builds with warnings as errors for the native Godot
extension; a separate CPU-only configure/build and CTest suite pass. A relocated
project using the freshly compiled private DLL passes 25 parser, 18 actual-study
disclosure and 35 camera/UI checks with zero physical controls/substeps. No live
DLL or learned policy was replaced for these checks.

Current-study result and independent proof are now bundled byte-for-byte in
`live/research_data`. Portable resource paths locate those copies; the original
audit identity is retained only as provenance text and is never opened as a file.
The packaging utility and Git attributes preserve their exact SHA bindings on
fresh checkouts. All three rejected-study archive manifests were verified before
publication, including intentional historical files beneath nested build paths.

## November 23 aerial framing

Historical intermediate framing, superseded above: the aerial camera fitted the rounded pond rim, waterline and nearby reeds
instead of the full meadow extent. The viewing angle, landscape and creature
camera are preserved. Actual desktop, phone, tablet and short-landscape renders
were inspected at 1280x800, 390x844, 768x1024 and 844x390. All 100 existing
camera/UI assertions pass with zero physical controls/substeps. The camera check
now samples the real rounded waterline instead of obsolete rectangular corners.
Artifacts and the pre-change camera source are under
`build/aerial-zoom-2026-11-23/`. This is a presentation-only change; it does not
advance any learning or physical study.
The normal app was reopened as PID 11756 and the closer aerial view selected
through its Pond view control. The actual window was inspected and left open,
paused, without test/capture or automatic-exit arguments.

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

A small **Current swimming study** section appears only inside Details when
`live/research_data/current-study.json` binds a real closed result by path and
SHA-256. A qualified outcome additionally needs a hash-bound independent verifier
that identifies that exact result. Missing metadata adds no speculative status;
malformed or mismatched metadata cannot claim a successful study. The active
calm-water policy and primary controls are unchanged.

The latest closed v2 study is displayed as **Study complete. Current swimming
is not ready**, with **32 native PPO updates** and **4 → 8 / 16 current cases**.
Its exact aggregate and independent proof are SHA-bound. Withheld and current
endurance suites did not run, so their null results are not converted into
zero-of-sixteen failures. Details explicitly identifies the active preview as
the saved calm-water policy and does not adopt an experimental current actor.
The previous v1 numerical failure remains in the evidence archive and reports.

## Verification

Actual Windows Godot 4.4.1 rendering was inspected at **1280×800, 390×844,
768×1024 and 844×390**. The viewport sizes are recorded in the reports. The
rendered `tests/godot/pond_ui_smoke.gd` checks pass **31 assertions per size**.
A missing-policy case passes 25 assertions and a metadata/package mismatch case
passes 32: **181 assertions in total**.

The earlier v1 current-study disclosure had separate verification: **15 parser
assertions** in `tests/godot/current_study_metadata_smoke.gd`, plus **16 rendered
assertions at each of the same four sizes** in
`tests/godot/current_study_view_smoke.gd`, for **79 additional assertions**. All
passed with zero physical controls/substeps. The actual failed-study section was
inspected for readable wrapping, scrolling, modal bounds and keyboard focus;
synthetic parser fixtures remain clearly labeled under ignored
`build/current-ui-2026-11-22/test-only-fixtures`. Actual rendered screenshots and
counter reports are `build/current-ui-2026-11-22/*-current-details.png` and `.json`.

The final v2 disclosure has **22 parser assertions** and **18 rendered assertions
at each of the same four sizes**, for **94 checks**. They pass with zero native
controls/substeps, including exact independent-proof binding, measured 4→8/16
progress, explicit nonqualification, null unrun suites, calm-policy selection,
scrolling/bounds and focus isolation/restoration. Actual final screenshots were
inspected at every size; wrapping is readable and no horizontal overflow occurs.
Artifacts are under `build/current-ui-v2-2026-11-22/`. This supersedes the actual
status shown by the earlier v1 screenshots; the older checks remain historical
evidence rather than being counted again as new v2 verification.

After these hidden checks, the normal interactive app was launched as PID 15196,
without test/capture/auto-exit arguments. It stayed open with the Tadpole Pond
window title and clean startup logs. The Project Manager registration was checked
again and still points at this worktree's `live` project. The active policy is
paused at startup; user-controlled playback is separate from closed-study work.

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
