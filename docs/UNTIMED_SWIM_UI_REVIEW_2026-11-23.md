# Untimed Swim presentation review

The ordinary Swim selection uses the native interactive untimed request. Its
elapsed time and forward distance share the existing measurement reading;
there is no additional control or persistent banner. Rest retains its bounded
64-second check. The separate qualified Free swim mode remains unavailable for
the current saved policy.

The selector tooltip and Details explain that Swim runs until paused or stopped
by a bank/physical check. Details retains the limitation that reliable turns
around the pond are unverified, the unchanged saved-weight/no-learning scope,
and the existing closed comparison outcome. Restart and request changes still
explicitly start a new state. A startup capability check rejects an older
native module instead of silently pairing untimed UI with the old timed API.

The pond, creature, camera/zoom, high-contrast identity panel, colors, typography,
touch targets and existing drawer interactions are unchanged.
Visual verification found a separate inherited checked-state defect: Show
measurements used white pressed text on the cream drawer. Its normal, hover,
pressed, hover-pressed and focus text colors now share the established dark
text token; no new style or color was introduced.

## Verified private checks

- Explicit Godot4.4.1, isolated build/us26/project with the rebuilt native DLL.
  Private test windows were hidden/offscreen; this agent did not touch the
  user's Godot window or issue a native step() call.
- **160 assertions passed**,40 at each actual1280x800,390x844,768x1024 and844x390
  viewport. Requested and actual sizes match in every final receipt.
- The test disables the view process loop before changing modes. It verifies
  the real untimed native API/metadata, no completion deadline, separate
  qualified Free swim mode, bounded64sRest, no-learning scope, preserved
  comparison disclosure,44px controls, rail containment and modal focus.
- Time65.4s/distance1234.5mm are explicitly **synthetic presentation values**
  used to exercise formatting and narrow layouts. They are not a trajectory,
  measured distance or evidence that this creature swims for65seconds.
- The measurement display and Details screenshots were visually inspected
  across all four sizes. The combined clock/distance fits on the narrow phone;
  controls and text fit without horizontal overflow. Checked Show measurements
  is now legible. The five text states use#243e37 on#fdfbf5, a calculated
  WCAG contrast of11.16:1; all five actual-theme assertions pass4.5:1.
- Every run reports **zero native controls/substeps and optimizer steps**.
  Latest final .err files are empty. These are UI integration checks, separate
  from root's bounded physical continuation test.
- Initial viewport-harness failures are preserved: the engine overwrote the
  size assigned in _initialize with the project1100x760 default. Resizing the
  started window fixed the test only. The earlier35-check passing renders are
  also preserved before the checked-text contrast correction.

Final receipts and PNGs: build/us26/ui/verified-<size>.json/.png/.log/.err;
the screenshots use -swim.png and -details.png suffixes. The aggregate
build/us26/ui/verification.json binds all four final receipts and source SHA256s.
Main source SHA256:1d02036a09515a492a0547a6d0b7d4a14170b902ec40755a4f7fc8806be81cf6.
Test source SHA256:f6b7265b4eb482b9affdee9a5b9d368741b3d98340daff92f76c2420cc0c2945.

## Actual app handoff

Root gracefully closed the actual old app, installed the byte-matched reviewed
DLL, and reopened the correct live project with Godot4.4.1 (PID39492). The actual
window was visually inspected with the saved creature, selected Swim, opened
Details, enabled readable measurements and closed Details. It remains paused
with Swim and0:00/0.0mm visible, ready for the user to Play. No extra visible
physics steps were issued. The actual log has no errors; Godot Manager's favorite
entry still points to this worktree's live project. Root also inspected the
final phone measurements and desktop Details PNGs. The separate bounded native
check, not these paused screenshots, establishes continuation past eight seconds.

## Conservative critique

Scores are internal design-review heuristics, not measured quality claims.

| Dimension | Score | Evidence and prioritized recommendation |
| --- | ---: | --- |
| Usability |9| Existing Swim action becomes untimed; elapsed time and distance are visible together when measurements are enabled. |
| Hierarchy |8.5| The quiet pond and compact controls remain primary. Priority2: condense accumulated research history in a separately scoped cleanup; current movement scope requires scrolling in Details. |
| Accessibility |8.5| Checked-text contrast,44px targets and modal keyboard focus are checked. Priority2: verify native screen-reader and actual-device touch behavior. |
| Consistency |9| Uses existing tokens, controls, drawer and measurements rather than another banner or selector. |
| Responsiveness |9| Four actual viewport sizes fit, including the longer synthetic clock/distance value. |
| Feedback |9| The native untimed flag drives the display; bank/safety stops retain their actual reason, while turning remains explicitly unverified. |
| Performance |8.5| The presentation adds no physics or optimizer work. Priority2: measure sustained frame/input timing before making performance claims. |

No further redesign is implied. This does not establish infinite pond
navigation, turning, current competence, physical skill retention on an
experimental runtime, full accessibility or actual-device touch behavior.
