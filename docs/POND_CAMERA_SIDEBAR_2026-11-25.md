# Pond camera and sidebar — November 25, 2026

User direction: remove the white bottom dock and redundant controls; keep
controls in a left column and allow intermediate views between the tadpole
and a closer pond overview.

The default column contains Explore/Pause, current motion, Zoom in/out,
Pond view/Follow tadpole, and Details. Details contains Restart, the original
motor, recordings and trace loading. Recording controls use the same column.
The column scrolls on short displays and reserves camera space beside it.

One orthographic zoom range replaces the close/wide mode switch and the
former three-times close zoom limit. Buttons, wheel, +/− and pinch share this
range. Each button/wheel step scales the span by 1.16. The pond preset uses 85%
of the fitted span, capped to 1.5 times the projected pond height in portrait.
It intentionally crops outer bank rather than making the pond tiny beside
the column; Zoom out can still reveal the whole pond. Framing blends from swimmer to pond center as
the span increases, without moving or resizing the physical creature.

Verification used Godot 4.4.1 with the RTX 5070 OpenGL renderer. All 215 engine
checks passed at 390×844, 768×1024, 1100×760 and 844×390. Main, expanded Details and
pond frames were rendered and inspected. Checks cover44px button targets,
column containment, scroll reachability, subject framing, zoom limits,
intermediate scales, keyboard/wheel/pinch events, and the original motor and
recordings. The gesture check exposed duplicate identical injected packets;
identical pinch packets are coalesced within a render frame. Physical touch
hardware and screen readers were not tested. In the actual registered desktop
project, mouse clicks verified the closer Pond view and Follow presets, mouse
wheel intermediate zoom, and Explore. The final app was reopened and left
running learned exploration with the updated UI.

The suite performs zero physics controls and zero optimizer updates. This is
a presentation-only change: motor88, chooser88, original motor, DLL, habitat,
physics settings and research archives remain unchanged. The current deployment
manifest refreshes only the two presentation hashes and UI check count.

Evidence: build/pond-camera-sidebar/ui-result.json and the 12 rendered PNGs.
Run from the foundation checkout with an existing output directory:

```powershell
& C:/Users/aiand/Godot/Godot_console.exe --path flexible-tail-preview --script (Resolve-Path tests/godot/flexible_tail_exploration_ui.gd).Path -- "--output=$((Resolve-Path build/pond-camera-sidebar).Path)"
./tools/test-flexible-tail-package.ps1
```

Internal design review (judgment, not user research): hierarchy9/10,
consistency 9/10, feedback 9/10, desktop usability 9/10, accessibility 8/10,
small-screen usability 8/10, visual restraint 9/10. Text contrast is measured
against opaque button/readout surfaces; normal and pressed text exceed 4.5:1
(normal 10.87:1, primary 8.33:1, pressed secondary 5.55:1).
Follow-up priorities are screen-reader/text-enlargement verification and a
collapsible left column if phone users need more unobstructed pond width.
Short landscape currently requires scrolling to secondary controls. These
limits do not justify another redesign or changes to the learned controller.
