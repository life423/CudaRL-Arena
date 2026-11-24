# Shared pond geometry preparation — November 23, 2026

The user chose to **keep the current visible pond**. This change gives native
code and Godot one geometry description, while preserving the existing motor
and simulation. It does not yet resolve physical bank contact or teach turning.
Work is local on `codex/shoreline-alignment`, based on main `495caff`.

## Scope and preserved behavior

| Quantity | Value / status |
| --- | --- |
| Visible water radii | 0.52 / 0.34 m, unchanged |
| Flat bed radii | 0.22 / 0.14 m, unchanged |
| Floor height / model scale | −0.16 m / 0.04 m per model unit, unchanged |
| Active whole-body motion guard | 0.20 / 0.12 m, unchanged |
| Physical bank response / bank-aware policy input | Not implemented |
| Motor, chooser or optimizer updates | None |

`src/environment/pond_geometry_v1.h` defines a pure SI descriptor, the existing
organic rim, bed/slope/land height and analytic upward normal queries. Invalid
parameters and nonfinite queries return invalid results. The code selects no
actions and changes no body state. Its interpolation weights are mathematical
surface interpolation, not creature movement constraints. The active force,
torque, filter, integration and safety equations are unchanged.

The native bridge exposes `get_pond_geometry_descriptor`, `sample_pond_outline`
and `sample_pond_surface`. The descriptor explicitly reports `geometry_only`,
`used_by_contact=false`, `used_by_policy=false` and the separate legacy guard.
Godot batches queries for its 20,769 terrain vertices and 161-point water rim;
it no longer owns duplicate shoreline/height equations. A mismatched extension
reports an error instead of substituting a separate terrain formula.

The stop notice now says **“Paused at the current simulation limit”** and
Details explains the remaining bank/contact/navigation gap. The compact notice
reserves camera space so it does not cover the stopped creature in a narrow
landscape window. The original native stop payload is retained for evidence
compatibility. It does not mean the creature reached the visible shoreline.

## Verification and accounting

No dynamic qualification or retained-skill evaluation was executed for this
geometry preparation. The formal checks below executed no physics. Ordinary
interactive playback is accounted for separately below. All earlier research
allocations remain closed.

| Check | Observed result |
| --- | --- |
| Standalone CPU geometry executable | 43,405 assertions passed; 30,725 geometry queries |
| Independent height-reference maximum error | 4.4408920985006262e−16 m |
| Analytic-normal vs numerical-reference maximum error | 2.4744254605371663e−9 |
| Native extension CPU build | `/W4 /WX`, passed; no PPO/CUDA build |
| Godot final zero-step checks | 39 each at 1280×800, 390×844, 768×1024 and 844×390: 156 passed |
| Rendered terrain height maximum error | 6.30281978969416e−8 m, within 1 µm |
| Water-rim position error | 0 model units |
| Packed-normal vs independently packed reference error | 0 |
| Raw analytic-to-renderer normal quantization difference | 1.22816229122691e−4, retained in receipts |
| Formal-check physical controls / substeps / CUDA kernels / optimizer steps | 0 / 0 / 0 / 0 |

The root executed the standalone geometry check once; this report transcribes
that observed result without rerunning it. Final UI receipts are
`build/pg26/ui/attempt-02-{1280x800,390x844,768x1024,844x390}.json`, with initial
and historical-stop PNGs and empty error logs. The stopped pose is a display-only
fixture from the earlier untimed-swim evidence, not new physical execution.
The private test scene disables native stepping and checks zero controls.

The first UI attempt is preserved in `attempt-01-*`. It exposed a test mismatch:
uncompressed analytic normals were compared directly with renderer-packed
normals at a 1e−5 threshold. The final test preserves that threshold and compares
against an independent renderer-packing reference, while separately reporting
the raw quantization difference above. The first narrow-landscape attempt also
found the notice obscured the creature; the layout was corrected. A zoom
assertion was corrected to measure after the actual layout pass. These are
render/test corrections, not changes to physical acceptance tolerances.

The test expects a private project whose parent contains `saved-stop-state.json`:
copy `live` without its cache/bin into `build/pg26/project`, install the checked
DLL there and copy the test script into that project. Extract `final_state` from
the preserved `build/us26/native-check.json` into
`build/pg26/saved-stop-state.json`. This is the historical 509-control fixture,
not a new run. Run Godot4.4.1 hidden with `--path build/pg26/project
--script res://pond_native_geometry_view_smoke.gd -- --test-size=1280x800
--test-output=<absolute receipt path>`; substitute each of the four sizes above.
The script disables processing before readiness. These instructions document
the completed checks, not authorization to restart a physical study.

## Installed app and evidence binding

The checked DLL was installed in `live/addons/frozen_swimmer/bin/`; its SHA256 is
`50555dfcbcb9d1cfe59b3c526d20259b413fe57f7582e4c269307e95b2d48d23`.
The previous DLL is preserved at `build/pg26/previous-live-frozen_swimmer.dll`.
Godot4.4.1 was launched normally in the actual `live` project; the selected
Godot Manager project path still points there. Root inspected the aerial pond
and the updated shared-geometry/contact/navigation explanation in Details.
The app log contains the4.4.1/OpenGL startup without an error. OpenGL rendering
on the RTX5070 is not evidence of CUDA physics or training execution.

During this interactive inspection the ordinary Rest preview was seen playing
before it was paused, then was seen playing again before opening Details paused
it. No control/substep trace was captured for this interactive playback; those
counts are **unknown**, not zero. It was not a requested-turn, retained-skill or
shoreline-physics qualification and contributes no success evidence. No Play,
Restart or mode-change action was sent by the inspection automation. Source
starts paused; the playback observation alone does not establish an autoplay
defect. No training or optimizer runs in this viewer.

All three protected policy/checkpoint hashes matched again after inspection.
Final Godot source files match the privately tested copies. The source, final
receipt and installation bindings are recorded in `build/pg26/verification.json`
(SHA256 `962ff68adf509e73e11b54a392f637470731f6b5336c85373111b05ab1e38e22`).
`preserved-policies.json`, `post-policy-verification.json`, `app-install.json`,
all failed/final UI receipts/screenshots and the visible-app log remain beside
it. CUDA compilation/runtime parity is not tested by this geometry preparation.

Internal UI review (conservative heuristics, not measured quality): usability8,
hierarchy8, accessibility8, consistency9, responsiveness8, feedback9 and
performance8. No new blocking defect was found. Prioritized follow-ups are to
put the latest world limitation before the long historical results in Details,
verify actual assistive-technology navigation beyond the tested focus/contrast,
make the creature more prominent in the portrait default framing, and measure
startup/query costs before adding more terrain detail. Existing manual zoom,
opaque high-contrast labels and44px controls are preserved. These follow-ups
do not justify another broad redesign or establish physical capability.

## Remaining work and retained ideas

The visible world is still larger than the active diagnostic motion domain.
Native geometry queries alone do not establish a physical bank, larger safe
swimming area, a successful interface comparison or autonomous navigation.
Real bank/bed response and whole-body clearance need a separately versioned
physical design and finite qualification; at swimming depth the slope may be
encountered before the surface waterline. Do not transfer an ellipse corner
certificate to curved-bed clearance or hide the gap with forced steering,
position correction, bounce, teleport, reset or a smaller visible pond.

All three original reviews and alternatives remain in
`docs/IDEA_REGISTER_2026-11-23.md` and `docs/design-reviews/`. Applicable next
ideas are shared geometry for future sensing/contact, persistent world-frame
requests, one learned chooser deciding when/which way to turn, and one reusable
motor executing physical turns. Judge actual COM course/speed, turning radius,
response delay, depth/tilt and minimum whole-body clearance through repeated
turns. If later motor learning is authorized, preserve original checkpoints and
measure swimming/rest retention on the actual candidate; unchanged weights or
good-looking alignment alone cannot establish unchanged behavior.

No reward schedule, new force model, training run, reopened comparison,
publication or merge is authorized by this preparation.
