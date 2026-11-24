# Godot Swim slowdown and current-state inspection

Later correction: this preserved trace used runtime3, whose sloped-gap certificate
was subsequently found to reverse the horizontal gap-gradient signs. Its reported
clearance intervals are historical outputs, not valid independently certified
bounds. Runtime4 corrects that arithmetic and passed a fresh36-case physical check;
see `SHORELINE_RUNTIME_V4_RESULTS_2026-11-23.md`. This trace remains evidence of
the observed slowdown but cannot isolate a learning improvement against a later
runtime4 candidate. No new default-Godot trajectory has been added here.

The single authorized headless probe reproduced the ordinary Godot Swim path:
`main.tscn` → Swim selection signal → `reset_untimed_swim(0)` → Play signal →
unchanged `_process(.05)`/native saved-motor steps. It completed 1,200 controls /
12,000 physical substeps (60 simulated seconds) in 25.788 seconds. No product
stop, timer completion, error or reset occurred during Swim. No weights changed;
PPO updates, optimizer calls and CUDA kernels were zero.

The creature swims near its requested 8 mm/s until the upward-sloping underwater
bed first contacts it at 47.40 seconds. COM normalized shoreline coordinate is
0.6714, versus 1.0 at the water rim. Contact activity ends at 50.50 seconds. By
60 seconds the body is **clear** by 5.967935–5.977932 mm, yet horizontal COM speed
has fallen to 0.568 mm/s, tilt reaches 117.300 degrees and depth error is 38.630 mm.
The final motor requests settle into an almost static bent shape. It still drifts
upward, so this is loss of useful swimming rather than complete immobility.

This trace supports failed learned recovery following bed contact. Continuing
contact, a scripted clamp, the old smaller guard or an 8-second timer do not
explain its late slowdown. One trace does not isolate every possible neural or
physical cause, and does not establish autonomous navigation. The moving actor's
72 mm request position is not used as a forward-distance stop by its observation
adapter. This product path is still water; no flow-velocity field was tested.

The shared native terrain supplies mesh vertices. Comparing actual containing
triangles against native analytic heights at COM/three link centres throughout
the trace finds at most 0.136287 mm interpolation error. That is not a global
triangle/witness bound and does not account for the 38.6 mm depth excursion or
the apparent interior collision. The underwater slope begins inside the water
rim. Neither pond shader displaces vertices; native link poses are drawn directly.

Evidence: `build/pw29/stall/protocol.json`, `run-01/controls.csv`,
`run-01/states.jsonl`, `run-01/result.json`, `REPORT.md` and `verification.json`.
The protocol was sealed before one dynamic attempt; all 16 input bindings were
verified unchanged afterward and copied to `frozen-inputs`. The zero-step product
preflight and two static parser invocations used no integration. Surface diagnostic
queries were 4 points in preflight and 4,804 in the trajectory. The bridge exports
certified endpoint body clearance/potential and cumulative contact work/damping,
but not per-5 ms contact forces or closest witness points.

After preserving that trace, Details gained a compact **Right now** section:
current certified whole-body clearance/contact status, measured horizontal COM
speed versus requested speed, tilt and depth error. Historical negative minima or
old damping totals cannot label a currently clear body as in contact. Uncertain
or unavailable current queries are labeled explicitly. No permanent HUD was added.

UI validation: 220/220 checks passed across 1280×800, 390×844, 768×1024 and 844×390
in a private hidden project. Native processing remained disabled: zero additional
physical controls/substeps or optimizer work. The four verified Details views and
portrait/landscape missing-proof views were visually inspected; all 16 screenshots
are retained in `build/pw29/ui`, with source/file bindings in `verification.json`.
Modal focus, readable contrast, controls, zoom and original saved motor remain
covered. Short landscape Details uses its scroll area.

No computer input, closing or relaunch of the user's open app occurred in this
task. The Details source update applies on the next normal Project Manager Run.
The installed runtime DLL and original motor remain unchanged. Next development
should address learned turning/recovery and anticipatory bank information while
preserving clear-water swimming and holding, without scripted turns or weakened
retention criteria.
