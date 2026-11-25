# Rate response PPO candidate — November 25, 2026

The separate candidate improves moderate turning but does not fix bank avoidance.
Godot's working motor, offset, native library and habitat remain unchanged.

| Measurement | Initial motor | Selected update88 |
|---|---:|---:|
| Fixed rate passes |4/12|8/12|
| Mean course lag |15.12°|9.48°|
| Requested−8°/s, actual two-phase mean |−2.95°/s|−3.94°/s|
| Requested+8°/s, actual two-phase mean |+5.18°/s|+6.90°/s|
| Straight/from-rest cases |12/12|12/12|
| Gentle/30° main-stroke keys |8+8|8+8|
| Pond cases with preserved chooser16 |8/12 archived|7/12 fresh|

All four±2°/s and four±4°/s probes pass at88. All four±8°/s probes fail.
Right-turn lag at the fast request remains about34.7°, left about16.3°.
The two uninterrupted30/60/90 sequences pass4/6 requests before guard stops;
the fixed bank motor probes pass1/2. The full pond test records five certified
clearance guard stops, no resets, no other native faults and zero endurance
attempts. This is not qualified autonomous exploration. No chooser was retrained.
The archived initial pond result uses the same physical motor and chooser, but
this session does not provide an independently retrained control arm.

## Change and execution

Study6 uses signed rate error from whole-beat horizontal COM courses0.5 seconds
apart and a bounded Huber score instead of a20ms Gaussian rate score. Correct,
undershooting and wrong-way movement receive graded feedback. Other rewards,
inputs, body/water equations, joint/torque/filter limits and PPO settings stay
the same. Old Study5 remains available. No food, artificial forces, scripted
turns or additional behavioral network were introduced.

Native CUDA PPO completed128 updates/8,388,608 transitions on1,024 worlds in
103.435s internally,103.992s external process time. Compilation took12.470s
after8.143s configuration; bank evaluator configuration/build took9.069/2.783s.
GPU physics92.050s, inference0.518s, PPO3.891s, evidence6.204s. Transfer/wait
timing overlaps GPU work and is not additive. All16 scheduled evaluations ran.
An exact-resume prerequisite executed3 additional updates. Initial commands
and CPU/CUDA observations passed equivalence/parity checks. Every assessed
checkpoint retained12/12 swimming cases, including8/8 starts from rest.
Hold remains an inactive bypass; physical station-holding was not retested.

Total physical work573,608,204 attempted substeps is within the579,000,000
preregistered ceiling, including diagnostics, assessments and the pond test.
No failed attempt was restarted. All checkpoints remain in the local evidence
root. The selected complete state, all actor exports, learning curves, window
scores, source snapshot and bindings are copied into[data](data/manifest.json).
Large native traces and unselected complete states remain under
`build/flexible-tail/rate-response-v1`, as inventoried by absolute source paths.

Selection required12 swims and all gentle/30° keys, then maximized rate passes
and minimized rate-course lag. Raw `selection.json.rule` inherited an outdated
seven-stage label; the frozen executed FocusRate branch and selected metrics
establish the actual rule. The label is corrected for future executions without
rewriting raw evidence. Selecting88 instead of final128 was predeclared.

The next unresolved target is fast, symmetric rate tracking and requalification
of the chooser with the changed motor response. This result establishes a
measured improvement, not a completed fix. No further training is running.
