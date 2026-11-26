# Shoreline-warning room — frozen diagnostic, November 25, 2026

**The12-second soft warning is too late for the previously qualified moderate
turn requests in these deliberately outward approaches.** Earlier20/24s cues
help some cases but do not make both moderate directions reliably fit the bank.
The working Godot chooser/motor, all11 deployment hashes and all weights remain
unchanged. No training, mouse/keyboard input or Godot publication occurred.

## What was measured

One native C++ CPU diagnostic uses the same physical simulation and frozen
motor88 as Godot. No separate physical equations, force, stabilization, route
or scripted deployed behavior is introduced. The test issues explicit constant
requests only to measure the motor's available maneuvering room. Two saved
moving phases, requested6/8/12mm/s, and initial headings0/45° from+X give12
center-to-bank prefixes. These are two world approach directions, not specified
45° incidence to the local bank. Each prefix keeps its complete physical and
actuator/filter/history state. At first12/20/24s warning crossings it waits for
the next actual0.8s chooser decision and saves the complete state.

Each of36 captures forks identical state into five requests:0,−4,+4,−8,+8°/s,
retaining the prefix's requested speed. There are180 branches. Each integrates
up to40s continuously, with no reset within a branch. ±4 requests passed the
archived calm motor rate probes; ±8 remained unqualified there and are included
as capacity diagnostics, not recommended actions. Captures and raw traces are
archived so no missing warm-up/body/filter state needs to be reconstructed.

Horizontal whole-body COM displacement over1s measures course and speed.
Response means a5° correct course change maintained for at least0.5s; it includes
the measurement window and is not an isolated actuator latency measurement.
Inward means COM motion toward the pond interior at>=6mm/s sustained for1s.
Safe40s requires positive certified clearance and the unchanged depth/tilt
limits throughout. A90° global course change alone does not establish success:
the bank curves and its local outward direction changes along the path.

## Results

| Warning horizon | Moderate right−4: inward / safe40s | Moderate left+4: inward / safe40s | Faster−8: inward / safe40s | Faster+8: inward / safe40s |
| --- | ---: | ---: | ---: | ---: |
|12s|0/12 ·0/12|0/12 ·0/12|2/12 ·2/12|0/12 ·0/12|
|20s|6/12 ·6/12|0/12 ·0/12|8/12 ·6/12|1/12 ·1/12|
|24s|6/12 ·6/12|0/12 ·0/12|12/12 ·12/12|10/12 ·10/12|

All36 zero-rate controls reach the clearance guard, establishing that continued
straight travel actually encounters the tested boundary. All24 moderate12s
branches also stop before sustained inward motion. Current-warning sensor
speeds are8.79–11.73mm/s despite requests6–12mm/s. At the first warning, straight
whole-body clearance range is104.98–140.63mm. The chooser decision delay is
0–0.72s; remaining range at the actual decision is102.05–140.14mm. Thus requested
speed alone understates runway consumption in some cases.

Example: moving_a,8mm/s request,toward+X. Actual cue speed10.153mm/s and remaining
range117.68mm give11.591s estimated straight travel. Right−4 registers5° response
at2.64s, but stops12.669s after cue after only49.31° of actual course change.
Left+4 responds at2.24s, stops at11.760s and reaches45.32°. Neither establishes
inward swimming. The same frozen state, filter and momentum are used in both.

In the24s-cued moderate branches that reach90°, measured90° times are22.24–23.76s
and effective moving-path radii138.54–182.51mm, with measured post-response
rates3.84–4.10°/s. These measurements corroborate a room deficit; the conclusion
does not rest solely on90/requested-rate or an ideal circle. Earlier cues alone
still do not solve the curved-bank maneuver. For example, at24s with8mm/s
requested and+4°/s, the swimmer reaches90° at22.24s but loses its clearance
certificate at34.765s before qualifying sustained inward motion. Requested+8
also changes propulsion: mean actual speed across its24s branches is17.51mm/s,
so more requested turn rate cannot be treated as a pure reduction in radius.

All measured depth errors and tilts are zero in this unchanged fixed-depth
runtime. There are137 clearance stops and43 safe40s branches. A guard stop is
loss of the conservative whole-body separation certificate, not independently
confirmed physical contact at the visible shoreline. The fixed-shape straight
translation sensor does not predict rotated-tail sweep or the entire future
curved path. Current code can reject an unsafe substep while its last returned
pose still has a small positive gap; minimum interval clearance and fault code
are therefore reported together.

## Interpretation and next decision

The installed appended16 chooser's four30-minute safe runs remain valid. These
are deliberate outward capability probes, not new failures of that deployed
chooser. They show that its12s shaping horizon is not a generally adequate
turning-room margin if exploration begins using more bankward trajectories.
This also supports the concern that safe interior circling can leave the pond
underexplored, without proving that a single reward term caused it.

Recommend a maneuver-aware risk signal based on measured speed, turn response,
whole-body sweep and bank curvature, followed by learned early avoidance using
qualified rate/speed choices. Do not simply replace12 with24 and assume safety,
force a turn, clamp position, rely on unqualified±8, or weaken clearance gates.
The missing evidence for a replacement is a validated maneuver envelope across
speed, phase, approach angle and both directions. No sensor/reward correction,
new learning or adoption is started by this diagnostic.

The one run completes in24.179s native/24.233s externally, within50/65s deadlines.
All12 prefixes,36 captures and180 branches complete with3794995 attempted /
3794858 accepted substeps, below6912000. Zero optimizer calls, zero resets within
branches, zero new CUDA work and zero desktop input occur. The analysis script
checks that prefix+branch work equals the native ledger. Executable/source/model
bindings, full captures, every branch trace and the earlier motor probes are
under`data`; original build evidence remains in`build/flexible-tail/shoreline-warning-v1`.
