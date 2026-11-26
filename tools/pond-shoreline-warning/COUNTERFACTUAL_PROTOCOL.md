# Same-state chooser comparison — November 25, 2026

User authorizes offline comparison of actual chooser decisions and safer speed/
turn sequences from identical native states. Frozen motor, no training, no
scripted avoidance in Godot, no mouse or keyboard input. Protect all11 active
bundle hashes. Use the current shared native CPU simulator, as Godot does.

Three fixed archived research failures are selected before execution: frontier
update8 northwest_diagonal (71.495s), frontier128 east_shelf_tangent (95.11s),
and frontier128 northeast_to_center (35.32375s). These rejected research policies
are NOT the installed appended16 chooser. Replay each original learned prefix
once, requiring original decision actions, all saved COM samples within1e-6m,
exact final accepted substep count and the same clearance fault. A parity failure
stops the study before any alternative branches; preserve partial evidence.

Capture the last chooser decision at/before20,12 and4s before each archived
failure. Save exact World bytes with body/momentum/servo/filter/heading/COM and
visitation history intact, before that decision. Branches memcpy this whole
state and verify its checksum. Do not reinitialize or restore within a branch.
Both policies are feedforward; their temporal history lives in the saved World.

Each of9 states has26 branches: original research policy, installed appended16
policy, and24 predeclared sequences. The sequence library is3 requested speeds
{6,8,12}mm/s ×4 requested rates{−8,−4,+4,+8}deg/s ×2 initial durations{4,12}s.
Then hold0deg/s at that same speed for8s, then return to the original research
policy. Every branch lasts at most40s. Requests use the existing0.8s chooser
interface and80ms motor updates; no force/action/limit/physics substitutions.
These fixed request sequences are diagnostic probes, not a controller installed
in Godot or labels used for training. ±8 remains unqualified as a general skill.

Record every decision, initial35 inputs and both policies'15 logits, actual COM
motion, certified clearance/stop, depth/tilt, stalls, inward response and course
change. A useful40s escape requires no stop/reset, positive interval clearance,
tilt<=10deg, depth error<=1mm, mean half-second COM speed>=6mm/s, maximum rolling
half-second speed stall<=8.14s and inward COM movement>=6mm/s sustained1s. The
last condition uses1s COM course. Report safe-but-stalled/non-inward separately.
Success is a finite counterfactual witness, not whole-pond qualification. No
successful sequence among this library is not proof that no safe sequence exists.

Maximum3×120×800 prefix +9×26×40×800 branch =7776000 attempted substeps.
Internal90s/external100s deadlines. Stop native/nonfinite/invalid-use faults;
clearance stops are recorded outcomes. No silent rerun, extra sequences, budget
extension or criterion changes. No optimizer calls. Reuse earlier physics and
motor validation. Stage no policies/DLLs and leave Godot unchanged regardless
of the diagnostic result. Archive exact state, source/build/model bindings and
all attempted outcomes. Stop after reporting the causal evidence.
