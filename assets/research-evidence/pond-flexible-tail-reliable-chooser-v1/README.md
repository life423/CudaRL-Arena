# Reliable-turn chooser and Godot adoption — November 25, 2026

Chooser checkpoint88 passes all12 fixed two-minute pond cases and all four
five-minute endurance cases, with zero guard stops, other native faults or
resets. It is now available in the registered **Tadpole Pond | Flexible Tail**
Godot project: click Run, then **Explore pond**. The viewer performs native
physical simulation with saved policies; it does not run PPO in the background.

## Training and selection

Options1,2,7 are implemented:15 rate/speed actions including±4deg/s; frozen
rate-response motor88; and a competence-based bank-approach curriculum. The
existing native CUDA rollout/MLP/GEMM/categorical PPO/Adam pipeline trained one
chooser across1,024 worlds. Body/water equations and actuator limits are intact;
no food, scripted turns, forced rotation or artificial propulsion were added.

The session completed128 updates and1,048,576 macro transitions in173.815s
native/174.095s external process time. Each update collects8 decisions/world,
uses1024-sample minibatches and four passes. GPU physics165.397s, inference
0.217s, PPO1.580s and evidence4.131s. Transfer/wait165.556s overlaps physics
and is not additive. A separate exact-resume diagnostic executed3 actual updates.

Checkpoint selection follows the fixed pass/endurance/coverage rule, rather
than adopting the final optimizer state. Untrained15-action chooser scores3/12;
update16 scores10/12;88 qualifies12/12+4/4;final128 regresses to7/12. All17
scheduled assessments are preserved. The selected diagnostic union covers
191/345 cells(55.4%); endurance union209/345(60.6%). These are unions across
separate starts, not proof that one creature explores the entire pond.

The unchanged gates require actual COM movement>=6mm/s, full horizon, depth
error<=1mm, tilt<=10deg, certified whole-body separation, visitation and turn
requirements. All12 and diagnostic union coverage>=20% precede endurance.
No tolerance was loosened. Existing motor88 evidence retains12/12 swimming
cases, including8 starts from rest. The frozen motor has no optimizer in this
chooser session. Station holding, currents, other depths, disturbance recovery
and general90deg command qualification remain separate work.

## Curriculum and prerequisite evidence

Half training starts remain central; bank approaches are paired by turn sign.
Nominal easy/middle/hard slants are25–30/30–36/34–44deg, with progressively
less runway. Promotions require60 uninterrupted simulated seconds,8 new cells,
300mm actual movement and no stall longer than2s. Promotions preserve the
current body, command, servo and history; difficulty affects the next spawn.
Two failures can demote a lane. Final start counts are2676/154/160 by level,
with2088 promotions and53 demotions. Training terminals are explicit; they are
never deployed recoveries.

The original easy geometry failed level0/lane4 before any physics or optimizer
work. The single reserved correction moves only initial gentle placements
inland when required by the unchanged20mm static warmup certificate. Original
hard placements remain exact. All3072 corrected geometry fixtures pass, as do
15-action CPU/GPU probability/gradient/sampling/canary checks and full-state
resume. The failed original root remains preserved.

Research work totals703,480,543 attempted substeps:672,374,533 main,
16,389,900 diagnostic and14,716,110 fixed assessments. The initial failed
preflight spent zero. This is below the frozen1,409,538,560 ceiling. All
collection failures and curriculum counters remain in the native results.

## Godot qualification

The extension uses exactly the evaluated chooser observer, CPU MLP inference,
argmax policy,15-action decoder and shared native motor runtime. It holds motor
actions for80ms and selects chooser requests every0.8s, independently of render
frame rate. There is no automatic reset or endpoint replay. Play/Pause and
Restart are explicit user controls. Original motor/offset files are unchanged,
and **Details → Manual turns · Original motor** retains the older manual mode.
Original DLL and scenes/scripts are backed up in the local evidence root.

Initial integration accidentally centered the saved start, shifting it by a
few millimetres. Its failed300s replay is retained. The corrected extension
preserves the saved initial COM and matches the qualified five-minute trace
at all3750 sampled states with ZERO COM or requested-speed difference, no
stops and no resets. Both simulated attempts total480,000 extra substeps;
initial argument/header test mistakes spent zero. No extra optimizer ran.

Godot4.4.1 rendered390x844,768x1024,1100x760 and844x390, with collapsed and
expanded controls. Final50 UI assertions pass with zero physics. Tests cover
load, play/pause, explicit restart, original manual requests, mode switching,
44px targets and dock/header containment. A missing unique label identifier
was fixed and assertions strengthened; the earlier UI result is preserved.
Mouse overview/follow and Details interactions were verified in the running
app; play/pause/manual/restart signals were checked by the engine test. Project
Manager registration points to this project; its minimized manager window was
not successfully automated. The running app was verified directly.

Conservative internal UI review: hierarchy9, usability9, contrast9, consistency9,
responsive layout9, feedback9, performance9, accessibility8. Accessibility next
work: screen-reader narration and touch-device hardware testing remain unverified.
These are design heuristics, not measured user-study scores. Native Godot focus
states and engine controls remain enabled; phone/touch claims are limited to
rendered layout and engine behavior, not real touch hardware.

The original qualification trace writer had nine logit column names despite15
values. Raw traces are retained; future writer headers now enumerate all15.
Scores and the Godot comparison use unaffected leading COM/request columns.

## Saved artifacts and limits

See[data/manifest.json](data/manifest.json), learning curve, case summaries,
protocol, source bindings, selected actor/full checkpoint, frozen motor,
curriculum counters, deployment hashes, replay/UI receipts and rendered images.
All original controls/traces and unselected full checkpoints remain in the
local `build/flexible-tail/reliable-chooser-v1-corrected` root. No trainer is
left running. The live viewer is intentionally open for the user.

Passing16 finite cases supports adoption of this pair in the tested calm pond.
It is not a guarantee of infinite survival or whole-pond coverage. Continued
longer and varied evaluation is future work, not silently started here.
