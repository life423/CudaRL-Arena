# Untimed Godot Swim

The user approved removing the interactive Swim session timer without changing
the saved motor, physical limits or pond checks. A separate reset_untimed_swim
API now keeps the original request, clocks, body, joint targets, reference rates
and torque history running. The original finite reset(id,false) API and research
scorers remain unchanged; Rest keeps its64-second duration. The new bridge-only
recorder retains the original endpoint/order/finite checks and measurement
bookkeeping while omitting the episode deadline. No task is revived or reset
every eight seconds. An explicit integer-overflow guard remains.

## Bounded product verification

The [protocol](UNTIMED_SWIM_PROTOCOL_2026-11-23.md) allowed one new trajectory,
at most800attempts/8000substeps/40simulatedseconds. The single native Godot4.4.1
check passed and closed without a retry:

| Measurement | Actual result |
| --- | ---: |
| Accepted displayed controls |509|
| Attempted physical controls |510|
| Attempted/executed5ms substeps |5100|
| Retained displayed substeps |5090|
| Displayed simulated time |25.45000076s|
| Forward COM displacement |188.907057mm|
| Final accepted COM speed |8.159997mm/s|
| Final accepted depth error |0.198727mm|
| Final accepted tilt |3.98015degrees|
| Original first160 pose maximum difference |4.99756e-12 model/quaternion/joint units|
| Required existing pose-fidelity tolerance |2e-6|
| Pause/resume at control170 |Exact state preserved|
| External process time |0.540291s|
| Motor/chooser training or optimizer updates |0|
| Research comparison cases |0|

The510th candidate interval reached the existing supported pond-domain limit.
Its10substeps count as executed work, but the candidate was not committed to
the displayed world. The last accepted whole-body state remains inside the
original domain, paused, finished=false, with the exact existing stop reason.
This is an unchanged boundary stop, not learned braking or a turn.

Every endpoint was saved. World/task/control clocks continue through159,160,
161,162 with the same request origin/position/heading/terminal velocity and
continuous servo state. Pause/get_state/resume at170 changes only paused and
continues171. The original160poses were read from existing CPU evidence; no
second physical reference was run. Rest and finite Swim metadata were checked
at zero controls. The333-assertion bookkeeping check uses327synthetic endpoints,
zero physics, and checks horizon crossing, original nonterminal fields and
rejection/overflow guards. Its first synthetic nonfinite fixture reached an
earlier valid rejection guard; only the test fixture was corrected and its
failed source/executable/receipt preserved. The real helper was unchanged.

The production module builds with warnings treated as errors. All68bound
functional inputs and three saved-policy/checkpoint files remain byte-identical
after the check. No physical equations, source contracts, NN weights, torque,
filter/servo settings or integration settings changed. The isolated CUDA
arithmetic research profile is not installed in this CPU viewer.

The independent saved-data audit passes (SHA256
605a94ec0d5469676812a69852c74673a271e69163451d40e5d54946be9630da).
It reconstructs original160pose fidelity, clocks/request/servo continuity and
the unchanged conservative pond-domain endpoint guard: maximum accepted
ellipse-box value0.997360337, rejected candidate1.005677992. It performs no
force replay or integration and makes no continuous swept-volume claim.
Pause/resume equality was checked by the bound native test; separate paused
snapshots were not retained, so the saved trace and assertion receipt provide
the available corroboration rather than a separate pause-state replay.

## Product scope and evidence

Untimed Swim continues until the user pauses or an existing physical/domain
check stops it. This test establishes removal of the8second cutoff for one
saved start, not indefinite autonomous navigation, general turning or current
swimming. The supported physical pond domain remains smaller than the visual
outer water; no area or body was enlarged. Old research allocations remain
closed, including the0/88 interface comparison.

Evidence is under build/us26: original DLL/source snapshots, saved-policy hashes,
verification-inputs.json, native-check.json (all511states), product-process.json,
bookkeeping-check.json and empty native-check.err. Private UI verification and
actual installed app inspection are recorded in
[the presentation review](UNTIMED_SWIM_UI_REVIEW_2026-11-23.md).
The reviewed DLL (SHA256d5f6a815e46152cd730036dbd95ab01b17d31d1f71bddfbffb8c953f0864ee07)
is installed in the correct live project. Root inspected actual Godot4.4.1
paused with Swim selected and elapsed/distance measurements enabled. No extra
visible physical step was issued.160rendered/assertion checks passed across
four sizes with zero physics; the checked measurement-label contrast was
corrected to11.16:1. The actual app log contains no errors.

The exact source/input bytes, raw trajectory, historical reference, process and
audit receipts and four UI receipts are also preserved in
[the portable evidence archive](../assets/research-evidence/untimed-swim-2026-11-23/evidence.zip)
with a per-file SHA256
[manifest](../assets/research-evidence/untimed-swim-2026-11-23/manifest.json).
This retains the original bytes even if a later Git checkout normalizes source
line endings. No external compiler or SDK binary is bundled.
