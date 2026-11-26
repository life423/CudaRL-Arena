# Close-shore chooser: one recovered start, no qualified adoption

November 25, 2026. The user authorized checking two failed starts, GPU PPO
chooser learning, frozen motor, no scripted live avoidance and no mouse input.
This finite study is closed. **The chooser is not fixed sufficiently for Godot.**

## Exact-state diagnostic

The two installed-policy failures replay with zero COM error over106 saved
samples and nine exact action decisions. Six full native World captures, before
decisions at0.8/1.6/2.4s, preserve body, servo, request, temporal and visit state.
186 uninterrupted40s branches compare the installed chooser and fixed speed/
turn/straightening probes. No diagnostic sequence becomes a deployed controller
or a training label. The motor and physical equations are unchanged.

46 branches pass the declared finite escape gate. From the first decision,
there are8 qualifying sequences for phase8 and7 for phase9. Moderate requests
of -4deg/s work for phase8 and +4deg/s for phase9, at multiple speeds. Stronger
turning is not automatically safer. The complete sequence includes subsequent
straightening and a return to the installed policy; this does not establish a
globally safe constant-rate controller. Some passing probes use other rates.

This establishes **avoidable chooser sequence failures with the existing motor**.
It does not prove every near-bank state is recoverable. Diagnostic work:
2,234,245 attempted /2,234,103 accepted substeps,14.145s native/14.215s external,
zero optimizer updates. All186 outcomes, failed guards and raw captures survive.

## Focused learning change

One opt-in continuation starts from curved-bank-v1 full checkpoint80, which
passes12 original short cases and4 five-minute cases. Actor, critic and Adam
moments/ages import byte exactly before explicit creation of new training worlds.
All39 inputs, curved reward,15 speed/rate choices, PPO settings, motor88 and
physics stay unchanged. No scripted avoidance or artificial forces are added.

Half the1,024 initial worlds use random shoreline azimuths, both tangent
directions and15–30mm certified gap. A quarter use earlier bank approaches and
a quarter interior starts. Training source indices8/9 are excluded from the
256-row archive; these index exclusions alone are not proof of unseen physical
gait phases across different archives. Fixed evaluation placements remain
separate and are not fed into training. New initial placement is confined to
training initialization; subsequent resets occur only after scored terminals.
No time-limit episode reset was added, and Godot has no rescue/reset controller.

Placement checks cover4,096 CPU/CUDA cases with maximum position difference
0.245756um and yaw difference2.38419e-7rad. Curved sensory parity, parent import
and a three-actual-update full-state resume pass. Initial16 original traces and
the fixed/close case tables match the parent byte for byte. Changing training
starts did not silently change the motor or evaluator.

## Measured learning result

Native CUDA PPO completes **128 updates,1,048,576 macro transitions,4,096 actor
and4,096 critic Adam calls**. GPU physics/task collection123.338s, observation/
inference0.837s,PPO1.465s,evidence writing3.494s. Trainer wall131.419s, external
process131.752s:7,979 main transitions/s. Transfer/wait124.135s overlaps GPU work
and is not additive. Compilation and CPU evaluations are separate. The resume
diagnostic adds three updates/24,576 transitions and takes5.113s externally.

| Evaluation | Parent0 | Final128 |
|---|---:|---:|
| Phase8/north/negative tangent | Guard at4.33375s | **Pass180s** |
| Phase9/north/positive tangent | Guard at3.88875s | Guard at3.75875s |
| All eight close-shore cases | 6/8 | **4/8** |
| Original short navigation suite | 12/12 | **4/12** |
| Five-minute endurance | 4/4 | Not eligible |
| Cases with15s certified near-shore swimming | 0/8 | 0/8 |

Phase8 first passes at update88 and remains a180s physical pass thereafter.
Final128 makes14.264mm/s mean horizontal COM speed in that case. Its early decisions
change from repeated12mm/s,-8deg/s to -8,-8,-4,-4,0,+4deg/s, then continue
autonomously. Phase9 still selects overly strong+8deg/s requests; the diagnostic
had successful+4deg/s sequences. These are observed decisions, not installed
rules. Recovering one case while losing others is **not** a qualified fix.

No trained checkpoint retains12+4 original gates. All17 fixed and17 close
assessments are retained, including every regression (`data/comparison.csv`).
No new30-minute tests or Godot parity physics run, because no contender qualifies.
The original criteria remain unchanged; higher union coverage on failed runs
does not establish successful individual exploration.

The run records1,178 episode starts, including1,024 new initial worlds, and154
clearance guard stops. Thus only154 subsequent training respawns occurred.
Initial close-state exposure is demonstrated; frequent repeated close-bank
practice throughout collection is not. The failure cannot be assigned solely
to insufficient exposure without measuring that experience and the advantages.
Do not repeat this configuration and expect improvement just from more updates.

## Preservation, bounds and evidence

All11 deployed Godot hashes still pass, including motor88
`01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E`.
The app remains appended16/motor88. No DLL/model/scene was staged and no desktop
input occurred. Motor weights, fitted stroke and physical limits never train.
No fresh standalone resting test ran; previous motor retention is preserved,
not presented as a new resting measurement.

`data/work-ledger.json` counts **718,083,153 attempted /718,082,596 accepted
substeps**, including the186 diagnostic branches, resume check, main training,
17 fixed and17 close assessments. Limits in the frozen
[protocol](data/source/tools/pond-flexible-tail-chooser/CLOSE_SHORE_PROTOCOL.md)
are respected. Guard stops mean loss of a clearance certificate, not proven
terrain contact. No native numerical failure occurred.

One wrapper binding error and one CUDA compilation error were corrected before
their respective native launches; both logs remain, with zero physics/optimizer
work from those failed preparations. There was one physical diagnostic and one
main PPO session, without automatic restart or extension.

Archived: exact executed training/probe sources and bindings, all129 portable
actors, four full checkpoints0/8/88/128, resume proof, all assessments, diagnostic
captures/traces and curves. All129 full checkpoints remain locally at
`build/flexible-tail/close-shore-v2/training/checkpoints`, bound by the inventory.
The diagnostic executable is preserved locally as
`build/flexible-tail/close-shore-v2/diagnostic/executed-evaluator.exe`; its original
hash is in probe bindings. `data/manifest.json` hashes the committed evidence.

The next learning decision should examine meaningful bank-encounter frequency
and safe-action advantages/retention before another run. Properly bootstrapped
training time limits are an option for repeated practice, not a deployed reset.
No further training or Godot adoption is implicit in this closed result.
