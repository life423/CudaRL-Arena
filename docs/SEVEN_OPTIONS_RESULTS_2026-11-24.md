# Seven-option PPO program — November 24, 2026

All seven authorized stages are closed. The strongest motor improvement is
actual 60° left/right travel in all eight fixed movement cases. Autonomous
bank avoidance remains unqualified: the strongest bank chooser passes8/12
two-minute cases, with four clearance guard stops. No endurance cases ran.
Godot's working motor, native library, habitat and saved original policies are
unchanged. No further learning session is running or authorized by this result.

## What was tested

Native C++/CUDA PPO used1,024 physical worlds and the existing GPU-resident
rollouts, MLP/GEMM, PPO and Adam. One reusable motor and one chooser remain the
behavioral architecture. There is no food, prescribed route, scripted body
rotation or artificial propulsion. Body/water motion still comes from the
existing internal actuators. The material variant is explicit and opt-in.

Motor stages used64 decisions/world/update,4096-sample minibatches and four
passes. Chooser updates used8 macro decisions/world,1024-sample minibatches
and four passes. Physical state, histories, optimizer state and sampling state
were preserved through exact, version-bound checkpoints. New actor coordinates
were initialized without changing the initial function; the new nonlinear
actor used fresh Adam state and retained the critic independently.

| Stage | Completed updates | Selected checkpoint | Measured result |
|---|---:|---:|---|
| Main-stroke rhythm |128|128|All8 gentle and8 30° main-stroke keys;12/12 swims|
| Joint stroke/steering outputs |128|56|60° mean course error26.55°→13.92°;12/12 swims|
| Hidden16 motor |128|128|8/8 actual60° movement,1.08° mean error,13.82mm/s;12/12 swims|
| Requested speed |115|72|4/6 speed probes;6/8/12mm/s requests produce8.40/9.23/10.64mm/s|
| Continuous requested course rate |128|0|No learned checkpoint outranks initial safe motor;4/12 rate probes|
| Earlier bank information and chooser PPO |63|16|6/12→8/12 pond cases; four guard stops remain|
| Half yaw stiffness |128|120|12/12 swims; frozen chooser drops to4/12 pond cases|

The speed run stopped cooperatively after the scheduled112 assessment failed
swimming retention6/12; checkpoint115 preserves the completed in-flight work.
Checkpoint72 still passes12/12. Selecting an earlier safe actor is explicit:
the last optimizer update is not automatically the best motor.

## Movement, rhythm and sustained navigation are separate results

The hidden16 motor passes all eight60° COM movement cases, including speed,
depth and tilt requirements. The original matched rhythm criteria still reject
these cases: the candidate's principal passive beat is about1Hz while the
underturning reference is about1.7Hz. The fixed80% frequency threshold was not
waived. Legacy crossing counts, prominence-based main-stroke counts and actual
movement scores are all preserved. No selected motor qualifies90° movement.

The selected baseline-material motor completes4/6 requests across two continuous
30°/60°/90° sequences. Later guard stops remain failures in the denominator;
no creature state is reset between requests. Its separate bank request probes
pass1/2 and do not establish autonomous navigation. Continuous-rate tracking is
still weak and asymmetric: requests±8°/s produce roughly−2.95/+5.18°/s, with
33–43° course lag. This limits how rapidly the chooser can steer away from shore.

The chooser's strongest update16 visits201/345 scored cells across twelve
separate starts:58.3% union coverage. This is not one creature covering58.3%
of the pond. The untrained chooser scored6/12; final63 regressed to4/12.
All12 must pass before the four five-minute endurance tests; none qualified.
The frozen chooser on the half-stiffness motor also scores4/12, so the material
change does not support adoption.

The hard-body hidden16 motor's60° cases show1.92s mean initial COM-course
response and165.3mm effective complete-beat turning radius. Response is the
first measured course change, not completion of the turn. The selected softer
body completes only4/8 actual60° movement cases and3/6 alternating requests.

The shoreline uses conservative whole-body contact-free certificates. A failed
certificate produces a recorded native guard stop; this is not measured bank
contact or a bank-contact solver. Tests in calm, initially horizontal states
show zero tilt/depth error but do not demonstrate disturbance recovery,
different-depth control or swimming against current. Station holding was not
physically retested. Inactive Hold continues to bypass the experimental motor;
this equivalence is not a fresh physical holding qualification.

## Repairs and preserved failed work

1. The first rhythm resume diagnostic spent3 updates before failing a1e-6
   CPU/CUDA critic-observation check. Duplicate snapshot trigonometry changed
   COM by7.45nm and normalized course by1.01328e-6. Using the accepted cached
   COM measurements removes the redundant arithmetic while preserving the
   equations, history, timestep and tolerance. The saved-state maximum is then
   4.76837e-7. The reserved second diagnostic passed; both attempts count.
2. The first chooser diagnostic met the shared categorical eight-action limit
   while the new interface has nine choices. It spent655,360 warmup substeps,
   zero collected actions and zero Adam calls. Shared capacity is now16, with
   passing GPU math checks at3/8/9/16 actions and rejection at17. Original
   three-action arithmetic remains bit-exact. The reserved second diagnostic
   passed3 actual updates; no third diagnostic ran.
3. The runner misclassified the evaluator's valid exit1 (unqualified6/12)
   as a process fault and killed the first chooser main invocation after
   1.0926767 seconds. Partial rollout/optimizer work is unknown. Preserve its
   receipt and checkpoint0. A byte-identical copy continued under a tightened
   63-update budget; reserve one original update for the lost work, up to8192
   transitions,32 Adam calls per optimizer and10,485,760 substeps. The interrupted
   wall time is charged to the native budget and cumulative wall total. The
   valid baseline and successful diagnostic were reused. The fixed runner
   validates complete case evidence, rejects timeouts/native faults and stops
   training cooperatively on orchestration errors.

Raw stage3 `study_actor_steps` subtracts the inherited actor age even though
the new nonlinear actor resets Adam age. Its negative reporting field is
preserved; full checkpoint and invocation audits establish8192 actual new Adam
calls for each optimizer. It is a reporting defect, not negative optimizer work.

The finite protocol, raw curves, selections, compact case/window tables,
source/input hashes, checkpoint evidence and failed attempts are retained in
[the evidence archive](../assets/research-evidence/pond-flexible-tail-seven-options-v6/README.md).
Full local controls, traces and unselected checkpoints remain in the original
build directories; the archive inventories their locations. Reused validation
is counted once, and unknown interrupted work is bounded rather than reported
as zero.

## Execution and budget

The final saved-state audit verifies761 checkpoint/export pairs across all six
motor stages:755 main updates and49,479,680 motor decisions. The chooser adds
63 main updates and516,096 macro decisions. Separately counted diagnostics spend
21 motor updates and3 chooser updates, including the failed first motor probe.
The failed categorical probe and interrupted chooser allowance remain separate.
Known total physical work is3,722,460,954 attempted substeps; including the
interrupted invocation's full upper bound gives3,732,946,714, below the frozen
4,241,983,616 ceiling.

Unique training-process wall time totals685.833s, diagnostic verification23.279s
and CPU evaluation184.600s. CPU evaluation overlapped training; these are not
additive end-to-end times and exclude compilation and manual orchestration gaps.
The final chooser's GPU physics time is86.568s, inference0.0966s, PPO0.7329s
and evidence1.7068s. Its86.548s transfer/wait field includes waiting on the same
physics work and must not be added again. Native wall91.270s includes the prior
interrupted1.0927s; the successful resumed process receipt is90.481s. Raw timing
fields and the unique-receipt ledger are archived.

## Next decision

The useful next experiment would improve symmetric continuous course-rate
tracking while retaining the successful stroke and swimming, then retest
early-bank navigation with the same fixed pond cases. A stronger bank chooser
cannot assume motor requests are executed accurately. This is a recommendation,
not another started run. The seven-option authorization is complete and stopped.
