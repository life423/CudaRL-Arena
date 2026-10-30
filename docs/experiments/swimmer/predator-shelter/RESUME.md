# Finish the calm-holding step

## Newest: bounded paired motor v1 stopped on a fin/bed safety breach

Read [PAIRED_MOTOR_RESULTS.md](../../../PAIRED_MOTOR_RESULTS.md) and the
[two-policy interface](../../../CREATURE_CONTROLLER_INTERFACE.md). The user
authorized one new paired64-update study, preserving the old128-update work.
The fixed servo passed8/8 CPU plus8/8 CUDA joint-regulation cases in3200controls;
recorded-data readiness also passed with0extra integration. Torque PPO completed
6updates, then rejected rollout7 beforeGAE/Adam. Firstfailure was distalfin
−0.19335mm belowbed while nativecontact samples remained+4.61863mm clear.
No numericalfailure; no safetythreshold change. Lastgoodcheckpoint/export
matchtheir pre-update source. Joint-target PPO0updates; no finalbehavioral
evaluations, no motorwinner and no adoption. Actualtraining229318controls,
314withheldfixturecapturecontrols; do not reopen thisrun or extendits budget.

The narrow nextdecision is whether to authorize a targeted fin/contact-geometry
alignment before another preregistered comparison. It is outside the completed
run's unchangedphysics scope and hasnotbeen implemented. Keep the successful
gains, fullglobal-stop and allfailed results. No whole-app rewrite, per-skill
motornetwork, rescue or newtraining campaign is warranted by this result.
The actual6/64 partialmotor is available in the isolated livebench, labelled
unaccepted, alongside an empty/nottrained jointtargetarm. The productionpond
and both originaldecisioncontrollers remain unchanged. Evidence/rollback:
`build/paired-motor/2026-10-29-v1/`.

## Newest: bounded braking probe finished; no research admission

Read [MOTOR_BRAKING_RESULTS.md](../../../MOTOR_BRAKING_RESULTS.md). The latest
user removed the conventional16/16 prerequisite for research, preserving strict
final gates. The one authorized probe spent61,440+1,600 ticks and failed its
two-state alternative: bestzero torque, safe9.35/11.35s stopping, no10s quiet
hold. Three nonselected slow sequences brake head/COM but not the tail/body;
no extra replay or final-suite tuning. No PPO pilot ran or actor adopted.

Current set_states already had correct same-stream dependency; focused actual
nonblocking/continuation checks pass. Final scratch learner checks pass and
retain/reject unsafe PREstates before GAE/Adam. True600-tick finite-horizon
terminals now use done1/truncated0. Numerical results and originals preserved.
New evidence/failed diagnostic logs are in `build/motor-braking/2026-10-29-v1/`.
One proposed next comparison is a validated fast joint-target servo under the
same four-output PPO/torque caps; NOT implemented or automatically authorized.
No repeat/search extension, whole-app rewrite, rescued creature or weaker gates.

## Newest checkpoint: bounded motor milestone completed; physical gate failed

Read [MOTOR_CONTROL_RESULTS.md](../../../MOTOR_CONTROL_RESULTS.md). The independent
saved-actor CPU-double/GPU loss, gradient, finite-difference and fresh Adam check
passed. All32 registered open-water reference trials ran: conventional0/16 full
passes; passive2/16. Eight actual-motion braking starts qualified; passive
stopping4.9–11.3s exceeds the proposed3s envelope. This does not prove physical
uncontrollability. Do not extend that spent budget or silently relax the gates.

The isolated persistent-command motor task, capped trainer and separate live
Godot bench compile. CPU-only `--prepare` completed; `--run` with the false gate
was explicitly rejected before CUDA work. **No PPO pilot ran, no actor adopted.**
All180 original protected files still match. Original32-trial evidence and
executed binary remain under `build/motor-control/2026-10-29/reference-v1/`;
a post-build stronger geometric proof has a separately labelled offline review,
not an undisclosed rerun. The original failed gate remains unchanged.

For live frozen comparison, open `godot-project/motor_bench/motor_bench.tscn`.
63 functional/render checks pass; prepared untrained actor is unavailable there.
The working pond and overhang work remain parked/preserved. Discuss the braking
reference/control boundary before any new experiment, learning or redesign.
Pending/not-run claims in older sections below are historical and superseded.

## Latest direction: dependable locomotion motor before predator behavior

Read [the motor replan](../../../MOTOR_CONTROL_PLAN.md). The user explicitly
rejects food rate, inherited-output matching, component passes and counters as
proof of swimming. Preserve old systems/results for rollback; replacing the
isolated body motor/training task is allowed if supported. Keep the goal chooser
and one reusable motor conceptually separate. Whole-app overhaul is a last resort.

The bounded supervised study completed3×1,000 steps at
`build/holding-supervised/2026-10-29-v5/run-v1/`. Cover-only5/16 unseen holds
with lost feeding; joint1/16 with retained feeding; both0/4 nominal and full
gate failure. Tiny initial errors create joint/rate/previous-action feedback
outside the stationary cover labels, followed by command growth and exposure
before penetration. This does not prove a capacity or numerical-core defect.

Proposed next slice: numerical-reference preflight (still not run), physically
supported calm open-water acceptance harness, then one capped persistent-command
motor pilot. Move→brake→stop→hold10s is the milestone; predator choice and food
integration are later claims. No new pilot or physical reference was run for
the replan. See the exact scope/gates/budgets before doing any implementation.

## Completed controlled diagnosis — do not extend automatically

October 29: [holding-diagnosis.md](holding-diagnosis.md) is now the current
evidence/decision record. The first invalid gradient audit remains marked
INVALID; corrected stream-ordered diagnostics pass fixed-batch replay,
gradient consistency and exact full optimizer continuation. Protocol-5 gains
and their required derivatives are correct and untouched.

The predeclared four-arm comparison finished exactly 16 iterations/arm (48 PPO
updates total), with identical initial full states/RNG and frozen control.
All arms fail 0/4 nominal and 0/24 blind near five-second holds. Holding-only
improves partial return/clearance but destroys feeding retention; combined
retains food in the small screen but not holding. Final held-out traces for
holding-only and combined also fail all 44 deterministic/common-noise/saved-noise
cases. Six near starts were already exposed. First visibility/margin failure
precedes penetration in all 38 initially safe learned deterministic cases.

This rules out noise alone and a missing old units fix; it does not isolate a
unique learning cause or establish a capacity limit. No new learning fix is
justified yet. The diagnostic ordering fix is the only corrected defect with
repeatability evidence. No budget extension, failed actor adoption or production
learning change was made. Frozen-observation fitting of safe cover while
preserving food outputs is a proposed next discriminator, **not executed**.
Discuss a larger targeted redesign before implementing it.

Sources, personal saves and all 51 protected files remain unchanged. Keep all
raw runs under `build/holding-diagnosis/2026-10-29-protocol5/`, including the
invalid first audit and three final disposable update-316 runs with Adam10,112.
Compact metadata is in the three `holding-*-results.json` documents. Preserve
the parked larger-bevel overhang/default-menu work; don't mistake this diagnosis
for completion of that UI request or of learned predator hiding.

## Current requested diagnostic — no new training variant

The newest user request specifies matched zero/calibrated-inherited/frozen-v5
rollouts from exact nominal, near-calm and explicit velocity starts; per-step
action/reward/safety traces; and a food-versus-cover gradient/Adam audit on one
fixed rollout with disposable state. Keep physical safety thresholds intact.
Protocol 5 already fixes the earlier final-layer units mismatch. Its failed
300-update result remains evidence of failure, not a reason to redo that fix.
Do not alter rewards, architecture, gains or physical contact during diagnosis.

Both tools use the saved protocol-5 task and existing layout 2. Unfinished
overhang work (including the later unverified larger-bevel shape) is parked and
must be preserved. The working personal learner is never an output destination.
Baseline hashes/diff are saved in
`build/holding-diagnosis/2026-10-29-protocol5/`. Results will be recorded in
`holding-diagnosis.md` after execution, with confirmed defects, ruled-out claims
and hypotheses separated. No large sweep or failed actor deployment is allowed.

## Latest direction: diagnose before more variants

The user requested a decisive failure diagnosis before further feature expansion,
reward tuning or network changes. Implementation and research runs are paused
for that discussion. The proposed protocol-6 multi-head body change is **design
only**, not implemented or validated. Preserve the coherent protocol-5 boundary.

- Physical feasibility is established only at rest: all four nominal legal
  cover poses remain safe for ten calm seconds under zero torques. No controlled
  rollout yet establishes braking/recovery from initial linear or angular
  velocity. Calibration samples containing velocities are not such a test.
- Protocols 4 and 5 both completed 300 updates and still achieve **0/24**
  five-second holds on the near-calm suite. Protocol 5 averages only 0.458 seconds
  for each case's longest safe streak. Its matched calm feeding is 11.500/min
  versus the inherited 12.500/min. It is not a deployable improvement.
- The protocol-5 focused GPU suite passed 19/19 before the proposed protocol-6
  test edits. These checks verify components, not learned holding or end-to-end
  predator behavior. The full live holding scene is still unqualified/unverified.
- Next diagnostic: fixed small rest/velocity/heading cases, unchanged physics and
  shelter rules, comparing zero torques, a simple diagnostic stabilizer if needed,
  and the frozen neural motor. Trace physical actions, saturation, clearance,
  exposure and safe-hold duration. Then audit sampling/log-probability, scaling,
  gradients and full optimizer continuation together. Only after that choose one
  controlled learning change against a matched reference.
- Whole-body predator visibility and positive physical clearance define shelter;
  low current behind a rock is not predator cover. An explicitly assigned shelter
  goal diagnoses body control; it does not prove learning when to stop eating.
  Final behavior remains approach, stay concealed, resume feeding, retain swimming.

Checkpoint commit `5ea5d45` already saves the earlier work. Later protocol-5 and
holding-smoke changes remain local; no push, deployment or failed actor adoption.

The user requested a checkpoint commit, then explicitly asked to keep working.
This note records the unfinished step; it does not declare shelter learning
complete. Main player navigation still opens the working boulder pond.

## Current result and files

- Original full source: `build/predator-shelter/source-boulder.bin`, learner
  7,856/applied actor 7,568, SHA-256 D9A5C0FD…668D8A6. Keep it immutable.
  The newer personal boulder learner is separate; never roll it back.
- Protocols 1/2 both failed 0/24 five-second holds. Raw neural joint requests
  around 100 were clipped to the physical maximum of 1.
- Protocol 3 adds cover-only actuator-mean calibration **[8,4,1024,2]**.
  Food means/sampling/log probabilities are unchanged. Gaussian PPO uses the
  calibrated physical means; actor backprop applies the exact inverse scales.
  `shelter_action.cu`, `shelter_motor_ppo.h` and `shelter_calibration.h` contain it.
- The five calibration/gradient tests and prior shelter tests passed **14/14**
  on GPU. Generic PPO and the physical equations were not changed.
- Protocol-3 300-update trial: baseline 6 arrivals/0 holds; trained 0 arrivals/
  0 holds on the far/current suite, calm food 12.250 vs12.500 and flowing food
  11.500 vs11.625. Saved as `motor-v3-training.bin`, log `motor-v3-300.log`.
  Normalization fixes exploration, but does not itself teach this harder skill.
- Source is now protocol **4**: half food retention unchanged; one quarter
  starts at a legal still cover pose, one quarter has tiny pose errors. Cover
  practice is calm. `shelter_hold_cases.h` defines independent near-calm tests.
  No protocol-4 research run has been started at this checkpoint. Its native
  build and all **18/18 focused GPU checks** now pass.

## Next actions

1. Wire the near-calm suite into the CLI, measuring calibrated frozen baseline
   and trained actor with `probe_shelter_motor(...,1,200,saved_scales)`.
   Protected holding still means ≤5% actual whole-body exposure and ≥0.08
   model units clearance for five consecutive seconds. This lesson starts near
   cover; do not call it escape, predator choice or current stationkeeping.
2. Fix practice reporting: every25 updates ×64 steps lands exactly on a
   400-step cover reset. Episode peak counters then read zero by construction.
   Report before reset (e.g. update mod25==24) or accumulate completed episodes;
   do not use the old reset-boundary numbers as evidence that learning failed.
3. Run a bounded fresh protocol-4 trial, preserving v1/v2/v3 outputs. Confirm
   real improvement and old food retention; freeze a candidate before fresh
   final seeds. Keep the harder 24-case approach/current gate unchanged.
4. Verify `holding_session.cu` and `holding_pond.gd/.tscn` plus the seconds-based
   curve. Their source is now written and the native backend compiles, but no
   qualified policy or rendered live holding demonstration has been verified.
   It is a new explicitly assigned practice world, not a reset or
   scripted recovery of the user's creature. Native PPO must continue/save its
   full state; visible torques come from the actual calibrated neural actor.
5. Build/check the added `create_hold_session` / `DeepPondLab.start_hold_learning`
   bindings and holding metrics. Keep the original two-controller design and untouched default pond.
   A marked assigned goal is not evidence of learned fear or goal selection.
6. Verify live save/resume, preserved reference hashes, actual weight changes,
   promotion without a live reset, and rendered desktop/portrait/landscape UI.
   Open the practice scene only when there is a useful measured skill to watch.
7. Later train/validate autonomous food-versus-cover decisions and broaden
   approach/current tasks. Do not add lilies, legs or another decision network
   merely because those ideas were discussed.

Old `predator_readiness_smoke.gd` and `predator_visual_smoke.gd` use protocol-2
failed fixtures. Update their fixture generation/path for the new protocol
before claiming those old visual results validate protocol4. The original
predator route remains gated and absent from the normal menu. No new demo has
been opened yet. Raw logs/full training snapshots are ignored build artifacts;
do not commit them or publish a failed candidate.
