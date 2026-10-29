# Finish the calm-holding step

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
