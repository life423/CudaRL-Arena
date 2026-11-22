# Hold-from-rest trial: completed, candidate rejected

The numerical preparation gates passed and the run completed80critic-only steps
plus exactly8PPOupdates. It retained4/4short swims and28/30long windows but passed
0/6holding tests. The actor still starts swimming from its rest/zero-goal state.
Original336 and prior evidence remain intact. Training is stopped; no adoption.

## Reporting requirement missed

The native calibration report was generated and the run paused before PPO.
A local commentary update was emitted and the release marker was then written.
The delegated update did NOT reach Drew until after PPO. Local emission was
incorrectly treated as delivery, so the requested reporting timing was missed.
Parent delegation explicitly confirmed this. Do not claim Drew received or
acknowledged calibration results before PPO. reporting-deviation.json records
the distinction. The native reporting pause/marker are preserved as actual
evidence, not proof of user receipt. No retry or additional learning is performed
to address this procedural failure.

## Actual calibration and fixed gates

One frozen32x320batch,16holding/16swimming, balanced across poses0/3. Hold goal0
from the first control, centered-80mm target, original poses/filters/time feature,
all4NNoutputs. Hold gain1/1024; swimming unchanged. Calibration action seed
2718281828. Fit worlds world%8<4, validation the other16worlds, no episode leakage.

| Return metric | Holding | Swimming |
|---|---:|---:|
| Mean |-0.524979 |-0.286416 |
| SD |0.151353 |0.201980 |
| Absolute99th percentile |0.759137 |1.035973 |
| Maximum absolute |0.772349 |1.059665 |

Both pass the unchanged<=5percentile/<=20maximum scale gates. The calibration
used exactly4epochs x20minibatches=80critic-only steps with the actor and its
complete Adam state unchanged. An independent checkpoint byte/parameter comparison
confirms that actor freeze. The calibration data were not reused for actor PPO.

| Held-out validation | Before | After | Approved limit |
|---|---:|---:|---:|
| HoldingRMSE |0.291129 |0.189920 |<=.218347 AND<=.281087 |
| Absolute holding bias |0.046748 |0.043179 |<=.140543 |
| SwimmingRMSE |0.231082 |0.206897 |<=.254190 |

All specified validation gates passed. Holding explained variance improved from
-2.502336 to-.450847 and remains negative; this is a limitation, not a passed EV
threshold. The approved protocol did not make EV an additional gate. It would be
incorrect to silently change the criteria after seeing it. Independently recomputed
validationRMSE/bias agree with the native report on all2560rows/task.

The first fresh PPO batch's holding share of squared globally normalized advantage
was61.239161%,
within the fixed10..90% gate. It was evaluated before actor updates. No reward,
threshold, normalization, seed, episode length or update budget was retuned.

## Physical results

Baseline and final each:4original8s swims,6HOLD-FROM-REST64s tests(two poses x
depth0,+/-.5mm),2long60s swims. There is no32s pre-swim or holding grace period.

| Metric | Copied336 | After8PPOupdates |
|---|---:|---:|
| Short swims |4/4 |4/4 |
| Mean short speed |7.579388mm/s |7.423402mm/s |
| Long windows |28/30 |28/30 |
| Previously passing4..60s windows |28/28 |28/28 |
| Hold-from-rest cases |0/6 |0/6 |
| First speed>.5mm/s |.35s in all6 |.35s in all6 |
| Final8s mean3Dspeed across cases |8.536..8.565mm/s |8.275..8.306mm/s |
| Final8s forward drift |67.642..67.866mm |65.567..65.812mm |
| Worst whole64s anchor error |522.523mm |507.951mm |
| Worst depth error |0.796480mm |0.895972mm |
| Worst lane error |1.253396mm |2.071668mm |
| Worst head tilt |8.501427degrees |8.298105degrees |

Final speed, anchor and final8s excursion fail all6cases. Depth and tilt pass6/6;
lane passes1/6and fails5/6. Both centered cases also fail holding. The small speed/
drift reduction does not satisfy stationary holding, and time to first speed
violation did not improve. Both formal long cases still fail the two original
2..4s startup-speed windows;28/30 is not full long-swim acceptance.

Independent analysis from saved50ms traces agrees with all12baseline/final case
pass/fails and all maxima. Per-case gate results, initial violation times, speed
ranges, signed depth drift and all4joint command/angle/rate/torque ranges are in
hold-details.json. Joint movement is permitted; failure is body motion/drift.

Saved PPO holding observations include933/40960pre-action samples<=.5mm/s
(2.278%), including128true zero-velocity initial states. Thus the new
lesson did include low-speed states, unlike the old moving-stop pilot, but the
policy did not sustain them. These correlated samples are not independent trials.
The result does not establish whether a different curriculum would learn holding,
or whether more updates would help. Neither is authorized by unused resources.

## Integrity, budget and preservation

21focused tests passed,0skipped with actualCUDArequired. Zero-step32worldCPU/GPU
observations and old40/new41outputs matched exactly; signed input, unchanged40
meanings, from-control0gain,320horizon and asymmetric80critic-age fixtures pass.
On all10240actual calibration states, old40/new41GPUoutput error0 andCPU/GPU
error1.430511e-6. No additional physical replay was used.

All9calibration/PPO batches pass actual goal, anchor, elapsed-time,320control/
3200substep, terminal, component-sum, independent GAE and oldlogp checks.
Maximum GAE error8.032223e-7 within the existing tolerance; oldlogp rescore error0.
Actual holding rows and gradients were checked in shared PPO minibatches; no
shadow updater. All8updates used both epochs,640actorPPOsteps. Critic total720
includes80calibration. Final old/new ages: actor22340/640,critic22420/720, all
checkpoints reload with independent clocks. No optimizer clock/moment reset.

Exactly113664controls/1136640substeps: axis64,baseline10720,calibration10240,
PPO81920,final10720. Native39.174126s/process39.293611s, below120s including the
31.66s native reporting pause. Native process exit0; no numerical/physics fault,
retry, extension, extra seed, intermediate physical candidate test or adoption.
User-facing local execution time: November4,2026,2:41:51..2:42:30pm America/Chicago.
Raw UTC timestamps remain inexecution.json for audit. Critic fitting phase began
20:41:54.039UTC; PPO collection phase began20:42:25.747UTC after local release.
These phase timestamps do not certify delivery of the report to Drew.

Preservation checks verify4494prior evidence file hashes,15unrelated Godot/
exploration files and61current frozen input/source hashes unchanged. Original336:
810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
All10full training-state checkpoints(initial,calibrated,eight PPO), migrated
initial state, nine full rollout batches, source/executable snapshots, protocol,
validation rows and traces remain preserved. Full raw run:
build/ppo/rest-hold-336to344-v1. Branch codex/minimal-foundation; no push or merge.

The remaining behavioral blocker is0/6holding acceptance. The communication
timing requirement was also missed. Training is stopped and the candidate is
rejected; no further source/runtime changes or learning were made during
finalization. Future work needs a new approved bounded protocol and a reporting
handoff that verifies delivery rather than relying on a local progress message.
