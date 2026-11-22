# Balanced holding reward-scale comparison: completed results

Both approved eight-update arms completed; neither learned acceptable stopping
or holding. Arm B improved critic conditioning and retained the existing swim
regressions, but its holding policy still travels near8.1mm/s and one case
exceeds10degrees tilt. Keep checkpoint336; no policy was adopted.

The [frozen protocol](HOLD_SCALE_2026-11-04.md) was approved before execution.
Durable evidence is in
`assets/ppo-four-axis-checkpoints/balanced-hold-scale-336to344`; full source,
executable, all16 checksummed rollout batches and logs remain in
`build/ppo/balanced-hold-scale-336to344-v1`.
Branch `codex/minimal-foundation`; no push or merge.

## Measured behavior after8updates per arm

| Metric | A:holding gain1 | B:holding gain1/1024 |
|---|---:|---:|
| Original short swims |0/4 |4/4 |
| Mean short speed |7.0390mm/s |7.3542mm/s |
| Long60s windows |28/30 |28/30 |
| Previously passing4..60s windows |28/28 |28/28 |
| Complete formal60s cases |0/2 |0/2 |
| Stop+hold cases |0/6 |0/6 |
| Final8s mean3Dspeed across cases |8.239..8.251mm/s |8.085..8.104mm/s |
| Final8s forward drift |65.255..65.363mm |64.053..64.203mm |
| Worst final8s fixed-anchor error |257.180mm |257.764mm |
| Whole64s maximum depth error |1.006360mm |0.609293mm |
| Whole64s maximum lane error |1.671209mm |1.855716mm |
| Whole64s maximum head tilt |9.358492degrees |10.421790degrees |

Neither arm ever completes a1s stopped streak or permanent stop in any case.
The3Dspeed<=.5mm/s deadline and2mm anchor/excursion gates fail all6cases.
Pre-stop swimming passes6/6 in each arm. A exceeds1mm depth in one case;
B exceeds10degrees tilt in one case. Each passes the whole64s depth/lane/tilt
envelope in5/6cases. Baseline336's preserved two2..4s startup misses remain;
28/30 long windows is not full formal long acceptance.

All4joint commands, angles, rates and torques, signed roll/pitch, drift,
overshoot/reversal and speed ranges/RMS are saved in controls.csv and
hold-drift-details.json. Independent analysis recomputed every physical gate
from saved50ms control samples and agrees with all native per-case pass/fails.
Final8s signed depth net drift ranges are A
-0.260615..-0.189714mm,
B -0.231881..-0.225449mm.

## Critic conditioning and interpretation

On update344's actual active-holding samples:

| Diagnostic | A | B |
|---|---:|---:|
| Return mean |-1476.524114 |-1.510114 |
| Return SD |620.954001 |0.624838 |
| Critic mean after optimization |-141.459551 |-1.552346 |
| Critic halfMSE after |1083955.478230 |0.115591 |
| Variance-normalized MSE |5.622411 |0.592134 |
| Explained variance after |0.000182327 |0.412434384 |

B/A normalizedMSE=0.105317,
EVdifference=0.412252. The
prespecified <=.75MSE ratio and>=.20EV improvement both pass. Normalized error
and EV improve, so this is stronger evidence than the mechanically smaller raw
MSE after scaling. It remains same-batch fit, one seed and only8updates. Both
arms use a balanced goal/pose mix; this does not isolate the mix's effect from
historical data or prove the cause of earlier instability. Input saturation,
critic time/history information and holding-start coverage remain hypotheses.
Scaling changes task weighting; it does not guarantee an effective hold reward.

## Integrity, execution and preservation

Both independent copies migrated original336 with identical old parameters,
moments and21700old/0new clocks. CPU signed-feature/Adam/checkpoint fixtures
and CPU/CUDA goal/gain probes passed;23focused tests passed,0skipped with actual
CUDA required. Both historical40/41input trainer targets also built.
Both first40960row batches were collected before ANY PPO optimization, compared
exactly across original+appended observations, history, actions, values, logp,
dones and final state fields. Actual all5costs/rewards match exactly outside
holding and by1/1024 inside it. These cached first batches were then consumed
once, with no extra replay or physical samples.
On all40960actual initial states, original40/new41CPU actor/critic error0;
CPU/CUDA max2.02655792236e-6. All8batches/arm pass independent return/component,
actual goal/pose/horizon and optimizer permutation-coverage checks. Old-policy
logp rescore error0 in every batch. Maximum GAE errors A.001752921 and
B.000001556 are within the frozen tolerance. Existing first optimizer-minibatch
late-return checks pass, including126late rows with nonzero policy gradients.
All8updates actually used2epochs,2560Adam steps/network/arm; final old age24260,
new column age2560. Final checkpoints reload with exact networks and ages.

Process launched 2026-11-04T19:24:18.3930709Z, ended
2026-11-04T19:24:53.1806763Z, exit0,34.787837s. Actual PPO optimization
started A 2026-11-04T19:24:25.612164+00:00, B 2026-11-04T19:24:39.377271+00:00.
Native accumulated time A17.384945s/B17.272369s, below180s each; own process
below360s. Exactly16PPOupdates,655360trainingcontrols; axis128 and
assessment42880 give698368controls/6983680substeps. No retry, extension,
extra seed, intermediate physical assessment, physics change or hard fault.
Build/config67.966457s within300s; zero-step preparation1.971340s within60s.

All36prior checkpoint/protocol binary hashes,15unrelated Godot/exploration
file hashes, and67frozen input/source hashes verified unchanged after execution.
Original336 SHA256:
810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
All18numbered checkpoints and16rollout batches preserved. Training is stopped;
no native trainer remains. No candidate adopted, no push/merge.

The remaining blocker is behavioral: stationary holding still fails all6cases.
The scaled objective is better conditioned in this comparison, but neither arm
meets acceptance. Preserve336 and this evidence; any further training needs a
new bounded, approved protocol. Finite tests establish no indefinite performance.
