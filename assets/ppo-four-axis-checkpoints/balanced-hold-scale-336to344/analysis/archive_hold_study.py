from pathlib import Path
import json,hashlib,shutil,datetime,subprocess
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
s=r/'build/ppo/balanced-hold-scale-336to344-v1'
asset=r/'assets/ppo-four-axis-checkpoints/balanced-hold-scale-336to344'
assert not asset.exists(),'Archive identity exists; no overwrite'
asset.mkdir()
summary=json.loads((s/'comparison-summary.json').read_text())
assert summary['preservation']['unrelated_files_hash_verified']==15
for name in ['comparison-summary.json','paired-first-batch.json','readiness.json','checks.json','execution.json','launch.json','result.json','frozen-protocol.md','frozen-hashes.json','source-manifest.json','evidence-manifest.json']:
 shutil.copy2(s/name,asset/name)
for arm in ['A','B']:
 dst=asset/arm;dst.mkdir();src=s/arm
 for p in src.iterdir():
  if p.is_file() and p.suffix in ['.bin','.json','.jsonl','.csv'] and not p.name.startswith('batch-'):
   shutil.copy2(p,dst/p.name)
 for folder in ['baseline-stop','final-stop','baseline-long','final-long']:
  shutil.copytree(src/folder,dst/folder)
failures={}
for arm in ['A','B']:
 details=json.loads((s/arm/'hold-drift-details.json').read_text())
 failures[arm]=[{'case':d['case'],'pose':d['pose'],'initial_depth_offset_mm':d['initial_depth_offset_mm'],'failed_gates':[k for k,v in d['gates'].items() if not v]} for d in details if d['stage']=='final']
prov={'status':'Completed exact approved two-arm trial; critic conditioning improved in B; both physical candidates rejected; no adoption',
 'source_run':str(s.relative_to(r)),'branch':'codex/minimal-foundation','source_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),
 'checkpoint336_SHA256':'810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0',
 'source_manifest':'source-manifest.json','raw_rollout_batches':'All16 checksummed batches remain in source_run/{A,B}/batch-{336..343}.bin; hashes and byte sizes in evidence-manifest.json',
 'protocol':'frozen-protocol.md','measured_results':'comparison-summary.json','independent_final_gates':failures,
 'checkpoint_schema':5,'observation_schema':4,'observations':41,'reward_tag':9,'reward_gain_bound_to_arm_protocol_checksum':True,
 'holding_gain_A':1,'holding_gain_B':1/1024,'all4joints_NN_controlled':True,'tilt_limit_deg':10,'posture_clamp':False,'physics_changed':False,
 'PPO_updates_each':8,'training_controls_combined':655360,'controls_combined':698368,'substeps_combined':6983680,
 'checkpoints_each':9,'rollout_batches_each':8,'prior_checkpoint_protocol_files_hash_verified':36,'unrelated_files_hash_verified':15,
 'runtime_adopted':False,'training_stopped':True,'retries':0,'extensions':0,'hard_faults':0,
 'remaining_blocker':'Neither candidate stops/holds in any of six tests. A loses short swimming and one depth gate; B exceeds10degree tilt in one case. No candidate accepted.',
 'inference':'Scaling holding reward improved the prespecified same-batch critic conditioning diagnostic under balanced goal/pose mix for this seed. It did not establish the cause of earlier failures or solve holding.'}
(asset/'provenance.json').write_text(json.dumps(prov,indent=2))
A=summary['arms']['A'];B=summary['arms']['B'];af=A['stages'][1];bf=B['stages'][1]
text=f'''# Balanced holding reward-scale comparison: completed results

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
| Mean short speed |{af['mean_short_speed_mm_s']:.4f}mm/s |{bf['mean_short_speed_mm_s']:.4f}mm/s |
| Long60s windows |28/30 |28/30 |
| Previously passing4..60s windows |28/28 |28/28 |
| Complete formal60s cases |0/2 |0/2 |
| Stop+hold cases |0/6 |0/6 |
| Final8s mean3Dspeed across cases |{af['hold_speed_mean_min_mm_s']:.3f}..{af['hold_speed_mean_max_mm_s']:.3f}mm/s |{bf['hold_speed_mean_min_mm_s']:.3f}..{bf['hold_speed_mean_max_mm_s']:.3f}mm/s |
| Final8s forward drift |{af['hold_forward_drift_min_mm']:.3f}..{af['hold_forward_drift_max_mm']:.3f}mm |{bf['hold_forward_drift_min_mm']:.3f}..{bf['hold_forward_drift_max_mm']:.3f}mm |
| Worst final8s fixed-anchor error |{af['maximum_hold_anchor_error_mm']:.3f}mm |{bf['maximum_hold_anchor_error_mm']:.3f}mm |
| Whole64s maximum depth error |{af['maximum_stop_depth_mm']:.6f}mm |{bf['maximum_stop_depth_mm']:.6f}mm |
| Whole64s maximum lane error |{af['maximum_stop_lane_mm']:.6f}mm |{bf['maximum_stop_lane_mm']:.6f}mm |
| Whole64s maximum head tilt |{af['maximum_stop_tilt_deg']:.6f}degrees |{bf['maximum_stop_tilt_deg']:.6f}degrees |

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
{A['final_depth_net_drift_range_mm'][0]:.6f}..{A['final_depth_net_drift_range_mm'][1]:.6f}mm,
B {B['final_depth_net_drift_range_mm'][0]:.6f}..{B['final_depth_net_drift_range_mm'][1]:.6f}mm.

## Critic conditioning and interpretation

On update344's actual active-holding samples:

| Diagnostic | A | B |
|---|---:|---:|
| Return mean |{A['final_holding_diagnostic']['return_mean']:.6f} |{B['final_holding_diagnostic']['return_mean']:.6f} |
| Return SD |{A['final_holding_diagnostic']['return_std']:.6f} |{B['final_holding_diagnostic']['return_std']:.6f} |
| Critic mean after optimization |{A['final_holding_diagnostic']['critic_mean_after']:.6f} |{B['final_holding_diagnostic']['critic_mean_after']:.6f} |
| Critic halfMSE after |{A['final_holding_diagnostic']['half_MSE_after']:.6f} |{B['final_holding_diagnostic']['half_MSE_after']:.6f} |
| Variance-normalized MSE |{A['final_holding_diagnostic']['variance_normalized_MSE']:.6f} |{B['final_holding_diagnostic']['variance_normalized_MSE']:.6f} |
| Explained variance after |{A['final_holding_diagnostic']['EV_after']:.9f} |{B['final_holding_diagnostic']['EV_after']:.9f} |

B/A normalizedMSE={summary['conditioning_diagnostic']['B_to_A_normalized_MSE']:.6f},
EVdifference={summary['conditioning_diagnostic']['B_minus_A_EV']:.6f}. The
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

Process launched {summary['execution']['started_utc']}, ended
{summary['execution']['ended_utc']}, exit0,34.787837s. Actual PPO optimization
started A {A['training_started_utc']}, B {B['training_started_utc']}.
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
'''
(r/'docs/HOLD_SCALE_RESULTS_2026-11-04.md').write_text(text,newline='\n')
shutil.copy2(r/'docs/HOLD_SCALE_RESULTS_2026-11-04.md',asset/'RESULTS.md')
readme=r/'README.md';old=readme.read_text();needle='## Current state\n\n';assert old.count(needle)==1
update='''The approved balanced reward-scale comparison is complete and training is stopped.
Both arms ran exactly8updates. Gain1 lost short swimming(0/4); holding gain1/1024
retained4/4short and28/30long windows and improved critic conditioning, but both
passed0/6holding cases. The scaled arm still travels~8.1mm/s with~64mm drift in
the final8s and reaches10.42degrees tilt in one case, failing the10degree limit.
Neither candidate is accepted or installed. Checkpoint336/all prior data remain
unchanged. [Measured results](docs/HOLD_SCALE_RESULTS_2026-11-04.md) and
[frozen protocol](docs/HOLD_SCALE_2026-11-04.md) contain exact gates and budgets.

'''
readme.write_text(old.replace(needle,needle+update),newline='\n')
print(json.dumps({'archive':str(asset),'results_doc':str(r/'docs/HOLD_SCALE_RESULTS_2026-11-04.md'),'independent_final_gate_failures':failures},indent=2))
