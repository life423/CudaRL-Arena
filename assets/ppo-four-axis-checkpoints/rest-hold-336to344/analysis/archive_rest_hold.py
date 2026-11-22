from pathlib import Path
import json,hashlib,shutil,datetime
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation');local=Path(__file__).parent
run=r/'build/ppo/rest-hold-336to344-v1';asset=r/'assets/ppo-four-axis-checkpoints/rest-hold-336to344'
assert not asset.exists(),'Archive identity exists; no overwrite'
x=json.loads((run/'comparison-summary.json').read_text());asset.mkdir()
for p in run.iterdir():
 if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.bin','.md','.txt','.log','.err'] and not p.name.startswith('batch-'):shutil.copy2(p,asset/p.name)
for name in ['baseline-rest','final-rest','baseline-long','final-long','initial-training-state','critic-calibrated']+[f'update-{u}' for u in range(337,345)]:shutil.copytree(run/name,asset/name)
for name in ['initial-function-parity.json','actual-GPU-migration.json','gpu-reset-starts.csv']:shutil.copy2(run/'calibration'/name,asset/('calibration-'+name))
(asset/'analysis').mkdir()
for name in ['analyze_rest_hold.py','archive_rest_hold.py','implement_rest_hold.py','freeze_rest_hold.py','build_rest_hold.ps1']:shutil.copy2(local/name,asset/'analysis'/name)
b,f=x['stages'];v=x['calibration_report'];hold=v['hold_validation_after'];swim=v['swim_validation_after']
low=sum(q['hold_pre_action_speed_le05'] for q in x['training_hold_coverage']);total=sum(q['hold_rows'] for q in x['training_hold_coverage'])
report=f'''# Hold-from-rest trial: completed, candidate rejected

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
| Mean |{v['holding_returns']['return_mean']:.6f} |{v['swimming_returns']['return_mean']:.6f} |
| SD |{v['holding_returns']['return_std']:.6f} |{v['swimming_returns']['return_std']:.6f} |
| Absolute99th percentile |{v['holding_returns']['abs_return_p99']:.6f} |{v['swimming_returns']['abs_return_p99']:.6f} |
| Maximum absolute |{v['holding_returns']['abs_return_max']:.6f} |{v['swimming_returns']['abs_return_max']:.6f} |

Both pass the unchanged<=5percentile/<=20maximum scale gates. The calibration
used exactly4epochs x20minibatches=80critic-only steps with the actor and its
complete Adam state unchanged. An independent checkpoint byte/parameter comparison
confirms that actor freeze. The calibration data were not reused for actor PPO.

| Held-out validation | Before | After | Approved limit |
|---|---:|---:|---:|
| HoldingRMSE |{v['hold_validation_before']['RMSE']:.6f} |{hold['RMSE']:.6f} |<=.218347 AND<=.281087 |
| Absolute holding bias |{abs(v['hold_validation_before']['bias']):.6f} |{abs(hold['bias']):.6f} |<=.140543 |
| SwimmingRMSE |{v['swim_validation_before']['RMSE']:.6f} |{swim['RMSE']:.6f} |<=.254190 |

All specified validation gates passed. Holding explained variance improved from
-2.502336 to-.450847 and remains negative; this is a limitation, not a passed EV
threshold. The approved protocol did not make EV an additional gate. It would be
incorrect to silently change the criteria after seeing it. Independently recomputed
validationRMSE/bias agree with the native report on all2560rows/task.

The first fresh PPO batch's holding share of squared globally normalized advantage
was{100*x['signal_gate']['holding_fraction_squared_normalized_advantage']:.6f}%,
within the fixed10..90% gate. It was evaluated before actor updates. No reward,
threshold, normalization, seed, episode length or update budget was retuned.

## Physical results

Baseline and final each:4original8s swims,6HOLD-FROM-REST64s tests(two poses x
depth0,+/-.5mm),2long60s swims. There is no32s pre-swim or holding grace period.

| Metric | Copied336 | After8PPOupdates |
|---|---:|---:|
| Short swims |4/4 |4/4 |
| Mean short speed |{b['mean_short_speed_mm_s']:.6f}mm/s |{f['mean_short_speed_mm_s']:.6f}mm/s |
| Long windows |28/30 |28/30 |
| Previously passing4..60s windows |28/28 |28/28 |
| Hold-from-rest cases |0/6 |0/6 |
| First speed>.5mm/s |.35s in all6 |.35s in all6 |
| Final8s mean3Dspeed across cases |{b['final8_mean_speed_range_mm_s'][0]:.3f}..{b['final8_mean_speed_range_mm_s'][1]:.3f}mm/s |{f['final8_mean_speed_range_mm_s'][0]:.3f}..{f['final8_mean_speed_range_mm_s'][1]:.3f}mm/s |
| Final8s forward drift |{b['final8_forward_drift_range_mm'][0]:.3f}..{b['final8_forward_drift_range_mm'][1]:.3f}mm |{f['final8_forward_drift_range_mm'][0]:.3f}..{f['final8_forward_drift_range_mm'][1]:.3f}mm |
| Worst whole64s anchor error |{b['max_anchor_error_mm']:.3f}mm |{f['max_anchor_error_mm']:.3f}mm |
| Worst depth error |{b['max_depth_mm']:.6f}mm |{f['max_depth_mm']:.6f}mm |
| Worst lane error |{b['max_lane_mm']:.6f}mm |{f['max_lane_mm']:.6f}mm |
| Worst head tilt |{b['max_tilt_deg']:.6f}degrees |{f['max_tilt_deg']:.6f}degrees |

Final speed, anchor and final8s excursion fail all6cases. Depth and tilt pass6/6;
lane passes1/6and fails5/6. Both centered cases also fail holding. The small speed/
drift reduction does not satisfy stationary holding, and time to first speed
violation did not improve. Both formal long cases still fail the two original
2..4s startup-speed windows;28/30 is not full long-swim acceptance.

Independent analysis from saved50ms traces agrees with all12baseline/final case
pass/fails and all maxima. Per-case gate results, initial violation times, speed
ranges, signed depth drift and all4joint command/angle/rate/torque ranges are in
hold-details.json. Joint movement is permitted; failure is body motion/drift.

Saved PPO holding observations include{low}/{total}pre-action samples<=.5mm/s
({100*low/total:.3f}%), including128true zero-velocity initial states. Thus the new
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
'''
(asset/'REPORT.md').write_text(report,newline='\n');shutil.copy2(asset/'REPORT.md',run/'REPORT.md')
prov={'status':'Completed80critic-only steps and8PPOupdates; holding failed0/6; rejected; reporting delivery timing missed',
 'source_run':str(run.relative_to(r)),'branch':'codex/minimal-foundation','controls':113664,'substeps':1136640,'actor_optimizer_steps':640,'critic_optimizer_steps':720,
 'holding_passes':0,'short_swim_passes':4,'long_window_passes':28,'retained_long_windows':28,'runtime_adopted':False,'training_stopped':True,'retries':0,'extensions':0,
 'numerical_gates_passed':True,'reporting_delivery_requirement_met':False,'reporting_deviation':'Native report/pause occurred before PPO; delegated update reached Drew only after PPO. Local emission was incorrectly treated as delivery.',
 'all_prior_evidence_hashes_verified':4494,'unrelated_hashes_verified':15,'remaining_blockers':['Stationary holding not learned','Lane limit failed5of6','Reporting delivery timing requirement missed']}
(asset/'provenance.json').write_text(json.dumps(prov,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
for h in json.loads((run/'preservation/prior-evidence-hashes.json').read_text()):assert sha(r/h['path'])==h['sha256'],h['path']
for h in json.loads((run/'frozen-hashes.json').read_text()):assert sha(Path(h['path']))==h['sha256'],h['path']
rawmanifest=[{'path':str(p.relative_to(run)),'sha256':sha(p),'bytes':p.stat().st_size} for p in run.rglob('*') if p.is_file() and p.name!='evidence-manifest.json']
(run/'evidence-manifest.json').write_text(json.dumps(rawmanifest,indent=2));shutil.copy2(run/'evidence-manifest.json',asset/'raw-evidence-manifest.json')
manifest=[{'path':str(p.relative_to(asset)),'sha256':sha(p),'bytes':p.stat().st_size} for p in asset.rglob('*') if p.is_file()]
(asset/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'report':str(asset/'REPORT.md'),'archive_files':len(manifest),'raw_files':len(rawmanifest),'provenance':prov},indent=2))
