from pathlib import Path
import json,csv,statistics,hashlib,shutil,math
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
run=r/'build/diagnostics/frozen-brake-B344-v1'
pres=r/'build/review-preservation/2026-11-04-before-frozen-brake'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def rows(p):return list(csv.DictReader(p.open(newline='')))
def n(q,k):return float(q[k])
result=json.loads((run/'result.json').read_text());execution=json.loads((run/'execution.json').read_text())
assert result['controls']==6240 and result['substeps']==62400 and result['completed_branches']==60 and not result['fault']
trace=rows(run/'controls.csv');br=rows(run/'branches.csv');prefix=rows(run/'prefixes.csv')
assert len(trace)==6240 and len(br)==60 and len(prefix)==6
independent=[];goals=[];pulse_branches=[q for q in br if int(q['branch'])>=2]
for case in range(6):
 ss=[q for q in br if int(q['case'])==case];zero=next(q for q in ss if int(q['branch'])==1);eight=next(q for q in ss if int(q['branch'])==0)
 goals.append({'case':case,'pose':int(zero['pose']),'offset_mm':n(zero,'offset_mm'),
  'continue8_last05_speed_mm_s':n(eight,'last05_mean_speed_mm_s'),'zero_last05_speed_mm_s':n(zero,'last05_mean_speed_mm_s'),
  'zero_speed_reduction_vs8_mm_s':n(eight,'last05_mean_speed_mm_s')-n(zero,'last05_mean_speed_mm_s'),
  'zero_displacement_reduction_vs8_mm':n(eight,'forward_displacement_mm')-n(zero,'forward_displacement_mm')})
 #32s prefix must reproduce prior frozenB trajectory; this uses no extra controls.
 for q in ss:
  branch=int(q['branch']);tt=[x for x in trace if x['phase']=='branch' and int(x['case'])==case and int(x['branch'])==branch]
  assert len(tt)==40
  late=tt[-10:];speed=statistics.mean(n(x,'COM_speed_mm_s') for x in late)
  assert abs(speed-n(q,'last05_mean_speed_mm_s'))<2e-9
  # Include start using previous prefix endpoint for exact whole-branch gates.
  start=next(x for x in trace if x['phase']=='prefix' and int(x['case'])==case and int(x['local_control'])==640)
  dep=max(abs(n(x,'depth_mm')) for x in [start]+tt);lane=max(n(x,'lane_mm') for x in [start]+tt);tilt=max(n(x,'tilt_deg') for x in [start]+tt)
  displacement=n(tt[-1],'forward_mm')-n(start,'forward_mm');assert abs(displacement-n(q,'forward_displacement_mm'))<2e-8
  for k,actual in [('max_depth_mm',dep),('max_lane_mm',lane),('max_tilt_deg',tilt)]:assert abs(actual-n(q,k))<2e-8,(case,branch,k)
  axis=int(q['pulse_axis']);delta=n(q,'pulse_latent')
  for t,x in enumerate(tt):
   for j in range(4):
    expected=delta if j==axis and t<10 else 0
    assert abs(n(x,f'latent{j}')-n(x,f'mean{j}')-expected)<2e-7
    assert abs(math.tanh(n(x,f'latent{j}'))-n(x,f'command{j}'))<2e-7
  speed_reduction=n(zero,'last05_mean_speed_mm_s')-speed;disp_reduction=n(zero,'forward_displacement_mm')-displacement
  envelope=dep<=1 and lane<=2 and tilt<=10;threshold=speed_reduction>=.5 and disp_reduction>=.5;useful=branch>=2 and threshold and envelope
  assert useful==bool(int(q['useful_local_braking'])) and envelope==bool(int(q['branch_envelope_pass']))
  independent.append({'case':case,'branch':branch,'envelope':envelope,'speed_reduction_mm_s':speed_reduction,'displacement_reduction_mm':disp_reduction,'useful':useful})
prior=rows(r/'build/ppo/balanced-hold-scale-336to344-v1/B/final-stop/controls.csv')
prior_prefix={(int(q['case']),int(q['control'])):q for q in prior if int(q['control'])<=640}
maxdiff={k:0 for k in ['COM_speed_mm_s','forward_velocity_mm_s','forward_mm','depth_mm','lane_mm','tilt_deg']+[f'command{j}' for j in range(4)]}
for q in trace:
 if q['phase']!='prefix':continue
 old=prior_prefix[int(q['case']),int(q['local_control'])]
 for k in maxdiff:maxdiff[k]=max(maxdiff[k],abs(n(q,k)-n(old,k)))
for h in json.loads((pres/'prior-evidence-hashes.json').read_text()):assert sha(r/h['path'])==h['sha256'],h['path']
for h in json.loads((pres/'unrelated-hashes.json').read_text()):assert sha(r/h['path'])==h['sha256'],h['path']
frozen=json.loads((run/'frozen-hashes.json').read_text())
for h in frozen:assert sha(Path(h['path']))==h['sha256'],h['path']
summary={'result':result,'execution':execution,'independent_all60_gates_agree':True,
 'pulse_branches':48,'pulse_useful_count':sum(int(q['useful_local_braking']) for q in pulse_branches),
 'pulse_envelope_passes':sum(int(q['branch_envelope_pass']) for q in pulse_branches),'all_branch_envelope_passes':sum(int(q['branch_envelope_pass']) for q in br),
 'pulse_speed_improvement_count':sum(n(q,'speed_reduction_vs_zero_mm_s')>0 for q in pulse_branches),
 'pulse_both_speed_displacement_improvement_count':sum(n(q,'speed_reduction_vs_zero_mm_s')>0 and n(q,'displacement_reduction_vs_zero_mm')>0 for q in pulse_branches),
 'best_pulse_speed_reduction_mm_s':max(n(q,'speed_reduction_vs_zero_mm_s') for q in pulse_branches),
 'best_pulse_displacement_reduction_mm':max(n(q,'displacement_reduction_vs_zero_mm') for q in pulse_branches),
 'best_speed_branch':max(pulse_branches,key=lambda q:n(q,'speed_reduction_vs_zero_mm_s')),
 'best_displacement_branch':max(pulse_branches,key=lambda q:n(q,'displacement_reduction_vs_zero_mm')),
 'branch_maximum_depth_mm':max(n(q,'max_depth_mm') for q in br),'branch_maximum_lane_mm':max(n(q,'max_lane_mm') for q in br),'branch_maximum_tilt_deg':max(n(q,'max_tilt_deg') for q in br),
 'branch_maximum_torque_microNm':10*max(n(q,'max_torque_normalized') for q in br),'branch_minimum_speed_mm_s':min(n(q,'min_speed_mm_s') for q in br),
 'zero_goal_late_speed_min_mm_s':min(g['zero_last05_speed_mm_s'] for g in goals),'zero_goal_late_speed_max_mm_s':max(g['zero_last05_speed_mm_s'] for g in goals),
 'zero_goal_displacement_range_mm':[min(n(q,'forward_displacement_mm') for q in br if int(q['branch'])==1),max(n(q,'forward_displacement_mm') for q in br if int(q['branch'])==1)],
 'zero_vs8':goals,'prefixes':prefix,'prefix_max_abs_difference_from_prior_B':maxdiff,
 'preservation':{'prior_evidence_files_hash_verified':len(json.loads((pres/'prior-evidence-hashes.json').read_text())),'unrelated_files_verified':15,'frozen_inputs_verified':len(frozen),'checkpoint336_unchanged':True},
 'new_optimizer_steps':0,'runtime_adopted':False,'new_training':False,'remaining_uncertainty':'Fixed one-axis .2latent half-second pulses at six captured phases did not demonstrate threshold-sized braking; this does not establish physical impossibility or full hold feasibility.'}
(run/'analysis.json').write_text(json.dumps(summary,indent=2));(run/'independent-gates.json').write_text(json.dumps(independent,indent=2))
print(json.dumps(summary,indent=2))
