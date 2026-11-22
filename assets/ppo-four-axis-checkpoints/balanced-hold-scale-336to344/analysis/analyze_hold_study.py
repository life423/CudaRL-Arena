from pathlib import Path
import json,csv,math,statistics,hashlib,datetime,subprocess,shutil
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
s=r/'build/ppo/balanced-hold-scale-336to344-v1'
pres=r/'build/review-preservation/2026-11-04-before-balanced-scale'
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open(newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def utc(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
def num(d,k):return float(d[k])
prior=read(pres/'prior-checkpoint-hashes.json')
for h in prior:assert sha(r/h['path'])==h['sha256'],h['path']
unrelated=[h for h in read(pres/'manifest.json') if h['path'].replace('\\','/').startswith(('live/','assets/ppo-smooth-exploration-v1'))]
for h in unrelated:assert sha(r/h['path'])==h['sha256'],h['path']
frozen=read(s/'frozen-hashes.json')
for h in frozen:assert sha(Path(h['path']))==h['sha256'],h['path']
allresults={}
for arm in ['A','B']:
 p=s/arm;result=read(p/'result.json');assert result['PPO_updates']==8 and result['controls']==349184 and not result['fault']
 result['training_started_utc']=utc(result['training_started_unix'])
 result['initial_function_parity']=read(p/'initial-function-parity.json')
 result['stages']=[];detail=[]
 for stage in ['baseline','final']:
  short=rows(p/f'{stage}-evaluation.csv');stops=rows(p/f'{stage}-stop/summary.csv');controls=rows(p/f'{stage}-stop/controls.csv');long=rows(p/f'{stage}-long/sustained-windows.csv')
  assert len(short)==4 and len(stops)==6 and len(controls)==7680 and len(long)==30
  d={'stage':stage,'short_passes':sum(int(x['passed']) for x in short),'mean_short_speed_mm_s':statistics.mean(num(x,'steady_mm_s') for x in short),
     'stopping_passes':sum(int(x['passed']) for x in stops),'long_windows_passed':sum(int(list(x.values())[-1]) for x in long),
     'retained4to60_windows':sum(int(list(x.values())[-1]) for x in long if float(list(x.values())[1])>4),
     'maximum_stop_depth_mm':max(num(x,'max_depth_mm') for x in stops),'maximum_stop_lane_mm':max(num(x,'max_lane_mm') for x in stops),
     'maximum_stop_tilt_deg':max(num(x,'peak_tilt_deg') for x in stops),'hold_speed_mean_min_mm_s':min(num(x,'hold_mean_speed_mm_s') for x in stops),
     'hold_speed_mean_max_mm_s':max(num(x,'hold_mean_speed_mm_s') for x in stops),'hold_forward_drift_min_mm':min(num(x,'hold_net_drift_mm') for x in stops),
     'hold_forward_drift_max_mm':max(num(x,'hold_net_drift_mm') for x in stops),'maximum_hold_anchor_error_mm':max(num(x,'hold_anchor_max_error_mm') for x in stops),
     'permanent_stop_cases':sum(num(x,'permanent_stop_relative_s')>=0 for x in stops),'pre_swim_passes':sum(int(x['pre_swim_pass']) for x in stops),
     'state_envelope_passes':sum(int(x['state_pass']) for x in stops)}
  result['stages'].append(d)
  for case in range(6):
   rr=[x for x in controls if int(x['case'])==case];late=[x for x in rr if int(x['control'])>=960];hold=[x for x in rr if int(x['control'])>=1120];last=[x for x in rr if int(x['control'])>1120]
   speeds=[num(x,'COM_speed_mm_s') for x in late]
   drift=num(rr[-1],'depth_mm')-num(hold[0],'depth_mm')
   excursion=max(abs(num(x,'forward_mm')-num(hold[0],'forward_mm')) for x in hold)
   gates={'speed48to64':max(speeds)<=.5,'anchor56to64':max(num(x,'hold_error_mm') for x in hold)<=2,'excursion56to64':excursion<=2,
    'depth':max(abs(num(x,'depth_mm')) for x in rr)<=1,'lane':max(num(x,'lane_mm') for x in rr)<=2,'tilt':max(num(x,'tilt_deg') for x in rr)<=10,
    'pre_swim':bool(int(stops[case]['pre_swim_pass']))}
   passed=all(gates.values());assert passed==bool(int(stops[case]['passed']))
   joints=[]
   for j in range(4):
    joints.append({'axis':j,**{key:[min(num(x,key+str(j)) for x in last),max(num(x,key+str(j)) for x in last)] for key in ['command','angle','rate','last_torque']}})
   detail.append({'stage':stage,'case':case,'pose':int(rr[0]['pose']),'initial_depth_offset_mm':num(rr[0],'initial_depth_offset_mm'),
      'passed':passed,'gates':gates,'speed48to64_min_mm_s':min(speeds),'speed48to64_max_mm_s':max(speeds),'speed48to64_RMS_mm_s':math.sqrt(statistics.mean(v*v for v in speeds)),
      'final8_speed_mean_mm_s':statistics.mean(num(x,'COM_speed_mm_s') for x in last),'final8_speed_min_mm_s':min(num(x,'COM_speed_mm_s') for x in last),'final8_speed_max_mm_s':max(num(x,'COM_speed_mm_s') for x in last),
      'final8_depth_net_drift_mm':drift,'anchor56to64_max_error_mm':max(num(x,'hold_error_mm') for x in hold),'forward56to64_max_excursion_mm':excursion,
      'first1s_stopped_relative_s':num(stops[case],'first20samples_stop_relative_s'),'permanent_stop_relative_s':num(stops[case],'permanent_stop_relative_s'),
      'post_command_forward_mm':num(stops[case],'post_command_forward_mm'),'reverse_distance_mm':num(stops[case],'reverse_distance_mm'),
      'final8_roll_range_deg':[min(num(x,'roll_deg') for x in last),max(num(x,'roll_deg') for x in last)],'final8_pitch_range_deg':[min(num(x,'pitch_deg') for x in last),max(num(x,'pitch_deg') for x in last)],'final8_joints':joints})
 (p/'hold-drift-details.json').write_text(json.dumps(detail,indent=2))
 groups=[json.loads(x) for x in (p/'group-diagnostics.jsonl').read_text().splitlines()]
 result['initial_holding_diagnostic']=next(x for x in groups if x['update']==337 and x['group']==8)
 result['final_holding_diagnostic']=next(x for x in groups if x['update']==344 and x['group']==8)
 integrities=[read(p/f'return-integrity-{u}.json') for u in range(336,344)]
 rescores=[read(p/f'old-policy-rescore-{u}.json') for u in range(336,344)]
 result['integrity']={'all8_batches_passed':all(x['passed'] for x in integrities),'max_GAE_error':max(x['max_GAE_error'] for x in integrities),'max_component_error':max(x['max_component_error'] for x in integrities),
  'max_old_logp_rescore_error':max(x['max_abs_logp_error'] for x in rescores),'all8_batches_saved':len(list(p.glob('batch-*.bin')))==8,'numbered_checkpoints_saved':len(list(p.glob('ppo-append-update-*.bin')))}
 assert result['integrity']['all8_batches_passed'] and result['integrity']['all8_batches_saved']
 result['final_hold_speed_min_mm_s']=min(x['final8_speed_min_mm_s'] for x in detail if x['stage']=='final')
 result['final_hold_speed_max_mm_s']=max(x['final8_speed_max_mm_s'] for x in detail if x['stage']=='final')
 result['final_depth_net_drift_range_mm']=[min(x['final8_depth_net_drift_mm'] for x in detail if x['stage']=='final'),max(x['final8_depth_net_drift_mm'] for x in detail if x['stage']=='final')]
 result['checkpoint_hashes']=[{'name':x.name,'sha256':sha(x)} for x in sorted(p.glob('ppo-append-update-*.bin'))]
 allresults[arm]=result
ad=allresults['A']['final_holding_diagnostic'];bd=allresults['B']['final_holding_diagnostic']
diagnostic={'defined':ad['variance_defined'] and bd['variance_defined'],'B_to_A_normalized_MSE':bd['variance_normalized_MSE']/ad['variance_normalized_MSE'],'B_minus_A_EV':bd['EV_after']-ad['EV_after']}
diagnostic['passed']=diagnostic['defined'] and diagnostic['B_to_A_normalized_MSE']<=.75 and diagnostic['B_minus_A_EV']>=.2
summary={'study':read(s/'result.json'),'execution':read(s/'execution.json'),'paired_first_batch':read(s/'paired-first-batch.json'),'checks':read(s/'checks.json'),'arms':allresults,'conditioning_diagnostic':diagnostic,
 'preservation':{'prior_checkpoint_protocol_files_hash_verified':len(prior),'unrelated_files_hash_verified':len(unrelated),'frozen_inputs_hash_verified':len(frozen),'checkpoint336_unchanged':True},
 'runtime_adopted':False,'no_more_training':True,'retries':0,'extensions':0,'limitations':['One seed and eight updates per arm','Reward gain compared under balanced goal/pose mix, not a full factorial design','No holding pass; no proof of indefinite swimming','Two original startup window misses remain in accepted swim regression definition']}
(s/'comparison-summary.json').write_text(json.dumps(summary,indent=2))
evidence=[{'path':str(p.relative_to(s)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in s.rglob('*') if p.is_file() and p.name!='evidence-manifest.json']
(s/'evidence-manifest.json').write_text(json.dumps(evidence,indent=2))
out={'study':summary['study'],'execution':summary['execution'],'conditioning_diagnostic':diagnostic,'preservation':summary['preservation']}
for arm in ['A','B']:out[arm]={k:allresults[arm][k] for k in ['training_started_utc','old_Adam_age','new_column_age','stages','integrity','final_hold_speed_min_mm_s','final_hold_speed_max_mm_s','final_depth_net_drift_range_mm']}
print(json.dumps(out,indent=2))
