from pathlib import Path
import json,csv,hashlib,struct,math,statistics
import numpy as np
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation');run=r/'build/ppo/rest-hold-336to344-v1';pres=r/'build/review-preservation/2026-11-04-before-rest-hold'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def rows(p):return list(csv.DictReader(p.open(newline='')))
def n(q,k):return float(q[k])
result=json.loads((run/'result.json').read_text());report=json.loads((run/'pre-PPO-report.json').read_text());assert result['PPO_updates']==8 and result['critic_only_steps']==80 and not result['fault'] and not result['gate_failed']
class Reader:
 def __init__(self,b):self.b=b;self.at=0
 def v(self,f):x=struct.unpack_from('<'+f,self.b,self.at)[0];self.at+=struct.calcsize('<'+f);return x
 def array(self,dtype='<f4'):
  dt=np.dtype(dtype);count=self.v('Q');x=np.frombuffer(self.b,dtype=dt,count=count,offset=self.at);self.at+=count*dt.itemsize;return x
def payload(p):
 b=p.read_bytes();magic,length,checksum=struct.unpack_from('<QQQ',b);assert magic==0x31504B434C524443 and length==len(b)-24;return b[24:]
def checkpoint(p):
 q=Reader(payload(p));assert q.v('I')==0x35415050 and q.v('I')==5;protocol=q.v('Q');assert q.v('I')==1247037243;update=q.v('i')
 for x in [6,6,3,9,4,41,4,2532,2433]:assert q.v('I')==x
 q.v('f');q.array();actor=q.array().copy();critic=q.array().copy();optim=[]
 for count in [2532,2433]:
  start=q.at;config=q.b[q.at:q.at+20];q.at+=20;age=q.v('i');m=q.array();v=q.array();assert len(m)==count and len(v)==count
  optim.append({'age':age,'bytes':q.b[start:q.at],'m':m,'v':v,'config':config})
 anew=q.v('i');cnew=q.v('i');assert q.at==len(q.b);return {'actor':actor,'critic':critic,'optim':optim,'actor_new_age':anew,'critic_new_age':cnew,'update':update,'protocol':protocol}
initial=checkpoint(run/'initial-training-state/ppo-append-update-336.bin');calibrated=checkpoint(run/'critic-calibrated/ppo-append-update-336.bin');final=checkpoint(run/'update-344/ppo-append-update-344.bin')
assert np.array_equal(initial['actor'],calibrated['actor']) and initial['optim'][0]['bytes']==calibrated['optim'][0]['bytes']
assert [q['age'] for q in initial['optim']]==[21700,21700] and [q['age'] for q in calibrated['optim']]==[21700,21780] and calibrated['actor_new_age']==0 and calibrated['critic_new_age']==80
assert [q['age'] for q in final['optim']]==[22340,22420] and final['actor_new_age']==640 and final['critic_new_age']==720
value=rows(run/'calibration-value-rows.csv');assert len(value)==10240
validation={}
for hold in [0,1]:
 rr=[q for q in value if int(q['hold'])==hold and int(q['fit'])==0];assert len(rr)==2560
 for phase in ['before','after']:
  errors=[n(q,'value_'+phase)-n(q,'return') for q in rr];expected=report[('hold' if hold else 'swim')+'_validation_'+phase]
  rmse=math.sqrt(statistics.mean(e*e for e in errors));bias=statistics.mean(errors)
  assert abs(rmse-expected['RMSE'])<1e-10 and abs(bias-expected['bias'])<1e-10
  validation[f'{hold}_{phase}']={'RMSE':rmse,'bias':bias}
details=[];stages=[]
for stage in ['baseline','final']:
 summaries=rows(run/f'{stage}-rest/summary.csv');controls=rows(run/f'{stage}-rest/controls.csv');short=rows(run/f'{stage}-evaluation.csv');long=rows(run/f'{stage}-long/sustained-windows.csv');assert len(summaries)==6 and len(controls)==7680 and len(short)==4 and len(long)==30
 for case in range(6):
  rr=[q for q in controls if int(q['case'])==case];ss=summaries[case];late=[q for q in rr if int(q['control'])>1120];at56=next(q for q in rr if int(q['control'])==1120)
  gates={'speed':max(n(q,'speed_mm_s') for q in rr)<=.5,'anchor':max(n(q,'anchor_error_mm') for q in rr)<=2,'depth':max(abs(n(q,'depth_mm')) for q in rr)<=1,'lane':max(n(q,'lane_mm') for q in rr)<=2,'tilt':max(n(q,'tilt_deg') for q in rr)<=10,
   'final8_excursion':max(abs(n(q,'forward_mm')-n(at56,'forward_mm')) for q in rr if int(q['control'])>=1120)<=2}
  assert all(gates.values())==bool(int(ss['passed']))
  for field,key,absolute in [('max_speed_mm_s','speed_mm_s',False),('max_anchor_error_mm','anchor_error_mm',False),('max_depth_mm','depth_mm',True),('max_lane_mm','lane_mm',False),('max_tilt_deg','tilt_deg',False)]:
   actual=max(abs(n(q,key)) if absolute else n(q,key) for q in rr);assert abs(actual-n(ss,field))<2e-7,(stage,case,field)
  first=next((n(q,'time_s') for q in rr if n(q,'speed_mm_s')>.5),None)
  details.append({'stage':stage,'case':case,'pose':int(ss['pose']),'offset_mm':n(ss,'offset_mm'),'gates':gates,'first_speed_violation_s':first,
    'final8_mean_speed_mm_s':statistics.mean(n(q,'speed_mm_s') for q in late),'final8_speed_min_mm_s':min(n(q,'speed_mm_s') for q in late),'final8_speed_max_mm_s':max(n(q,'speed_mm_s') for q in late),
    'final8_forward_drift_mm':n(rr[-1],'forward_mm')-n(at56,'forward_mm'),'final8_signed_depth_drift_mm':n(rr[-1],'depth_mm')-n(at56,'depth_mm'),
    'final8_joint_ranges':[{'axis':j,**{key:[min(n(q,key+str(j)) for q in late),max(n(q,key+str(j)) for q in late)] for key in ['command','angle','rate','last_torque']}} for j in range(4)]})
 ds=[q for q in details if q['stage']==stage]
 stages.append({'stage':stage,'short_passes':sum(int(q['passed']) for q in short),'mean_short_speed_mm_s':statistics.mean(n(q,'steady_mm_s') for q in short),
   'long_windows_passed':sum(int(list(q.values())[-1]) for q in long),'retained4to60_windows':sum(int(list(q.values())[-1]) for q in long if float(list(q.values())[1])>4),
   'rest_passes':sum(int(q['passed']) for q in summaries),'depth_passes':sum(q['gates']['depth'] for q in ds),'lane_passes':sum(q['gates']['lane'] for q in ds),'tilt_passes':sum(q['gates']['tilt'] for q in ds),
   'max_speed_mm_s':max(n(q,'max_speed_mm_s') for q in summaries),'max_anchor_error_mm':max(n(q,'max_anchor_error_mm') for q in summaries),'max_depth_mm':max(n(q,'max_depth_mm') for q in summaries),'max_lane_mm':max(n(q,'max_lane_mm') for q in summaries),'max_tilt_deg':max(n(q,'max_tilt_deg') for q in summaries),
   'final8_mean_speed_range_mm_s':[min(q['final8_mean_speed_mm_s'] for q in ds),max(q['final8_mean_speed_mm_s'] for q in ds)],'final8_forward_drift_range_mm':[min(q['final8_forward_drift_mm'] for q in ds),max(q['final8_forward_drift_mm'] for q in ds)],
   'first_speed_violation_range_s':[min(q['first_speed_violation_s'] for q in ds),max(q['first_speed_violation_s'] for q in ds)]})
def batch_observations(p):
 q=Reader(payload(p));assert q.v('I')==1;q=Reader(q.array('u1').tobytes());assert [q.v('I') for _ in range(4)]==[1,10240,41,4];return q.array().reshape(10240,41)
training=[];world=np.arange(10240)%32;hold=(world//2)%2==1
for u in range(336,344):
 x=batch_observations(run/f'batch-{u}.bin');speed=np.linalg.norm(x[:,12:15],axis=1)*20
 training.append({'global_update_collected':u,'hold_rows':int(hold.sum()),'hold_pre_action_speed_le05':int(np.sum(hold&(speed<=.5))),'hold_pre_action_speed_mean_mm_s':float(speed[hold].mean()),'hold_initial_zero_velocity_rows':int(np.sum(hold&(np.arange(10240)//32==0)&(speed==0)))})
integrity=[json.loads((run/f'PPO-{u}-integrity.json').read_text()) for u in range(336,344)]+[json.loads((run/'calibration-integrity.json').read_text())]
assert all(q['passed'] for q in integrity)
frozen=json.loads((run/'frozen-hashes.json').read_text())
for h in frozen:assert sha(Path(h['path']))==h['sha256'],h['path']
previous=json.loads((pres/'prior-evidence-hashes.json').read_text())
for h in previous:assert sha(r/h['path'])==h['sha256'],h['path']
unrelated=json.loads((pres/'unrelated-hashes.json').read_text())
for h in unrelated:assert sha(r/h['path'])==h['sha256'],h['path']
summary={'result':result,'execution':json.loads((run/'execution.json').read_text()),'calibration_report':report,'signal_gate':json.loads((run/'first-PPO-signal-gate.json').read_text()),'stages':stages,
 'calibration_actor_parameters_and_Adam_unchanged':True,'asymmetric_final_ages':{'actor_old':22340,'actor_new':640,'critic_old':22420,'critic_new':720},
 'all9_batch_integrity_passed':True,'max_GAE_error':max(q['max_GAE_error'] for q in integrity),'max_old_logp_rescore_error':max(q['max_old_logp_error'] for q in integrity),
 'independent_all12_physical_cases_and_validation_metrics_agree':True,'training_hold_coverage':training,
 'preservation':{'prior_evidence_files_hash_verified':len(previous),'unrelated_files_verified':len(unrelated),'current_frozen_files_verified':len(frozen),'checkpoint336_unchanged':True},
 'reporting_deviation':{'native_report_generated_before_PPO':True,'native_reporting_pause_occurred':True,'report_delivered_to_Drew_before_PPO':False,'requested_delivery_timing_met':False,
  'explanation':'The native report was generated and a local commentary message emitted before release, but the delegated update did not reach Drew until after PPO. The reporting requirement was missed, as confirmed by parent delegation. Do not describe local emission as verified user receipt.'},
 'runtime_adopted':False,'no_further_training':True,'retries':0,'extensions':0}
(run/'comparison-summary.json').write_text(json.dumps(summary,indent=2));(run/'hold-details.json').write_text(json.dumps(details,indent=2));(run/'reporting-deviation.json').write_text(json.dumps(summary['reporting_deviation'],indent=2))
print(json.dumps({k:summary[k] for k in ['result','signal_gate','stages','asymmetric_final_ages','max_GAE_error','preservation','reporting_deviation','training_hold_coverage']},indent=2))
