import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json,struct,hashlib,csv,math,statistics,time
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
s=r/'build/ppo/balanced-hold-scale-336to344-v1'
out=Path(r'C:\Users\aiand\Documents\Codex\2026-11-04\task\hold-diagnosis')
out.mkdir(exist_ok=True)
evidence={x['path'].replace('\\','/'):x for x in json.loads((s/'evidence-manifest.json').read_text())}
verified=[]
def load(p):
 raw=p.read_bytes();rel=str(p.relative_to(s)).replace('\\','/')
 if rel in evidence:assert hashlib.sha256(raw).hexdigest().upper()==evidence[rel]['sha256'],rel;verified.append(rel)
 magic,n,checksum=struct.unpack_from('<QQQ',raw);assert magic==0x31504B434C524443 and len(raw)==n+24
 return raw[24:]
class Reader:
 def __init__(self,data):self.data=data;self.at=0
 def v(self,fmt):v=struct.unpack_from('<'+fmt,self.data,self.at)[0];self.at+=struct.calcsize('<'+fmt);return v
 def array(self,dtype='<f4'):
  n=self.v('Q');dt=np.dtype(dtype);v=np.frombuffer(self.data,dtype=dt,count=n,offset=self.at);self.at+=n*dt.itemsize;return v
def network(p):
 b=Reader(load(p));assert b.v('I')==0x35415050 and b.v('I')==5
 b.v('Q');assert b.v('I')==1247037243;update=b.v('i')
 for v in [6,6,3,9,4,41,4,2532,2433]:assert b.v('I')==v
 assert abs(b.v('f')-.8)<1e-6
 logsigma=b.array();actor=b.array();critic=b.array();assert actor.size==2532 and critic.size==2433
 return actor.copy(),critic.copy(),update
def batch(p):
 outer=Reader(load(p));assert outer.v('I')==1;inner=Reader(outer.array('u1').tobytes());assert [inner.v('I') for _ in range(4)]==[1,40960,41,4]
 names=['obs','prevobs','prevaction','history','action','logp','value','reward','done'];d={k:inner.array() for k in names};assert inner.at==len(inner.data)
 for k,w in [('obs',41),('prevobs',41),('action',4),('prevaction',4)]:d[k]=d[k].reshape(-1,w)
 d['costs']=outer.array().reshape(-1,5);return d
def split(p,output):
 layers=[];at=0
 for ni,no in [(41,32),(32,32),(32,output)]:
  w=p[at:at+ni*no].reshape(no,ni);at+=ni*no;b=p[at:at+no];at+=no;layers.append((w,b))
 assert at==p.size;return layers
def forward(p,x,output=4,cache=False):
 acts=[x]
 for j,(w,b) in enumerate(split(p,output)):
  y=acts[-1]@w.T+b
  acts.append(np.tanh(y) if j<2 else y)
 return (acts[-1],acts) if cache else acts[-1]
def backward(p,acts,g):
 layers=split(p,4);gr=[]
 for j in [2,1,0]:
  if j<2:g=g*(1-acts[j+1]**2)
  dw=g.T@acts[j];db=np.sum(g,axis=0);gr.insert(0,np.r_[dw.ravel(),db]);g=g@layers[j][0]
 return np.concatenate(gr)
def gae(d):
 rew=d['reward'].reshape(1280,32);v=d['value'].reshape(1280,32);adv=np.zeros_like(v);last=np.zeros(32,np.float32)
 # Match gae_one_env's TD recursion; lambda1, full terminal episode.
 for t in range(1279,-1,-1):
  nv=v[t+1] if t<1279 else 0
  delta=rew[t]+np.float32(.99)*nv-v[t];last=delta+np.float32(.99)*last;adv[t]=last
 return adv.ravel(),(adv+v).ravel()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def stats(x):
 x=np.asarray(x,dtype=np.float64);return {'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max()),'RMS':rms(x)}
row=np.arange(40960);tick=row//32;world=row%32;hold=(world//2)%2==1;active=hold&(tick>=640);other=~active
def sensitivity(p,x,mask):
 z=x[mask].copy();base=forward(p,z);cmd=np.tanh(base)
 zero=z.copy();zero[:,3:6]=0
 eight=z.copy();eight[:,3:6]=eight[:,6:9]*np.float32(.4)
 diff=np.tanh(forward(p,zero))-np.tanh(forward(p,eight))
 pos0=z.copy();pos0[:,40]=0;posdiff=cmd-np.tanh(forward(p,pos0))
 pp=z.copy();pm=z.copy();pp[:,40]=np.clip(pp[:,40]+.001,-1,1);pm[:,40]=np.clip(pm[:,40]-.001,-1,1)
 denom=pp[:,40]-pm[:,40];der=(np.tanh(forward(p,pp))-np.tanh(forward(p,pm)))/denom[:,None]
 phys=der*(1-z[:,40,None]**2)/20 # radians target /mm forward error
 sweep={}
 for val in [-1,-.5,0,.5,1]:
  zz=z.copy();zz[:,40]=val;sweep[str(val)]=np.mean(np.tanh(forward(p,zz))-np.tanh(forward(p,pos0)),axis=0).tolist()
 return {'rows':len(z),'command_RMS':rms(cmd),'goal0_minus8_command_RMS':rms(diff),'goal0_minus8_RMS_by_axis':np.sqrt(np.mean(diff.astype(np.float64)**2,axis=0)).tolist(),'goal0_minus8_mean_by_axis':diff.mean(axis=0).tolist(),
  'goal0_minus8_command_abs_max':float(abs(diff).max()),'position_feature_minus_zero_command_RMS':rms(posdiff),'position_feature_minus_zero_RMS_by_axis':np.sqrt(np.mean(posdiff.astype(np.float64)**2,axis=0)).tolist(),'position_feature_derivative_RMS':rms(der),
  'physical_position_derivative_rad_per_mm_RMS':rms(phys),'feature_sweep_mean_change_from0_by_axis':sweep}
result={'scope':'Saved data, network inference and unapplied analytical gradients only; no physics integration, optimizer steps or runtime edits','arms':{}}
for arm in ['A','B']:
 records=[];gradients=[];sens=[]
 for u in range(336,344):
  d=batch(s/arm/f'batch-{u}.bin');pa,pc,up=network(s/arm/f'ppo-append-update-{u}.bin');assert up==u
  x=d['obs'];mu,cache=forward(pa,x,cache=True);pmu,pcache=forward(pa,d['prevobs'],cache=True)
  cond=mu+np.float32(.8)*d['history'][:,None]*(d['prevaction']-pmu)
  var=np.exp(np.float32(-2))*(1-np.float32(.8)**2*d['history'])
  score=-.5*np.sum((d['action']-cond)**2/var[:,None]+np.log(np.float32(2*np.pi)*var[:,None]),axis=1)
  rescore=float(abs(score-d['logp']).max());assert rescore<1e-4,(arm,u,rescore)
  adv,ret=gae(d);na=(adv-adv.mean())/(adv.std()+np.float32(1e-8))
  speed=np.linalg.norm(x[:,12:15],axis=1)*20
  fv=np.sum(x[:,12:15]*x[:,6:9],axis=1)*20
  tilt=np.degrees(np.arccos(np.clip(x[:,10],-1,1)))
  vertical=np.sum(x[:,:3]*x[:,9:12],axis=1)*-2
  feat=x[:,40];held=feat[active]
  rawgain=1 if arm=='A' else 1/1024
  costs=d['costs'].astype(np.float64).sum(axis=0)*.05
  holdcost=d['costs'][active].astype(np.float64).sum(axis=0)*.05
  tmp={
   'update_collected':u,'actual_logp_rescore_max_error':rescore,'active_hold_rows':int(active.sum()),
   'hold_speed_mm_s':stats(speed[active]),'hold_forward_velocity_mm_s':stats(fv[active]),'hold_speed_le05_count':int(np.sum(active&(speed<=.5))),
   'hold_forward_velocity_negative_count':int(np.sum(active&(fv<0))),
   'hold_low_speed_and_depth_tilt_count':int(np.sum(active&(speed<=.5)&(abs(vertical)<=1)&(tilt<=10))),
   'feature_near99_fraction':float(np.mean(abs(held)>=.99)),'feature_exact1_fraction':float(np.mean(abs(held)==1)),
   'feature_positive_count':int(np.sum(held>0)),'feature_negative_fraction':float(np.mean(held<0)),
   'after2s_time_feature_nonzero':int(np.sum(x[tick>=40,34]!=0)),
   'holding_cost_fraction':float(holdcost.sum()/costs.sum()),'holding_components_fraction':(holdcost/holdcost.sum()).tolist(),
   'normalized_adv_hold':stats(na[active]),'normalized_adv_hold_positive_fraction':float(np.mean(na[active]>0)),
   'normalized_adv_other':stats(na[other]),'hold_reward_discount_at16s':float(.99**320),
  }
  noise=d['action']-cond;g=-na[:,None]*noise/var[:,None]/40960
  groupgr=[]
  for mask in [active,other]:
   gc=g*mask[:,None];gp=-np.float32(.8)*d['history'][:,None]*gc
   groupgr.append(backward(pa,cache,gc)+backward(pa,pcache,gp))
  gh,go=groupgr;total=gh+go
  # Reconstruct initial full-batch, unclipped, current-policy gradient; NOT stored minibatch gradients after updates.
  gradients.append({'update_collected':u,'held_grad_norm':float(np.linalg.norm(gh)),'other_grad_norm':float(np.linalg.norm(go)),
   'cosine_held_other':float(np.dot(gh,go)/(np.linalg.norm(gh)*np.linalg.norm(go))),
   'held_column40_norm':float(np.linalg.norm(gh[:1312].reshape(32,41)[:,40])),
   'other_column40_norm':float(np.linalg.norm(go[:1312].reshape(32,41)[:,40])),
   'held_goal_columns_norm':float(np.linalg.norm(gh[:1312].reshape(32,41)[:,3:6])),
   'total_grad_norm':float(np.linalg.norm(total)),
   'held_row_score_gradient_nonzero_fraction':float(np.mean(np.linalg.norm(g[active],axis=1)>1e-12)),
   'mean_directional_score_gain_held':float(-np.dot(gh,total)),'mean_directional_score_gain_other':float(-np.dot(go,total))})
  if u in [336,343]:
   final=network(s/arm/'ppo-append-update-344.bin')[0]
   sens.append({'states_from_batch':u,'before_policy':sensitivity(pa,x,active),'after344_policy':sensitivity(final,x,active),
    'command_boundary_before':sensitivity(pa,x,hold&(tick==640)),'command_boundary_after344':sensitivity(final,x,hold&(tick==640))})
   windows=[]
   for lo,hi in [(32,34),(34,40),(40,48),(48,56),(56,64)]:
    mask=hold&(tick*.05>=lo)&(tick*.05<hi);flat=mask&(abs(feat)==1)
    residual=d['value']-ret
    windows.append({'start_s':lo,'end_s':hi,'speed_mean_mm_s':float(speed[mask].mean()),'return_mean':float(ret[mask].mean()),'critic_before_mean':float(d['value'][mask].mean()),
      'residual_mean':float(residual[mask].mean()),'feature_exact1_fraction':float(np.mean(abs(feat[mask])==1))})
   tmp['windows']=windows
   flat=active&(abs(feat)==1)
   tmp['flat_feature_critic_residual_correlation_with_remaining_time']=float(np.corrcoef((1280-tick[flat])*.05,(d['value']-ret)[flat])[0,1])
  records.append(tmp)
  del d,cache,pcache
 # Quantify policy parameter changes, especially the new column.
 initial=network(s/arm/'ppo-append-update-336.bin')[0];final=network(s/arm/'ppo-append-update-344.bin')[0]
 result['arms'][arm]={'batches':records,'initial_unclipped_gradient_reconstruction':gradients,'matched_state_sensitivity':sens,
   'actor_total_parameter_change_RMS':rms(final-initial),'actor_new_column40_RMS':rms(final[:1312].reshape(32,41)[:,40])}
 # Frozen deterministic trajectories: compare gait activity across phase windows without integrating.
 frozen=[]
 for stage in ['baseline','final']:
  rr=list(csv.DictReader((s/arm/f'{stage}-stop/controls.csv').open(newline='')))
  for case in range(6):
   cr=[q for q in rr if int(q['case'])==case];phases=[]
   for lo,hi in [(24,32),(32,40),(48,56),(56,64)]:
    ss=[q for q in cr if lo<float(q['time_s'])<=hi]
    phases.append({'start_s':lo,'end_s':hi,'speed_mean_mm_s':statistics.mean(float(q['COM_speed_mm_s']) for q in ss),
     'command_std_by_axis':[statistics.pstdev(float(q[f'command{j}']) for q in ss) for j in range(4)],
     'joint_rate_RMS_rad_s':rms([[float(q[f'rate{j}']) for j in range(4)] for q in ss]),
     'last_torque_normalized_RMS':rms([[float(q[f'last_torque{j}']) for j in range(4)] for q in ss])})
   frozen.append({'stage':stage,'case':case,'phases':phases})
 result['arms'][arm]['frozen_gait_activity']=frozen
result['verified_inputs_count']=len(set(verified));result['verified_inputs']=sorted(set(verified))
(out/'diagnosis.json').write_text(json.dumps(result,indent=2))
short={}
for arm in ['A','B']:
 a=result['arms'][arm];b=a['batches'];g=a['initial_unclipped_gradient_reconstruction']
 short[arm]={'training_hold_rows':sum(x['active_hold_rows'] for x in b),'hold_speed_le05_total':sum(x['hold_speed_le05_count'] for x in b),
  'negative_forward_velocity_total':sum(x['hold_forward_velocity_negative_count'] for x in b),'first_last_batch':[b[0],b[-1]],
  'first_last_gradients':[g[0],g[-1]],'actor_new_column40_RMS':a['actor_new_column40_RMS'],
  'sensitivity_on_first_batch':a['matched_state_sensitivity'][0]}
print(json.dumps(short,indent=2))
