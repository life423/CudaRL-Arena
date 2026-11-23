"""Offline balanced-gait measurements and evidence verification.

Reads saved native states only; never steps a body or applies an optimizer.
The immutable earlier verifier supplies articulated kinematics/NN arithmetic.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

import numpy as np
import verify_current_swim as reference
import verify_current_swim_v2 as prior_audit

SIGMA=float(np.float32(.35))
PITCH_LIMIT=10.
OSCILLATION_LIMIT=5.
WORLD_COUNT=8192
EPISODE_CONTROLS=160
GAIT_START_CONTROL=80
CONTROL_CAP=85295280
SUBSTEP_CAP=852952800
HASHES,VALIDATED=reference.HASHES,reference.VALIDATED
read,js,rows,near,envelope=reference.read,reference.js,reference.rows,reference.near,reference.envelope
digest,payload_checksum=prior_audit.file_digest,prior_audit.payload_checksum
PARTS=(('pre-state',33),('post-state',23),('base',4),('joint-latent',4),('reward',1))


def tail_angles(angles):
    """Head-relative tail axis; full yaw/pitch kinematics, not pitch addition."""
    angles=np.asarray(angles,float)
    assert angles.shape[-1]==4 and np.isfinite(angles).all()
    q=np.zeros(angles.shape[:-1]+(4,));q[...,3]=1
    for joint in range(2):
        q=reference.qmul(q,reference.axis_rotation([0,1,0],angles[...,joint*2]))
        q=reference.qmul(q,reference.axis_rotation([0,0,1],angles[...,joint*2+1]))
    axis=reference.rotate(q,[1,0,0])
    elevation=np.degrees(np.arctan2(axis[...,1],np.linalg.norm(axis[...,[0,2]],axis=-1)))
    geodesic=np.degrees(np.arccos(np.clip(axis[...,0]/np.linalg.norm(axis,axis=-1),-1,1)))
    return elevation,geodesic


def gait_metrics(angles):
    """Population moments for an explicit window of actual joint positions."""
    angles=np.asarray(angles,float)
    assert angles.ndim==2 and angles.shape[1]==4 and len(angles)>1
    elevation,kink=tail_angles(angles)
    means=angles.mean(axis=0);centered=angles-means
    variance=np.mean(centered*centered,axis=0)
    joint_std=np.degrees(np.sqrt(variance))
    rms=float(np.degrees(np.sqrt(np.mean(variance))))
    pitch=np.degrees(means[[1,3]])
    bend=float(elevation.mean())
    gates={'pitch1':bool(abs(pitch[0])<=PITCH_LIMIT),'pitch3':bool(abs(pitch[1])<=PITCH_LIMIT),
           'net_vertical_bend':bool(abs(bend)<=PITCH_LIMIT),'oscillation':bool(rms>=OSCILLATION_LIMIT)}
    bias_cost=.05*((pitch[0]/10)**2+(pitch[1]/10)**2+(bend/10)**2)
    return {'samples':len(angles),'joint_mean_deg':np.degrees(means).tolist(),'joint_std_deg':joint_std.tolist(),
            'joint_range_deg':np.degrees(np.ptp(angles,axis=0)).tolist(),
            'pitch_mean_deg':pitch.tolist(),'pitch_RMS_deg':np.degrees(np.sqrt(np.mean(angles[:,[1,3]]**2,axis=0))).tolist(),
            'pitch_peak_abs_deg':np.degrees(np.max(abs(angles[:,[1,3]]),axis=0)).tolist(),
            'net_vertical_bend_mean_deg':bend,'net_vertical_bend_RMS_deg':float(np.sqrt(np.mean(elevation**2))),
            'net_vertical_bend_peak_abs_deg':float(abs(elevation).max()),
            'geodesic_kink_mean_deg':float(kink.mean()),'geodesic_kink_peak_deg':float(kink.max()),
            'centered_four_joint_RMS_deg':rms,'bias_cost':float(bias_cost),'gates':gates,'passed':bool(all(gates.values()))}


def gait_window(body,start_control=80,end_control=160):
    body=np.asarray(body,float)
    assert body.ndim==2 and body.shape[1]>=11 and 0<=start_control<end_control<len(body)
    result=gait_metrics(body[start_control+1:end_control+1,7:11])
    result.update(start_control=start_control,end_control=end_control,
                  start_s=start_control*.05,end_s=end_control*.05)
    return result


def swim_latents(latent,backbone):
    latent=np.asarray(latent,float);backbone=np.asarray(backbone,float)
    assert latent.shape[-1]==8 and backbone.shape[-1]==4
    result=np.exp(math.log(2)*np.tanh(latent[...,4:]))*backbone+.75*np.tanh(latent[...,:4])
    return np.where(np.all(latent==0,axis=-1,keepdims=True),backbone,result)


def log_probability(means,latent):
    return np.sum(-.5*((np.asarray(latent,float)-means)/SIGMA)**2-math.log(SIGMA)-.5*math.log(2*math.pi),axis=-1)


def leave_one_out(returns,scale):
    returns=np.asarray(returns,float);n=len(returns)
    assert n==WORLD_COUNT and math.isfinite(scale) and scale>0
    result=np.zeros(n)
    for case in range(4):
        values=returns[case::4];assert len(values)==2048
        result[case::4]=(values-(values.sum()-values)/2047)/scale
    return result


def validate_envelopes(root,helper):
    paths=sorted(root.rglob('*.bin'))
    directory=Path('build/gait-pond-preparation-2026-11-22/offline-audits');directory.mkdir(parents=True,exist_ok=True)
    listing=directory/'envelope-paths.txt';listing.write_text(''.join(str(p.resolve())+'\n' for p in paths),encoding='utf-8')
    read(helper)
    process=subprocess.run([str(helper.resolve()),str(listing.resolve())],capture_output=True,text=True,check=True,
        creationflags=0x08000000 if os.name=='nt' else 0)
    result=json.loads(process.stdout)
    assert result['passed'] and result['files']==len(paths) and result['physical_controls']==result['optimizer_steps']==0
    VALIDATED.update(p.resolve() for p in paths);return result


def file_fnv(path):return reference.shared.fnv(read(path))


def read_protocol(root):
    r=envelope(root/'protocol.bin');protocol=payload_checksum(root/'protocol.bin')
    assert [r.value('I'),r.value('I')]==[0x31524747,1]
    assert r.value('Q')==file_fnv(root/'approved-protocol.md')
    parent,hold=r.value('Q'),r.value('Q')
    assert [r.value('i') for _ in range(18)]==[6,1,3,4,41,4,336,8,8192,160,80,64,1024,4,2048,CONTROL_CAP,SUBSTEP_CAP,900]
    near([r.value('f') for _ in range(12)],np.float32([.35,.75,.5,2,.01,.5,.2,.01,.05,1,10,5]),0,0)
    assert [r.value('I'),r.value('I')]==[0x67a13c85,0xa2d54913]
    assert [r.value('Q') for _ in range(3)]==[35*1024**3,8*1024**3,24*1024**3];r.finish()
    return protocol,parent,hold


def read_policy(path,protocol,inherited):
    frozen,holding,gain,parent,hold=inherited;r=envelope(path)
    assert [r.value('I'),r.value('I')]==[0x31504747,1]
    assert [r.value('Q') for _ in range(3)]==[protocol,parent,hold]
    assert [r.value('I') for _ in range(8)]==[6,1,3,4,41,4,336,8]
    n,update,age=[r.value('i') for _ in range(3)]
    assert n==8192 and 0<=update<=64 and 0<=age<=1024 and age%4==0
    assert np.array_equal(r.array(size=2532),frozen) and np.array_equal(r.array(size=2),holding) and r.value('f')==gain
    means=r.array(size=8).copy();near([r.value('f') for _ in range(3)],np.float32([.35,.75,2]),0,0)
    optimizer=reference.shared.Reader(r.array('u1').tobytes());r.finish()
    near([optimizer.value('f') for _ in range(5)],np.float32([.01,.9,.999,.00001,.5]),0,0)
    assert optimizer.value('i')==age
    first,second=optimizer.array(size=8),optimizer.array(size=8);optimizer.finish()
    assert np.isfinite(means).all() and np.isfinite(first).all() and np.all(np.isfinite(second)&(second>=0))
    if update==0:assert age==0 and np.all(means==0) and np.all(first==0) and np.all(second==0)
    return means,{'update':update,'Adam_steps':age,'means':means.tolist(),'bias':(.75*np.tanh(means[:4].astype(float))).tolist(),
                  'gain':np.exp(math.log(2)*np.tanh(means[4:].astype(float))).tolist(),'sha256':HASHES[str(path.resolve())]}


def gait_type(layout):
    g=layout['gait']
    names=['pitch_sum','tail_bend_sum','pitch_square_sum','tail_bend_square_sum','angle_sum','angle_square_sum',
           'pitch_peak','tail_bend_peak','yaw_min','yaw_max','samples']
    formats=[('<f8',2),'<f8',('<f8',2),'<f8',('<f8',4),('<f8',4),('<f4',2),'<f4',('<f4',2),('<f4',2),'<i4']
    assert g['size']==144 and [g[name] for name in names]==[0,16,24,40,48,80,112,120,124,132,140]
    return np.dtype({'names':names,'formats':formats,'offsets':[g[name] for name in names],'itemsize':g['size']})


def read_batch(directory,protocol,frozen,expected_means,layout,purpose,index):
    r=envelope(directory/'meta.bin');binding=payload_checksum(directory/'meta.bin');n=8192;ticks=160
    assert [r.value('I'),r.value('I'),r.value('Q')]==[0x31424747,1,protocol]
    assert [r.value('i') for _ in range(6)]==[n,160,160,purpose,index,0]
    means=r.array(size=8);latent=r.array(size=n*8).reshape(n,8);logp=r.array(size=n);returns=r.array(size=n)
    assert np.array_equal(means,expected_means)
    c=layout['episode_counts'];assert r.value('I')==c['size']
    names=['controls','substeps','reward_sum','gait','failure']
    counts_type=np.dtype({'names':names,'formats':['<u8','<u8','<f8',gait_type(layout),'<i4'],
                         'offsets':[c['controls'],c['substeps'],c['reward_sum'],c['gait'],c['failure_code']],'itemsize':c['size']})
    counts=r.array(counts_type,n)
    assert r.value('I')==layout['world_size']==696;world=r.array(prior_audit.WORLD_TYPE,n)
    assert r.value('I')==layout['task_size']==152;task=r.array(prior_audit.TASK_TYPE,n);r.finish()
    assert np.all(counts['controls']==160) and np.all(counts['substeps']==1600) and np.all(counts['failure']==0)
    assert np.all(counts['gait']['samples']==80)
    assert np.all(world['controls']==160) and np.all(world['substeps']==1600) and np.all(world['attempted']==1600)
    assert np.all(world['failure']==0) and np.all(world['unsafe']==0) and np.all(world['invalid']==0) and np.all(world['current']==0)
    assert np.all(task['controls']==160) and np.all((task['outcome']==1)|(task['outcome']==2))
    probability_error=near(logp,log_probability(means,latent),1e-4,0)
    seed=0x67a13c85 if purpose==0 else 0xa2d54913
    rng_error=near(latent,means+np.float32(.35)*prior_audit.normal(seed,index,np.arange(n*8,dtype=np.uint32).reshape(n,8)),2e-6,2e-6)
    manifest=js(directory/'manifest.json')
    assert manifest['schema']==1 and manifest['meta_checksum']==binding and manifest['worlds']==n and manifest['ticks']==160 and manifest['purpose']==purpose
    assert len(manifest['parts'])==5
    arrays={}
    for field,((name,columns),entry) in enumerate(zip(PARTS,manifest['parts'])):
        path=directory/(name+'.bin');digest(path);assert path.resolve() in VALIDATED
        mm=np.memmap(path,dtype='u1',mode='r');magic,size,checksum=struct.unpack_from('<QQQ',mm)
        assert magic==0x31504b434c524443 and size==len(mm)-24 and size<=512*1024**2
        assert entry==dict(field=field,name=path.name,columns=columns,payload_bytes=size,payload_checksum=checksum)
        part=reference.shared.Reader(memoryview(mm)[24:])
        assert [part.value('I'),part.value('I'),part.value('i'),part.value('Q'),part.value('i'),part.value('i'),part.value('i')]==[0x31424747,1,field,binding,n,160,columns]
        arrays[name]=part.array(size=n*160*columns).reshape(160,n,columns);part.finish()
    poses=np.arange(n)%4;sums=np.zeros(n);squares=np.zeros(n);origin=np.zeros((n,3));peaks=np.zeros((n,3))
    angle_sums=np.zeros((n,4));angle_squares=np.zeros((n,4));bend_sums=np.zeros(n)
    pitch_sums=np.zeros((n,2));maximum={key:0. for key in ('NN','action','reward','continuity','filter_angle','filter_rate')}
    movement=np.zeros(n,dtype=bool)
    for k in range(160):
        for start in range(0,n,4096):
            sl=slice(start,min(n,start+4096));pre=arrays['pre-state'][k,sl];post=arrays['post-state'][k,sl]
            backbone=arrays['base'][k,sl];action=arrays['joint-latent'][k,sl];reward=arrays['reward'][k,sl,0]
            assert all(np.isfinite(x).all() for x in (pre,post,backbone,action,reward))
            assert np.all(abs(pre[:,7:11])<=1) and np.all(abs(post[:,7:11])<=1) and np.all(abs(pre[:,25:29])<=1)
            assert np.all((post[:,22]>=0)&(post[:,22]<=1)) and np.all(post[:,21]>=0)
            previous=reference.initial_body(poses[sl]) if k==0 else arrays['post-state'][k-1,sl,:21]
            maximum['continuity']=max(maximum['continuity'],near(pre[:,:21],previous,2e-7 if k==0 else 0,2e-6 if k==0 else 0))
            obs=reference.observations(pre,poses[sl],np.zeros((len(pre),3)),np.full(len(pre),k))[:,:41]
            maximum['NN']=max(maximum['NN'],near(backbone,reference.shared.forward(frozen.astype(float),obs),2e-5,2e-6))
            maximum['action']=max(maximum['action'],near(action,swim_latents(latent[sl],backbone),2e-5,2e-6))
            commands=np.tanh(action.astype(float));angle=pre[:,21:25].astype(float).copy();rate=pre[:,29:33].astype(float).copy()
            h,omega=float(np.float32(.005)),float(np.float32(2*np.float32(np.pi)))
            for _ in range(10):rate+=h*(omega*omega*(commands-angle)-2*omega*rate);angle+=h*rate
            next_angle=arrays['pre-state'][k+1,sl,21:25] if k<159 else world['held'][sl]
            next_rate=arrays['pre-state'][k+1,sl,29:33] if k<159 else world['rates'][sl]
            maximum['filter_angle']=max(maximum['filter_angle'],near(next_angle,angle,2e-6,2e-5))
            maximum['filter_rate']=max(maximum['filter_rate'],near(next_rate,rate,2e-5,2e-5))
            metrics=reference.metrics(post,poses[sl]);center,_=reference.motion(post)
            maximum['reward']=max(maximum['reward'],near(reward,reference.reward(metrics,post[:,22],np.full(len(pre),k)),2e-5,3e-5))
            sums[sl]+=reward;peaks[sl]=np.maximum(peaks[sl],np.stack((metrics['lane'],metrics['depth'],metrics['tilt']),axis=-1))
            if k==39:origin[sl]=center
            if k>=40:
                delta=center-origin[sl];forward=np.sum(delta*metrics['heading'],axis=-1)
                speed=forward/((k-39)*float(np.float32(.05)))
                squares[sl]+=np.sum((metrics['velocity']-metrics['heading']*float(np.float32(.008)))**2,axis=-1)
                rmse=np.sqrt(squares[sl]/(k-39));side=delta-metrics['heading']*forward[:,None]
                path=np.degrees(np.arctan2(np.linalg.norm(side[:,[0,2]],axis=-1),forward))
                if k==159:
                    work=post[:,21]/np.maximum(metrics['forward'],1e-9)
                    movement[sl]=(speed>=.0072)&(speed<=.0088)&(rmse<=.0025)&(peaks[sl,0]<=.002)&(peaks[sl,1]<=.001)&(peaks[sl,2]<=10)&(path<=5)&(work<=.001)
            if k>=80:
                angles=post[:,7:11].astype(float);angle_sums[sl]+=angles;angle_squares[sl]+=angles**2
                pitch_sums[sl]+=np.degrees(angles[:,[1,3]]);bend_sums[sl]+=tail_angles(angles)[0]
    near(counts['reward_sum'],sums,1e-7,1e-7);assert np.array_equal(task['outcome']==1,movement)
    g=counts['gait'];near(g['angle_sum'],angle_sums,1e-12,1e-12);near(g['angle_square_sum'],angle_squares,1e-12,1e-12)
    near(g['pitch_sum']/80,pitch_sums/80,2e-4,2e-5);near(g['tail_bend_sum']/80,bend_sums/80,2e-4,2e-5)
    native_pitch=g['pitch_sum']/80;native_bend=g['tail_bend_sum']/80
    penalty=float(np.float32(.05))*(np.sum((native_pitch/10)**2,axis=1)+(native_bend/10)**2)
    near(returns,np.float32(sums/8-penalty-(~movement)),2e-6,2e-6)
    variance=np.maximum(0,angle_squares/80-(angle_sums/80)**2);rms=np.degrees(np.sqrt(variance.mean(axis=1)))
    gait=(np.all(abs(pitch_sums/80)<=10,axis=1))&(abs(bend_sums/80)<=10)&(rms>=5)
    near(world['body'][:,:21],arrays['post-state'][-1,:,:21],0,0)
    near(world['body'][:,21],arrays['post-state'][-1,:,21],0,0);near(world['mean_sq'],arrays['post-state'][-1,:,22],0,0)
    near(world['raw'],np.tanh(arrays['joint-latent'][-1].astype(float)),2e-7,2e-6)
    assert np.all(abs(world['actual'])<=1) and np.isfinite(world['requested']).all()
    near(world['mean_sq'],np.mean(world['actual'].astype(float)**2,axis=1),2e-7,2e-6)
    table=rows(directory/'episodes.csv');assert len(table)==n
    for i,row in enumerate(table):
        assert [row[key] for key in ('episode','case','pose','controls','substeps','failure_code')]==[i,i%4,i%4,160,1600,0]
        assert bool(row['movement_passed'])==bool(movement[i])
        near(row['old_logp'],logp[i],1e-8,1e-10);near(row['reward_sum'],sums[i],1e-8,1e-10);near(row['return'],returns[i],1e-8,1e-10)
        near([row['mean_pitch1_deg'],row['mean_pitch3_deg']],native_pitch[i],1e-8,1e-10)
        near(row['mean_tail_bend_deg'],native_bend[i],1e-8,1e-10)
        near(row['centered_joint_RMS_deg'],rms[i],2e-5,2e-6)
        near([row[f'latent{j}'] for j in range(8)],latent[i],1e-8,1e-10)
    return returns.astype(float).copy(),{'controls':n*160,'substeps':n*1600,'probability_error':probability_error,
        'rng_error':rng_error,'maximum_errors':maximum,'sampled_movement_passes':int(movement.sum()),
        'sampled_gait_passes':int(gait.sum()),'sampled_joint_passes':int((gait&movement).sum())}


def compare_gait(row,metrics):
    near([row['mean_pitch1_deg'],row['mean_pitch3_deg']],metrics['pitch_mean_deg'],2e-4,2e-5)
    near(row['mean_tail_vertical_bend_deg'],metrics['net_vertical_bend_mean_deg'],2e-4,2e-5)
    near(row['centered_joint_RMS_deg'],metrics['centered_four_joint_RMS_deg'],2e-5,2e-5)
    assert bool(row['gait_passed'])==metrics['passed']


def cpu_case(directory,case,pose,offset,controls,frozen,means,gain,hold=False):
    table=rows(directory/'controls.csv');selected=[r for r in table if r['case']==case]
    assert [row['control'] for row in selected]==list(range(1,controls+1))
    def columns(prefix,count):return np.asarray([[row[prefix+str(j)] for j in range(count)] for row in selected],dtype=np.float32)
    pre,post,backbone,latent=columns('pre',33),columns('post',23),columns('base',4),columns('latent',4)
    native_reward=np.float32([row['reward'] for row in selected])
    initial=reference.initial_body(pose);initial[1]+=float(np.float32(offset))/float(np.float32(.04))
    near(pre[0,:21],initial,3e-7,2e-6);near(pre[1:,:21],post[:-1,:21],0,0)
    body=np.zeros((controls+1,22));body[0,:21]=pre[0,:21];body[1:]=post[:,:22]
    original_center,_=reference.motion(reference.initial_body(pose));anchor=reference.motion(pre[0])[0].copy();anchor[1]=original_center[1]
    poses=np.full(controls,pose);obs=reference.observations(pre,poses,np.zeros((controls,3)),np.arange(controls))[:,:41]
    if hold:
        obs[:,3:6]=0;center,_=reference.motion(pre)
        heading=reference.rotate(reference.initial_body(pose)[3:7],[1,0,0]);heading/=np.linalg.norm(heading)
        obs[:,40]=np.tanh(np.sum((anchor-center)*heading,axis=-1)/float(np.float32(.02)))
    nn_error=near(backbone,reference.shared.forward(frozen.astype(float),obs),2e-5,2e-6)
    expected=gain*backbone.astype(float) if hold else swim_latents(np.broadcast_to(means,(controls,8)),backbone)
    action_error=near(latent,expected,2e-5,2e-6)
    angle=pre[:,21:25].astype(float).copy();rate=pre[:,29:33].astype(float).copy();command=np.tanh(latent.astype(float))
    h,omega=float(np.float32(.005)),float(np.float32(2*np.float32(np.pi)))
    for _ in range(10):rate+=h*(omega*omega*(command-angle)-2*omega*rate);angle+=h*rate
    near(pre[1:,21:25],angle[:-1],2e-6,2e-5);near(pre[1:,29:33],rate[:-1],2e-5,2e-5)
    values=reference.metrics(body,pose);step={key:(value if key=='heading' else value[1:]) for key,value in values.items()}
    if hold:
        center,velocity=reference.motion(body);error=center-anchor
        desired=-error[1:,[0,2]]/2;mag=np.linalg.norm(desired,axis=-1)
        desired*=np.minimum(1,.002/np.maximum(mag,1e-30))[:,None]
        horizontal=np.sum((velocity[1:,[0,2]]-desired)**2,axis=-1)/.001**2
        vertical=(velocity[1:,1]-np.clip(-values['signed_depth'][1:]/2,-.001,.001))**2/.001**2
        cost=horizontal+vertical+.05*(values['lane'][1:]/.002)**2+.05*np.sum(error[1:,[0,2]]**2,axis=-1)/.002**2
        cost+=.05*(values['depth'][1:]/.001)**2+.05*(values['roll'][1:]/10)**2+.01*(values['pitch'][1:]/10)**2+.002*post[:,22]
        reward=-float(np.float32(.05))*cost
    else:reward=reference.reward(step,post[:,22],np.arange(controls))
    reward_error=near(native_reward,reward,2e-5,3e-5)
    assert np.all(abs(body[:,7:11])<=1) and np.all(body[:,21]>=0) and np.all((post[:,22]>=0)&(post[:,22]<=1))
    return body,values,native_reward,{'NN_error':nn_error,'action_error':action_error,'reward_error':reward_error}


def assess_swim(directory,frozen,means,gain,long_windows=False,recovery=False):
    summary=rows(directory/'summary.csv');initial=rows(directory/'initial.csv');windows=rows(directory/'windows.csv')
    assert len(summary)==len(initial)
    details=[];count=0;old_keys=set();gait_keys=set()
    for case,native in enumerate(summary):
        assert native['case']==case;pose=int(native['pose']);offset=native['offset_mm']/1000;controls=int(native['controls'])
        body,values,rewards,errors=cpu_case(directory,case,pose,offset,controls,frozen,means,gain)
        measured,calculated=reference.swim_metrics(body,pose);gates=reference.swim_gates(measured);movement=all(gates.values())
        for key,value in measured.items():near(native[key],value,.003 if key in ('tilt_deg','path_deg') else .0003,3e-5)
        assert bool(native['movement_passed'])==movement
        gait=gait_window(body,80,controls);compare_gait(native,gait)
        mean_cost=float(rewards.astype(float).sum()/(controls*.05)-float(np.float32(.05))/.05*gait['bias_cost']-(not movement))
        near(native['return'],mean_cost,2e-5,3e-5)
        recovered=movement and (not recovery or offset==0 or np.all(abs(values['signed_depth'][-160:])<abs(offset)*.5))
        assert bool(native['recovery_passed'])==recovered
        fixture=initial[case];assert fixture['case']==case and fixture['pose']==pose and fixture['controls']==controls
        near(fixture['offset_mm'],np.float32(offset)*np.float32(1000),1e-6,1e-6)
        near(fixture['arena_half_width_m'],3. if controls==6000 else float(np.float32(.6)),1e-10,1e-10)
        near(fixture['arena_half_depth_m'],float(np.float32(.4)),1e-10,1e-10)
        center,_=reference.motion(reference.initial_body(pose))
        near([fixture['start_com_'+axis] for axis in 'xyz'],center/.04,2e-6,2e-6)
        case_windows=[r for r in windows if r['case']==case]
        assert len(case_windows)==(controls//80 if long_windows else 0)
        center,velocity=reference.motion(body);heading=values['heading']
        for row,end in zip(case_windows,range(80,controls+1,80)):
            start=40 if end==80 else end-80
            delta=center[end]-center[start];forward=float(np.dot(delta,heading));side=delta-heading*forward
            speed=forward/((end-start)*float(np.float32(.05)))*1000
            rmse=float(np.sqrt(np.mean(np.sum((velocity[start+1:end+1]-heading*float(np.float32(.008)))**2,axis=-1)))*1000)
            value={'steady_mm_s':speed,'RMSE_mm_s':rmse,'lateral_mm':float(values['lane'][:end+1].max()*1000),
                'depth_mm':float(values['depth'][:end+1].max()*1000),'tilt_deg':float(values['tilt'][:end+1].max()),
                'path_deg':float(np.degrees(np.arctan2(np.linalg.norm(side[[0,2]]),forward))),
                'work_J_per_m':float((body[end,21]-body[start,21])/max(forward,1e-9))}
            for key,v in value.items():near(row[key],v,.003 if key in ('tilt_deg','path_deg') else .0003,3e-5)
            assert row['pose']==pose and row['end_s']==end*.05
            passed=all(reference.swim_gates(value).values());assert bool(row['movement_passed'])==passed
            if passed:old_keys.add((pose,end//20))
            assert bool(row['gait_scored'])==(end>=160)
            if end>=160:
                window_gait=gait_window(body,end-80,end);compare_gait(row,window_gait)
                if window_gait['passed']:gait_keys.add((pose,end//20))
        details.append({'case':case,'pose':pose,'offset_m':offset,'controls':controls,'movement_passed':movement,
                        'gait':gait,'recovery_passed':bool(recovered),'movement_gates':gates,'metrics':measured,'errors':errors})
        count+=controls
    return {'cases':details,'controls':count,'substeps':count*10,'movement_passes':sum(c['movement_passed'] for c in details),
            'gait_passes':sum(c['gait']['passed'] for c in details),'recovery_passes':sum(c['recovery_passed'] for c in details),
            'passing_window_keys':sorted(old_keys),'gait_window_keys':sorted(gait_keys),'mean_cost':-float(np.mean([r['return'] for r in summary]))}


def assess_holds(directory,frozen,means,gain):
    summary=rows(directory/'summary.csv');details=[];count=0
    for case,native in enumerate(summary):
        assert native['case']==case;pose=int(native['pose']);offset=native['offset_mm']/1000;controls=int(native['controls'])
        body,values,rewards,errors=cpu_case(directory,case,pose,offset,controls,frozen,means,gain,True)
        center,velocity=reference.motion(body);target,_=reference.motion(reference.initial_body(pose))
        anchor=center[0].copy();anchor[1]=target[1]
        speed=np.linalg.norm(velocity,axis=-1);distance=np.linalg.norm(center-anchor,axis=-1)
        peaks={'max_speed_mm_s':float(speed.max()*1000),'max_anchor_mm':float(distance.max()*1000),
               'max_depth_mm':float(values['depth'].max()*1000),'max_lane_mm':float(values['lane'].max()*1000),
               'max_tilt_deg':float(values['tilt'].max()),'final8_excursion_mm':float(np.max(abs(values['forward'][-161:]-values['forward'][-161]))*1000)}
        for key,value in peaks.items():near(native[key],value,.003 if key=='max_tilt_deg' else .0003,3e-5)
        gates={key:peaks[key]<=limit for key,limit in (('max_speed_mm_s',.5),('max_anchor_mm',2),('max_depth_mm',1),('max_lane_mm',2),('max_tilt_deg',10),('final8_excursion_mm',2))}
        passed=all(gates.values());assert bool(native['passed'])==passed
        details.append({'case':case,'pose':pose,'offset_m':offset,'controls':controls,'passed':passed,'gates':gates,'peaks':peaks,'errors':errors});count+=controls
    return {'passes':sum(c['passed'] for c in details),'cases':details,'controls':count,'substeps':count*10}


def read_ready(root,protocol,phase):
    name=('gate','baseline','calibration')[phase]
    bindings=([root/'gate-result.json',root/'gate-started/endpoints.bin',root/'gate-started/result.json'] if phase==0 else
        [root/'baseline-result.json',root/'baseline-short/summary.csv',root/'baseline-hold/summary.csv',root/'baseline-long/windows.csv',root/'gate-ready.bin'] if phase==1 else
        [root/'calibration-result.json',root/'calibration-report.json',root/'fixed-scale.bin',root/'calibration-started/calibration-batch-0/manifest.json',root/'baseline-ready.bin'])
    r=envelope(root/(name+'-ready.bin'))
    assert [r.value('I'),r.value('I'),r.value('Q'),r.value('i')]==[0x31594747,1,protocol,phase]
    controls,substeps=r.value('i'),r.value('i');seconds=r.value('d');hashes=r.array('<u8',len(bindings));r.finish()
    assert controls==(1200,11920,1322640)[phase] and substeps==controls*10 and 0<seconds<900
    assert list(hashes)==[file_fnv(p) for p in bindings]
    return {'controls':controls,'substeps':substeps,'active_seconds':seconds}


def verify_freeze(root,required=True):
    paths=[root/name for name in ('frozen-inputs.json','study-frozen-inputs.json','frozen-causal-inputs.json') if (root/name).exists()]
    if not paths:assert not required,'Frozen causal-input manifest absent';return None
    details=[]
    for path in paths:
        entries=js(path)
        if isinstance(entries,dict):entries=entries['files']
        for entry in entries:assert digest(entry['path'])==entry['sha256'].lower(),entry['path']
        details.append({'manifest':str(path),'files':len(entries),'changed':0})
    return details


def verify_gate(root,protocol):
    report=js(root/'gate-started/result.json');native=js(root/'gate-result.json')
    assert report['passed'] and report['controls']==1200 and report['substeps']==12000 and report['fixtures']==10 and report['backends']==3
    assert not native['fault'] and native['PPO_updates']==native['actor_steps']==0 and native['controls']==1200 and native['substeps']==12000
    r=envelope(root/'gate-started/endpoints.bin')
    assert [r.value('I'),r.value('I'),r.value('Q'),r.value('I'),r.value('I')]==[0x31454747,1,protocol,696,152]
    initial=r.array(prior_audit.WORLD_TYPE,10);initial_task=r.array(prior_audit.TASK_TYPE,10)
    assert np.all(initial['controls']==0) and np.all(initial_task['controls']==0)
    count_type=np.dtype({'names':['controls','substeps','failure'],'formats':['<u8','<u8','<i4'],'offsets':[0,8,16],'itemsize':208})
    energy_type=np.dtype({'names':['values'],'formats':[('<f8',6)],'offsets':[640],'itemsize':696})
    maximum=0.
    for control in range(1,41):
        assert r.value('i')==control
        groups=[r.array(prior_audit.WORLD_TYPE,10) for _ in range(3)]
        tasks=[r.array(prior_audit.TASK_TYPE,10) for _ in range(3)];counts=r.array(count_type,10)
        legacy,identity,gpu=groups
        for group,t in zip(groups,tasks):
            assert np.all(group['controls']==control) and np.all(group['substeps']==control*10) and np.all(group['attempted']==control*10)
            assert np.all(t['controls']==control) and np.all(t['outcome']==0)
            assert np.all(group['invalid']==0) and np.all(group['unsafe']==0) and np.all(group['failure']==0)
            assert np.isfinite(group['body']).all() and np.all(abs(group['body'][:,7:11])<=1) and np.all(abs(group['actual'])<=1)
        assert np.all(counts['controls']==control) and np.all(counts['substeps']==control*10) and np.all(counts['failure']==0)
        for name in prior_audit.WORLD_TYPE.names:assert np.array_equal(legacy[name],identity[name]),name
        for name in ('controls','outcome'):assert np.array_equal(tasks[0][name],tasks[1][name])
        a=np.concatenate((legacy['body'][:,:21],legacy['held'],legacy['last_torque'],legacy['rates']),axis=1)
        b=np.concatenate((gpu['body'][:,:21],gpu['held'],gpu['last_torque'],gpu['rates']),axis=1)
        errors=abs(a.astype(float)-b);maximum=max(maximum,float(errors.max()))
        tolerances=np.full(33,2e-5);tolerances[11:14]=2e-7
        assert np.all(errors<=tolerances)
        near(gpu['actual'],legacy['actual'],2e-5,0)
        energy=[np.frombuffer(group,dtype=energy_type)['values'] for group in groups]
        near(energy[1],energy[0],0,0)
        assert np.all(abs(energy[2]-energy[0])<=1e-12+1e-4*np.maximum(abs(energy[2]),abs(energy[0])))
    r.finish();near(maximum,report['maximum_state_difference'],1e-12,1e-10)
    return {'controls':1200,'substeps':12000,'maximum_state_difference':maximum,'release':read_ready(root,protocol,0)}


def verify_gate_failure(root,protocol):
    native=js(root/'gate-result.json')
    assert native['closed'] and not native['qualified'] and native['fault']=='Gait compatibility state mismatch'
    assert native['PPO_updates']==native['actor_steps']==native['critic_steps']==native['critic_forward_calls']==0
    assert not (root/'gate-ready.bin').exists() and not (root/'baseline-started').exists() and not (root/'calibration-started').exists() and not (root/'training-started').exists()
    r=envelope(root/'gate-started/endpoints.bin')
    assert [r.value('I'),r.value('I'),r.value('Q'),r.value('I'),r.value('I')]==[0x31454747,1,protocol,696,152]
    initial=r.array(prior_audit.WORLD_TYPE,10);initial_task=r.array(prior_audit.TASK_TYPE,10)
    assert np.all(initial['controls']==0) and np.all(initial_task['controls']==0)
    count_type=np.dtype({'names':['controls','substeps','failure'],'formats':['<u8','<u8','<i4'],'offsets':[0,8,16],'itemsize':208})
    energy_type=np.dtype({'names':['values'],'formats':[('<f8',6)],'offsets':[640],'itemsize':696})
    names=['x','y','z','qx','qy','qz','qw']+[f'joint{j}' for j in range(4)]+[f'velocity{j}' for j in range(10)]+[f'held{j}' for j in range(4)]+[f'lasttorque{j}' for j in range(4)]+[f'refrate{j}' for j in range(4)]
    limits=np.full(33,2e-5);limits[11:14]=2e-7
    violations=[];ticks=0;maximum_position=0.;maximum_command=0.
    while r.at<len(r.data):
        control=r.value('i');assert control==ticks+1;ticks=control
        groups=[r.array(prior_audit.WORLD_TYPE,10) for _ in range(3)]
        tasks=[r.array(prior_audit.TASK_TYPE,10) for _ in range(3)];counts=r.array(count_type,10)
        legacy,identity,gpu=groups
        for group,t in zip(groups,tasks):
            assert np.all(group['controls']==control) and np.all(group['substeps']==control*10) and np.all(group['attempted']==control*10)
            assert np.all(t['controls']==control) and np.all(t['outcome']==0)
            assert np.all(group['failure']==0) and np.all(group['invalid']==0) and np.all(group['unsafe']==0)
            assert np.isfinite(group['body']).all() and np.all(abs(group['body'][:,7:11])<=1) and np.all(abs(group['actual'])<=1)
        assert np.all(counts['controls']==control) and np.all(counts['substeps']==control*10) and np.all(counts['failure']==0)
        for name in prior_audit.WORLD_TYPE.names:assert np.array_equal(legacy[name],identity[name]),name
        for name in ('controls','outcome'):assert np.array_equal(tasks[0][name],tasks[1][name])
        a=np.concatenate((legacy['body'][:,:21],legacy['held'],legacy['last_torque'],legacy['rates']),axis=1).astype(float)
        b=np.concatenate((gpu['body'][:,:21],gpu['held'],gpu['last_torque'],gpu['rates']),axis=1).astype(float)
        error=abs(a-b);maximum_position=max(maximum_position,float(error[:,:3].max()))
        maximum_command=max(maximum_command,float(abs(legacy['raw'].astype(float)-gpu['raw']).max()))
        for case,column in np.argwhere(error>limits):
            violations.append({'control':control,'time_s':control*.05,'case':int(case),'component':names[column],
                'CPU':float(a[case,column]),'CUDA':float(b[case,column]),'difference':float(error[case,column]),'tolerance':float(limits[column])})
        energy=[np.frombuffer(group,dtype=energy_type)['values'] for group in groups]
        near(energy[1],energy[0],0,0)
        assert np.all(abs(energy[2]-energy[0])<=1e-12+1e-4*np.maximum(abs(energy[2]),abs(energy[0])))
        near(gpu['actual'],legacy['actual'],2e-5,0)
    r.finish()
    assert ticks==32 and native['controls']==ticks*10*3==960 and native['substeps']==9600
    assert len(violations)==4 and all(v['control']==32 for v in violations)
    assert [(v['case'],v['component']) for v in violations]==[(0,'velocity8'),(0,'refrate2'),(2,'velocity8'),(2,'refrate2')]
    aggregate=js(root/'study-result.json')
    assert aggregate['study']=='gait_pond_v1' and aggregate['closed'] and not aggregate['qualified']
    assert aggregate['controls']==960 and aggregate['substeps']==9600 and aggregate['PPO_updates']==aggregate['actor_steps']==0
    assert not aggregate['training_started'] and not aggregate['behavioral_suites_run']
    for key in ('short_movement_passes','short_gait_passes','long60_passing_windows','recovery_passes','long300_passing_windows','original_hold_passes','withheld_hold_passes','hold300_passes'):
        assert aggregate[key] is None,key
    return {'closed_failure_verified':True,'controls':960,'substeps':9600,'PPO_updates':0,'Adam_steps':0,
            'controls_per_fixture_backend':32,'CPU_identity_exact':True,'physical_step_faults':0,
            'maximum_position_error_model':maximum_position,'maximum_command_disagreement':maximum_command,
            'first_registered_violation':violations[0],'all_registered_violations':violations,
            'scope':'Independent closed NN/physics loops failed strict trajectory compatibility; no terminal movement/gait suite ran'}


def verify_baseline(root,protocol,inherited):
    frozen,holding,gain,parent,hold=inherited;zero=np.zeros(8)
    short=assess_swim(root/'baseline-short',frozen,zero,gain)
    holds=assess_holds(root/'baseline-hold',frozen,zero,gain)
    long=assess_swim(root/'baseline-long',frozen,zero,gain,True)
    expected={(pose,end) for pose in (0,3) for end in range(8,61,4)}
    assert short['movement_passes']==4 and holds['passes']==6 and expected<=set(map(tuple,long['passing_window_keys']))
    assert short['controls']+holds['controls']+long['controls']==10720
    native=js(root/'baseline-result.json');assert not native['fault'] and native['PPO_updates']==native['actor_steps']==0
    assert native['short_movement_passes']==4 and native['short_gait_passes']==short['gait_passes']
    return {'short':short,'holds':holds,'long':long,'release':read_ready(root,protocol,1)}


def verify_calibration(root,protocol,inherited,layout):
    frozen,holding,gain,parent,hold=inherited
    returns,batch=read_batch(root/'calibration-started/calibration-batch-0',protocol,frozen,np.zeros(8,dtype=np.float32),layout,0,0)
    r=envelope(root/'fixed-scale.bin')
    assert [r.value('I'),r.value('I'),r.value('Q')]==[0x31534747,1,protocol]
    scale=r.value('d');means=np.asarray([r.value('d') for _ in range(4)]);r.finish()
    near(means,np.asarray([returns[c::4].mean() for c in range(4)]),1e-12,1e-12)
    expected=float(np.sqrt(np.mean((returns-means[np.arange(8192)%4])**2)))
    assert math.isfinite(scale) and scale>0;near(scale,expected,1e-12,1e-12)
    native=js(root/'calibration-result.json');report=js(root/'calibration-report.json')
    assert not native['fault'] and native['PPO_updates']==native['actor_steps']==0
    assert native['controls']==1322640 and native['substeps']==13226400
    return {'batch':batch,'scale':scale,'case_means':means.tolist(),'release':read_ready(root,protocol,2),'native_report':report}


def write_calibration_release(root,protocol,report_path):
    path=root/'calibration-verified.bin';assert not path.exists(),'Independent calibration release already exists'
    payload=struct.pack('<IIQQQQ',0x31564747,1,protocol,file_fnv(report_path),file_fnv(root/'fixed-scale.bin'),file_fnv(root/'calibration-ready.bin'))
    raw=struct.pack('<QQQ',0x31504b434c524443,len(payload),reference.shared.fnv(payload))+payload
    path.write_bytes(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'physical_controls':0,'optimizer_steps':0}


def verify_preparation(root,frozen,gain):
    prepared=js(root/'prepared.json');assert prepared['prepared'] and prepared['physical_controls']==prepared['optimizer_steps']==0
    details={}
    for prefix in ('prepare','verified'):
        r=envelope(root/(prefix+'-zero')/'initialization.bin')
        assert [r.value('I'),r.value('I'),r.value('i'),r.value('I')]==[0x314e4747,1,8192,696]
        world=r.array(prior_audit.WORLD_TYPE,8192);assert r.value('I')==152;task=r.array(prior_audit.TASK_TYPE,8192)
        obs=r.array(size=8192*41).reshape(8192,41);output=r.array(size=8192*4).reshape(8192,4)
        latent=r.array(size=8192*8).reshape(8192,8);logp=r.array(size=8192);r.finish()
        assert all(np.all(world[k]==0) for k in ('controls','substeps','attempted','failure','invalid','unsafe'))
        assert np.all(task['controls']==0) and np.all(task['outcome']==0)
        pre=np.concatenate((world['body'][:,:21],world['held'],world['last_torque'],world['rates']),axis=-1)
        expected=reference.observations(pre,np.arange(8192)%4,np.zeros((8192,3)),np.zeros(8192))[:,:41]
        observation_error=near(obs,expected,3e-5,3e-5)
        nn_error=near(output,reference.shared.forward(frozen.astype(float),obs.astype(float)),2e-5,2e-6)
        rng_error=near(latent,np.float32(.35)*prior_audit.normal(0x67a13c85,0,np.arange(8192*8,dtype=np.uint32).reshape(8192,8)),2e-6,2e-6)
        probability_error=near(logp,log_probability(np.zeros(8),latent),1e-4,0)
        bulk=js(root/(prefix+'-zero')/'bulk-policy-zero.json')
        assert bulk['passed'] and bulk['checked_rows']==8192 and bulk['checked_last_rows']==4 and bulk['minibatch_episodes']==2048
        assert bulk['physical_controls']==bulk['optimizer_steps']==0
        m=envelope(root/(prefix+'-math')/'bulk-gradient.bin')
        assert [m.value('I'),m.value('I')]==[0x314d4747,1]
        means=m.array(size=8);u=m.array(size=2048*8).reshape(2048,8);oldp=m.array(size=2048);adv=m.array(size=2048)
        gradients=m.array(size=2048*8).reshape(2048,8);stats=m.array(size=2048*5).reshape(2048,5);m.finish()
        change=log_probability(means,u)-oldp;ratio=np.exp(change);clip=np.clip(ratio,1-float(np.float32(.2)),1+float(np.float32(.2)))
        active=adv*ratio<=adv*clip
        expected_gradient=np.where(active[:,None],-adv[:,None]*ratio[:,None]*(u-means)/SIGMA**2/2048,0)
        gradient_error=near(gradients,expected_gradient,2e-7,0)
        entropy=8*(math.log(SIGMA)+.5*(math.log(2*math.pi)+1))
        expected_stats=np.stack((-np.minimum(adv*ratio,adv*clip),np.zeros(2048),np.full(2048,entropy),ratio-1-change,(abs(ratio-1)>float(np.float32(.2))).astype(float)),axis=-1)
        stats_error=near(stats,expected_stats,2e-4,0)
        d=envelope(root/(prefix+'-math')/'decode-zero.bin')
        assert [d.value('I'),d.value('I')]==[0x31444747,1] and d.value('f')==gain
        values=d.array(size=96*20).reshape(96,20);d.finish()
        decode_error=near(values[:,12:16],swim_latents(values[:,:8],values[:,8:12]),2e-6,0)
        near(values[:,16:20],np.float32(gain)*values[:,8:12],0,0)
        native=js(root/(prefix+'-math')/'math-zero.json')
        assert native['passed'] and native['physical_controls']==native['optimizer_steps']==0
        details[prefix]={'worlds':8192,'observations_error':observation_error,'NN_error':nn_error,'rng_error':rng_error,
                         'probability_error':probability_error,'gradient_error':gradient_error,'stats_error':stats_error,'decode_error':decode_error,'bulk':bulk}
    read(root/'preflight-passed.txt');return details


def audit_run(args):
    root=args.run.resolve();phase='closed_compatibility_gate_failure' if args.gate_failure else 'preparation' if args.prepared_only else 'gate' if args.gate_only else 'calibration' if args.calibration_only else 'training'
    output=root/('independent-'+phase+'-verification.json' if phase not in ('training','closed_compatibility_gate_failure') else 'independent-verification.json')
    report={'passed':False,'qualified':False,'phase':phase,'scope':'Independent saved-evidence audit; zero new physical or optimizer steps'}
    protocol=None
    try:
        for path in (Path(__file__),Path(reference.__file__),Path(reference.shared.__file__),Path(prior_audit.__file__)):read(path)
        report['envelopes']=validate_envelopes(root,args.helper)
        protocol,parent,hold=read_protocol(root);inherited=reference.parents(root,protocol);assert inherited[3:]==(parent,hold)
        means,policy=read_policy(root/'initial-policy/gait-policy-update-0.bin',protocol,inherited);report['initial_policy']=policy
        layout=js(root/'layout.json');gait_type(layout)
        report['freeze']=verify_freeze(root,phase!='preparation')
        report['preparation']=verify_preparation(root,inherited[0],inherited[2])
        report['counted_controls']=0
        if phase=='closed_compatibility_gate_failure':
            report['closed_gate']=verify_gate_failure(root,protocol);report['counted_controls']=960
        elif phase!='preparation':
            report['gate']=verify_gate(root,protocol);report['counted_controls']=1200
        if phase in ('calibration','training'):
            report['baseline']=verify_baseline(root,protocol,inherited)
            report['calibration']=verify_calibration(root,protocol,inherited,layout);report['counted_controls']=1322640
        if phase=='training':raise NotImplementedError('Final closed gait reader is being reconciled; calibration release is implemented')
        report['counted_substeps']=report['counted_controls']*10
        for name,value in list(HASHES.items()):assert digest(name)==value,name
        report['passed']=True
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['input_sha256']=HASHES;output.write_text(json.dumps(report,indent=2)+'\n')
    if phase=='calibration':
        release=write_calibration_release(root,protocol,output)
        print(json.dumps({'passed':True,'phase':phase,'controls':report['counted_controls'],'release':release}))
    else:print(json.dumps({'passed':True,'phase':phase,'controls':report['counted_controls'],'inputs':len(HASHES)}))
    directory=Path('build/gait-pond-preparation-2026-11-22/offline-audits/verified-scripts');directory.mkdir(parents=True,exist_ok=True)
    source=Path(__file__);(directory/(hashlib.sha256(source.read_bytes()).hexdigest()+'.py')).write_bytes(source.read_bytes())


def audit_csv(path,category='short',start_control=80,end_control=160):
    table=reference.rows(path);cases=sorted({int(row['case']) for row in table if row.get('category',category)==category})
    assert cases,'No requested saved-state cases'
    details=[]
    for case in cases:
        sequence=[row for row in table if row.get('category',category)==category and row['case']==case]
        assert [row['control'] for row in sequence]==list(range(len(sequence)))
        body=reference.body_array(sequence)
        metrics=gait_window(body,start_control,end_control);metrics['case']=case;details.append(metrics)
    return {'scope':'Saved-state gait measurements; zero new physical or optimizer steps',
            'input_sha256':{str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest(),
                            str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'category':category,'cases':details,'gait_passes':sum(case['passed'] for case in details),
            'physical_controls':0,'optimizer_steps':0}


def self_test():
    t=np.arange(80)*2*np.pi/20
    moving=np.zeros((80,4));moving[:,0]=.25*np.sin(t);moving[:,2]=-.25*np.sin(t)
    assert gait_metrics(moving)['passed']
    curled=moving.copy();curled[:,3]=np.radians(-27)
    assert not gait_metrics(curled)['gates']['pitch3']
    assert not gait_metrics(curled)['gates']['net_vertical_bend']
    assert not gait_metrics(np.zeros((80,4)))['gates']['oscillation']
    backbone=np.arange(12,dtype=float).reshape(3,4)
    assert np.array_equal(swim_latents(np.zeros((3,8)),backbone),backbone)
    reference.near(swim_latents(np.full((3,8),1000.),backbone),2*backbone+.75,1e-12,1e-12)
    returns=np.arange(WORLD_COUNT,dtype=float)/1000;advantages=leave_one_out(returns,3)
    assert all(abs(advantages[c::4].mean())<1e-12 for c in range(4))
    index=17;before=returns[index]-3*advantages[index];returns[index]+=5
    reference.near(returns[index]-3*leave_one_out(returns,3)[index],before,1e-12,1e-12)
    return {'passed':True,'scope':'Independent gait/mapping/LOO arithmetic only','physical_controls':0,'optimizer_steps':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states',type=Path)
    parser.add_argument('--category',default='short')
    parser.add_argument('--start-control',type=int,default=GAIT_START_CONTROL)
    parser.add_argument('--end-control',type=int,default=EPISODE_CONTROLS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--helper',type=Path,default=Path('build/learned-amplitude-preparation-2026-11-22/check-evidence.exe'))
    parser.add_argument('--prepared-only',action='store_true')
    parser.add_argument('--gate-only',action='store_true')
    parser.add_argument('--calibration-only',action='store_true')
    parser.add_argument('--gate-failure',action='store_true')
    args=parser.parse_args()
    if args.run:audit_run(args);return
    if args.self_test:report=self_test()
    else:
        assert args.states is not None,'Supply saved --states CSV or --self-test'
        report=audit_csv(args.states,args.category,args.start_control,args.end_control)
    if args.output:
        assert not args.output.exists(),'Preserve the existing evidence identity'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
