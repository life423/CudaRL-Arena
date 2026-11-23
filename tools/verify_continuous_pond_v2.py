"""Independent saved-data checks for proposed continuous-pond version two.

Preparation only until the fresh physical study is explicitly authorized.
This program never invokes a simulator, GPU kernel, learner or optimizer and
never creates an authorization, physical-gate, or calibration-release marker.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct
import inspect
import platform
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import verify_continuous_pond as v1

reference=v1.reference;previous=v1.previous
read,js,near,envelope=reference.read,reference.js,reference.near,reference.envelope
HASHES,VALIDATED=reference.HASHES,reference.VALIDATED
digest,payload_checksum,file_fnv=v1.digest,v1.payload_checksum,v1.file_fnv
json_text=v1.json_text
N,T,P,MB=2048,1200,80,512
SIGMA,LIMIT=float(np.float32(.15)),float(np.float32(.75))
CAP_CONTROLS,CAP_SUBSTEPS=62286640,622866400
V1_AUDITOR_SHA='c032485c0922899571c1277319ade71fc5fd1bca7e79b7742451dd7ef7be62c7'
ACTIVE_SOURCES=None


def derived_features(raw):
    raw=np.asarray(raw,float);assert raw.shape[-1]==20
    output=np.tanh(raw);output[...,:8]=v1.features(raw[...,:8]);return output


def action_latents(coefficients,base,raw):
    coefficients=np.asarray(coefficients,float);base=np.asarray(base,float)
    terms=np.tanh(coefficients).reshape(coefficients.shape[:-1]+(4,20))
    result=base+LIMIT*np.tanh(np.sum(terms*derived_features(raw)[...,None,:],axis=-1))
    return np.where(np.all(coefficients==0,axis=-1,keepdims=True),base,result)


def raw_sensors(body,depth):
    body=np.asarray(body,float);metrics=v1.measured(body,depth)
    result=np.zeros(body.shape[:-1]+(20,));result[...,:8]=v1.raw_sensors(body)
    result[...,8]=metrics['depth']/float(np.float32(.001))
    result[...,9]=metrics['velocity'][...,1]/float(np.float32(.001))
    result[...,10]=metrics['roll']/10;result[...,11]=metrics['pitch']/10
    result[...,12:16]=body[...,7:11];result[...,16:20]=body[...,17:21]/12
    return result


def log_probability(means,latent):
    latent=np.asarray(latent,float)
    return np.sum(-.5*((latent-means)/SIGMA)**2-math.log(SIGMA)-.5*math.log(2*math.pi),axis=-1)


def leave_one_out(returns,scale):
    returns=np.asarray(returns,float);assert returns.shape==(N,) and math.isfinite(scale) and scale>0
    group=returns.reshape(-1,16)
    return ((group-(group.sum(axis=0)-group)/127)/scale).reshape(-1)


def ppo_terms(means,latent,oldp,adv):
    delta=log_probability(means,latent)-oldp;ratio=np.exp(delta)
    clipped=np.clip(ratio,1-float(np.float32(.2)),1+float(np.float32(.2)))
    active=adv*ratio<=adv*clipped
    gradient=np.where(active[...,None],-adv[...,None]*ratio[...,None]*(latent-means)/SIGMA**2/MB,0)
    entropy=P*(math.log(SIGMA)+.5*(math.log(2*math.pi)+1))
    stats=np.stack((-np.minimum(adv*ratio,adv*clipped),np.zeros_like(adv),np.full_like(adv,entropy),
                    ratio-1-delta,(abs(ratio-1)>float(np.float32(.2))).astype(float)),axis=-1)
    return gradient,stats


def task_type(layout):
    fields=[('controls','<u8'),('last_world_control','<u8'),('maximum_controls','<i4'),('reason','<i4'),
        ('fixed_depth_model','<f4'),('maximum_depth_m','<f4'),('maximum_tilt_deg','<f4'),('minimum_clearance_m','<f4'),
        ('last_com_model',('<f4',3)),('elapsed_seconds','<f8'),('reward_sum','<f8'),('path_length_m','<f8'),
        ('motion_window_arc',('<f4',40)),('motion_window_dt',('<f4',40)),('motion_window_next','<i4'),
        ('motion_window_count','<i4'),('rolling_horizontal_speed_m_s','<f4'),('done','u1'),
        ('boundary_terminal','u1'),('behavior_terminal','u1')]
    return v1.dtype_from_layout(layout,'task',fields)


def read_protocol(root):
    r=envelope(root/'protocol.bin');protocol=payload_checksum(root/'protocol.bin')
    assert [r.value('I'),r.value('I'),r.value('Q')]==[0x32525054,2,file_fnv(root/'approved-protocol.md')]
    parent,hold=r.value('Q'),r.value('Q')
    assert [r.value('i') for _ in range(26)]==[6,2,3,4,2,41,20,4,336,80,N,T,24,384,4,MB,2480,10720,46240,16,1200,8,36000,CAP_CONTROLS,CAP_SUBSTEPS,900]
    near([r.value('f') for _ in range(26)],np.float32([.15,.75,.003,.5,.2,.01,.20,.12,.195,.115,.008,.001,10,
        .006,.003,.12,.06,.20,.12,4,60,6250,.005,20,.002,2]),0,0)
    assert [r.value('i'),r.value('i')]==[40,80]
    assert [r.value('I') for _ in range(3)]==[0x8e25a643,0x137be951,0x04a71369]
    assert [r.value('Q'),r.value('Q')]==[21*1024**3//2,8*1024**3]
    for withheld,count in ((False,16),(True,8)):
        for case in range(count):
            center=np.array([r.value('f') for _ in range(3)]);yaw=r.value('f')
            pose,identifier,flag=r.value('i'),r.value('i'),r.value('I')
            expected,angle,joint_pose=v1.case_spec(case,withheld)
            near(center,expected,2e-8,2e-6);near(yaw,angle,1e-6,1e-6)
            assert pose==joint_pose and identifier==case and flag==int(withheld)
    r.finish();return protocol,parent,hold


def read_policy(path,protocol,inherited):
    frozen,holding,gain,parent,hold=inherited;r=envelope(path)
    assert [r.value('I'),r.value('I'),r.value('Q'),r.value('Q'),r.value('Q')]==[0x32505054,2,protocol,parent,hold]
    assert [r.value('I') for _ in range(10)]==[6,2,3,4,2,41,20,4,336,80]
    n,update,age=[r.value('i') for _ in range(3)]
    assert n==N and 0<=update<=24 and 0<=age<=384 and age%4==0
    assert np.array_equal(r.array(size=2532),frozen) and np.array_equal(r.array(size=2),holding) and r.value('f')==gain
    means=r.array(size=P).copy();assert [r.value('f'),r.value('f')]==[SIGMA,LIMIT]
    a=reference.shared.Reader(r.array('u1'));r.finish()
    near([a.value('f') for _ in range(5)],np.float32([.003,.9,.999,.00001,.5]),0,0)
    assert a.value('i')==age;first=a.array(size=P);second=a.array(size=P);a.finish()
    assert np.isfinite(means).all() and np.isfinite(first).all() and np.all(np.isfinite(second)&(second>=0))
    if update==0:assert age==0 and np.all(means==0) and np.all(first==0) and np.all(second==0)
    return means,{'update':update,'Adam_steps':age,'sha256':HASHES[str(path.resolve())],'means':means.tolist()}


def initial_arrays(path,layout,n,magic,withheld,frozen,sampled=True):
    r=envelope(path);assert [r.value('I'),r.value('I'),r.value('i'),r.value('I')]==[magic,2,n,696]
    worlds=r.array(previous.WORLD_TYPE,n);ttype=task_type(layout);assert r.value('I')==ttype.itemsize
    tasks=r.array(ttype,n);obs=r.array(size=n*41).reshape(n,41);raw=r.array(size=n*20).reshape(n,20)
    base=r.array(size=n*4).reshape(n,4)
    latent=logp=None
    if sampled:
        latent=r.array(size=n*P).reshape(n,P);logp=r.array(size=n)
    r.finish()
    for key in ('controls','substeps','attempted','failure','invalid','unsafe'):assert np.all(worlds[key]==0)
    assert np.all(worlds['current']==0)
    for key in ('controls','last_world_control','reason','elapsed_seconds','reward_sum','path_length_m','done','boundary_terminal',
                'behavior_terminal','maximum_depth_m','maximum_tilt_deg','motion_window_arc','motion_window_dt',
                'motion_window_next','motion_window_count','rolling_horizontal_speed_m_s'):
        assert np.all(tasks[key]==0),key
    assert np.all(tasks['maximum_controls']==(T if sampled else 0))
    cases=8 if withheld else 16
    for case in range(cases):
        actual=worlds['body'][case::cases]
        near(actual,np.broadcast_to(v1.expected_initial_body(case,withheld),actual.shape),3e-6,3e-6)
    context=np.concatenate((worlds['held'],worlds['rates'],worlds['last_torque']),axis=1)
    input_error=near(obs,v1.observations(worlds['body'],context,tasks['fixed_depth_model'],0),3e-5,3e-5)
    sensor_error=near(raw,raw_sensors(worlds['body'],tasks['fixed_depth_model']),3e-5,3e-5)
    nn_error=near(base,reference.shared.forward(frozen.astype(float),obs.astype(float)),2e-5,2e-6)
    errors={'observation':input_error,'raw_sensors':sensor_error,'backbone':nn_error}
    if sampled:
        errors['RNG']=near(latent,np.float32(SIGMA)*previous.normal(0x8e25a643,0,np.arange(n*P,dtype=np.uint32).reshape(n,P)),2e-6,2e-6)
        errors['log_probability']=near(logp,log_probability(np.zeros(P),latent),1e-4,0)
    return {'worlds':worlds,'tasks':tasks,'obs':obs,'raw':raw,'base':base,'latent':latent,'logp':logp,'errors':errors}


def verify_math(directory,gain):
    d=envelope(directory/'decode-zero.bin')
    assert [d.value('I'),d.value('I')]==[0x32445054,2] and d.value('f')==gain and d.value('i')==116
    values=d.array(size=96*116).reshape(96,116);d.finish()
    decode_error=near(values[:,104:108],action_latents(values[:,:80],values[:,80:84],values[:,84:104]),2e-6,0)
    near(values[:,108:112],np.float32(gain)*values[:,80:84],0,0);near(values[:,112:116],values[:,80:84],0,0)
    assert np.array_equal(values[0,104:108],values[0,80:84])
    g=envelope(directory/'gradient-zero.bin');assert [g.value('I'),g.value('I')]==[0x324d5054,2]
    means=g.array(size=P);latent=g.array(size=MB*P).reshape(MB,P);oldp=g.array(size=MB);adv=g.array(size=MB)
    gradients=g.array(size=MB*P).reshape(MB,P);stats=g.array(size=MB*5).reshape(MB,5);g.finish()
    expected_gradient,expected_stats=ppo_terms(means,latent,oldp,adv)
    gradient_error=near(gradients,expected_gradient,2e-7,0);stats_error=near(stats,expected_stats,2e-4,0)
    maximum_fd=0.
    for i in range(24):
        for j in range(P):
            plus=means.astype(float).copy();minus=plus.copy();plus[j]+=1e-6;minus[j]-=1e-6
            lp=ppo_terms(plus,latent[i:i+1],oldp[i:i+1],adv[i:i+1])[1][0,0]/MB
            lm=ppo_terms(minus,latent[i:i+1],oldp[i:i+1],adv[i:i+1])[1][0,0]/MB
            maximum_fd=max(maximum_fd,abs((lp-lm)/2e-6-expected_gradient[i,j]))
    assert maximum_fd<2e-7
    native=js(directory/'math-zero.json');diagnostic=js(directory/'diagnostic.json')
    assert native['passed'] and native['CUDA_decode_cases']==96 and native['PPO_rows']==MB
    assert native['physical_controls']==native['optimizer_steps']==0 and diagnostic['max_mean_loss_FD_error']<2e-7
    return {'decode_error':decode_error,'gradient_error':gradient_error,'stats_error':stats_error,'independent_mean_loss_FD_error':maximum_fd}


def permutation(n,seed):
    bits=2
    while (1<<bits)<n:bits+=2
    half=bits//2;mask=np.uint32((1<<half)-1)
    with np.errstate(over='ignore'):
        keys=[previous.hash_u32(np.uint32(seed)^previous.hash_u32(np.uint32(r+0x51ed270b))) for r in range(4)]
    def walk(value):
        left,right=value>>half,value&mask
        for key in keys:left,right=right,left^(previous.hash_u32(right^key)&mask)
        return (left<<half)|right
    result=walk(np.arange(n,dtype=np.uint32));count=0
    while np.any(result>=n):
        bad=result>=n;result[bad]=walk(result[bad]);count+=1;assert count<1000
    assert len(np.unique(result))==n
    return result.astype(int)


def verify_bulk(directory,protocol,initial):
    order=permutation(N,0x7138b2a9);synthetic=np.float32(.25)+np.float32(np.arange(N)%19-9)*np.float32(.07)
    seen=np.zeros(N,dtype=int);details=[]
    for batch in range(4):
        r=envelope(directory/('bulk-minibatch-'+str(batch)+'.bin'))
        assert [r.value('I'),r.value('I'),r.value('Q')]==[0x324b5054,2,protocol]
        assert [r.value('i') for _ in range(4)]==[batch,N,MB,batch*MB]
        indices=r.array('<i4',MB);means=r.array(size=P)
        latent=r.array(size=MB*P).reshape(MB,P);oldp=r.array(size=MB);adv=r.array(size=MB)
        gradients=r.array(size=MB*P).reshape(MB,P);stats=r.array(size=MB*5).reshape(MB,5)
        reduced=r.array(size=P);reduced_stats=r.array(size=5);r.finish()
        assert np.array_equal(indices,order[batch*MB:(batch+1)*MB]);seen[indices]+=1
        assert np.all(means==0) and np.array_equal(latent,initial['latent'][indices])
        assert np.array_equal(oldp,initial['logp'][indices]) and np.array_equal(adv,synthetic[indices])
        expected_gradient,expected_stats=ppo_terms(means,latent,oldp,adv)
        row_error=near(gradients,expected_gradient,1e-5,0);stats_error=near(stats,expected_stats,2e-4,0)
        reduced_error=near(reduced,expected_gradient.sum(axis=0),2e-5,0)
        reduced_stats_error=near(reduced_stats/MB,expected_stats.mean(axis=0),2e-5,0)
        assert np.sum(abs(expected_gradient[-1]))>1e-8 and np.sum(abs(gradients[-1]))>1e-8
        details.append({'minibatch':batch,'rows':MB,'last_row_checked':True,'row_gradient_error':row_error,
            'row_stats_error':stats_error,'reduced_gradient_error':reduced_error,'mean_stats_error':reduced_stats_error})
    assert np.all(seen==1)
    return {'episodes_checked':N,'last_rows_checked':4,'indices_cover_each_episode_once':True,'minibatches':details}


def verify_gate_preparation(root,protocol,layout,inherited):
    r=envelope(root/'gate-preparation/initial-probes.bin')
    assert r.value('I')==2;size=r.value('I');ttype=task_type(layout)
    observation_offset=696+ttype.itemsize;raw_offset=observation_offset+41*4;valid_offset=raw_offset+20*4
    assert size==(valid_offset+4+7)//8*8
    probe_type=np.dtype({'names':['world','task','observation','raw','valid'],
        'formats':[previous.WORLD_TYPE,ttype,('<f4',41),('<f4',20),'<i4'],
        'offsets':[0,696,observation_offset,raw_offset,valid_offset],'itemsize':size})
    cpu=r.array(probe_type,24);gpu=r.array(probe_type,24);r.finish();details={}
    for name,probes in (('CPU',cpu),('CUDA',gpu)):
        assert np.all(probes['valid']==1)
        for key in ('controls','substeps','attempted','failure','invalid','unsafe'):assert np.all(probes['world'][key]==0)
        assert np.all(probes['task']['controls']==0) and np.all(probes['task']['maximum_controls']==T)
        for i in range(24):near(probes['world']['body'][i],v1.expected_initial_body(i if i<16 else i-16,i>=16),3e-6,3e-6)
        context=np.concatenate((probes['world']['held'],probes['world']['rates'],probes['world']['last_torque']),axis=1)
        obs=near(probes['observation'],v1.observations(probes['world']['body'],context,probes['task']['fixed_depth_model'],0),3e-5,3e-5)
        raw=near(probes['raw'],raw_sensors(probes['world']['body'],probes['task']['fixed_depth_model']),3e-5,3e-5)
        details[name]={'cases':24,'observation_error':obs,'raw_sensor_error':raw}
    near(cpu['observation'],gpu['observation'],2e-5,0);near(cpu['raw'],gpu['raw'],2e-5,0)
    marker=envelope(root/'gate-prepared.bin')
    assert [marker.value('I'),marker.value('I'),marker.value('Q'),marker.value('Q'),marker.value('Q')]==[0x325a4754,2,protocol,inherited[3],inherited[4]]
    marker.finish();native=js(root/'gate-preparation/result.json')
    assert native['passed'] and native['physical_controls']==native['physical_substeps']==native['optimizer_steps']==0
    assert native['synthetic_measurement_checks']>0 and native['synthetic_measurements_are_physics'] is False
    return details


def bind_native_inputs(directory):
    files=[directory/'Release/ppo_turning_swim_v2.exe',directory/'Release/turning_physics_gate_v2.exe',
        directory/'CMakeCache.txt',directory/'ppo_turning_swim_v2.vcxproj',directory/'turning_physics_gate_v2.vcxproj',
        Path('tools/turning_swim_v2_study.cu'),Path('tools/turning_retention_v2.h'),Path('tools/turning_physics_gate_v2.cu'),
        Path('tools/turning_study_authorization.h'),Path('tools/rest_hold_study.cu'),Path('tools/hold_learned_amplitude_math.h'),
        Path('docs/CONTINUOUS_POND_V2_PROTOCOL_2026-11-23.md')]
    contract=read(Path('src/nn/motor/controller_contract.txt')).decode().splitlines()
    assert len(contract)==12
    files += [Path(p.strip()) for p in contract if p.strip()]
    files += [Path('src/nn/motor/turning_policy_v2.h'),Path('src/nn/motor/turning_task_v2.h')]
    entries=[]
    for path in sorted(set(files)):
        assert path.is_file(),str(path)
        entries.append({'path':str(path.resolve()),'sha256':digest(path)})
    return entries


def validate_envelopes(root,helper,phase='preparation'):
    paths=sorted(p for p in root.rglob('*.bin') if 'frozen-causal-inputs' not in p.parts)
    prep=Path('build/continuous-pond-v2-preparation-2026-11-23/offline-audits');prep.mkdir(parents=True,exist_ok=True)
    listing=prep/('envelope-paths-'+phase+'.txt');listing.write_text(''.join(str(p.resolve())+'\n' for p in paths),encoding='utf-8')
    read(helper);process=subprocess.run([str(helper.resolve()),str(listing.resolve())],capture_output=True,text=True,check=True,
        creationflags=0x08000000 if os.name=='nt' else 0)
    result=json.loads(process.stdout)
    assert result['passed'] and result['files']==len(paths) and result['physical_controls']==result['optimizer_steps']==0
    VALIDATED.update(p.resolve() for p in paths);return result


def preserve_sources(phase='preparation'):
    base=Path('build/continuous-pond-v2-preparation-2026-11-23/offline-audits');path=base/('code-'+phase);index=1
    while path.exists():path=base/('code-'+phase+'-'+str(index));index+=1
    path.mkdir(parents=True)
    sources=[]
    files=[Path(__file__),Path(v1.__file__),Path(reference.__file__),Path(previous.__file__),Path(v1.retained.__file__),Path(reference.shared.__file__)]
    helper=Path('tools/continuous_pond_v2_batch_audit.py')
    if phase!='preparation':
        assert helper.is_file();files.append(helper)
        files.append(Path(__file__).with_name('authorize_continuous_pond_v2.py'))
    for source in files:
        raw=read(source);snapshot=path/source.name;snapshot.write_bytes(raw)
        sources.append({'source':str(source.resolve()),'snapshot':str(snapshot.resolve()),'sha256':HASHES[str(source.resolve())]})
    (path/'manifest.json').write_text(json_text(sources,indent=2)+'\n',encoding='utf-8')
    return sources


def read_authorization(root,protocol):
    r=envelope(root/'authorization.bin')
    assert [r.value('I'),r.value('I'),r.value('Q')]==[0x32554150,1,protocol]
    message=r.array('u1');r.finish();assert 0<len(message)<=65536
    text=message.tobytes().decode('utf-8');assert text.strip()
    receipt=js(root/'authorization.json')
    assert receipt['schema']==1 and receipt['study']=='continuous_pond_v2' and receipt['explicit_user_grant_recorded'] is True
    assert receipt['protocol_checksum']==f'{protocol:016X}' and receipt['protocol_sha256']==digest(root/'protocol.bin')
    assert receipt['authorization_sha256']==digest(root/'authorization.bin')
    assert receipt['prepared_verification_sha256']==digest(root/'independent-preparation-verification.json')
    assert receipt['frozen_inputs_manifest_sha256']==digest(root/'frozen-inputs.json')
    assert receipt['physical_controls_executed']==receipt['optimizer_steps_executed']==0
    assert receipt['writer_sha256']==digest(Path(__file__).with_name('authorize_continuous_pond_v2.py'))
    assert digest(receipt['approval_message_file'])==receipt['approval_message_sha256']
    assert read(receipt['approval_message_file'])==message.tobytes()
    return {'present':True,'protocol_bound':True,'sha256':HASHES[str((root/'authorization.bin').resolve())]}


def reuse_preparation(root):
    path=root/'independent-preparation-verification.json';proof=js(path)
    assert proof['schema']==2 and proof['phase']=='preparation' and proof['passed'] and not proof['qualified']
    assert proof['physical_run_approved'] is False
    mutable=str(Path(__file__).resolve())
    for source in proof['auditor_sources']:assert digest(source['snapshot'])==source['sha256']
    for source,value in proof['input_sha256'].items():
        if str(Path(source).resolve())!=mutable:assert digest(source)==value,source
    return {'passed':True,'sha256':HASHES[str(path.resolve())],'executed_code_preserved':True}


def verify_gate(root,protocol,inherited):
    native=js(root/'gate-result.json')
    assert native['closed'] and native['matched_commands'] and not native['closed_loop_trajectory_parity_claimed']
    r=envelope(root/'gate-started/states.bin');version,size,worldbytes,infobytes=[r.value('I') for _ in range(4)]
    assert version==2 and worldbytes==696 and size==native['step_bytes'] and infobytes==native['step_info_bytes']
    dtype=np.dtype({'names':['world','energy'],'formats':[previous.WORLD_TYPE,('<f8',6)],'offsets':[0,640],'itemsize':size})
    csv=reference.rows(root/'gate-started/states.csv');records={};controls=substeps=0;violations=[];maxima={}
    bodykeys=['x','y','z','qx','qy','qz','qw']+[f'j{j}' for j in range(4)]+[f'v{j}' for j in range(10)]
    energykeys=['signed_work','positive_work','fluid_loss','numerical_loss','external_work','energy_error']
    for row in csv:
        key=tuple(int(row[k]) for k in ('group','case','backend','control'))
        assert tuple(r.value('i') for _ in range(4))==key and key not in records
        data=np.frombuffer(r.data,dtype=dtype,count=1,offset=r.at)[0].copy();r.at+=size;records[key]=data
        world=data['world'];near([row[k] for k in bodykeys],world['body'][:21],0,0)
        near([row[k] for k in energykeys],data['energy'],0,0)
        assert row['attempted_substeps']==world['attempted'] and row['accepted_substeps']==world['substeps']
        if key[3]:
            controls+=1;before=records[key[:3]+(key[3]-1,)]['world'];substeps+=int(world['attempted']-before['attempted'])
            if row['advanced']!=1 or row['last_substeps']!=10 or row['boundary_terminal']!=0:
                violations.append({'state':key,'kind':'step_status'})
        if world['failure'] or world['invalid'] or world['unsafe']:violations.append({'state':key,'kind':'world_failure'})
        assert np.isfinite(world['body']).all() and np.isfinite(world['actual']).all()
        if np.any(abs(world['actual'])>1):violations.append({'state':key,'kind':'torque_cap'})
    r.finish();assert controls==native['controls'] and substeps==native['substeps']
    for (group,case,backend,control),gpu in records.items():
        if backend==1 and group<2:
            old=records[(group,case,0,control)]
            for field in previous.WORLD_TYPE.names:assert np.array_equal(gpu['world'][field],old['world'][field]),field
            assert np.array_equal(gpu['energy'],old['energy'])
        if backend!=2 or (group,case,1,control) not in records:continue
        cpu=records[(group,case,1,control)]
        for name,sl,tolerance in [('position',slice(0,3),2e-5),('orientation_joints',slice(3,11),2e-5),
                                  ('linear_velocity',slice(11,14),2e-7),('angular_joint_velocity',slice(14,21),2e-5)]:
            error=float(abs(gpu['world']['body'][sl]-cpu['world']['body'][sl]).max());maxima[name]=max(maxima.get(name,0),error)
            if error>tolerance:violations.append({'state':(group,case,control),'kind':name,'error':error,'limit':tolerance})
        for field in ('actual','last_torque','held','rates'):
            error=float(abs(gpu['world'][field]-cpu['world'][field]).max());maxima[field]=max(maxima.get(field,0),error)
            if error>2e-5:violations.append({'state':(group,case,control),'kind':field,'error':error,'limit':2e-5})
        difference=abs(gpu['energy']-cpu['energy']);bound=1e-12+1e-4*np.maximum(abs(gpu['energy']),abs(cpu['energy']))
        if np.any(difference>bound):violations.append({'state':(group,case,control),'kind':'energy'})
    commands=reference.rows(root/'gate-started/same-observation.csv')
    for row in commands:
        obs=np.array([row[f'observation{j}'] for j in range(41)]);base=reference.shared.forward(inherited[0].astype(float),obs[None])[0]
        for backend in ('cpu','gpu'):
            observed=np.array([row[f'{backend}_base{j}'] for j in range(4)]);near(observed,base,2e-5,2e-6)
            latent=observed*inherited[2] if row['group']==1 else observed
            near([row[f'{backend}_latent{j}'] for j in range(4)],latent,2e-7,2e-6)
            near([row[f'{backend}_command{j}'] for j in range(4)],np.tanh(latent),2e-7,2e-6)
        for field in ('base','latent','command'):
            near([row[f'cpu_{field}{j}'] for j in range(4)],[row[f'gpu_{field}{j}'] for j in range(4)],2e-5,0)
    if native['passed']:
        assert not violations and controls==2480 and substeps==24800 and len(csv)==2542 and len(commands)==1040 and native['completed_cases']==26
        marker=envelope(root/'gate-passed.bin')
        assert [marker.value('I'),marker.value('I'),marker.value('I'),marker.value('Q'),marker.value('Q'),marker.value('Q')]==[0x32504754,2,1,2480,24800,protocol];marker.finish()
    else:
        assert native['fault'] and not (root/'gate-passed.bin').exists() and controls<=2480 and substeps<=24800
        read(root/'gate-started/first-failure.bin')
    return {'passed':bool(native['passed']),'controls':controls,'substeps':substeps,'states':len(csv),
        'same_observation_rows':len(commands),'maximum_errors':maxima,'violations':violations,
        'closed_loop_trajectory_parity_claimed':False}


def verify_old_retention(directory,inherited,means,full):
    assert np.asarray(means).shape==(P,) and np.isfinite(means).all()
    # The fixed requested branches bypass all eighty values. The immutable V1
    # scorer's sixteen-value argument is therefore an unused compatibility slot.
    result=v1.verify_retention(directory,inherited[0],np.zeros(16),inherited[2],full)
    result['eighty_parameter_head_bypassed_by_requested_modes']=True
    return result


def read_ready(root,protocol,phase,controls,substeps):
    name='baseline' if phase==1 else 'calibration'
    files=([root/'baseline-result.json',root/'baseline-retention/retention.json',root/'baseline-free/summary.csv',
        root/'baseline-free/manifest.json',root/'gate-passed.bin',root/'gate-result.json'] if phase==1 else
        [root/'calibration-result.json',root/'calibration-report.json',root/'fixed-scale.bin',
         root/'calibration-started/calibration-batch-0/manifest.json',root/'baseline-ready.bin'])
    r=envelope(root/(name+'-ready.bin'))
    assert [r.value('I'),r.value('I'),r.value('Q'),r.value('i')]==[0x32595054,2,protocol,phase]
    assert [r.value('i'),r.value('i')]==[controls,substeps]
    seconds=r.value('d');assert 0<seconds<900
    assert list(r.array('<u8',len(files)))==[file_fnv(p) for p in files];r.finish()
    return {'controls':controls,'substeps':substeps,'active_seconds':seconds}


def verify_baseline(root,protocol,inherited,layout,io):
    old=verify_old_retention(root/'baseline-retention',inherited,np.zeros(P),False);assert old['passed']
    batch,collection=io.audit_batch(root/'baseline-free',protocol,layout,inherited[0],np.zeros(P))
    assert batch['purpose']==2 and batch['horizon']==1200 and batch['n']==16
    scores=io.score_evaluation(root/'baseline-free',batch)
    controls=2480+old['controls']+collection['controls'];substeps=24800+old['substeps']+collection['substeps']
    native=js(root/'baseline-result.json')
    assert native['fault']=='' and native['controls']==controls and native['substeps']==substeps and native['selection_passes']==scores['passes']
    assert native['PPO_updates']==native['actor_steps']==0
    return {'retention':old,'collection':collection,'continuous':scores,'ready':read_ready(root,protocol,1,controls,substeps)}


def verify_calibration(root,protocol,inherited,layout,baseline,io):
    assert baseline['continuous']['passes']<16
    batch,collection=io.audit_batch(root/'calibration-started/calibration-batch-0',protocol,layout,inherited[0],np.zeros(P))
    assert batch['purpose']==0 and batch['index']==0 and batch['n']==N and batch['horizon']==T
    grouped=batch['returns'].astype(float).reshape(-1,16);means=grouped.mean(axis=0)
    independent=float(np.sqrt(np.mean((grouped-means)**2)));assert np.isfinite(independent) and independent>0
    r=envelope(root/'fixed-scale.bin');assert [r.value('I'),r.value('I'),r.value('Q')]==[0x32535054,2,protocol]
    scale=r.value('d');case_means=np.array([r.value('d') for _ in range(16)]);r.finish()
    near(scale,independent,1e-12,1e-12);near(case_means,means,1e-12,1e-12)
    report=js(root/'calibration-report.json');assert report['passed'] and report['episodes']==N and report['cases']==16 and report['episodes_per_case']==128
    near(report['fixed_global_residual_RMS'],scale,1e-12,1e-12)
    for field,measured in [('ordinary_boundary_terminals','task_boundary_episodes'),('deadline_terminals','deadline_episodes'),
        ('depth_terminals','task_depth_episodes'),('tilt_terminals','task_tilt_episodes'),('stall_terminals','task_stalled_episodes')]:
        assert report[field]==collection[measured]
    controls=baseline['ready']['controls']+collection['controls'];substeps=baseline['ready']['substeps']+collection['substeps']
    native=js(root/'calibration-result.json');assert native['fault']=='' and native['controls']==controls and native['substeps']==substeps
    assert native['PPO_updates']==native['actor_steps']==0 and not native['qualified']
    return {'collection':collection,'fixed_global_residual_RMS':scale,'independent_residual_RMS':independent,
        'case_means':means.tolist(),'ready':read_ready(root,protocol,2,controls,substeps),'training_started':False,'LOO_own_return_excluded':True}


def write_calibration_release(root,protocol,report_path):
    path=root/'calibration-verified.bin';assert not path.exists()
    payload=struct.pack('<IIQQQQ',0x32565054,2,protocol,*[file_fnv(p) for p in
        (report_path,root/'fixed-scale.bin',root/'calibration-ready.bin')])
    path.write_bytes(struct.pack('<QQQ',0x31504b434c524443,len(payload),reference.shared.fnv(payload))+payload)
    return digest(path)


def verify_calibration_release(root,protocol):
    files=[root/'independent-calibration-verification.json',root/'fixed-scale.bin',root/'calibration-ready.bin']
    r=envelope(root/'calibration-verified.bin');assert [r.value('I'),r.value('I'),r.value('Q')]==[0x32565054,2,protocol]
    assert [r.value('Q') for _ in range(3)]==[file_fnv(p) for p in files];r.finish()
    proof=js(files[0]);assert proof['passed'] and proof['phase']=='calibration' and not proof['qualified']
    for source in proof['auditor_sources']:assert digest(source['snapshot'])==source['sha256']
    return {'passed':True,'proof_sha256':HASHES[str(files[0].resolve())],
        'marker_sha256':HASHES[str((root/'calibration-verified.bin').resolve())]}


def batch_worker(arguments):
    import continuous_pond_v2_batch_audit as io
    directory,protocol,layout,frozen,means=arguments
    HASHES.clear();VALIDATED.clear()
    VALIDATED.update((directory/(name+'.bin')).resolve() for name in ('meta','row-substeps','pre-filter','post-body','joint-latent','reward'))
    batch,report=io.audit_batch(directory,protocol,layout,frozen,means)
    assert batch['purpose']==1
    return batch['returns'].astype(float).copy(),report,dict(HASHES)


def training_batches(root,protocol,inherited,layout,updates,io):
    identity={'schema':2,'protocol':protocol,'helper_sha256':digest(io.__file__),'V1_auditor_sha256':V1_AUDITOR_SHA,
        'executed_math_sources':{str(Path(path).resolve()):digest(path) for path in
            (__file__,v1.__file__,reference.__file__,previous.__file__,v1.retained.__file__,reference.shared.__file__)},
        'contract_math':'\n'.join(inspect.getsource(fn) for fn in (raw_sensors,derived_features,action_latents,log_probability,task_type)),
        'constants':[N,T,P,MB,SIGMA,LIMIT],'Python':platform.python_version(),'NumPy':np.__version__,
        'layout_sha256':digest(root/'layout.json'),'frozen_inputs_sha256':digest(root/'frozen-inputs.json')}
    key=hashlib.sha256(json_text(identity,sort_keys=True).encode()).hexdigest()
    cache=Path('build/continuous-pond-v2-preparation-2026-11-23/offline-audits/batch-receipts')/f'{protocol:016x}'/key
    means,_=read_policy(root/'training-started/turning-policy-update-0.bin',protocol,inherited)
    pending=[];results={}
    for update in range(1,updates+1):
        path=cache/f'update-{update}.json';directory=root/'training-started'/f'training-batch-{update}'
        if path.exists():
            proof=js(path)
            assert proof['schema']==2 and proof['raw_audit_identity']==key and proof['protocol']==protocol and proof['update']==update
            assert proof['all_actual_control_rows_reconstructed'] is True and np.array_equal(np.float32(proof['expected_means']),means)
            for source in proof['executed_auditor_sources']:assert digest(source['snapshot'])==source['sha256']
            for source,value in proof['input_sha256'].items():assert digest(source)==value
            values=np.asarray(proof['returns'],float);assert values.shape==(N,) and np.isfinite(values).all()
            results[update]=(values,proof['report'])
            print(json_text({'reused_hash_bound_full_row_receipt':update,'physical_controls':0,'optimizer_steps':0}),flush=True)
        else:pending.append((update,(directory,protocol,layout,inherited[0],means.copy())))
        means,_=read_policy(root/'training-started'/f'turning-policy-update-{update}.bin',protocol,inherited)
    available=previous.available_ram_bytes();workers=min(4,os.cpu_count() or 1,max(1,len(pending)),max(1,available//(4*1024**3)))
    memory={'available_RAM_bytes':available,'workers':workers,'maximum_rows_per_chunk':4096,'receipt_identity':key}
    print(json_text({'offline_audit_memory_budget':memory}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs=[(update,args,executor.submit(batch_worker,args)) for update,args in pending]
        for update,args,job in jobs:
            returns,report,hashes=job.result()
            for path,value in hashes.items():assert path not in HASHES or HASHES[path]==value;HASHES[path]=value
            receipt={'schema':2,'protocol':protocol,'raw_audit_identity':key,'update':update,'expected_means':args[4].tolist(),
                'all_actual_control_rows_reconstructed':True,'physical_controls_executed':0,'optimizer_steps_executed':0,
                'returns':returns.tolist(),'report':report,'executed_auditor_sources':ACTIVE_SOURCES,
                'input_sha256':dict(sorted(hashes.items()))}
            path=cache/f'update-{update}.json';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8') as stream:stream.write(json_text(receipt,indent=2,sort_keys=True)+'\n')
            digest(path);results[update]=(returns,report)
            print(json_text({'offline_batch_audit':update,'total':updates,'physical_controls':0,'optimizer_steps':0}),flush=True)
    return results,memory


def verify_process_ledger(root,inherited_only):
    records=js(root/'process-execution.json');expected=['Gate','Baseline']+([] if inherited_only else ['Calibrate'])+['Train']
    assert [r['phase'] for r in records]==expected;seconds=0.
    for entry in records:
        assert entry['exit_code']==0 and entry['terminated_at_cap'] is False and math.isfinite(entry['seconds']) and entry['seconds']>0
        seconds+=entry['seconds'];path=Path(entry['executable'])
        assert path.name==('turning_physics_gate_v2.exe' if entry['phase']=='Gate' else 'ppo_turning_swim_v2.exe')
        assert digest(path)==entry['executable_sha256'].lower();read(entry['stdout']);read(entry['stderr'])
    assert seconds<900
    return {'passed':True,'combined_native_process_seconds':seconds,'maximum_seconds':900,'phases':records}


def verify_training(root,protocol,inherited,layout,baseline,calibration,io):
    native=js(root/'training-result.json')
    assert native['schema']==2 and native['study']=='continuous_pond_v2' and native['closed'] and native['phase_closed']
    assert not native['fault'] and not native['runtime_adopted'] and native['retries']==0
    assert native['critic_steps']==native['critic_forward_calls']==0 and native['worlds']==N
    inherited_only=baseline['continuous']['passes']==16;assert native['inherited_only']==inherited_only
    updates=int(native['PPO_updates']);assert updates==0 if inherited_only else 1<=updates<=24
    prior=baseline['ready'] if inherited_only else calibration['ready']
    near(native['prior_active_seconds'],prior['active_seconds'],1e-10,1e-10)
    assert 0<native['active_seconds']+native['prior_active_seconds']<900
    entries=[json.loads(line) for line in read(root/'updates.jsonl').decode().splitlines()];assert len(entries)==updates
    means,policy=read_policy(root/'training-started/turning-policy-update-0.bin',protocol,inherited)
    assert np.all(means==0) and policy['update']==policy['Adam_steps']==0
    collections,memory=training_batches(root,protocol,inherited,layout,updates,io) if updates else ({},{'workers':0})
    controls,substeps=prior['controls'],prior['substeps'];selection=baseline['continuous']['passes'];details=[]
    for update,entry in enumerate(entries,1):
        returns,collected=collections[update];a=envelope(root/'training-started'/f'advantages-{update}.bin')
        assert [a.value('I'),a.value('I'),a.value('i'),a.value('i')]==[0x32415054,2,update,N]
        assert a.value('d')==calibration['fixed_global_residual_RMS']
        assert np.array_equal(a.array(size=N),returns.astype(np.float32))
        near(a.array(size=N),np.float32(leave_one_out(returns,calibration['fixed_global_residual_RMS'])),2e-6,2e-6);a.finish()
        current,info=read_policy(root/'training-started'/f'turning-policy-update-{update}.bin',protocol,inherited)
        assert info['update']==update and 4<=info['Adam_steps']-policy['Adam_steps']<=16 and (info['Adam_steps']-policy['Adam_steps'])%4==0
        directory=root/'training-started'/f'assessment-{update}'
        batch,assessment=io.audit_batch(directory,protocol,layout,inherited[0],current)
        assert batch['purpose']==2 and batch['n']==16 and batch['horizon']==1200
        scores=io.score_evaluation(directory,batch);selection=scores['passes']
        controls+=collected['controls']+assessment['controls'];substeps+=collected['substeps']+assessment['substeps']
        assert entry['update']==update and entry['worlds']==N and entry['minibatch_episodes']==MB
        assert entry['selection_passes']==selection and entry['selection_survivors']==sum(c['gates']['uninterrupted'] for c in scores['cases'])
        assert entry['actor_steps']==info['Adam_steps'] and entry['controls']==controls and entry['substeps']==substeps
        near(entry['means'],current,0,0);assert math.isfinite(entry['KL']) and entry['KL']>=-1e-7
        near(entry['minimum_mean_speed_mm_s'],min(c['metrics']['mean_speed_mm_s'] for c in scores['cases']),2e-4,5e-5)
        near(entry['minimum_window_speed_mm_s'],min(c['metrics']['min_window_speed_mm_s'] for c in scores['cases']),2e-4,5e-5)
        if update<updates:assert selection<16
        details.append({'update':update,'collection':collected,'assessment_collection':assessment,'assessment':scores,'policy':info,'KL':entry['KL']})
        means,policy=current,info
        print(json_text({'offline_assessment_audit':update,'total':updates,'physical_controls':0,'optimizer_steps':0}),flush=True)
    assert native['actor_steps']==policy['Adam_steps'] and native['final_policy_update']==updates and native['selection_passes']==selection
    withheld=None
    if selection==16:
        directory=root/'withheld-validation';batch,collected=io.audit_batch(directory,protocol,layout,inherited[0],means)
        assert batch['purpose']==2 and batch['withheld'] and batch['n']==8 and batch['horizon']==36000
        withheld=io.score_evaluation(directory,batch);withheld['collection']=collected
        controls+=collected['controls'];substeps+=collected['substeps']
        assert native['withheld_continuous_passes']==withheld['passes'] and native['withheld_seconds']==1800
    else:
        assert updates==24 and not (root/'withheld-validation').exists() and native['withheld_continuous_passes'] is None
    retained=verify_old_retention(root/'final-retention',inherited,means,True)
    controls+=retained['controls'];substeps+=retained['substeps'];rt=retained['totals']
    qualification={'selection_passes':selection,'withheld_continuous_passes':withheld['passes'] if withheld else None,
        'withheld_seconds':1800,'short_swim_passes':rt['short_passes'],'long60_retained_windows':rt['long60_retained'],
        'recovery_passes':rt['recovery_passes'],'long300_retained_windows':rt['long300_retained'],
        'original_hold_passes':rt['original_holds'],'withheld_hold_passes':rt['withheld_holds'],'hold300_passes':rt['endurance_holds']}
    for name,value in qualification.items():assert native[name]==value
    qualified=selection==16 and withheld is not None and withheld['passes']==8 and retained['passed'];assert native['qualified']==qualified
    ablation=None
    if qualified:
        mask=np.zeros(P,dtype=np.float32);mask[::20]=means[::20]
        directory=root/'constant-only-ablation';batch,collected=io.audit_batch(directory,protocol,layout,inherited[0],mask)
        assert batch['purpose']==4 and batch['n']==16 and batch['horizon']==1200
        ablation=io.score_evaluation(directory,batch);ablation['collection']=collected
        controls+=collected['controls'];substeps+=collected['substeps'];assert native['ablation_passes']==ablation['passes']
    else:assert native['ablation_passes'] is None and not (root/'constant-only-ablation').exists()
    assert native['controls']==controls and native['substeps']==substeps and controls<=CAP_CONTROLS and substeps<=CAP_SUBSTEPS
    assert sorted(int(p.name.split('-')[-1]) for p in (root/'training-started').glob('training-batch-*'))==list(range(1,updates+1))
    return {'qualified':qualified,'controls':controls,'substeps':substeps,'PPO_updates':updates,'Adam_steps':policy['Adam_steps'],
        'assessment_sha256':HASHES[str((root/'training-result.json').resolve())],'final_policy_sha256':policy['sha256'],
        'qualification':qualification,'inherited_only':inherited_only,'updates':details,'retention':retained,
        'withheld':withheld,'ablation':ablation,'offline_audit':memory}


def audit_preparation(args):
    root=args.root.resolve();assert root.is_dir()
    assert digest(v1.__file__)==V1_AUDITOR_SHA,'The closed version-one auditor must remain immutable'
    sources=preserve_sources();envelopes=validate_envelopes(root,args.helper)
    protocol,_,_=read_protocol(root);inherited=reference.parents(root,protocol);layout=js(root/'layout.json')
    means,policy=read_policy(root/'initial-policy/turning-policy-update-0.bin',protocol,inherited);assert np.all(means==0)
    native=js(root/'prepared.json')
    assert native['schema']==2 and native['prepared'] and native['physical_run_approved'] is False
    assert native['physical_controls']==native['optimizer_steps']==0
    assert [native[k] for k in ('worlds','maximum_controls','PPO_updates_max','Adam_steps_max','controls_cap','substeps_cap')]==[N,T,24,384,CAP_CONTROLS,CAP_SUBSTEPS]
    assert not (root/'authorization.bin').exists()
    for name in ('gate-started','gate-result.json','baseline-started','calibration-started','training-started'):
        assert not (root/name).exists(),'Preparation cannot claim physical work was authorized or started'
    inputs=bind_native_inputs(args.native_dir.resolve());details={}
    for prefix in ('prepare','verified'):
        initial=initial_arrays(root/(prefix+'-zero')/'initialization.bin',layout,N,0x324e5054,False,inherited[0])
        withheld=initial_arrays(root/(prefix+'-zero')/'withheld-zero.bin',layout,8,0x32575054,True,inherited[0],False)
        math_report=verify_math(root/(prefix+'-math'),inherited[2])
        bulk=js(root/(prefix+'-zero')/'bulk-policy-zero.json')
        assert bulk['passed'] and [bulk[k] for k in ('worlds','minibatch_episodes','loss_grid_blocks','checked_rows','checked_last_rows')]==[N,MB,4,N,4]
        assert bulk['physical_controls']==bulk['optimizer_steps']==0
        independent_bulk=verify_bulk(root/(prefix+'-zero'),protocol,initial)
        gpu=js(root/(prefix+'-zero')/'gpu-kernel-metadata.json')
        assert gpu['physical_controls']==gpu['optimizer_steps']==0
        assert gpu['advance_block_threads']==32 and gpu['grid_blocks']==64
        assert gpu['SM_count']>0 and gpu['VRAM_bytes']>0 and gpu['active_blocks_per_SM']>0
        details[prefix]={'initial_errors':initial['errors'],'withheld_errors':withheld['errors'],'math':math_report,
            'bulk_native_report':bulk,'independent_bulk':independent_bulk,'GPU_kernel_metadata':gpu}
    gate=verify_gate_preparation(root,protocol,layout,inherited)
    read(root/'preflight-passed.txt')
    output=args.report.resolve() if args.report else root/'independent-preparation-verification.json'
    assert not output.exists()
    result={'schema':2,'study':'continuous_pond_v2','phase':'preparation','passed':True,'qualified':False,
        'physical_run_approved':False,'physical_controls_executed_by_verifier':0,'optimizer_steps_executed_by_verifier':0,
        'auditor_sources':sources,'native_inputs':inputs,'envelopes':envelopes,'initial_policy':policy,'preparation':details,'pure_gate':gate,
        'input_sha256':dict(sorted(HASHES.items()))}
    output.write_text(json_text(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json_text({'passed':True,'phase':'preparation','physical_run_approved':False,'physical_controls':0,
        'optimizer_steps':0,'report':str(output),'sha256':digest(output)}))


def audit_physical(args):
    global ACTIVE_SOURCES
    import continuous_pond_v2_batch_audit as io
    root=args.root.resolve();phase=args.phase
    assert digest(v1.__file__)==V1_AUDITOR_SHA
    ACTIVE_SOURCES=preserve_sources(phase)
    envelopes=validate_envelopes(root,args.helper,phase)
    protocol,_,_=read_protocol(root);inherited=reference.parents(root,protocol);layout=js(root/'layout.json')
    initial,policy=read_policy(root/'initial-policy/turning-policy-update-0.bin',protocol,inherited);assert np.all(initial==0)
    preparation=reuse_preparation(root);authorization=read_authorization(root,protocol)
    frozen=v1.retained.verify_freeze(root,required=True)
    gate=verify_gate(root,protocol,inherited)
    result={'schema':2,'study':'continuous_pond_v2','phase':phase,'passed':True,'qualified':False,
        'physical_controls_executed_by_verifier':0,'optimizer_steps_executed_by_verifier':0,
        'auditor_sources':ACTIVE_SOURCES,'envelopes':envelopes,'preparation':preparation,'authorization':authorization,
        'freeze':frozen,'initial_policy':policy,'gate':gate}
    baseline=calibration=None
    if not gate['passed']:
        assert phase=='gate' and not (root/'baseline-started').exists()
        result.update(closed=True,outcome='physics_gate_failed',controls=gate['controls'],substeps=gate['substeps'],PPO_updates=0,Adam_steps=0)
    if phase in ('baseline','calibration','training'):
        assert gate['passed'];baseline=verify_baseline(root,protocol,inherited,layout,io);result['baseline']=baseline
    if phase in ('calibration','training') and baseline['continuous']['passes']<16:
        calibration=verify_calibration(root,protocol,inherited,layout,baseline,io);result['calibration']=calibration
    if phase=='training':
        training=verify_training(root,protocol,inherited,layout,baseline,calibration,io);result['training']=training
        for key in ('qualified','controls','substeps','PPO_updates','Adam_steps','final_policy_sha256','assessment_sha256','qualification'):
            result[key]=training[key]
        if not training['inherited_only']:result['calibration_release']=verify_calibration_release(root,protocol)
        result['process_ledger']=verify_process_ledger(root,training['inherited_only']);result['closed']=True
    output=args.report.resolve() if args.report else root/('independent-verification.json' if phase=='training' else f'independent-{phase}-verification.json')
    assert not output.exists(),'Preserve an already sealed phase proof'
    result['input_sha256']=dict(sorted(HASHES.items()))
    output.write_text(json_text(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if phase=='calibration':
        assert calibration is not None and output==root/'independent-calibration-verification.json'
        write_calibration_release(root,protocol,output)
    print(json_text({'passed':True,'phase':phase,'qualified':result['qualified'],'report':str(output),'sha256':digest(output),
                    'physical_controls':0,'optimizer_steps':0}))


def self_test():
    body=np.stack([v1.expected_initial_body(i) for i in range(16)]);depth=reference.motion(body)[0][:,1]/reference.SCALE
    raw=raw_sensors(body,depth);base=np.ones((16,4));u=np.zeros((16,P))
    assert raw.shape==(16,20) and np.array_equal(action_latents(u,base,raw),base)
    corrected=action_latents(u+.2,base,raw);assert np.all(abs(corrected-base)<=LIMIT)
    returns=np.arange(N,dtype=float)/N;adv=leave_one_out(returns,1)
    assert np.all(abs(adv.reshape(-1,16).sum(axis=0))<1e-10)
    assert np.array_equal(np.sort(permutation(N,0x7138b2a9)),np.arange(N))
    before=leave_one_out(returns,2);baseline=returns[19]-before[19]*2;returns[19]+=7
    assert abs((returns[19]-leave_one_out(returns,2)[19]*2)-baseline)<1e-12
    return {'passed':True,'physical_controls':0,'optimizer_steps':0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path,nargs='?');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--phase',choices=('preparation','gate','baseline','calibration','training'),default='preparation');parser.add_argument('--report',type=Path)
    parser.add_argument('--helper',type=Path,default=Path('build/learned-amplitude-preparation-2026-11-22/check-evidence.exe'))
    parser.add_argument('--native-dir',type=Path,default=Path('build/continuous-pond-v2-native-2026-11-23'))
    args=parser.parse_args()
    if args.self_test:print(json_text(self_test()));return
    assert args.root is not None
    if args.phase=='preparation':audit_preparation(args)
    else:audit_physical(args)


if __name__=='__main__':main()
