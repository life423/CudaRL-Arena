"""Offline v2 current-study evidence audit. Never integrates bodies or trains.

Geometry/CSV arithmetic is reused from the immutable v1 evidence verifier; v2
policy, runtime-size, split-trace, profile and release schemas are explicit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import os
import ctypes
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")

import numpy as np
import verify_current_swim as base

read, js, rows, near = base.read, base.js, base.rows, base.near
envelope, verify_assembly, motion = base.envelope, base.verify_assembly, base.motion
HASHES, VALIDATED = base.HASHES, base.VALIDATED
shared = base.shared
SIZES = (2048,4096,8192,16384)
CAP_CONTROLS, CAP_SUBSTEPS = 89571440,895714400

def verify_gate(root):
    gate = root/"physics-gate"
    approved=read(gate/"approved-protocol.md")
    assert hashlib.sha256(approved).hexdigest() == "770342c47155e9314c7cffe4529d2d58bad7897588e0cd236db6e869a635cf1e"
    if (root/"approved-protocol.md").exists(): assert approved == read(root/"approved-protocol.md")
    frozen=js(root/"gate-frozen-inputs.json")
    for item in frozen: assert hashlib.sha256(read(Path(item["path"]))).hexdigest() == item["sha256"].lower()
    zero = js(gate/"zero-started/result.json")
    assert zero["passed"] and zero["physical_controls"] == zero["physical_substeps"] == 0
    assembly = verify_assembly(gate/"zero-started/assembly-probes.bin")
    gate_protocol = shared.fnv(envelope(gate/"protocol.bin").data)
    marker = envelope(gate/"zero-passed.bin")
    assert marker.value("I") == 2 and marker.value("Q") == gate_protocol; marker.finish()
    report = js(gate/"run-started/result.json")
    assert report["passed"] and not report["fault"]
    assert report["PPO_updates"] == report["optimizer_steps"] == 0
    assert report["attempted_controls"] == report["completed_controls"] == 3760
    assert report["physical_substeps"] == 37600 and report["finished_cases"] == 26
    marker = envelope(gate/"gate-passed.bin")
    assert [marker.value("I") for _ in range(3)] == [0x32474343,2,1]
    assert [marker.value("Q") for _ in range(3)] == [3760,37600,gate_protocol]; marker.finish()
    seconds = float(read(gate/"run-started/active-seconds.txt"))
    near(seconds, report["active_seconds"], 1e-12, 1e-12)
    assert 0 < seconds < 900
    table = rows(gate/"run-started/states.csv")
    index = {(row["group"], int(row["case"]), int(row["backend"]), int(row["control"])): row for row in table}
    assert len(index) == len(table) == 3760+62
    groups = [("zero_swim", 4, 40, (0,1,2)), ("zero_hold", 6, 40, (0,1,2)),
              ("flow_saved_motor", 8, 80, (1,2)), ("flow_zero_target_servo", 8, 80, (1,2))]
    counted, maximum_position = 0, 0.
    for group, cases, count, backends in groups:
        for case in range(cases):
            for backend in backends:
                sequence = [index[group,case,backend,control] for control in range(count+1)]
                assert [row["substeps"] for row in sequence] == list(range(0,count*10+1,10))
                assert [row["attempted_substeps"] for row in sequence] == list(range(0,count*10+1,10))
                counted += count
                if group == "flow_zero_target_servo":
                    body = np.asarray([[row[key] for key in (["x","y","z","qx","qy","qz","qw"]
                                      +[f"j{i}" for i in range(4)]+[f"v{i}" for i in range(10)])] for row in sequence])
                    center, _ = motion(body)
                    direction = np.asarray([(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)][case%4])
                    assert float(np.dot(center[-1]-center[0], direction)) > 0
            for control in range(count+1):
                cpu, gpu = index[group,case,1,control], index[group,case,2,control]
                if group.startswith("zero_"):
                    old = index[group,case,0,control]
                    for key in cpu:
                        if key != "backend": assert old[key] == cpu[key], (group,case,control,key)
                for key in ("x","y","z"):
                    difference = abs(cpu[key]-gpu[key]); maximum_position = max(maximum_position,difference)
                    assert difference <= 2e-5
                for key in ["qx","qy","qz","qw"]+[f"j{i}" for i in range(4)]:
                    assert abs(cpu[key]-gpu[key]) <= 2e-5
                for i in range(10):
                    assert abs(cpu[f"v{i}"]-gpu[f"v{i}"]) <= (2e-7 if i < 3 else 2e-5)
                for key in ("joint_work","positive_work","relative_water_loss","external_work","numerical_loss","energy_error"):
                    assert abs(cpu[key]-gpu[key]) <= 1e-12+1e-4*max(abs(cpu[key]),abs(gpu[key]))
    proof=envelope(gate/"run-started/states.bin")
    assert [proof.value("I") for _ in range(3)] == [1,696,112]
    width=824;count=(len(proof.data)-proof.at)//width
    dtype=np.dtype({"names":["group","case","backend","control","body","actual","world_controls","substeps","unsafe","invalid","failure"],
        "formats":["<i4","<i4","<i4","<i4",("<f4",23),("<f4",40),"<u8","<u8","u1","u1","<i4"],
        "offsets":[0,4,8,12,16,216,544,552,569,570,704],"itemsize":width})
    raw=np.frombuffer(proof.data,dtype=dtype,count=count,offset=proof.at);proof.at+=count*width;proof.finish()
    assert count==len(table) and np.isfinite(raw["body"]).all()
    assert np.all(abs(raw["body"][:,7:11])<=1) and np.all(abs(raw["actual"])<=1)
    assert np.all(raw["unsafe"]==0) and np.all(raw["invalid"]==0) and np.all(raw["failure"]==0)
    assert np.array_equal(raw["world_controls"],raw["control"]) and np.array_equal(raw["substeps"],raw["control"]*10)
    names=["zero_swim","zero_hold","flow_saved_motor","flow_zero_target_servo"]
    for saved,state in zip(table,raw):
        assert saved["group"]==names[int(state["group"])] and all(saved[k]==state[k] for k in ("case","backend","control"))
        near([saved[k] for k in ["x","y","z","qx","qy","qz","qw"]+[f"j{i}" for i in range(4)]+[f"v{i}" for i in range(10)]],state["body"][:21],0,0)
    assert int(np.sum(raw["control"]>0)) == counted == 3760
    near(maximum_position, report["maximum_position_error_model"], 1e-12, 1e-8)
    return {"passed": True, "controls": counted, "substeps": counted*10, "active_seconds": seconds,
            "assembly": assembly, "maximum_position_error_model": maximum_position}


def verify_neutral_probes(root):
    reader=envelope(root/"physics-gate/zero-started/neutral-force-probes.bin")
    assert [reader.value("I") for _ in range(3)] == [1,120,3616]
    input_type=np.dtype({"names":["body","fluid","flow","source"],
                         "formats":[("<f4",23),("<f4",3),("<f4",3),"<i4"],
                         "offsets":[0,92,104,116],"itemsize":120})
    output_type=np.dtype({"names":["systems","old_flow","new_flow"],
                          "formats":[("<f4",(4,220)),("<f4",10),("<f4",10)],
                          "offsets":[0,3520,3568],"itemsize":3616})
    inputs=reader.array(input_type,12)
    groups=[reader.array(output_type,12),reader.array(output_type,12)];reader.finish()
    near(inputs["fluid"],np.broadcast_to(np.float32([1000,.001,9.81]),(12,3)),0,0)
    details=[]
    for name,block in zip(("CPU","CUDA"),groups):
        assert np.isfinite(block["systems"]).all()
        old,new,new_zero,legacy_zero=np.moveaxis(block["systems"],1,0)
        assert np.array_equal(old[:,:210],new[:,:210])
        assert np.array_equal(new_zero,legacy_zero)
        old_static=old[:,210:220].astype(float)-block["old_flow"]
        new_static=new[:,210:220].astype(float)-block["new_flow"]
        assert np.all(new_static==0)
        details.append({"backend":name,"probes":12,"old_static_force_max":float(abs(old_static).max()),
                        "v2_static_force_max":float(abs(new_static).max()),"mass_resistance_inertia_unchanged":True,
                        "legacy_zero_delegation_exact":True})
    return details


def file_digest(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    HASHES[str(path)]=digest.hexdigest();return digest.hexdigest()


def validate_envelopes(root,helper):
    paths=sorted(root.rglob('*.bin'))
    directory=Path('build/current-swim-v2-preparation-2026-11-22/offline-audits');directory.mkdir(exist_ok=True)
    listing=directory/'envelope-paths.txt'
    listing.write_text(''.join(str(path.resolve())+'\n' for path in paths),encoding='utf-8')
    read(helper)
    process=subprocess.run([str(helper.resolve()),str(listing.resolve())],capture_output=True,text=True,check=True,
                           creationflags=0x08000000 if os.name=='nt' else 0)
    report=json.loads(process.stdout)
    assert report['passed'] and report['files']==len(paths) and report['physical_controls']==report['optimizer_steps']==0
    VALIDATED.update(path.resolve() for path in paths);return report


def verify_study_freeze(root):
    frozen=js(root/'study-frozen-inputs.json')
    for item in frozen:assert file_digest(item['path'])==item['sha256'].lower(),item['path']
    return {'frozen_inputs':len(frozen),'changed':0}


def reuse_verified(root,phase):
    """Reuse completed arithmetic only after rehashing every exact input byte."""
    path=root/('independent-'+phase+'-verification.json')
    previous=js(path)
    assert previous['passed'] and previous['phase']==phase and phase in previous
    current=Path(__file__).resolve()
    for name,digest in previous['input_sha256'].items():
        if Path(name).resolve()==current and hashlib.sha256(current.read_bytes()).hexdigest()!=digest:
            snapshot=Path('build/current-swim-v2-preparation-2026-11-22/offline-audits/verified-scripts')/(digest+'.py')
            assert file_digest(snapshot)==digest,'Executed verifier snapshot missing or changed'
        else:assert file_digest(name)==digest,name
    section=previous[phase]
    section['reused_verified_arithmetic']={'report_sha256':HASHES[str(path.resolve())],
        'inputs_rehashed':len(previous['input_sha256']),'physical_controls':0,'optimizer_steps':0}
    return section


def payload_checksum(path):
    path=Path(path).resolve()
    assert path in VALIDATED,path
    with path.open('rb') as stream:magic,size,checksum=struct.unpack('<QQQ',stream.read(24))
    assert magic==0x31504B434C524443 and size==path.stat().st_size-24
    return checksum


def file_fnv(path):
    return shared.fnv(read(path))


def read_protocol(root):
    reader=envelope(root/'protocol.bin');protocol=payload_checksum(root/'protocol.bin')
    assert [reader.value('I'),reader.value('I')]==[0x32505343,2]
    assert reader.value('Q')==file_fnv(root/'approved-protocol.md')
    parent,hold=reader.value('Q'),reader.value('Q')
    assert [reader.value('Q') for _ in range(3)]==[file_fnv(root/name) for name in ('gate-frozen-inputs.json','physics-gate/gate-passed.bin','independent-gate-verification.json')]
    assert [reader.value('i') for _ in range(21)]==[6,2,3,5,44,41,4,16,336,7,16384,160,32,512,10720,2560,17040,2949120,CAP_CONTROLS,CAP_SUBSTEPS,900]
    assert [reader.value('i') for _ in range(4)]==list(SIZES)
    assert reader.value('I')==0x57c0a199
    assert [reader.value('Q'),reader.value('Q')]==[52*1024**3,8*1024**3]
    near([reader.value('f') for _ in range(9)],np.float32([.5,.5,.002,.0025,.01,.5,.2,.01,8]),0,0)
    assert [reader.value('I'),reader.value('I')]==[0x36b91a57,0x84e62c13]
    endurance=np.float32([[0,-.002,0,0],[3,0,0,.002]])
    for cases,ticks in ((base.case_specs(),160),(base.case_specs(True),160),(endurance,1280)):
        for pose,fx,fy,fz in cases:
            assert reader.value('i')==int(pose)
            near([reader.value('f'),reader.value('f')],[fx,fz],0,0)
            assert reader.value('i')==ticks
    method=reader.array('u1').tobytes().decode();reader.finish()
    assert 'nativePPOonceepisode' in method and '.5*tanh(u)' in method
    return protocol,parent,hold


def read_policy(path,protocol,inherited,selection=0):
    frozen,holding,gain,parent,hold=inherited;reader=envelope(path)
    assert [reader.value('I'),reader.value('I')]==[0x32575343,2]
    assert [reader.value('Q') for _ in range(4)]==[protocol,parent,hold,selection]
    assert [reader.value('I') for _ in range(10)]==[6,2,3,5,44,41,4,16,336,7]
    n,update,age=[reader.value('i') for _ in range(3)];assert n in SIZES
    assert np.array_equal(reader.array(size=2532),frozen)
    assert np.array_equal(reader.array(size=2),holding) and reader.value('f')==gain
    means=reader.array(size=16).copy()
    assert [reader.value('f'),reader.value('f')]==[.5,.5]
    adam=shared.Reader(reader.array('u1').tobytes());reader.finish()
    near([adam.value('f') for _ in range(5)],np.float32([.01,.9,.999,.00001,.5]),0,0)
    assert adam.value('i')==age
    first,second=adam.array(size=16),adam.array(size=16);adam.finish()
    assert 0<=update<=32 and 0<=age<=512 and age%4==0
    assert np.isfinite(means).all() and np.isfinite(first).all() and np.all(np.isfinite(second)&(second>=0))
    if update==0:assert age==0 and np.all(means==0) and np.all(first==0) and np.all(second==0)
    return means,{'worlds':n,'update':update,'Adam_steps':age,'means':means.tolist(),
                  'deterministic_coefficients':(.5*np.tanh(means.astype(float))).tolist(),
                  'sha256':HASHES[str(path.resolve())]}


def hash_u32(x):
    x=np.asarray(x,dtype=np.uint32);x=x^(x>>16);x=x*np.uint32(0x7feb352d)
    x=x^(x>>15);x=x*np.uint32(0x846ca68b);return x^(x>>16)


def normal(seed,stream,index):
    def uniform(i):
        bits=hash_u32(np.uint32(seed)^hash_u32(np.uint32(stream)^hash_u32(i)))
        return ((bits>>8).astype(np.float32)+np.float32(.5))*np.float32(1/16777216)
    index=np.asarray(index,dtype=np.uint32)
    u1,u2=uniform(2*index),uniform(2*index+1)
    return np.sqrt(np.float32(-2)*np.log(u1))*np.cos(np.float32(6.28318530718)*u2)


WORLD_TYPE=np.dtype({'names':['body','current','held','last_torque','raw','actual','requested','mean_sq',
                             'controls','substeps','time','unsafe','invalid','rates','attempted','failure'],
    'formats':[('<f4',23),('<f4',3),('<f4',4),('<f4',4),('<f4',4),('<f4',40),('<f4',40),'<f4',
               '<u8','<u8','<f8','u1','u1',('<f4',4),'<u8','<i4'],
    'offsets':[0,104,152,168,184,200,360,520,528,536,544,553,554,616,632,688],'itemsize':696})
TASK_TYPE=np.dtype({'names':['controls','outcome'],'formats':['<i4','<i4'],'offsets':[92,148],'itemsize':152})
PARTS=(('observations',41),('flow',3),('base',4),('joint-latent',4),('reward',1),('metrics',16),('pre-state',33),('post-state',23))


def read_split_batch(directory,protocol,frozen,expected_means,layout,purpose,index,expected_n=None):
    """Audit every saved control, bounded RAM, with no body step or optimizer."""
    reader=envelope(directory/'meta.bin');meta_hash=payload_checksum(directory/'meta.bin')
    assert [reader.value('I'),reader.value('I'),reader.value('Q')]==[0x32425343,2,protocol]
    n,horizon,ticks,obs_columns,flow_columns,actions,metrics_columns,coefficients,got_purpose,got_index,fault=[reader.value('i') for _ in range(11)]
    assert n in SIZES and (expected_n is None or n==expected_n)
    assert [horizon,obs_columns,flow_columns,actions,metrics_columns,coefficients,got_purpose,got_index,fault]==[160,41,3,4,16,16,purpose,index,0]
    assert ticks==(32 if purpose==2 else 160)
    means=reader.array(size=16);latent=reader.array(size=n*16).reshape(n,16)
    old_logp=reader.array(size=n);returns=reader.array(size=0 if purpose==2 else n)
    assert np.array_equal(means,expected_means)
    abi=reader.value('I');offsets=layout['episode_counts']
    assert abi==offsets['size']==216 and [offsets[k] for k in ('controls','substeps','reward_sum','failure_code')]==[0,8,16,24]
    counts_type=np.dtype({'names':['controls','substeps','reward_sum','failure'],'formats':['<u8','<u8','<f8','<i4'],'offsets':[0,8,16,24],'itemsize':abi})
    counts=reader.array(counts_type,n)
    assert reader.value('I')==layout['world_size']==696;world=reader.array(WORLD_TYPE,n)
    assert reader.value('I')==layout['task_size']==152;task=reader.array(TASK_TYPE,n);reader.finish()
    assert np.all(counts['controls']==ticks) and np.all(counts['substeps']==ticks*10) and np.all(counts['failure']==0)
    assert np.all(world['controls']==ticks) and np.all(world['substeps']==ticks*10) and np.all(world['attempted']==ticks*10)
    assert np.all(world['failure']==0) and np.all(world['invalid']==0) and np.all(world['unsafe']==0)
    assert np.all(task['controls']==ticks)
    assert np.all(task['outcome']==0) if purpose==2 else np.all((task['outcome']==1)|(task['outcome']==2))
    probability_error=near(old_logp,base.log_probability(means,latent),1e-4,0)
    seed=(0x36b91a57,0x84e62c13,0x57c0a199)[purpose]
    sampled=means+np.float32(.5)*normal(seed,index,np.arange(n*16,dtype=np.uint32).reshape(n,16))
    rng_error=near(latent,sampled,2e-6,2e-6)
    manifest=js(directory/'manifest.json')
    assert manifest['schema']==2 and manifest['meta_checksum']==meta_hash and manifest['worlds']==n and manifest['ticks']==ticks and manifest['purpose']==purpose
    assert len(manifest['parts'])==8
    mapped={}
    for field,((name,columns),entry) in enumerate(zip(PARTS,manifest['parts'])):
        path=directory/(name+'.bin');file_digest(path);mm=np.memmap(path,dtype='u1',mode='r')
        magic,size,checksum=struct.unpack_from('<QQQ',mm)
        assert magic==0x31504B434C524443 and size==len(mm)-24 and size<=512*1024**2
        assert entry==dict(field=field,name=path.name,columns=columns,payload_bytes=size,payload_checksum=checksum)
        part=shared.Reader(memoryview(mm)[24:])
        assert [part.value('I'),part.value('I'),part.value('i'),part.value('Q'),part.value('i'),part.value('i'),part.value('i')]==[0x32425343,2,field,meta_hash,n,ticks,columns]
        mapped[name]=part.array(size=n*ticks*columns).reshape(ticks,n,columns);part.finish()
    specifications=base.case_specs()[np.arange(n)%16];poses=specifications[:,0].astype(int);current=specifications[:,1:4]
    near(world['current'],current,0,0)
    error={key:0. for key in ('observations','backbone','actions','rewards','continuity','filter_angle','filter_rate','geometry','steady')}
    sums=np.zeros(n);sq_speed=np.zeros(n);steady_start=np.zeros((n,3))
    peaks=np.zeros((n,3));passed=np.zeros(n,dtype=bool)
    for k in range(ticks):
        for start in range(0,n,4096):
            sl=slice(start,min(n,start+4096));pre=mapped['pre-state'][k,sl];post=mapped['post-state'][k,sl]
            obs=mapped['observations'][k,sl];flow=mapped['flow'][k,sl];backbone=mapped['base'][k,sl]
            action=mapped['joint-latent'][k,sl];reward=mapped['reward'][k,sl,0];recorded=mapped['metrics'][k,sl]
            assert all(np.isfinite(x).all() for x in (pre,post,obs,flow,backbone,action,reward,recorded))
            assert np.all(abs(pre[:,7:11])<=1) and np.all(abs(post[:,7:11])<=1) and np.all(abs(pre[:,25:29])<=1)
            assert np.all((post[:,22]>=0)&(post[:,22]<=1)) and np.all(post[:,21]>=0)
            previous=base.initial_body(poses[sl]) if k==0 else mapped['post-state'][k-1,sl,:21]
            error['continuity']=max(error['continuity'],near(pre[:,:21],previous,2e-7 if k==0 else 0,2e-6 if k==0 else 0))
            predicted=base.observations(pre,poses[sl],current[sl],np.full(len(pre),k))
            error['observations']=max(error['observations'],near(np.concatenate((obs,flow),axis=-1),predicted,3e-5,3e-5))
            assert np.all(obs[:,40]==0)
            prediction=shared.forward(frozen.astype(float),obs.astype(float))
            error['backbone']=max(error['backbone'],near(backbone,prediction,2e-5,2e-6))
            predicted_action=base.joint_latents(latent[sl],backbone,flow)
            error['actions']=max(error['actions'],near(action,predicted_action,2e-5,2e-6))
            commands=np.tanh(action.astype(float));angle=pre[:,21:25].astype(float).copy();rate=pre[:,29:33].astype(float).copy()
            h,omega=float(np.float32(.005)),float(np.float32(2*np.float32(np.pi)))
            for _ in range(10):rate+=h*(omega*omega*(commands-angle)-2*omega*rate);angle+=h*rate
            next_angle=mapped['pre-state'][k+1,sl,21:25] if k+1<ticks else world['held'][sl]
            next_rate=mapped['pre-state'][k+1,sl,29:33] if k+1<ticks else world['rates'][sl]
            error['filter_angle']=max(error['filter_angle'],near(next_angle,angle,2e-6,2e-5))
            error['filter_rate']=max(error['filter_rate'],near(next_rate,rate,2e-5,2e-5))
            actual=base.metrics(post,poses[sl]);values=np.stack((actual['forward'],actual['lane'],actual['depth'],actual['tilt'],actual['signed_depth'],*np.moveaxis(actual['velocity'],-1,0),actual['roll'],actual['pitch'],actual['head_yaw']),axis=-1)
            for col in range(11):error['geometry']=max(error['geometry'],near(recorded[:,col],values[:,col],.003 if col in (3,8,9,10) else 2e-7,2e-5))
            near(recorded[:,11],post[:,22],0,0);near(recorded[:,15],np.full(len(pre),k+1),0,0)
            error['rewards']=max(error['rewards'],near(reward,base.reward(actual,post[:,22],np.full(len(pre),k)),2e-5,3e-5))
            sums[sl]+=reward
            peaks[sl]=np.maximum(peaks[sl],np.stack((actual['lane'],actual['depth'],actual['tilt']),axis=-1))
            center,_=motion(post)
            if k==39:steady_start[sl]=center
            if k<40:assert np.all(recorded[:,12:15]==0)
            else:
                displacement=center-steady_start[sl];forward=np.sum(displacement*actual['heading'],axis=-1)
                speed=forward/((k-39)*float(np.float32(.05)))
                sq_speed[sl]+=np.sum((actual['velocity']-actual['heading']*float(np.float32(.008)))**2,axis=-1)
                rmse=np.sqrt(sq_speed[sl]/(k-39));side=displacement-actual['heading']*forward[:,None]
                path=np.degrees(np.arctan2(np.linalg.norm(side[:,[0,2]],axis=-1),forward))
                error['steady']=max(error['steady'],near(recorded[:,12],speed,2e-7,2e-5),near(recorded[:,13],rmse,2e-7,2e-5),near(recorded[:,14],path,.003,2e-5))
                if k==ticks-1:
                    work=post[:,21]/np.maximum(actual['forward'],1e-9)
                    passed[sl]=(speed>=.0072)&(speed<=.0088)&(rmse<=.0025)&(peaks[sl,0]<=.002)&(peaks[sl,1]<=.001)&(peaks[sl,2]<=10)&(path<=5)&(work<=.001)
    near(counts['reward_sum'],sums,1e-7,1e-7)
    if purpose!=2:
        assert np.array_equal(returns,np.float32(sums/8)) and np.array_equal(task['outcome']==1,passed)
    near(world['body'][:,:21],mapped['post-state'][-1,:,:21],0,0)
    near(world['body'][:,21],mapped['post-state'][-1,:,21],0,0)
    near(world['mean_sq'],mapped['post-state'][-1,:,22],0,0)
    near(world['raw'],np.tanh(mapped['joint-latent'][-1].astype(float)),2e-7,2e-6)
    assert np.all(abs(world['actual'])<=1) and np.isfinite(world['requested']).all()
    near(world['mean_sq'],np.mean(world['actual'].astype(float)**2,axis=1),2e-7,2e-6)
    table=rows(directory/'episodes.csv');assert len(table)==n
    for i,row in enumerate(table):
        assert [row[key] for key in ('episode','case','pose','controls','substeps','failure_code')]==[i,i%16,int(poses[i]),ticks,ticks*10,0]
        near([row['current_x_mm_s'],row['current_z_mm_s']],np.float32(current[i,[0,2]])*np.float32(1000),1e-8,1e-8)
        near(row['old_logp'],old_logp[i],1e-8,1e-10);near(row['reward_sum'],sums[i],1e-8,1e-10)
        assert row['return_8s']=='' if purpose==2 else near(row['return_8s'],returns[i],1e-8,1e-10)>=0
        near([row[f'latent{j}'] for j in range(16)],latent[i],1e-8,1e-10)
    timing=js(directory/'timing.json')
    assert [timing[key] for key in ('worlds','ticks','purpose','sample','controls','optimizer_steps','PPO_data')]==[n,ticks,purpose,index,n*ticks,0,purpose!=2]
    assert all(math.isfinite(timing[k]) and timing[k]>=0 for k in ('observe_ms','network_ms','physics_ms','readback_wall_seconds','validation_seconds','evidence_seconds','end_to_end_seconds'))
    assert timing['end_to_end_seconds']>0
    assert timing['end_to_end_seconds']>=timing['readback_wall_seconds']+timing['validation_seconds']+timing['evidence_seconds']
    return returns.astype(float).copy(),{'worlds':n,'ticks':ticks,'purpose':purpose,'index':index,'controls':n*ticks,'substeps':n*ticks*10,
        'sampled_complete_passes':int(passed.sum()) if purpose!=2 else None,'probability_error':probability_error,'rng_error':rng_error,
        'maximum_errors':error,'timing':timing,'batch_manifest_checksum':file_fnv(directory/'manifest.json'),'timing_checksum':file_fnv(directory/'timing.json')}


def read_ready(root,phase,phase_id,protocol,bindings,expected_controls,expected_n):
    reader=envelope(root/(phase+'-ready.bin'))
    assert [reader.value('I'),reader.value('I'),reader.value('Q'),reader.value('i')]==[0x32525343,2,protocol,phase_id]
    n,controls,substeps=[reader.value('i') for _ in range(3)];seconds=reader.value('d')
    assert n==expected_n and controls==expected_controls and substeps==controls*10
    assert 0<seconds<900 and controls<=CAP_CONTROLS
    hashes=reader.array('<u8',len(bindings));reader.finish()
    assert list(hashes)==[file_fnv(root/name) for name in bindings]
    return {'controls':controls,'substeps':substeps,'active_seconds':seconds,'worlds':n}


def verify_zero_initialization(directory,n,frozen):
    reader=envelope(directory/'initialization.bin')
    assert [reader.value('I'),reader.value('I'),reader.value('i')]==[0x32495343,2,n]
    obs=reader.array(size=n*41).reshape(n,41);flow=reader.array(size=n*3).reshape(n,3)
    output=reader.array(size=n*4).reshape(n,4);latent=reader.array(size=n*16).reshape(n,16);logp=reader.array(size=n)
    assert reader.value('I')==696;world=reader.array(WORLD_TYPE,n)
    assert reader.value('I')==152;task=reader.array(TASK_TYPE,n);reader.finish()
    assert all(np.all(world[k]==0) for k in ('controls','substeps','attempted','failure','invalid','unsafe'))
    assert np.all(task['controls']==0) and np.all(task['outcome']==0)
    specs=base.case_specs()[np.arange(n)%16];poses=specs[:,0].astype(int)
    pre=np.zeros((n,33));pre[:,:21]=world['body'][:,:21];pre[:,21:25]=world['held'];pre[:,25:29]=world['last_torque'];pre[:,29:33]=world['rates']
    near(world['current'],specs[:,1:4],0,0);near(pre[:,:21],base.initial_body(poses),2e-7,2e-6)
    expected=base.observations(pre,poses,specs[:,1:4],np.zeros(n))
    observation_error=near(np.concatenate((obs,flow),axis=-1),expected,3e-5,3e-5)
    backbone_error=near(output,shared.forward(frozen.astype(float),obs.astype(float)),2e-5,2e-6)
    probability_error=near(logp,base.log_probability(np.zeros(16),latent),1e-4,0)
    rng_error=near(latent,.5*normal(0x36b91a57,0,np.arange(n*16,dtype=np.uint32).reshape(n,16)),2e-6,2e-6)
    native=js(directory/'initialization-zero.json');bulk=js(directory/'bulk-policy-zero.json')
    assert native['passed'] and native['worlds']==n and native['physical_controls']==native['optimizer_steps']==0
    assert bulk['passed'] and bulk['worlds']==n and bulk['minibatch_episodes']==n//4
    assert bulk['checked_rows']==n and bulk['checked_last_rows']==4 and bulk['physical_controls']==bulk['optimizer_steps']==0
    assert bulk['max_row_gradient_error']<1e-5 and bulk['max_reduced_gradient_error']<2e-5 and bulk['max_stats_error']<2e-5
    return {'worlds':n,'observation_error':observation_error,'backbone_error':backbone_error,'probability_error':probability_error,'rng_error':rng_error,'bulk_loss':bulk}


def verify_preparation(root,frozen):
    prepared=js(root/'prepared.json')
    assert prepared['schema']==2 and prepared['prepared'] and prepared['world_sizes']==list(SIZES)
    assert prepared['controls_cap']==CAP_CONTROLS and prepared['substeps_cap']==CAP_SUBSTEPS
    assert prepared['physical_controls']==prepared['optimizer_steps']==0
    phases={}
    for phase in ('prepare-zero','verified-zero'):
        directory=root/phase;summary=js(directory/'summary.json');math_report=js(directory/'math/math-zero.json')
        assert summary['passed'] and summary['supported_sizes']==4 and summary['withheld_configurations_zero_step']==16
        assert summary['physical_controls']==summary['optimizer_steps']==0
        assert math_report['passed'] and math_report['actual_CUDA_cases']==24 and math_report['physical_controls']==math_report['optimizer_steps']==0
        assert math_report['max_logp_error']<2e-5 and math_report['max_gradient_error']<.0001 and math_report['max_finite_difference_error']<.0001 and math_report['max_decode_error']<2e-6
        assert math_report['LOO_own_return_excluded'] and math_report['exact_calm_passthrough']
        phases[phase]=[verify_zero_initialization(directory/('N-'+str(n)),n,frozen) for n in SIZES]
        metadata=js(directory/'gpu-kernel-metadata.json')
        assert metadata['physical_controls']==metadata['optimizer_steps']==0
        assert metadata['block_threads']==128 and metadata['working_reserve_bytes']==8*1024**3 and metadata['initial_disk_guard_bytes']==52*1024**3
        near(metadata['theoretical_occupancy'],metadata['active_blocks_per_SM']*128/metadata['maximum_threads_per_SM'],1e-12,1e-12)
    read(root/'preflight-passed.txt')
    return phases


def phase_result(root,phase,prior,expected_controls):
    report=js(root/(phase+'-result.json'))
    assert report['schema']==2 and not report['fault'] and not report['closed'] and report['phase_closed']
    assert report['controls']==expected_controls and report['substeps']==expected_controls*10
    assert report['PPO_updates']==report['actor_steps']==report['critic_steps']==report['critic_forward_calls']==0
    near(report['prior_active_seconds'],prior['active_seconds'],1e-10,1e-10)
    assert report['active_seconds']>0 and report['active_seconds']+prior['active_seconds']<900
    assert not report['qualified'] and not report['runtime_adopted'] and report['retries']==0
    return report


def verify_baseline(root,protocol):
    calm=base.calm_assessment(root/'baseline-calm')
    assert calm['short_passes']==4 and calm['hold_passes']==6 and calm['long_passes']==28
    current=base.current_assessment(root/'baseline-current',np.zeros(16),base.case_specs())
    native=js(root/'baseline-result.json')
    assert native['baseline_current_passes']==current['passes'] and native['final_current_passes']==current['passes']
    assert native['PPO_updates']==native['actor_steps']==0 and not native['runtime_adopted']
    assert native['critic_steps']==native['critic_forward_calls']==0 and native['schema']==2 and native['phase_closed']
    assert not native['fault'] and native['retries']==0
    gate_seconds=float(read(root/'physics-gate/run-started/active-seconds.txt'))
    near(native['prior_active_seconds'],gate_seconds,1e-12,1e-12)
    assert 0<native['active_seconds']+native['prior_active_seconds']<900
    final_calm=withheld=endurance=None
    if current['passes']<16:
        assert not native['closed'] and not native['qualified'] and not native['baseline_only'] and not native['fault']
        assert native['controls']==17040 and native['substeps']==170400
        ready=read_ready(root,'baseline',1,protocol,['baseline-result.json','baseline-current/summary.csv','baseline-calm/long/sustained-windows.csv','physics-gate/gate-passed.bin'],17040,0)
    else:
        ready=None
        final_calm=base.calm_assessment(root/'final-calm')
        assert final_calm['short_passes']==4 and final_calm['hold_passes']==6 and final_calm['long_pass_keys']==calm['long_pass_keys']
        withheld=base.current_assessment(root/'reserved-withheld',np.zeros(16),base.case_specs(True))
        if withheld['passes']==16:
            endurance=base.current_assessment(root/'reserved-endurance',np.zeros(16),np.float32([[0,-.002,0,0],[3,0,0,.002]]),True)
        qualified=bool(withheld['passes']==16 and endurance is not None and endurance['passes']==2)
        assert native['closed'] and native['baseline_only'] and native['qualified']==qualified
        assert native['withheld_passes']==withheld['passes'] and native['endurance_passes']==(endurance['passes'] if endurance else 0)
        expected=17040+10720+2560+(2560 if endurance else 0)
        assert native['controls']==expected and native['substeps']==expected*10
        assert not (root/'profile-started').exists() and not (root/'calibration-started').exists() and not (root/'training-started').exists()
    return {'calm':calm,'current':current,'native':native,'release':ready,'final_calm':final_calm,'withheld':withheld,'endurance':endurance}


def verify_profile(root,protocol,frozen,layout,baseline):
    assert baseline['current']['passes']<16
    master=js(root/'profile-manifest.json')
    assert master['schema']==2 and master['protocol']==protocol and master['profile_seed']==0x57c0a199 and master['ticks']==32
    assert len(master['sizes'])==4
    details=[];controls=0;prefixes={}
    for n,entry in zip(SIZES,master['sizes']):
        assert entry['worlds']==n and len(entry['samples'])==3
        seconds=[];batches=[]
        for sample,saved in enumerate(entry['samples']):
            directory=root/'profile-started'/('N-'+str(n))/('profile-batch-'+str(sample))
            returns,detail=read_split_batch(directory,protocol,frozen,np.zeros(16,dtype=np.float32),layout,2,sample,n)
            assert len(returns)==0
            assert saved['sample']==sample and saved['warmup']==(sample==0)
            near(saved['end_to_end_seconds'],detail['timing']['end_to_end_seconds'],1e-12,1e-12)
            assert saved['batch_manifest_checksum']==detail['batch_manifest_checksum'] and saved['timing_checksum']==detail['timing_checksum']
            # Common-prefix sampling is checked against the fixed RNG above;
            # this additionally binds the actual small-N trace to larger Ns.
            meta=envelope(directory/'meta.bin');meta.at=60
            meta.array(size=16);latent=meta.array(size=n*16).reshape(n,16)
            if sample in prefixes:near(latent[:2048],prefixes[sample],0,0)
            else:prefixes[sample]=latent[:2048].copy()
            seconds.append(saved['end_to_end_seconds']);batches.append(detail);controls+=detail['controls']
        timed=sum(seconds[1:]);score=2*n*32/timed
        near(entry['timed_seconds'],timed,1e-12,1e-12);near(entry['controls_per_second'],score,1e-8,1e-12)
        details.append({'worlds':n,'controls_per_second':score,'batches':batches})
    assert controls==2949120
    selected=max(details,key=lambda x:(x['controls_per_second'],-x['worlds']))['worlds']
    assert master['selected_worlds']==selected
    marker=envelope(root/'selected-size.bin')
    assert [marker.value('I'),marker.value('I'),marker.value('Q'),marker.value('i')]==[0x325a5343,2,protocol,selected]
    assert [marker.value('Q'),marker.value('Q')]==[file_fnv(root/'profile-manifest.json'),file_fnv(root/'baseline-ready.bin')];marker.finish()
    native=phase_result(root,'profile',baseline['release'],17040+controls)
    assert native['selected_worlds']==selected
    ready=read_ready(root,'profile',2,protocol,['profile-result.json','profile-manifest.json','selected-size.bin','baseline-ready.bin'],17040+controls,selected)
    assert ready['active_seconds']>=native['prior_active_seconds']+native['active_seconds']
    return {'selected_worlds':selected,'selection_payload_checksum':payload_checksum(root/'selected-size.bin'),
            'controls':controls,'sizes':details,'release':ready}


def verify_calibration(root,protocol,frozen,layout,baseline,profile):
    n=profile['selected_worlds']
    returns,batch=read_split_batch(root/'calibration-started/calibration-batch-0',protocol,frozen,np.zeros(16,dtype=np.float32),layout,0,0,n)
    reader=envelope(root/'fixed-scale.bin')
    assert [reader.value('I'),reader.value('I'),reader.value('Q')]==[0x32535343,2,protocol]
    scale=reader.value('d');means=np.asarray([reader.value('d') for _ in range(16)])
    assert [reader.value('i') for _ in range(16)]==[n//16]*16;reader.finish()
    expected_means=np.asarray([returns[c::16].mean() for c in range(16)])
    near(means,expected_means,1e-12,1e-12)
    expected_scale=float(np.sqrt(np.mean((returns-means[np.arange(n)%16])**2)))
    assert math.isfinite(scale) and scale>0;near(scale,expected_scale,1e-12,1e-12)
    report=js(root/'calibration-report.json')
    assert report['schema']==2 and report['passed'] and report['episodes']==n and report['cases']==16 and report['episodes_per_case']==n//16
    assert report['sampled_complete_passes']==batch['sampled_complete_passes']
    assert report['actor_steps']==report['critic_steps']==0
    near(report['fixed_global_residual_RMS'],scale,1e-12,1e-12)
    near(report['return_abs_max'],abs(returns).max(),1e-12,1e-12)
    near(report['return_abs_p99'],np.sort(abs(returns))[math.ceil(.99*n)-1],1e-12,1e-12)
    count=17040+2949120+n*160
    native=phase_result(root,'calibration',profile['release'],count)
    assert native['selected_worlds']==n
    ready=read_ready(root,'calibration',3,protocol,['calibration-result.json','calibration-report.json','fixed-scale.bin','selected-size.bin','profile-ready.bin','physics-gate/gate-passed.bin'],count,n)
    assert ready['active_seconds']>=native['prior_active_seconds']+native['active_seconds']
    return {'batch':batch,'scale':scale,'case_means':means.tolist(),'release':ready}


def leave_one_out(returns,scale):
    returns=np.asarray(returns,float);n=len(returns)
    assert n in SIZES and math.isfinite(scale) and scale>0
    result=np.zeros(n)
    for case in range(16):
        values=returns[case::16];assert len(values)==n//16
        result[case::16]=(values-(values.sum()-values)/(len(values)-1))/scale
    return result


def audit_saved_training_batch(arguments):
    """Worker receives only validated files; never imports a native runtime."""
    directory,protocol,frozen,means,layout,index,n=arguments
    HASHES.clear();VALIDATED.clear()
    VALIDATED.update((directory/(name+'.bin')).resolve() for name in ('meta',)+tuple(name for name,_ in PARTS))
    returns,detail=read_split_batch(directory,protocol,frozen,means,layout,1,index,n)
    return returns,detail,dict(HASHES)


def available_ram_bytes():
    if os.name=='nt':
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(name,ctypes.c_uint64) for name in
                ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended')]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.available_physical)
    return int(os.sysconf('SC_AVPHYS_PAGES'))*int(os.sysconf('SC_PAGE_SIZE'))


def audit_training_batches(root,protocol,inherited,layout,n,selection,updates):
    arguments=[];previous,_=read_policy(root/'training-started/current-policy-update-0.bin',protocol,inherited,selection)
    for update in range(1,updates+1):
        directory=root/'training-started'/('training-batch-'+str(update))
        assert all((directory/(name+'.bin')).resolve() in VALIDATED for name in ('meta',)+tuple(name for name,_ in PARTS))
        arguments.append((directory,protocol,inherited[0],previous.copy(),layout,update,n))
        previous,_=read_policy(root/'training-started'/('current-policy-update-'+str(update)+'.bin'),protocol,inherited,selection)
    available=available_ram_bytes();workers=min(4,os.cpu_count() or 1,updates,max(1,available//(4*1024**3)))
    memory_budget={'available_RAM_bytes':available,'workers':workers,'reserve_per_worker_bytes':4*1024**3,
                   'rows_per_chunk':4096,'physical_controls':0,'optimizer_steps':0}
    print(json.dumps({'offline_audit_memory_budget':memory_budget}),flush=True)
    results={}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs=[executor.submit(audit_saved_training_batch,argument) for argument in arguments]
        for update,job in enumerate(jobs,1):
            returns,detail,hashes=job.result()
            for path,digest in hashes.items():
                assert path not in HASHES or HASHES[path]==digest,path
                HASHES[path]=digest
            results[update]=(returns,detail)
            print(json.dumps({'offline_batch_audit':update,'total':updates,'workers':workers,'physical_controls':0,'optimizer_steps':0}),flush=True)
    return results,memory_budget


def verify_training(root,protocol,inherited,layout,baseline,profile,calibration):
    native=js(root/'training-result.json');n=profile['selected_worlds'];selection=profile['selection_payload_checksum']
    assert native['schema']==2 and native['closed'] and native['phase_closed'] and not native['fault']
    assert not native['runtime_adopted'] and native['retries']==0 and native['selected_worlds']==n
    assert native['critic_steps']==native['critic_forward_calls']==0
    near(native['prior_active_seconds'],calibration['release']['active_seconds'],1e-10,1e-10)
    assert 0<native['active_seconds']+native['prior_active_seconds']<900
    updates=int(native['PPO_updates']);assert 1<=updates<=32
    entries=[json.loads(line) for line in read(root/'updates.jsonl').decode().splitlines()]
    assert len(entries)==updates
    previous,previous_policy=read_policy(root/'training-started/current-policy-update-0.bin',protocol,inherited,selection)
    assert previous_policy['worlds']==n
    batches,memory_budget=audit_training_batches(root,protocol,inherited,layout,n,selection,updates)
    details=[];count=calibration['release']['controls']
    for update,entry in enumerate(entries,1):
        returns,batch=batches[update]
        advantages=envelope(root/'training-started'/('advantages-'+str(update)+'.bin'))
        assert [advantages.value('I'),advantages.value('I'),advantages.value('i'),advantages.value('i')]==[0x32415343,2,update,n]
        assert advantages.value('d')==calibration['scale']
        assert np.array_equal(advantages.array(size=n),returns)
        expected=np.float32(leave_one_out(returns,calibration['scale']))
        near(advantages.array(size=n),expected,2e-6,2e-6);advantages.finish()
        current,policy=read_policy(root/'training-started'/('current-policy-update-'+str(update)+'.bin'),protocol,inherited,selection)
        assert policy['worlds']==n and policy['update']==update
        assert 4<=policy['Adam_steps']-previous_policy['Adam_steps']<=16
        assert (policy['Adam_steps']-previous_policy['Adam_steps'])%4==0
        assessment=base.current_assessment(root/'training-started'/('assessment-'+str(update)),current,base.case_specs())
        count+=n*160+2560
        assert entry['update']==update and entry['worlds']==n and entry['minibatch_episodes']==n//4
        assert entry['current_passes']==assessment['passes'] and entry['baseline_current_passes']==baseline['current']['passes']
        assert entry['actor_steps']==policy['Adam_steps'] and entry['controls']==count and entry['calm_passthrough_exact']
        assert math.isfinite(entry['KL']) and entry['KL']>=0
        near(entry['means'],current,0,0);near(entry['current_mean_cost'],assessment['mean_cost'],2e-7,2e-6)
        near(entry['baseline_mean_cost'],baseline['current']['mean_cost'],2e-7,2e-6)
        near(entry['collection_seconds'],batch['timing']['end_to_end_seconds'],1e-12,1e-12)
        if update<updates:assert assessment['passes']<16
        details.append({'update':update,'batch':batch,'policy':policy,'current':assessment,'KL':entry['KL']})
        previous,previous_policy=current,policy
    assert native['actor_steps']==previous_policy['Adam_steps'] and native['final_current_passes']==details[-1]['current']['passes']
    assert native['baseline_current_passes']==baseline['current']['passes']
    calm=base.calm_assessment(root/'final-calm');count+=10720
    assert calm['short_passes']==4 and calm['hold_passes']==6 and calm['long_pass_keys']==baseline['calm']['long_pass_keys']
    final=details[-1]['current'];withheld=endurance=None
    if final['passes']==16:
        withheld=base.current_assessment(root/'reserved-withheld',previous,base.case_specs(True));count+=2560
        if withheld['passes']==16:
            endurance=base.current_assessment(root/'reserved-endurance',previous,np.float32([[0,-.002,0,0],[3,0,0,.002]]),True);count+=2560
        qualified=bool(withheld['passes']==16 and endurance is not None and endurance['passes']==2 and final['passes']>baseline['current']['passes'])
        assert native['gate']==('qualified_current_swimming_no_adoption' if qualified else 'reserved_current_validation_failed')
    else:
        assert updates==32 and not (root/'reserved-withheld').exists() and not (root/'reserved-endurance').exists()
        qualified=False;assert native['gate']=='bounded_updates_exhausted_current_not_qualified'
    assert native['withheld_passes']==(withheld['passes'] if withheld else 0)
    assert native['endurance_passes']==(endurance['passes'] if endurance else 0)
    assert native['qualified']==qualified and native['controls']==count and native['substeps']==count*10
    assert count<=CAP_CONTROLS and count*10<=CAP_SUBSTEPS
    return {'qualified':qualified,'controls':count,'substeps':count*10,'PPO_updates':updates,'Adam_steps':previous_policy['Adam_steps'],
            'final_policy_sha256':previous_policy['sha256'],'offline_audit':memory_budget,'updates':details,'calm':calm,'withheld':withheld,'endurance':endurance}


def self_test():
    for n in SIZES:
        returns=np.arange(n,dtype=float)*.001;advantages=leave_one_out(returns,3)
        assert all(abs(advantages[c::16].mean())<1e-12 for c in range(16))
        selected=17;old=returns[selected]-3*advantages[selected];returns[selected]+=7
        new=returns[selected]-3*leave_one_out(returns,3)[selected];near(old,new,1e-12,1e-12)
        draws=normal(0x57c0a199,1,np.arange(n*16,dtype=np.uint32).reshape(n,16))
        assert np.isfinite(draws).all()
        if n==2048:prefix=draws.copy()
        else:near(draws[:2048],prefix,0,0)
    print(json.dumps({'passed':True,'runtime_sizes':list(SIZES),'scope':'Verifier arithmetic only','physical_controls':0,'optimizer_steps':0}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("build/ppo/current-swim-v2-2026-11-22-v1"))
    parser.add_argument("--helper",type=Path,default=Path("build/learned-amplitude-preparation-2026-11-22/check-evidence.exe"))
    parser.add_argument("--gate-only",action="store_true")
    parser.add_argument("--baseline-only",action="store_true")
    parser.add_argument("--profile-only",action="store_true")
    parser.add_argument("--calibration-only",action="store_true")
    parser.add_argument("--prepared-only",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args();root=args.run.resolve()
    if args.self_test:self_test();return
    if args.gate_only and (root/'protocol.bin').exists():
        raise RuntimeError('Gate audit is sealed by the prepared learner protocol; use a later-phase audit without rewriting it')
    requested_phase=('physical_gate' if args.gate_only else 'preparation' if args.prepared_only else 'baseline' if args.baseline_only
                     else 'profile' if args.profile_only else 'calibration' if args.calibration_only else 'training')
    report={"passed":False,"qualified":False,"phase":"physical_gate",
            "scope":"Independent saved-evidence audit; zero new physical or optimizer steps"}
    try:
        read(Path(__file__));read(Path(base.__file__));read(Path(base.shared.__file__))
        report["envelopes"]=validate_envelopes(root,args.helper)
        report["gate"]=verify_gate(root)
        report["neutral_force_probes"]=verify_neutral_probes(root)
        report["counted_controls"]=3760;report["counted_substeps"]=37600
        if not args.gate_only:
            report['study_freeze']=verify_study_freeze(root)
            protocol,parent,hold=read_protocol(root);inherited=base.parents(root,protocol)
            assert inherited[3:]==(parent,hold)
            means,policy=read_policy(root/'initial-policy/current-policy-update-0.bin',protocol,inherited)
            report['initial_policy']=policy
            layout=js(root/'layout.json')
            report['preparation']=verify_preparation(root,inherited[0]);report['phase']='preparation'
            if args.prepared_only:
                for path,digest in list(HASHES.items()):assert file_digest(path)==digest
                report['passed']=True
                print(json.dumps({'passed':True,'phase':'preparation','controls':3760,'inputs':len(HASHES)}))
                return
            report['baseline']=verify_baseline(root,protocol);report['phase']='baseline'
            report['counted_controls']=report['baseline']['native']['controls']
            if report['baseline']['native']['baseline_only']:
                report['qualified']=report['baseline']['native']['qualified']
            elif not args.baseline_only:
                cached=root/'independent-profile-verification.json'
                report['profile']=reuse_verified(root,'profile') if not args.profile_only and cached.exists() else verify_profile(root,protocol,inherited[0],layout,report['baseline']);report['phase']='profile'
                report['counted_controls']=report['profile']['release']['controls']
                if not args.profile_only:
                    cached=root/'independent-calibration-verification.json'
                    report['calibration']=reuse_verified(root,'calibration') if not args.calibration_only and cached.exists() else verify_calibration(root,protocol,inherited[0],layout,report['baseline'],report['profile']);report['phase']='calibration'
                    report['counted_controls']=report['calibration']['release']['controls']
                    if not args.calibration_only:
                        report['training']=verify_training(root,protocol,inherited,layout,report['baseline'],report['profile'],report['calibration']);report['phase']='training'
                        report['qualified']=report['training']['qualified'];report['counted_controls']=report['training']['controls']
            report['counted_substeps']=report['counted_controls']*10
        for path,digest in list(HASHES.items()):assert file_digest(path)==digest
        report["passed"]=True
    except Exception as error:
        report["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        report["input_sha256"]=HASHES
        target='independent-gate-verification.json' if args.gate_only else 'independent-'+requested_phase+'-verification.json'
        (root/target).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"phase":report["phase"],"controls":report["counted_controls"],"inputs":len(HASHES)}))


if __name__=="__main__":main()
