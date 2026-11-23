"""Offline audit of the frozen requested-turn diagnostic; no body/GPU/learning calls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import numpy as np
import verify_frozen_requested_turn_preparation as prep
import continuous_pond_v2_batch_audit as cuda_reference

prior,ref,shared=prep.prior,prep.ref,prep.shared
read,envelope,near=prep.read,prep.envelope,ref.near
BODY=['x','y','z','qx','qy','qz','qw']+[f'j{i}' for i in range(4)]+[f'v{i}' for i in range(10)]+['body_work','body_travel']
OFFSETS=[0,5,-5,15,-15,30,-30,60,-60,90,-90]


def table(path):
    values=csv.reader(io.StringIO(read(path).decode('utf-8-sig')))
    header=next(values);assert len(header)==len(set(header))
    rows=[[float(x) if x not in ('null','') else np.nan for x in row] for row in values]
    data=np.asarray(rows,float).reshape(-1,len(header))
    return {'header':header,'index':{key:i for i,key in enumerate(header)},'data':data}


def cols(t,names):return t['data'][:,[t['index'][key] for key in names]]
def col(t,name):return t['data'][:,t['index'][name]]
def f32_equal(a,b):
    a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
    assert a.shape==b.shape and np.array_equal(a.view(np.uint32),b.view(np.uint32))


def world_type():
    old=prior.previous.WORLD_TYPE
    return np.dtype({'names':list(old.names)+['energy'],
        'formats':[old.fields[key][0] for key in old.names]+[('<f8',6)],
        'offsets':[old.fields[key][1] for key in old.names]+[640],'itemsize':696})


def record_dtype(layout):
    _,task=prep.types(layout);world=world_type()
    dtype=np.dtype([('group','<i4'),('case','<i4'),('backend','<i4'),('tick','<i4'),
        ('world',world),('task',task),('observation','<f4',(41,)),('base','<f4',(4,)),('command','<f4',(4,)),
        ('attempted','<i4'),('accepted','<i4'),('reason','<i4'),('failure','<i4')],align=False)
    return dtype,task


def records(root,layout):
    dtype,task=record_dtype(layout)
    r=envelope(root/'probe-started/states.bin')
    assert [r.value('I') for _ in range(3)]==[1,696,task.itemsize]
    assert (len(r.data)-r.at)%dtype.itemsize==0
    result=np.frombuffer(r.data,dtype=dtype,offset=r.at).copy()
    return result,task


def kinematics(body):
    """Independent articulated link velocities, geometry, clearance and kinetic energy."""
    b=np.asarray(body,float);lengths=list(map(lambda x:float(np.float32(x)),(.020,.026,.026)))
    centers=[b[...,:3]*ref.SCALE];qs=[b[...,3:7]];anchors=[];ya=[];pa=[]
    for link in (1,2):
        parent=link-1;anchor=centers[parent]+ref.rotate(qs[parent],[-lengths[parent]/2,0,0])
        yaw_axis=ref.rotate(qs[parent],[0,1,0]);yaw=ref.qmul(qs[parent],ref.axis_rotation([0,1,0],b[...,7+2*parent]))
        pitch_axis=ref.rotate(yaw,[0,0,1]);q=ref.qmul(yaw,ref.axis_rotation([0,0,1],b[...,8+2*parent]))
        anchors.append(anchor);ya.append(yaw_axis);pa.append(pitch_axis);qs.append(q)
        centers.append(anchor+ref.rotate(q,[-lengths[link]/2,0,0]))
    pi,r=float(np.float32(np.pi)),float(np.float32(.0015));h,w=float(np.float32(.0064)),float(np.float32(.008))
    membrane,thickness=float(np.float32(.0097)),float(np.float32(.0001))
    masses=[4*pi*(lengths[0]/2)*h*w/3*1000,pi*r*r*lengths[1]*1000,
            (pi*r*r*lengths[2]+membrane*lengths[2]*thickness)*1000]
    inertias=[masses[0]/5*np.array([h*h+w*w,(lengths[0]/2)**2+w*w,(lengths[0]/2)**2+h*h])]
    inertias.append(masses[1]*np.array([r*r/2,(lengths[1]**2+3*r*r)/12,(lengths[1]**2+3*r*r)/12]))
    rod=1000*pi*r*r*lengths[2];fin=1000*membrane*lengths[2]*thickness;half=membrane/2;offset=r+half/2
    inertias.append(np.array([rod*r*r/2+fin*((half*half+thickness*thickness)/12+offset*offset),
        rod*(lengths[2]**2+3*r*r)/12+fin*(lengths[2]**2+thickness*thickness)/12,
        rod*(lengths[2]**2+3*r*r)/12+fin*((lengths[2]**2+half*half)/12+offset*offset)]))
    ke=np.zeros(b.shape[:-1]);minimum=np.full_like(ke,np.inf)
    normals=[([1,0,0],.6),([-1,0,0],.6),([0,1,0],float(np.float32(.16))),([0,-1,0],0),([0,0,1],.4),([0,0,-1],.4)]
    for link in range(3):
        velocity=b[...,11:14]+np.cross(b[...,14:17],centers[link]-centers[0]);omega=b[...,14:17].copy()
        for j in range(link):
            velocity+=np.cross(ya[j],centers[link]-anchors[j])*b[...,17+2*j,None]
            velocity+=np.cross(pa[j],centers[link]-anchors[j])*b[...,18+2*j,None]
            omega+=ya[j]*b[...,17+2*j,None]+pa[j]*b[...,18+2*j,None]
        local=ref.rotate(qs[link],omega,True)
        ke+=.5*masses[link]*np.sum(velocity*velocity,axis=-1)+.5*np.sum(inertias[link]*local*local,axis=-1)
        height=h if link==0 else r if link==1 else float(np.float32((np.float32(.0015)*2+np.float32(.0097))/2))
        width=w if link==0 else r
        for normal,offset_bound in normals:
            u=ref.rotate(qs[link],normal,True)
            support=np.sqrt((lengths[link]*u[...,0]/2)**2+(height*u[...,1])**2+(width*u[...,2])**2) if link==0 else abs(u[...,0])*lengths[link]/2+np.sqrt((height*u[...,1])**2+(width*u[...,2])**2)
            minimum=np.minimum(minimum,np.sum(centers[link]*normal,axis=-1)+offset_bound-support)
    return ke,minimum


def guard_causes(body, fixed_depth_model, backend):
    """Literal binary32 observer/guard arithmetic; acceptance math stays independent."""
    body=np.asarray(body,np.float32);f=np.float32
    rotate=prior.rotate32 if backend else ref.rotate_float32
    multiply=prior.qmul32 if backend else prior.qmul_cpu32
    axis=prior.axis32 if backend else prior.axis_cpu32
    lengths=f([.020,.026,.026]);centers=[body[:,:3]*f(.04)];orientations=[body[:,3:7]]
    for link in (1,2):
        parent=link-1;anchor=centers[parent]+rotate(orientations[parent],f([-lengths[parent]*f(.5),0,0]))
        yaw=multiply(orientations[parent],axis([0,1,0],body[:,7+parent*2]));q=multiply(yaw,axis([0,0,1],body[:,8+parent*2]))
        orientations.append(q);centers.append(anchor+rotate(q,f([-lengths[link]*f(.5),0,0])))
    inset=np.zeros(len(body),np.float32)
    tail=f(f(f(.0015)*f(2)+f(.0097))*f(.5))
    for link in range(3):
        height=f(.0064) if link==0 else f(.0015) if link==1 else tail
        width=f(.008) if link==0 else f(.0015)
        for x in (-1,1):
            for y in (-1,1):
                for z in (-1,1):
                    point=centers[link]+rotate(orientations[link],f([f(x)*lengths[link]*f(.5),f(y)*height,f(z)*width]))
                    ix,iz=point[:,0]/f(.195),point[:,2]/f(.115)
                    radius=prior.fma32(iz,iz,ix*ix) if backend else ix*ix+iz*iz
                    inset=np.maximum(inset,radius)
    model=cuda_reference.cuda_com_model32(body) if backend else prior.motion_cpu32(body)[0]
    depth=abs((model[:,1]-f(fixed_depth_model))*f(.04))
    up=np.asarray(rotate(body[:,3:7],f([0,1,0])),np.float32).reshape(len(body),3)
    tilt=np.arccos(np.clip(up[:,1],f(-1),f(1))).astype(np.float32)*f(180)/f(np.pi)
    return np.where(inset>=f(1),1,np.where(depth>f(.005),4,np.where(tilt>f(20),5,0)))


def load_weights(root):
    assert hashlib.sha256(read(root/'parent-checkpoint.bin')).hexdigest().upper()==ref.PARENT_SHA
    assert hashlib.sha256(read(root/'holding-parent.bin')).hexdigest().upper()==ref.HOLD_SHA
    parent=envelope(root/'parent-checkpoint.bin');hold=envelope(root/'holding-parent.bin')
    assert [hold.value('I'),hold.value('I')]==[0x314d4148,1]
    assert hold.value('Q')!=0 and hold.value('Q')==shared.fnv(parent.data)
    assert [hold.value('I') for _ in range(7)]==[6,6,3,4,41,4,336]
    assert [hold.value('i'),hold.value('i')]==[7,112]
    return hold.array(size=2532).astype(float)


def csv_endpoints(t,r):
    assert len(t['data'])==len(r)
    for key,name in [('group','group'),('case','case'),('backend','backend'),('control','tick'),
                     ('last_attempted','attempted'),('last_accepted','accepted'),('behavior_reason','reason'),('hard_failure','failure')]:
        assert np.array_equal(col(t,key),r[name])
    for key,name in [('world_controls','controls'),('attempted_substeps','attempted'),('accepted_substeps','substeps')]:
        assert np.array_equal(col(t,key),r['world'][name])
    near(col(t,'seconds'),r['world']['time'],0,0)
    assert np.array_equal(col(t,'legacy_outcome'),r['task']['outcome'])
    assert np.array_equal(col(t,'task_controls'),r['task']['controls'])
    f32_equal(cols(t,BODY),r['world']['body'])
    for prefix,values in [('lane_origin',r['task']['start_com_model']),('target',r['task']['request']['position']),
        ('heading',r['task']['request']['heading']),('terminal_velocity',r['task']['request']['terminal_velocity'])]:
        f32_equal(cols(t,[prefix+'_'+axis for axis in 'xyz']),values)
    for prefix,width,values in [('held',4,r['world']['held']),('reference_rate',4,r['world']['rates']),('previous_torque',4,r['world']['last_torque']),
        ('observation',41,r['observation']),('base',4,r['base']),('command',4,r['command'])]:
        f32_equal(cols(t,[prefix+str(i) for i in range(width)]),values)
    near(cols(t,['energy'+str(i) for i in range(6)]),r['world']['energy'],0,0)


def head_cpu32(body):
    f=ref.rotate_float32(np.asarray(body,np.float32)[...,3:7],[1,0,0]);f[...,1]=0
    length=np.sqrt(f[...,0]*f[...,0]+f[...,2]*f[...,2])
    return f*(np.float32(1)/length)[...,None]


def course(heading,vector):
    heading,vector=np.asarray(heading,float),np.asarray(vector,float)
    return np.arctan2(heading[...,2]*vector[...,0]-heading[...,0]*vector[...,2],
                      heading[...,0]*vector[...,0]+heading[...,2]*vector[...,2])*180/float(np.float32(np.pi))


def check_group(rows,sub,weights,errors,cpu_rows=None):
    assert len(rows)>0 and rows[0]['tick']==0
    key=tuple(int(rows[0][x]) for x in ('group','case','backend'))
    assert all(tuple(int(r[x]) for x in ('group','case','backend'))==key for r in rows)
    ticks=rows['tick'];assert np.array_equal(ticks,np.arange(len(rows)))
    initial=rows[0];pose=key[1] if key[0]==0 else 0 if key[1]<11 else 3
    near(initial['world']['body'][:21],ref.initial_body(pose),2e-7,2e-6)
    assert initial['world']['controls']==initial['world']['attempted']==initial['world']['substeps']==0
    assert initial['task']['controls']==initial['task']['last_world_control']==0
    initial_heading=initial['task']['request']['heading'].copy();initial_origin=initial['task']['start_com_model'].copy()
    original_request=initial['task']['request'].copy();request=original_request.copy();origin=initial_origin.copy()
    offset=0 if key[0]==0 else OFFSETS[key[1]%11]
    consumed=0
    for j in range(1,len(rows)):
        before,after=rows[j-1],rows[j];w0,w1=before['world'],after['world'];tick=int(after['tick'])
        if tick==81 and offset:
            angle=math.radians(offset);x,y,z=initial_heading
            expected_heading=np.array([math.cos(angle)*x+math.sin(angle)*z,y,-math.sin(angle)*x+math.cos(angle)*z])
            request=after['task']['request'].copy();origin=after['task']['start_com_model'].copy()
            near(request['heading'],expected_heading,2e-7,2e-6)
            near(request['terminal_velocity'],expected_heading*float(np.float32(.008)),2e-8,2e-6)
            near(origin[[0,2]],ref.motion(w0['body'])[0][[0,2]]/ref.SCALE,3e-6,3e-6)
            assert origin[1]==initial_origin[1]
            near(request['position'],origin+request['heading']*(float(np.float32(.072))/ref.SCALE),3e-6,3e-6)
        assert after['task']['request'].tobytes()==request.tobytes()
        f32_equal(after['task']['start_com_model'],origin)
        if not after['failure']:
            assert after['task']['controls']==w1['controls'] and after['task']['last_world_control']==w1['controls']
        else:
            assert 0<=int(w1['controls'])-int(after['task']['controls'])<=1
            assert after['task']['last_world_control']==after['task']['controls']
        # Observe the PRE state with the POST-latched request and uninterrupted clock.
        observation_world=w0;observation_task=after['task'];observation_clock=before['task']['controls']
        if key[0]==0 and key[2]==1:
            assert cpu_rows is not None and j<len(cpu_rows)
            observation_world=cpu_rows[j-1]['world'];observation_task=cpu_rows[j]['task'];observation_clock=cpu_rows[j-1]['task']['controls']
        frame={'world':np.asarray([observation_world],dtype=observation_world.dtype),'task':np.asarray([observation_task],dtype=observation_task.dtype)};frame['task']['controls'][0]=observation_clock
        expected_obs=prep.fixed_observations(frame)[0]
        errors['observation']=max(errors['observation'],near(after['observation'],expected_obs,2e-5,2e-5))
        expected=shared.forward(weights,after['observation'].astype(float)[None,:])[0]
        errors['NN']=max(errors['NN'],near(after['base'],expected,2e-5,2e-6))
        errors['command']=max(errors['command'],near(after['command'],np.tanh(after['base'].astype(float)),2e-5,0))
        assert np.all(abs(after['command'])<=1)
        attempted,accepted=int(after['attempted']),int(after['accepted'])
        assert 0<=accepted<=attempted<=10
        assert int(w1['attempted'])-int(w0['attempted'])==attempted
        assert int(w1['substeps'])-int(w0['substeps'])==accepted
        if not after['failure']:
            assert accepted==attempted and 1<=accepted<=10 and w1['controls']==w0['controls']+1
            assert accepted==10 or after['reason'] in (1,4,5)
        else:assert j==len(rows)-1
        for field in ('actual','requested'):assert np.isfinite(w1[field]).all()
        assert np.all(abs(w1['actual'])<=1)
        if after['failure'] not in (101,102,103):
            assert np.all(w1['actual'][accepted*4:]==0) and np.all(w1['requested'][accepted*4:]==0)
        angle=w0['held'].astype(float).copy();rate=w0['rates'].astype(float).copy();h=float(np.float32(.005));omega=float(np.float32(2*np.float32(np.pi)))
        for _ in range(accepted):rate+=h*(omega*omega*(after['command']-angle)-2*omega*rate);angle+=h*rate
        if not after['failure']:
            errors['filter_angle']=max(errors['filter_angle'],near(w1['held'],angle,2e-6,2e-5))
            errors['filter_rate']=max(errors['filter_rate'],near(w1['rates'],rate,2e-5,2e-5))
            f32_equal(w1['last_torque'],w1['actual'][(accepted-1)*4:accepted*4])
        segment=sub[consumed:consumed+accepted];consumed+=accepted
        assert len(segment)==accepted
        if accepted:
            assert np.array_equal(segment[:,SUB_INDEX['control']],np.full(accepted,tick))
            assert np.array_equal(segment[:,SUB_INDEX['substep']],np.arange(1,accepted+1))
            assert np.array_equal(segment[:,SUB_INDEX['attempted_substeps']],int(w0['attempted'])+np.arange(1,accepted+1))
            assert np.array_equal(segment[:,SUB_INDEX['accepted_substeps']],int(w0['substeps'])+np.arange(1,accepted+1))
            body=segment[:,[SUB_INDEX[x] for x in BODY]].astype(np.float32)
            energy=segment[:,[SUB_INDEX['energy'+str(i)] for i in range(6)]]
            if not after['failure']:
                f32_equal(body[-1],w1['body'])
                near(energy[-1],w1['energy'],0,0)
            assert np.isfinite(body).all() and np.isfinite(energy).all()
            assert np.all(abs(np.sum(body[:,3:7].astype(float)**2,axis=1)-1)<=.01)
            assert np.all(abs(body[:,7:11])<=1)
            ke,gap=kinematics(body);active,inset,_=prior.boundary_values(body)
            if not after['failure']:
                assert np.all(gap>=.01-2e-7) and np.all(active<=1+2e-6)
            energy_sequence=np.vstack((w0['energy'],energy))
            for index in (1,2,3):assert np.all(np.diff(energy_sequence[:,index])>=-1e-18)
            identity=ke-energy[:,0]+energy[:,2]+energy[:,3]-energy[:,4]
            errors['energy_identity']=max(errors['energy_identity'],near(energy[:,5],identity,1e-12,1e-4))
            if not after['failure']:assert np.all(abs(energy[:,5])<=.02*(energy[:,1]+energy[:,2])+2e-10)
            center=ref.motion(body)[0];depth=abs(center[:,1]-float(initial_origin[1])*ref.SCALE)
            tilt=prior.measured(body,np.full(accepted,initial_origin[1]))['tilt']
            # The saved five-ms states allow independent first-crossing checks.
            cause=guard_causes(body,initial_origin[1],key[2])
            # Geometry/acos are reconstructed independently; near-threshold
            # disagreements are diagnosed rather than adding a relaxed gate.
            if not after['failure']:
                assert np.all(cause[:-1]==0)
                assert int(cause[-1])==int(after['reason'])
            assert np.all(body[:,21]>=np.r_[w0['body'][21],body[:-1,21]])
    assert consumed==len(sub)
    return {'group':key[0],'case':key[1],'backend':key[2],'physical_controls':len(rows)-1,
            'attempted_substeps':int(rows['attempted'][1:].sum()),'accepted_substeps':int(rows['accepted'][1:].sum()),
            'last_control':int(rows[-1]['tick']),'behavior_reason':int(rows[-1]['reason']),'hard_failure':int(rows[-1]['failure'])}


def descriptive(rows):
    tick=rows['tick'];last=rows[-1];off=OFFSETS[int(last['case'])%11];pose=0 if int(last['case'])<11 else 3
    result={'case':int(last['case']),'pose':pose,'offset_degrees':off,'controls':int(last['tick']),
            'seconds':float(last['world']['time']),'behavior_reason':int(last['reason']),
            'status':'hard_stop_prefix' if last['failure'] else 'ordinary_failure' if last['reason'] else 'completed' if last['tick']==320 else 'hard_stop_prefix',
            'turn_progress_m':0.,'turn_horizontal_arc_m':0.,'positive_work_J':float(last['world']['energy'][1])}
    bodies=rows['world']['body'];center,velocity=prior.motion_cpu32(bodies);center=center.astype(float)*ref.SCALE
    # Summary measurements are HOST float kinematics after each CUDA endpoint.
    depth=abs((prior.motion_cpu32(bodies)[0][:,1]-rows['task']['start_com_model'][:,1])*np.float32(.04))
    up=ref.rotate_float32(bodies[:,3:7],[0,1,0]);tilt=np.arccos(np.clip(up[:,1],-1,1)).astype(np.float32)*np.float32(180)/np.float32(np.pi)
    disp=center-rows['task']['start_com_model'].astype(float)*ref.SCALE
    heading=rows['task']['request']['heading'];forward=np.sum(disp*heading,axis=1);side=disp-heading*forward[:,None]
    result.update(max_depth_m=float(depth[1:].max(initial=0)),max_tilt_deg=float(tilt[1:].max(initial=0)),max_lane_m=float(np.linalg.norm(side[1:,[0,2]],axis=1).max(initial=0)))
    fields=['turn_displacement_course_error_deg','first_20_control_samples_head_error_below10_start_seconds','last4_velocity_x_m_s','last4_velocity_z_m_s','last4_velocity_course_error_deg','last4_head_circular_mean_error_deg','last4_head_error_RMS_deg','turn_mean_velocity_x_m_s','turn_mean_velocity_z_m_s','turn_mean_velocity_course_error_deg','turn_head_error_min_deg','turn_head_error_max_deg']
    result.update({key:None for key in fields})
    turn=np.flatnonzero(tick>80)
    if len(turn):
        origin=center[80];heading=last['task']['request']['heading'];delta=center[-1]-origin
        result['turn_progress_m']=float(np.dot(delta,heading));result['turn_horizontal_arc_m']=float(np.linalg.norm(np.diff(center[80:,[0,2]],axis=0),axis=1).sum())
        result['turn_displacement_course_error_deg']=float(course(heading,delta))
        dt=rows['accepted'][turn].astype(float)*.005;head=head_cpu32(bodies[turn]);errors=course(heading,head)
        mean=np.sum(velocity[turn]*dt[:,None],axis=0)/dt.sum()
        result.update(turn_mean_velocity_x_m_s=float(mean[0]),turn_mean_velocity_z_m_s=float(mean[2]),
            turn_mean_velocity_course_error_deg=float(course(heading,mean)) if np.linalg.norm(mean[[0,2]])>1e-9 else None,
            turn_head_error_min_deg=float(errors.min()),turn_head_error_max_deg=float(errors.max()))
        run=0
        for idx,err in zip(turn,errors):
            run=run+1 if abs(err)<=10 and rows['accepted'][idx]==10 else 0
            if run>=20:result['first_20_control_samples_head_error_below10_start_seconds']=float((int(tick[idx])-80-20)*.05);break
        late=np.flatnonzero(tick>240)
        if last['tick']==320 and not last['reason'] and len(late)==80:
            dt=rows['accepted'][late].astype(float)*.005;mean=np.sum(velocity[late]*dt[:,None],axis=0)/dt.sum()
            head=head_cpu32(bodies[late]);mean_head=np.sum(head*dt[:,None],axis=0)/dt.sum();err=course(heading,head)
            result.update(last4_velocity_x_m_s=float(mean[0]),last4_velocity_z_m_s=float(mean[2]),
                last4_velocity_course_error_deg=float(course(heading,mean)) if np.linalg.norm(mean[[0,2]])>1e-9 else None,
                last4_head_circular_mean_error_deg=float(course(heading,mean_head)) if np.linalg.norm(mean_head[[0,2]])>1e-9 else None,
                last4_head_error_RMS_deg=float(np.sqrt(np.sum(err*err*dt)/dt.sum())))
    return result


LAYOUT={};SUB_INDEX={}

def verify(root):
    global LAYOUT,SUB_INDEX
    native=json.loads(read(root/'result.json'));assert native['closed'] and native['study']=='frozen_requested_turn_probe'
    assert native['PPO_updates']==native['optimizer_steps']==0 and not native['qualified'] and not native['runtime_adopted']
    assert native['descriptive_probe_only'] and native['controls']<=7360 and native['attempted_substeps']<=73600
    assert 0<=native['accepted_substeps']<=native['attempted_substeps'] and np.isfinite(native['process_seconds']) and native['process_seconds']>0
    if not native['hard_fault']:assert native['process_seconds']<=120
    protocol=envelope(root/'protocol.bin');assert [protocol.value('I'),protocol.value('I')]==[0x31525446,1]
    checksum=shared.fnv(protocol.data)
    authorization=envelope(root/'authorization.bin')
    assert [authorization.value('I'),authorization.value('I'),authorization.value('Q')]==[0x31555446,1,checksum]
    assert authorization.value('Q')==shared.fnv(read(root/'frozen-inputs.json')) and len(authorization.array('u1'))>0;authorization.finish()
    freeze=json.loads(read(root/'frozen-inputs.json'));entries=freeze['files'] if isinstance(freeze,dict) else freeze
    for entry in entries:
        assert hashlib.sha256(read(entry['path'])).hexdigest()==entry['sha256'].lower()
        if 'snapshot' in entry:assert hashlib.sha256(read(entry['snapshot'])).hexdigest()==entry['sha256'].lower()
    assert protocol.value('Q')==shared.fnv(read(root/'approved-protocol.md'))
    assert protocol.value('Q')==shared.fnv(envelope(root/'parent-checkpoint.bin').data)
    assert protocol.value('Q')==shared.fnv(envelope(root/'holding-parent.bin').data)
    assert [protocol.value('i') for _ in range(12)]==[6,4,41,4,336,22,80,320,320,7360,73600,120]
    near([protocol.value('f') for _ in range(7)],np.float32([.008,.005,20,.20,.12,.195,.115]),0,0)
    cap=protocol.value('Q');assert cap in (64*1024**2,128*1024**2)
    assert [protocol.value('i') for _ in range(11)]==OFFSETS;protocol.finish()
    if not native['hard_fault']:assert native['evidence_bytes_before_close_report']+16384<=cap
    LAYOUT=json.loads(read(root/'layout.json'));r,task=records(root,LAYOUT);weights=load_weights(root)
    endpoints=table(root/'probe-started/controls.csv');substeps=table(root/'probe-started/substeps.csv');SUB_INDEX=substeps['index']
    csv_endpoints(endpoints,r)
    assert np.all(r['world']['current']==0)
    near(r['world']['time'],r['world']['substeps'].astype(float)*float(np.float32(.005)),0,0)
    assert len(substeps['data'])==int(r['accepted'].sum())
    errors={key:0. for key in ('observation','NN','command','filter_angle','filter_rate','energy_identity')}
    groups=[];by_key={}
    for key in dict.fromkeys(tuple(int(row[x]) for x in ('group','case','backend')) for row in r):
        assert key[0] in (0,1) and key[2] in (0,1)
        assert 0<=key[1]<(4 if key[0]==0 else 22) and (key[0]==0 or key[2]==1)
        select=(r['group']==key[0])&(r['case']==key[1])&(r['backend']==key[2]);rows=r[select]
        subselect=(col(substeps,'group')==key[0])&(col(substeps,'case')==key[1])&(col(substeps,'backend')==key[2])
        cpu_rows=by_key.get((0,key[1],0)) if key[0]==0 and key[2]==1 else None
        report=check_group(rows,substeps['data'][subselect],weights,errors,cpu_rows);groups.append(report);by_key[key]=rows
    assert sum(g['physical_controls'] for g in groups)==native['controls']
    assert sum(g['attempted_substeps'] for g in groups)==native['attempted_substeps']
    assert sum(g['accepted_substeps'] for g in groups)==native['accepted_substeps']
    # Matched-command endpoint comparison includes ALL40actual/requested torques.
    maxima={'pose':0.,'motor':0.,'energy':0.};paired=0
    for pose in range(4):
        left,right=by_key.get((0,pose,0)),by_key.get((0,pose,1))
        if left is None or right is None:continue
        count=min(len(left),len(right));a,b=left[:count],right[:count]
        valid=(a['failure']==0)&(b['failure']==0)&(a['reason']==0)&(b['reason']==0);a,b=a[valid],b[valid]
        f32_equal(a['command'],b['command']);f32_equal(a['base'],b['base']);f32_equal(a['observation'],b['observation'])
        tolerance=np.array([2e-5]*11+[2e-7]*3+[2e-5]*7)
        diff=abs(a['world']['body'][:,:21].astype(float)-b['world']['body'][:,:21]);assert np.all(diff<=tolerance)
        maxima['pose']=max(maxima['pose'],float(diff[:,:3].max(initial=0)))
        for field in ('held','rates','last_torque','actual','requested'):
            maxima['motor']=max(maxima['motor'],near(a['world'][field],b['world'][field],2e-5,0))
        ea,eb=a['world']['energy'],b['world']['energy'];error=abs(ea-eb)
        assert np.all(error<=1e-12+1e-4*np.maximum(abs(ea),abs(eb)));maxima['energy']=max(maxima['energy'],float(error.max(initial=0)))
        if count==41 and len(a)==41:paired+=1
    assert paired==native['precheck_completed_cases']
    point=table(root/'probe-started/same-observation.csv')
    # Header names are source-authoritative; positional widths are fixed.
    assert point['data'].shape[1]==59
    for row in point['data']:
        pose,tick=map(int,row[:2]);obs=row[2:43];cpu,gpu,command_cpu,command_gpu=np.split(row[43:],4)
        near(cpu,shared.forward(weights,obs[None,:])[0],2e-5,2e-6)
        near(command_cpu,np.tanh(cpu),2e-5,0);near(command_gpu,np.tanh(gpu),2e-5,0)
        near(cpu,gpu,2e-5,0);near(command_cpu,command_gpu,2e-5,0)
        for backend in (0,1):
            rr=by_key.get((0,pose,backend))
            if rr is not None and tick<len(rr):f32_equal(obs,rr[tick]['observation'])
    cases=table(root/'probe-started/cases.csv');summaries=[]
    for row in cases['data']:
        case=int(row[cases['index']['case']]);rr=by_key[(1,case,1)];metrics=descriptive(rr)
        for name,index in cases['index'].items():
            expected=metrics[name];actual=row[index]
            if expected is None:assert np.isnan(actual),(case,name,actual)
            else:near(actual,expected,.005 if 'deg' in name else 3e-6,5e-5)
        summaries.append(metrics)
    assert len(summaries)==native['probe_completed_cases']
    if not native['hard_fault']:
        assert paired==4 and len(summaries)==22
        assert all(g['last_control']==320 or g['behavior_reason'] in (1,4,5) for g in groups if g['group']==1)
    elif (root/'probe-started/first-failure.bin').exists():
        failure=envelope(root/'probe-started/first-failure.bin');assert failure.value('I')==1
        group,case,tick=[failure.value('i') for _ in range(3)]
        frame_dtype=prep.types(LAYOUT)[0];active=prep.raw_record(failure,frame_dtype);other=prep.raw_record(failure,frame_dtype);failure.finish()
        assert group in (0,1) and 0<=case<(4 if group==0 else 22) and tick>=0
        if active['failure']:
            assert (group,case,0 if group==0 else 1) in by_key
            failed_rows=[rows for key,rows in by_key.items() if key[:2]==(group,case)]
            assert any(rows[-1]['failure']==active['failure'] for rows in failed_rows)
    else:assert native['hard_fault'] in ('Final evidence/time cap exceeded','Evidence flush failed','Case summary write failed'),'Hard physical stop lacks first failure evidence'
    present={c['case'] for c in summaries};unrun=[i for i in range(22) if (1,i,1) not in by_key or len(by_key[(1,i,1)])==1]
    unfinished=[descriptive(rows) for key,rows in by_key.items() if key[0]==1 and key[1] not in present and len(rows)>1]
    for item in summaries:
        baseline=next((c for c in summaries if c['pose']==item['pose'] and c['offset_degrees']==0),None)
        item['same_pose_zero_case']=baseline['case'] if baseline else None
        item['late_course_error_delta_from_zero_deg']=(item['last4_velocity_course_error_deg']-baseline['last4_velocity_course_error_deg']) if baseline and item['last4_velocity_course_error_deg'] is not None and baseline['last4_velocity_course_error_deg'] is not None else None
    read(__file__)
    for module in (prep,prior,ref,shared,prior.previous,prior.retained,cuda_reference,cuda_reference.contract):read(module.__file__)
    return {'schema':1,'audit_status':'verified_hard_stop' if native['hard_fault'] else 'verified_descriptive_probe',
        'audit_passed':True,'full_probe_completed':not bool(native['hard_fault']) and len(summaries)==22,
        'qualified':False,'descriptive_only':True,'native_hard_fault':native['hard_fault'],'process_cap_exceeded':native['process_seconds']>120,
        'validation_scope':'Parser and synthetic tests only before the first authorized probe; actual probe rows are audited on execution.',
        'new_physical_controls_executed':0,'new_GPU_kernels_executed':0,'new_optimizer_steps':0,
        'recorded_controls':native['controls'],'recorded_attempted_substeps':native['attempted_substeps'],
        'recorded_accepted_substeps':native['accepted_substeps'],'unrun_cases':unrun,'unfinished_cases':unfinished,
        'matched_precheck_cases':paired,'maximum_errors':errors,'matched_backend_maxima':maxima,
        'cases':summaries,'groups':groups,'native_result':native,'protocol_checksum':checksum,
        'evidence_scope':'All recorded accepted5ms poses/endpoints, requests, NN/actions, filters and energy ledgers are reconstructed; force integration is not replayed. This is not continuous-swimming qualification.',
        'limitations':['A hard stop or unavailable case is not a pass.','The20control head band is sampled, not continuous between endpoints.'],
        'input_sha256':dict(sorted(prep.HASHES.items()))}


def self_test():
    heading=np.array([1.,0,0]);assert abs(course(heading,[0,0,-1])-90)<1e-4
    assert abs(course(heading,[0,0,1])+90)<1e-4
    body=ref.initial_body(np.arange(4));ke,gap=kinematics(body)
    assert np.all(ke==0) and np.all(gap>.01)
    moving=body.copy();moving[:,11]=.008;k,_=kinematics(moving);faster=moving.copy();faster[:,11]*=2;k2,_=kinematics(faster)
    near(k2,4*k,1e-18,1e-12)
    for backend in (0,1):assert np.all(guard_causes(body,np.float32(-2),backend)==0)
    assert np.all(k>0)
    return {'passed':True,'synthetic_cases':4,'kinetic_scaling_checked':True,'new_physical_controls':0,'GPU_kernels':0,'optimizer_steps':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',nargs='?',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:
        assert args.run is not None
        root=args.run.resolve()
        try:result=verify(root)
        except (AssertionError,ValueError,KeyError,EOFError) as error:
            native=json.loads(read(root/'result.json'))
            if not native.get('hard_fault'):raise
            result={'schema':1,'audit_status':'hard_stop_evidence_incomplete','audit_passed':False,
                'full_probe_completed':False,'qualified':False,'descriptive_only':True,
                'native_hard_fault':native['hard_fault'],'recorded_controls':native['controls'],
                'recorded_attempted_substeps':native['attempted_substeps'],
                'recorded_accepted_substeps':native['accepted_substeps'],
                'limitations':[f'Complete independent row verification was unavailable: {type(error).__name__}: {error}',
                    'No missing or unrun case is classified as a pass.'],
                'new_physical_controls_executed':0,'new_GPU_kernels_executed':0,'new_optimizer_steps':0,
                'input_sha256':dict(sorted(prep.HASHES.items()))}
        output=root/'independent-result-verification.json'
        with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({'audit_status':result['audit_status'],'full_probe_completed':result['full_probe_completed'],
                          'recorded_controls':result['recorded_controls'],'qualified':False,'new_physical_controls':0}))
