"""Independent saved-data audit for food-free continuous native pond PPO.

No simulator, learner, optimizer or viewer is invoked. Geometry, observations,
actions, probabilities, rewards and qualification are reconstructed from bytes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import inspect
import platform
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import verify_current_swim as reference
import verify_current_swim_v2 as previous
import verify_gait_pond as retained

N, HORIZON, PARAMETERS = 4096, 320, 16
CONTROL_CAP, SUBSTEP_CAP = 86811440, 868114400
SIGMA, LIMIT = float(np.float32(.20)), float(np.float32(.75))
RX, RZ, IX, IZ = map(lambda x: float(np.float32(x)), (.20,.12,.195,.115))
PI = float(np.float32(np.pi))
HASHES, VALIDATED = reference.HASHES, reference.VALIDATED
read, js, rows, near, envelope = reference.read, reference.js, reference.rows, reference.near, reference.envelope
digest, payload_checksum = previous.file_digest, previous.payload_checksum
PARTS = (('pre-filter',12),('post-body',23),('joint-latent',4),('reward',1))
ACTIVE_AUDITOR_SOURCES=None
LEGACY_CPU_CASE=retained.cpu_case


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported evidence JSON value: {type(value).__name__}')


def json_text(value,**kwargs):
    return json.dumps(value,default=json_default,**kwargs)


def file_fnv(path):
    return reference.shared.fnv(read(path))


def horizontal_forward(body):
    value = reference.rotate(np.asarray(body)[...,3:7], [1,0,0])
    value[...,1] = 0
    length = np.linalg.norm(value,axis=-1,keepdims=True)
    return np.where(length>1e-8,value/np.maximum(length,1e-30),[1,0,0])


def fma32(a,b,c):
    """Exact binary32 fused multiply/add via the wider binary64 significand."""
    return np.float32(np.float64(a)*np.float64(b)+np.float64(c))


def cross32(a,b):
    return np.stack((fma32(a[...,1],b[...,2],-a[...,2]*b[...,1]),
        fma32(a[...,2],b[...,0],-a[...,0]*b[...,2]),
        fma32(a[...,0],b[...,1],-a[...,1]*b[...,0])),axis=-1)


def rotate32(q,value):
    q=np.asarray(q,np.float32);value=np.asarray(value,np.float32)
    axis=q[...,:3];inner=cross32(axis,value)+value*q[...,3,None]
    return fma32(cross32(axis,inner),np.float32(2),value)


def qmul32(a,b):
    a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
    return np.stack((
        fma32(-a[...,2],b[...,1],fma32(a[...,1],b[...,2],fma32(a[...,0],b[...,3],a[...,3]*b[...,0]))),
        fma32(a[...,2],b[...,0],fma32(a[...,1],b[...,3],fma32(-a[...,0],b[...,2],a[...,3]*b[...,1]))),
        fma32(a[...,2],b[...,3],fma32(-a[...,1],b[...,0],fma32(a[...,0],b[...,1],a[...,3]*b[...,2]))),
        fma32(-a[...,2],b[...,2],fma32(-a[...,1],b[...,1],fma32(-a[...,0],b[...,0],a[...,3]*b[...,3])))),axis=-1)


def axis32(axis,angle):
    """Registered CUDA half-angle polynomial, for accepted |joint angle|<=1.

    Coefficients and fused order are independently transcribed from the PTX
    already embedded in the frozen executable, without executing a kernel.
    NumPy's system trigonometry may choose the adjacent binary32 value, which
    is amplified when the resulting near-zero COM is normalized.
    """
    half=np.asarray(angle,np.float32)*np.float32(.5)
    assert np.all(abs(half)<=.5)
    def bits(value):return np.array(value,dtype=np.uint32).view(np.float32)
    square=half*half
    sine_poly=fma32(bits(0xb94d4153),square,bits(0x3c0885e4))
    sine_poly=fma32(sine_poly,square,bits(0xbe2aaaa8))
    sine=fma32(sine_poly,fma32(square,half,np.float32(0)),half)
    cosine_poly=fma32(square,bits(0x37cbac00),bits(0xbab607ed))
    cosine_poly=fma32(cosine_poly,square,bits(0x3d2aaabb))
    cosine_poly=fma32(cosine_poly,square,bits(0xbeffffff))
    cosine=fma32(cosine_poly,square,np.float32(1))
    return np.concatenate((sine[...,None]*np.float32(axis),cosine[...,None]),axis=-1)


def sensor_com32(body):
    """Independent arithmetic reconstruction of the native binary32 COM sensor.

    Near the origin, normalizing a nanometre COM rounding difference amplifies
    it. Keep this sensor's registered float/FMA order; double COM remains the
    independent movement/acceptance measurement. No dynamics are evaluated.
    """
    body=np.asarray(body,np.float32);f=np.float32;lengths=f([.020,.026,.026])
    centers=[body[...,:3]*f(.04)];qs=[body[...,3:7]]
    for b in (1,2):
        p=b-1;anchor=centers[p]+rotate32(qs[p],f([-lengths[p]*f(.5),0,0]))
        yaw=qmul32(qs[p],axis32([0,1,0],body[...,7+p*2]))
        q=qmul32(yaw,axis32([0,0,1],body[...,8+p*2]));qs.append(q)
        centers.append(anchor+rotate32(q,f([-lengths[b]*f(.5),0,0])))
    pi=f(math.pi);r=f(.0015)
    masses=[f(4)*pi*(lengths[0]*f(.5))*f(.0064)*f(.008)/f(3)*f(1000),
        (pi*r*r*lengths[1])*f(1000),
        (pi*r*r*lengths[2]+f(.0097)*lengths[2]*f(.0001))*f(1000)]
    mass=f(0);weighted=np.zeros_like(centers[0])
    for m,center in zip(masses,centers):mass+=m;weighted=fma32(center,m,weighted)
    com_model=weighted*(f(1)/(mass*f(.04)))
    return com_model*f(.04)


def qmul_cpu32(a,b):
    a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
    return np.stack((a[...,3]*b[...,0]+a[...,0]*b[...,3]+a[...,1]*b[...,2]-a[...,2]*b[...,1],
        a[...,3]*b[...,1]-a[...,0]*b[...,2]+a[...,1]*b[...,3]+a[...,2]*b[...,0],
        a[...,3]*b[...,2]+a[...,0]*b[...,1]-a[...,1]*b[...,0]+a[...,2]*b[...,3],
        a[...,3]*b[...,3]-a[...,0]*b[...,0]-a[...,1]*b[...,1]-a[...,2]*b[...,2]),axis=-1)


def axis_cpu32(axis,angle):
    half=np.asarray(angle,np.float32)*np.float32(.5)
    return np.concatenate((np.sin(half)[...,None]*np.float32(axis),np.cos(half)[...,None]),axis=-1)


def motion_cpu32(body):
    """Non-contracted binary32 p::shape/motor::motion for recorded CPU inputs."""
    body=np.asarray(body,np.float32);f=np.float32;lengths=f([.020,.026,.026]);rot=reference.rotate_float32
    centers=[body[...,:3]*f(.04)];qs=[body[...,3:7]];anchors=[];yaw_axes=[];pitch_axes=[]
    for b in (1,2):
        p=b-1;anchor=centers[p]+rot(qs[p],f([-lengths[p]*f(.5),0,0]));anchors.append(anchor)
        yaw_axes.append(rot(qs[p],[0,1,0]));yaw=qmul_cpu32(qs[p],axis_cpu32([0,1,0],body[...,7+p*2]))
        pitch_axes.append(rot(yaw,[0,0,1]));q=qmul_cpu32(yaw,axis_cpu32([0,0,1],body[...,8+p*2]));qs.append(q)
        centers.append(anchor+rot(q,f([-lengths[b]*f(.5),0,0])))
    pi=f(math.pi);r=f(.0015)
    masses=[f(4)*pi*(lengths[0]*f(.5))*f(.0064)*f(.008)/f(3)*f(1000),
        (pi*r*r*lengths[1])*f(1000),(pi*r*r*lengths[2]+f(.0097)*lengths[2]*f(.0001))*f(1000)]
    mass=f(0);center=np.zeros_like(centers[0]);velocity=np.zeros_like(center);basis=np.eye(3,dtype=np.float32)
    for b in range(3):
        speed=np.zeros_like(center)
        for dof in range(10):
            if dof<3:column=basis[dof]
            elif dof<6:column=np.cross(basis[dof-3],centers[b]-centers[0])
            elif (dof-6)//2<b:
                joint=(dof-6)//2;axis=pitch_axes[joint] if dof%2 else yaw_axes[joint]
                column=np.cross(axis,centers[b]-anchors[joint])
            else:column=np.zeros_like(center)
            speed+=column*body[...,11+dof,None]
        mass+=masses[b];center+=centers[b]*masses[b];velocity+=speed*masses[b]
    return center*(f(1)/(mass*f(.04))),velocity*(f(1)/mass)


def requested_observations_cpu32(pre,start_model,heading,controls):
    pre=np.asarray(pre,np.float32);f=np.float32;center,velocity=motion_cpu32(pre)
    heading=np.asarray(heading,np.float32);displacement=(center-np.float32(start_model))*f(.04)
    dot=displacement[...,0]*heading[0]+displacement[...,1]*heading[1]+displacement[...,2]*heading[2]
    lane=heading*dot[...,None]-displacement;q=pre[...,3:7].copy();q[...,:3]*=f(-1)
    rot=reference.rotate_float32;out=np.zeros(pre.shape[:-1]+(44,),dtype=np.float32)
    out[...,:3]=rot(q,lane)*(f(1)/f(.002))
    out[...,3:6]=rot(q,heading*f(.008))*(f(1)/f(.02))
    out[...,6:9]=rot(q,heading);out[...,9:12]=rot(q,[0,1,0])
    out[...,12:15]=rot(q,velocity)*(f(1)/f(.02))
    out[...,15:18]=rot(q,pre[...,14:17])*(f(1)/f(4))
    out[...,18:22]=pre[...,7:11];out[...,22:26]=pre[...,17:21]/f(12)
    out[...,26:30]=pre[...,25:29];out[...,30:34]=pre[...,21:25]
    out[...,34]=np.clip(f(1)-np.asarray(controls,dtype=np.float32)/f(40),0,1)
    out[...,35]=1;out[...,36:40]=pre[...,29:33]/f(12)
    return out


def requested_cpu_case(directory,case,pose,offset,controls,frozen,means,gain,hold=False):
    """Keep old trace/scoring checks; reconstruct its CPU NN arithmetic precisely."""
    original=reference.observations
    initial=rows(directory/'initial.csv')[case] if (directory/'initial.csv').exists() else None
    def observer(pre,poses,world_flow,clock):
        reset=pre[0,:21].copy();reset[1]=np.float32(-2)
        start=motion_cpu32(reset)[0]
        heading=reference.rotate_float32(reset[3:7],[1,0,0])
        length=np.sqrt(heading[0]*heading[0]+heading[1]*heading[1]+heading[2]*heading[2])
        heading*=np.float32(1)/length
        if initial is not None:
            actual_start=np.float32([initial['start_com_'+a] for a in 'xyz'])
            actual_heading=np.float32([initial['heading_'+a] for a in 'xyz'])
            near(actual_start,start,2e-6,2e-6);near(actual_heading,heading,2e-7,2e-6)
            start,heading=actual_start,actual_heading
        return requested_observations_cpu32(pre,start,heading,clock)
    reference.observations=observer
    try:return LEGACY_CPU_CASE(directory,case,pose,offset,controls,frozen,means,gain,hold)
    finally:reference.observations=original


def articulated_corners(body):
    """All 24 registered OBB corners, including the tail fin-containing box."""
    body=np.asarray(body,float)
    centers=[body[...,:3]*reference.SCALE];quaternions=[body[...,3:7]]
    lengths=list(map(lambda x:float(np.float32(x)),(.020,.026,.026)))
    heights=[float(np.float32(.0064)),float(np.float32(.0015)),
             float(np.float32((np.float32(.0015)*2+np.float32(.0097))/2))]
    widths=list(map(lambda x:float(np.float32(x)),(.008,.0015,.0015)))
    for link in (1,2):
        parent=link-1
        anchor=centers[parent]+reference.rotate(quaternions[parent],[-lengths[parent]/2,0,0])
        q=reference.qmul(quaternions[parent],reference.axis_rotation([0,1,0],body[...,7+parent*2]))
        q=reference.qmul(q,reference.axis_rotation([0,0,1],body[...,8+parent*2]))
        centers.append(anchor+reference.rotate(q,[-lengths[link]/2,0,0]));quaternions.append(q)
    signs=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    return np.concatenate([centers[b][...,None,:]+reference.rotate(quaternions[b][...,None,:],
        signs*np.array([lengths[b]/2,heights[b],widths[b]])) for b in range(3)],axis=-2)


def boundary_values(body):
    box=articulated_corners(body)
    active=np.max((box[...,0]/RX)**2+(box[...,2]/RZ)**2,axis=-1)
    inset=np.max((box[...,0]/IX)**2+(box[...,2]/IZ)**2,axis=-1)
    return active,inset,RZ*(1-np.sqrt(active))


def directional_gap(box,direction):
    direction=np.asarray(direction,float)
    dx,dz=direction[...,0,None]/RX,direction[...,2,None]/RZ
    x,z=box[...,0]/RX,box[...,2]/RZ
    a=dx*dx+dz*dz;b=2*(x*dx+z*dz);c=x*x+z*z-1
    assert np.all(c<=2e-6) and np.all(a>0)
    disc=b*b-4*a*c
    assert np.all(disc>=-1e-9)
    return np.min((-b+np.sqrt(np.maximum(0,disc)))/(2*a),axis=-1)


def raw_sensors(body):
    body=np.asarray(body,float);center,velocity=reference.motion(body)
    forward=horizontal_forward(body);side=np.stack((-forward[...,2],np.zeros_like(forward[...,0]),forward[...,0]),axis=-1)
    box=articulated_corners(body)
    sensor_center=sensor_com32(body)
    normal=sensor_center/np.float32([np.float32(RX)*np.float32(RX),1,np.float32(RZ)*np.float32(RZ)])
    normal[...,1]=0
    length=np.sqrt(fma32(normal[...,0],normal[...,0],normal[...,2]*normal[...,2]))[...,None]
    radius2=fma32(sensor_center[...,0],sensor_center[...,0],sensor_center[...,2]*sensor_center[...,2])[...,None]
    normal=np.where((radius2>np.float32(1e-12))&(length>np.float32(1e-8)),
                    normal*(np.float32(1)/np.maximum(length,np.float32(1e-30))),0)
    value=np.empty(body.shape[:-1]+(8,))
    for i,direction in enumerate((forward,side,-side)):
        value[...,i]=np.clip(directional_gap(box,direction)/float(np.float32(.072)),0,4)
    value[...,3]=np.sum(normal*forward,axis=-1);value[...,4]=np.sum(normal*side,axis=-1)
    value[...,5]=body[...,15]/4
    value[...,6]=np.sum(velocity*forward,axis=-1)/float(np.float32(.008))
    value[...,7]=np.sum(velocity*side,axis=-1)/float(np.float32(.008))
    assert np.isfinite(value).all();return value


def features(raw):
    raw=np.asarray(raw,float);value=np.empty_like(raw)
    value[...,0]=1;value[...,1]=1/(1+raw[...,0])
    value[...,2]=1/(1+raw[...,1])-1/(1+raw[...,2])
    value[...,3:5]=raw[...,3:5];value[...,5]=np.tanh(8*raw[...,5])
    value[...,6]=np.tanh(2*(1-raw[...,6]));value[...,7]=np.tanh(raw[...,7]);return value


def action_latents(coefficients,backbone,raw):
    coefficients=np.asarray(coefficients,float);backbone=np.asarray(backbone,float)
    value=backbone.copy();terms=np.tanh(coefficients).reshape(coefficients.shape[:-1]+(2,8))
    value[...,[0,2]]+=LIMIT*np.tanh(np.sum(terms*features(raw)[...,None,:],axis=-1))
    return np.where(np.all(coefficients==0,axis=-1,keepdims=True),backbone,value)


def observations(body,context,fixed_depth_model,controls):
    body,context=np.asarray(body,float),np.asarray(context,float)
    center,velocity=reference.motion(body);q=body[...,3:7];forward=horizontal_forward(body)
    error=np.zeros_like(center);error[...,1]=np.asarray(fixed_depth_model)*reference.SCALE-center[...,1]
    value=np.zeros(body.shape[:-1]+(41,))
    value[...,:3]=reference.rotate(q,error,True)/float(np.float32(.002))
    value[...,3:6]=reference.rotate(q,forward*float(np.float32(.008)),True)/float(np.float32(.02))
    value[...,6:9]=reference.rotate(q,forward,True)
    value[...,9:12]=reference.rotate(q,[0,1,0],True)
    value[...,12:15]=reference.rotate(q,velocity,True)/float(np.float32(.02))
    value[...,15:18]=reference.rotate(q,body[...,14:17],True)/4
    value[...,18:22]=body[...,7:11];value[...,22:26]=body[...,17:21]/12
    value[...,26:30]=context[...,8:12];value[...,30:34]=context[...,:4]
    value[...,34]=np.maximum(0,1-np.minimum(controls,1000000)/40)
    value[...,35]=1;value[...,36:40]=context[...,4:8]/12
    return value


def measured(body,fixed_depth_model):
    body=np.asarray(body,float);center,velocity=reference.motion(body)
    forward=horizontal_forward(body);side=np.stack((-forward[...,2],np.zeros_like(forward[...,0]),forward[...,0]),axis=-1)
    up=reference.rotate_float32(body[...,3:7],[0,1,0],True)
    upright=reference.rotate_float32(body[...,3:7],[0,1,0])
    return {'center':center,'velocity':velocity,
        'forward':np.sum(velocity*forward,axis=-1),'side':np.sum(velocity*side,axis=-1),
        'depth':center[...,1]-np.asarray(fixed_depth_model)*reference.SCALE,
        'roll':np.degrees(np.arctan2(-up[...,2],up[...,1])),
        'pitch':np.degrees(np.arctan2(up[...,0],np.linalg.norm(up[...,1:3],axis=-1))),
        'tilt':np.degrees(np.arccos(np.clip(upright[...,1],-1,1)))}


def rewards(before,after,depth,substeps,boundary):
    values=measured(after,depth)
    vf=(values['forward']-float(np.float32(.008)))/float(np.float32(.008))
    vs=values['side']/float(np.float32(.008));dz=values['depth']/float(np.float32(.001))
    rate=1-vf*vf-.25*vs*vs-.05*dz*dz-.05*(values['roll']/10)**2-.01*(values['pitch']/10)**2
    dt=np.asarray(substeps,dtype=np.float32)*np.float32(.005)
    return dt*rate-6250*(np.asarray(after,float)[...,21]-np.asarray(before,float)[...,21])-4*np.asarray(boundary)


def log_probability(means,latent):
    return np.sum(-.5*((np.asarray(latent,float)-means)/SIGMA)**2-math.log(SIGMA)-.5*math.log(2*math.pi),axis=-1)


def leave_one_out(returns,scale):
    returns=np.asarray(returns,float);assert returns.shape==(N,) and math.isfinite(scale) and scale>0
    grouped=returns.reshape(-1,16);baseline=(grouped.sum(axis=0)-grouped)/255
    return ((grouped-baseline)/scale).reshape(-1)


def case_spec(case,withheld=False):
    assert 0<=case<(8 if withheld else 16)
    position=np.array([0.,float(np.float32(-.08)),0.]);pose=3 if case%2 else 0
    if case<4:yaw=(PI/4 if withheld else 0)+case*PI/2
    else:
        k=case-4 if withheld else (case-4)//3;g=0 if withheld else (case-4)%3
        phi=(PI/4 if withheld else 0)+k*PI/2
        position[[0,2]]=[RX*float(np.float32(.55))*math.cos(phi),RZ*float(np.float32(.55))*math.sin(phi)]
        normal=position*np.array([1/RX**2,0,1/RZ**2])
        yaw=math.atan2(-normal[2],normal[0])+((1 if k%2 else -1)*PI/4 if withheld else (g-1)*PI/3)
    return position,yaw,pose


def expected_initial_body(case,withheld=False):
    center,yaw,pose=case_spec(case,withheld);body=np.pad(reference.initial_body(pose),(0,2))
    body[3:7]=reference.axis_rotation([0,1,0],yaw)
    actual=reference.motion(body)[0];body[:3]+=(center-actual)/reference.SCALE
    return body


def turn_measurement(body):
    f=horizontal_forward(body);yaw=np.arctan2(f[...,2],f[...,0]);unwrapped=np.empty(len(yaw));unwrapped[0]=yaw[0]
    anchor=float(yaw[0]);positive=negative=raw=0.;event=PI/4
    for k in range(1,len(yaw)):
        delta=float(yaw[k]-yaw[k-1])
        while delta>PI:delta-=2*PI
        while delta< -PI:delta+=2*PI
        unwrapped[k]=unwrapped[k-1]+delta;raw+=abs(delta)
        while unwrapped[k]-anchor>=event:positive+=event;anchor+=event
        while unwrapped[k]-anchor<= -event:negative-=event;anchor-=event
    return {'positive_turn_rad':positive,'negative_turn_rad':negative,'net_unwrapped_yaw_rad':float(unwrapped[-1]-unwrapped[0]),
            'yaw_range_rad':float(np.ptp(unwrapped)),'raw_absolute_yaw_rad':raw}


def validate_envelopes(root,helper,phase):
    paths=sorted(p for p in root.rglob('*.bin') if 'frozen-causal-inputs' not in p.parts)
    directory=Path('build/continuous-pond-preparation-2026-11-23/offline-audits');directory.mkdir(parents=True,exist_ok=True)
    listing=directory/('envelope-paths-'+phase+'.txt')
    listing.write_text(''.join(str(p.resolve())+'\n' for p in paths),encoding='utf-8')
    read(helper);process=subprocess.run([str(helper.resolve()),str(listing.resolve())],capture_output=True,text=True,check=True,
        creationflags=0x08000000 if os.name=='nt' else 0)
    report=json.loads(process.stdout)
    assert report['passed'] and report['files']==len(paths) and report['physical_controls']==report['optimizer_steps']==0
    VALIDATED.update(p.resolve() for p in paths);return report


def read_policy(path,protocol,inherited):
    frozen,holding,gain,parent,hold=inherited;r=envelope(path)
    assert [r.value('I'),r.value('I'),r.value('Q'),r.value('Q'),r.value('Q')]==[0x31505054,1,protocol,parent,hold]
    assert [r.value('I') for _ in range(10)]==[6,1,3,4,1,41,8,4,336,16]
    n,update,age=[r.value('i') for _ in range(3)]
    assert n==N and 0<=update<=64 and 0<=age<=1024 and age%4==0
    assert np.array_equal(r.array(size=2532),frozen) and np.array_equal(r.array(size=2),holding) and r.value('f')==gain
    means=r.array(size=16).copy();assert [r.value('f'),r.value('f')]==[SIGMA,LIMIT]
    a=reference.shared.Reader(r.array('u1'));r.finish()
    near([a.value('f') for _ in range(5)],np.float32([.005,.9,.999,.00001,.5]),0,0)
    assert a.value('i')==age;first=a.array(size=16);second=a.array(size=16);a.finish()
    assert np.isfinite(means).all() and np.isfinite(first).all() and np.all(np.isfinite(second)&(second>=0))
    if update==0:assert age==0 and np.all(means==0) and np.all(first==0) and np.all(second==0)
    return means,{'update':update,'Adam_steps':age,'sha256':HASHES[str(path.resolve())],'means':means.tolist()}


def dtype_from_layout(layout,key,specs):
    fields=layout[key]
    return np.dtype({'names':[s[0] for s in specs],'formats':[s[1] for s in specs],
                     'offsets':[fields[s[0]] for s in specs],'itemsize':fields['size']})


def types(layout):
    counts=dtype_from_layout(layout,'counts',[
        ('controls','<u8'),('attempted_substeps','<u8'),('accepted_substeps','<u8'),('reward_sum','<f8'),
        ('seconds','<f8'),('last_substeps','<i4'),('reason','<i4'),('done','u1'),('truncated','u1')])
    task=dtype_from_layout(layout,'task',[
        ('controls','<u8'),('last_world_control','<u8'),('maximum_controls','<i4'),('reason','<i4'),
        ('fixed_depth_model','<f4'),('maximum_depth_m','<f4'),('maximum_tilt_deg','<f4'),('minimum_clearance_m','<f4'),
        ('last_com_model',('<f4',3)),('elapsed_seconds','<f8'),('reward_sum','<f8'),('path_length_m','<f8'),
        ('done','u1'),('boundary_terminal','u1')])
    return counts,task


def open_batch(directory,protocol,layout,expected_means=None):
    r=envelope(directory/'meta.bin');binding=payload_checksum(directory/'meta.bin')
    assert [r.value('I'),r.value('I'),r.value('Q')]==[0x31425054,1,protocol]
    n,horizon,ticks,purpose,index,cases,withheld,fault=[r.value('i') for _ in range(8)]
    assert n>0 and 0<ticks<=horizon and cases==(8 if withheld else 16) and fault==0
    means=r.array(size=16);latent=r.array(size=n*16).reshape(n,16);logp=r.array(size=n);returns=r.array(size=n)
    if expected_means is not None:assert np.array_equal(means,expected_means)
    ctype,ttype=types(layout);assert r.value('I')==ctype.itemsize;counts=r.array(ctype,n)
    assert r.value('I')==layout['world_size']==696
    initial=r.array(previous.WORLD_TYPE,n);final=r.array(previous.WORLD_TYPE,n)
    assert r.value('I')==ttype.itemsize
    initial_task=r.array(ttype,n);final_task=r.array(ttype,n);r.finish()
    probability_error=near(logp,log_probability(means,latent),1e-4,0)
    sampled=purpose<2
    expected=means+np.float32(SIGMA)*previous.normal(0x4e23b197 if purpose==0 else 0xa715c2d3,index,
                    np.arange(n*16,dtype=np.uint32).reshape(n,16)) if sampled else np.broadcast_to(means,(n,16))
    rng_error=near(latent,expected,2e-6,2e-6)
    manifest=js(directory/'manifest.json');assert manifest['schema']==1 and manifest['meta_checksum']==binding
    assert [manifest[k] for k in ('worlds','horizon','ticks','purpose')]==[n,horizon,ticks,purpose]
    assert len(manifest['parts'])==4;parts={};mapped=[]
    for field,((name,columns),entry) in enumerate(zip(PARTS,manifest['parts'])):
        path=directory/(name+'.bin');digest(path);assert path.resolve() in VALIDATED
        mm=np.memmap(path,dtype='u1',mode='r');mapped.append(mm)
        magic,size,checksum=struct.unpack_from('<QQQ',mm)
        assert magic==0x31504b434c524443 and size==len(mm)-24 and size<=512*1024**2
        assert entry==dict(field=field,name=path.name,columns=columns,payload_bytes=size,payload_checksum=checksum)
        part=reference.shared.Reader(memoryview(mm)[24:])
        assert [part.value('I'),part.value('I'),part.value('i'),part.value('Q'),part.value('i'),part.value('i'),part.value('i')]==[0x31425054,1,field,binding,n,ticks,columns]
        parts[name]=part.array(size=n*ticks*columns).reshape(ticks,n,columns);part.finish()
    f=envelope(directory/'row-substeps.bin')
    assert [f.value('I'),f.value('I'),f.value('Q'),f.value('i'),f.value('i')]==[0x31465054,1,binding,n,ticks]
    assert payload_checksum(directory/'row-substeps.bin')==manifest['row_substeps_checksum']
    steps=f.array('u1',n*ticks).reshape(ticks,n);f.finish()
    assert np.all(steps<=10)
    return dict(n=n,horizon=horizon,ticks=ticks,purpose=purpose,index=index,cases=cases,withheld=bool(withheld),
        means=means,latent=latent,logp=logp,returns=returns,counts=counts,initial=initial,final=final,
        initial_task=initial_task,task=final_task,parts=parts,steps=steps,mapped=mapped,
        probability_error=probability_error,rng_error=rng_error)


def audit_batch(directory,protocol,layout,frozen,expected_means=None):
    batch=open_batch(directory,protocol,layout,expected_means);n=batch['n'];counts=batch['counts'];steps=batch['steps']
    initial,final,task=batch['initial'],batch['final'],batch['task'];depth=batch['initial_task']['fixed_depth_model']
    for worlds in (initial,final):
        assert np.all(worlds['failure']==0) and np.all(worlds['unsafe']==0) and np.all(worlds['invalid']==0)
        assert np.all(worlds['current']==0) and np.all(abs(worlds['actual'])<=1) and np.isfinite(worlds['requested']).all()
    assert np.all(initial['controls']==0) and np.all(initial['substeps']==0) and np.all(initial['attempted']==0)
    for case in range(batch['cases']):
        actual=initial['body'][case::batch['cases']]
        near(actual,np.broadcast_to(expected_initial_body(case,batch['withheld']),actual.shape),3e-6,3e-6)
    assert np.array_equal(np.sum(steps>0,axis=0),counts['controls'])
    assert np.array_equal(steps.astype(np.uint64).sum(axis=0),counts['accepted_substeps'])
    assert np.array_equal(counts['attempted_substeps'],counts['accepted_substeps'])
    assert np.array_equal(final['substeps'],counts['accepted_substeps']) and np.array_equal(final['attempted'],counts['attempted_substeps'])
    assert np.array_equal(final['controls'],counts['controls']) and np.array_equal(task['controls'],counts['controls'])
    assert np.array_equal(task['last_world_control'],counts['controls'])
    # Every episode has a contiguous actual prefix. Padding contains no controls.
    assert np.array_equal(steps>0,np.arange(batch['ticks'])[:,None]<counts['controls'][None,:])
    assert np.all(counts['controls']>0)
    last=counts['controls'].astype(int)-1;episode=np.arange(n)
    assert np.array_equal(steps[last,episode],counts['last_substeps'])
    assert np.all(counts['accepted_substeps']==(counts['controls']-1)*10+counts['last_substeps'])
    if batch['purpose']<2:
        assert n==N and batch['horizon']==HORIZON and batch['cases']==16 and not batch['withheld']
        assert np.all(counts['done']==1) and np.all(counts['truncated']==0) and np.all(task['done']==1)
        assert np.all((counts['reason']==1)|(counts['reason']==2))
        assert np.all(task['maximum_controls']==320)
        deadline=counts['reason']==2;assert np.all(counts['controls'][deadline]==320)
        assert np.all(counts['last_substeps'][deadline]==10)
    else:
        survived=counts['reason']==0;failed=counts['reason']==1
        assert np.all(survived|failed) and np.all(counts['truncated'][survived]==1) and np.all(counts['done'][survived]==0)
        assert np.all(counts['controls'][survived]==batch['horizon']) and np.all(task['maximum_controls']==0)
        assert np.all(counts['last_substeps'][survived]==10)
        assert np.all(counts['done'][failed]==1) and np.all(counts['truncated'][failed]==0)
    assert np.array_equal(counts['reason'],task['reason']) and np.array_equal(counts['done'],task['done'])
    assert np.array_equal(task['boundary_terminal'],counts['reason']==1)
    assert np.array_equal(final['body'],batch['parts']['post-body'][last,episode])
    errors={key:0. for key in ('action','reward','filter_angle','filter_rate','initial_context')}
    reward_sum=np.zeros(n);seconds=np.zeros(n);peak_depth=np.zeros(n);peak_tilt=np.zeros(n);arc=np.zeros(n)
    min_clear=batch['initial_task']['minimum_clearance_m'].astype(float).copy()
    previous_body=initial['body'].copy();previous_center=reference.motion(previous_body)[0]
    for k in range(batch['ticks']):
        active=steps[k]>0;indices=np.flatnonzero(active)
        for start in range(0,len(indices),4096):
            selected=indices[start:start+4096]
            pre=previous_body[selected];post=batch['parts']['post-body'][k,selected]
            context=batch['parts']['pre-filter'][k,selected];action=batch['parts']['joint-latent'][k,selected]
            reward=batch['parts']['reward'][k,selected,0];substeps=steps[k,selected]
            assert all(np.isfinite(x).all() for x in (pre,post,context,action,reward))
            assert np.all(abs(pre[:,7:11])<=1) and np.all(abs(post[:,7:11])<=1) and np.all(abs(context[:,8:12])<=1)
            assert np.all(post[:,21]>=pre[:,21]) and np.all(post[:,22]>=0)
            if k==0:
                expected_context=np.concatenate((initial['held'][selected],initial['rates'][selected],initial['last_torque'][selected]),axis=1)
                errors['initial_context']=max(errors['initial_context'],near(context,expected_context,0,0))
            obs=observations(pre,context,depth[selected],k)
            base=reference.shared.forward(frozen.astype(float),obs)
            expected_action=action_latents(batch['latent'][selected],base,raw_sensors(pre))
            errors['action']=max(errors['action'],near(action,expected_action,3e-5,3e-6))
            angle=context[:,:4].astype(float).copy();rate=context[:,4:8].astype(float).copy();command=np.tanh(action.astype(float))
            h,omega=float(np.float32(.005)),float(np.float32(2*np.float32(np.pi)))
            for j in range(10):
                mask=substeps>j;rate[mask]+=h*(omega*omega*(command[mask]-angle[mask])-2*omega*rate[mask]);angle[mask]+=h*rate[mask]
            ending=(counts['controls'][selected]==k+1)
            next_angle=final['held'][selected].copy();next_rate=final['rates'][selected].copy()
            if k+1<batch['ticks']:
                next_angle[~ending]=batch['parts']['pre-filter'][k+1,selected[~ending],:4]
                next_rate[~ending]=batch['parts']['pre-filter'][k+1,selected[~ending],4:8]
            errors['filter_angle']=max(errors['filter_angle'],near(next_angle,angle,2e-6,2e-5))
            errors['filter_rate']=max(errors['filter_rate'],near(next_rate,rate,2e-5,2e-5))
            boundary=ending&(counts['reason'][selected]==1)
            errors['reward']=max(errors['reward'],near(reward,rewards(pre,post,depth[selected],substeps,boundary),3e-5,5e-5))
            active_radius,inset_radius,clearance=boundary_values(post)
            assert np.all(active_radius<=1+2e-6)
            assert np.all(inset_radius[~boundary]<1+2e-6) and np.all(inset_radius[boundary]>=1-2e-6)
            metrics=measured(post,depth[selected]);center=metrics['center']
            peak_depth[selected]=np.maximum(peak_depth[selected],abs(metrics['depth']))
            peak_tilt[selected]=np.maximum(peak_tilt[selected],metrics['tilt'])
            min_clear[selected]=np.minimum(min_clear[selected],clearance)
            arc[selected]+=np.linalg.norm((center-previous_center[selected])[:,[0,2]],axis=1)
            reward_sum[selected]+=reward;seconds[selected]+=substeps.astype(np.float32)*np.float32(.005)
            previous_body[selected]=post;previous_center[selected]=center
        inactive=~active
        for value in batch['parts'].values():assert np.all(value[k,inactive]==0)
    near(counts['reward_sum'],reward_sum,1e-7,1e-7);near(task['reward_sum'],reward_sum,1e-7,1e-7)
    near(counts['seconds'],seconds,1e-12,1e-12);near(task['elapsed_seconds'],seconds,1e-12,1e-12)
    # Learning episodes always use the registered fixed 16-second denominator.
    # Deterministic long captures carry a duration-normalized diagnostic receipt;
    # that receipt is never the selection metric or PPO training data.
    normalization_seconds=16 if batch['purpose']<2 else batch['horizon']*.05
    near(batch['returns'],np.float32(reward_sum/normalization_seconds),2e-6,2e-6)
    near(task['maximum_depth_m'],peak_depth,2e-7,2e-5);near(task['maximum_tilt_deg'],peak_tilt,.003,2e-5)
    near(task['minimum_clearance_m'],min_clear,2e-7,2e-5);near(task['path_length_m'],arc,1e-6,2e-5)
    table=rows(directory/'episodes.csv');assert len(table)==n
    for i,row in enumerate(table):
        assert [row[key] for key in ('episode','case','withheld')]==[i,i%batch['cases'],int(batch['withheld'])]
        for key in ('actual_controls','attempted_substeps','accepted_substeps','last_substeps','reason','done','truncated'):
            assert row[key]==counts['controls' if key=='actual_controls' else key][i]
        near([row['old_logp'],row['reward_sum'],row['return'],row['elapsed_seconds']],
             [batch['logp'][i],reward_sum[i],batch['returns'][i],seconds[i]],1e-7,1e-10)
        near([row[f'latent{j}'] for j in range(16)],batch['latent'][i],1e-8,1e-10)
    report={'controls':int(counts['controls'].sum()),'substeps':int(counts['attempted_substeps'].sum()),
        'worlds':n,'horizon':batch['horizon'],'purpose':batch['purpose'],'index':batch['index'],'means':batch['means'].tolist(),
        'accepted_substeps':int(counts['accepted_substeps'].sum()),'task_boundary_episodes':int(np.sum(counts['reason']==1)),
        'deadline_episodes':int(np.sum(counts['reason']==2)),'externally_truncated_episodes':int(np.sum(counts['truncated'])),
        'stored_rows':n*batch['ticks'],'actual_rows':int(np.sum(steps>0)),
        'probability_error':batch['probability_error'],'rng_error':batch['rng_error'],'maximum_errors':errors,
        'return_normalization_seconds':normalization_seconds,'learning_return_definition':batch['purpose']<2,
        'optimizer_training_data':batch['purpose']==1}
    timing=js(directory/'timing.json')
    assert [timing[k] for k in ('worlds','horizon','stored_ticks','actual_controls','actual_substeps','fault','optimizer_steps')]==[
        n,batch['horizon'],batch['ticks'],report['controls'],report['substeps'],0,0]
    for key in ('collection_readback_validation_seconds','evidence_seconds'):
        assert math.isfinite(timing[key]) and timing[key]>=0
    report['timing']=timing
    return batch,report


def score_evaluation(directory,batch):
    assert batch['purpose']>=2 and batch['n']==batch['cases']
    native=rows(directory/'summary.csv');windows=rows(directory/'motion-windows.csv');assert len(native)==batch['n']
    details=[];window_index=0
    for i,row in enumerate(native):
        count=int(batch['counts']['controls'][i]);steps=batch['steps'][:count,i]
        body=np.concatenate((batch['initial']['body'][i:i+1],batch['parts']['post-body'][:count,i]),axis=0)
        depth=batch['initial_task']['fixed_depth_model'][i];metrics=measured(body,depth)
        delta=np.diff(metrics['center'][:,[0,2]],axis=0);length=np.linalg.norm(delta,axis=1)
        arc=float(length.sum());seconds=float((steps.astype(np.float32)*np.float32(.005)).astype(float).sum())
        speeds=[]
        for end in range(240,count+1,160):
            start=end-160;duration=float((steps[start:end].astype(np.float32)*np.float32(.005)).astype(float).sum())
            distance=float(length[start:end].sum());speed=distance/duration
            w=windows[window_index];window_index+=1
            assert w['case']==i and w['start_s']==start*.05 and w['end_s']==end*.05
            near([w['actual_seconds'],w['horizontal_COM_arc_m'],w['mean_speed_mm_s']],
                 [duration,distance,speed*1000],2e-5,5e-5)
            assert bool(w['passed'])==(speed>=.003);speeds.append(speed)
        counts=batch['counts'][i];task=batch['task'][i]
        survived=bool(count==batch['horizon'] and counts['reason']==0 and not counts['done'] and counts['truncated'] and not task['done'] and task['maximum_controls']==0)
        minimum=min(speeds) if speeds else 0.;mean=arc/seconds
        spans=np.ptp(metrics['center'][:,[0,2]],axis=0);turn=turn_measurement(body)
        peak_depth=float(abs(metrics['depth'][1:]).max());peak_tilt=float(metrics['tilt'][1:].max())
        gates={'uninterrupted':survived,'depth':peak_depth<=.001,'tilt':peak_tilt<=10,'mean_speed':mean>=.006,
               'window_speed':minimum>=.003,'span_x':spans[0]>=(.20 if batch['withheld'] else .12),
               'span_z':spans[1]>=(.12 if batch['withheld'] else .06),
               'actual_turn':not batch['withheld'] or turn['positive_turn_rad']+abs(turn['negative_turn_rad'])>=2*PI}
        passed=all(gates.values());assert bool(row['passed'])==passed and bool(row['survived'])==survived
        assert [row[k] for k in ('case','withheld','actual_controls','attempted_substeps','accepted_substeps','task_done','external_truncated','reason')]==[
            i,int(batch['withheld']),count,int(counts['attempted_substeps']),int(counts['accepted_substeps']),int(counts['done']),int(counts['truncated']),int(counts['reason'])]
        measured_values={'elapsed_seconds':seconds,'horizontal_COM_arc_m':arc,'mean_speed_mm_s':mean*1000,
            'min_window_speed_mm_s':minimum*1000,'x_span_m':float(spans[0]),'z_span_m':float(spans[1]),
            'maximum_depth_mm':peak_depth*1000,'maximum_tilt_deg':peak_tilt,**turn}
        for key,value in measured_values.items():near(row[key],value,.003 if key=='maximum_tilt_deg' else 2e-4,5e-5)
        details.append({'case':i,'passed':passed,'gates':gates,'metrics':measured_values,'complete_motion_windows':len(speeds)})
    assert window_index==len(windows)
    return {'passes':sum(item['passed'] for item in details),'cases':details}


def assess_requested_swim(directory,frozen,legacy_unused,gain,long_windows=False,recovery=False):
    """Read the frozen retention CSV schema; reconstruct unlisted diagnostics."""
    summary=rows(directory/'summary.csv');initial=rows(directory/'initial.csv');windows=rows(directory/'windows.csv')
    assert len(summary)==len(initial)
    compared=('steady_mm_s','RMSE_mm_s','lateral_mm','depth_mm','tilt_deg','path_deg','work_J_per_m')
    expected_columns={'case','pose','offset_mm','controls','movement_passed','return','recovery_passed',
        'mean_pitch1_deg','mean_pitch3_deg','mean_tail_vertical_bend_deg','centered_joint_RMS_deg','gait_passed',*compared}
    details=[];count=0;passing=set();gait_passing=set()
    for case,native in enumerate(summary):
        assert set(native)==expected_columns and native['case']==case
        pose,controls=int(native['pose']),int(native['controls']);offset=native['offset_mm']/1000
        body,values,reward,errors=retained.cpu_case(directory,case,pose,offset,controls,frozen,legacy_unused,gain)
        metrics,_=reference.swim_metrics(body,pose);gates=reference.swim_gates(metrics);movement=all(gates.values())
        for key in compared:near(native[key],metrics[key],.003 if key in ('tilt_deg','path_deg') else .0003,3e-5)
        assert bool(native['movement_passed'])==movement
        gait=retained.gait_window(body,80,controls);retained.compare_gait(native,gait)
        episode_return=float(reward.astype(float).sum()/(controls*.05)-float(np.float32(.05))/.05*gait['bias_cost']-(not movement))
        near(native['return'],episode_return,2e-5,3e-5)
        recovered=movement and (not recovery or offset==0 or bool(np.all(abs(values['signed_depth'][-160:])<abs(offset)*.5)))
        assert bool(native['recovery_passed'])==recovered
        fixture=initial[case];assert fixture['case']==case and fixture['pose']==pose and fixture['controls']==controls
        near(fixture['offset_mm'],np.float32(offset)*np.float32(1000),1e-6,1e-6)
        near(fixture['arena_half_width_m'],3. if controls==6000 else float(np.float32(.6)),1e-10,1e-10)
        near(fixture['arena_half_depth_m'],float(np.float32(.4)),1e-10,1e-10)
        center,_=reference.motion(reference.initial_body(pose))
        near([fixture['start_com_'+axis] for axis in 'xyz'],center/.04,2e-6,2e-6)
        chosen=[r for r in windows if r['case']==case];assert len(chosen)==(controls//80 if long_windows else 0)
        center,velocity=reference.motion(body);heading=values['heading']
        for row,end in zip(chosen,range(80,controls+1,80)):
            start=40 if end==80 else end-80;delta=center[end]-center[start]
            forward=float(np.dot(delta,heading));side=delta-heading*forward
            speed=forward/((end-start)*float(np.float32(.05)))*1000
            rmse=float(np.sqrt(np.mean(np.sum((velocity[start+1:end+1]-heading*float(np.float32(.008)))**2,axis=-1)))*1000)
            value={'steady_mm_s':speed,'RMSE_mm_s':rmse,'lateral_mm':float(values['lane'][:end+1].max()*1000),
                'depth_mm':float(values['depth'][:end+1].max()*1000),'tilt_deg':float(values['tilt'][:end+1].max()),
                'path_deg':float(np.degrees(np.arctan2(np.linalg.norm(side[[0,2]]),forward))),
                'work_J_per_m':float((body[end,21]-body[start,21])/max(forward,1e-9))}
            for key in compared:near(row[key],value[key],.003 if key in ('tilt_deg','path_deg') else .0003,3e-5)
            assert row['pose']==pose and row['end_s']==end*.05
            passed=all(reference.swim_gates(value).values());assert bool(row['movement_passed'])==passed
            if passed:passing.add((pose,end//20))
            assert bool(row['gait_scored'])==(end>=160)
            if end>=160:
                window_gait=retained.gait_window(body,end-80,end);retained.compare_gait(row,window_gait)
                if window_gait['passed']:gait_passing.add((pose,end//20))
        details.append({'case':case,'pose':pose,'offset_m':offset,'controls':controls,'movement_passed':movement,
            'recovery_passed':bool(recovered),'movement_gates':gates,'metrics':metrics,'gait':gait,'errors':errors})
        count+=controls
    return {'cases':details,'controls':count,'substeps':count*10,'movement_passes':sum(x['movement_passed'] for x in details),
        'recovery_passes':sum(x['recovery_passed'] for x in details),'passing_window_keys':sorted(passing),
        'gait_window_keys':sorted(gait_passing),'gait_passes':sum(x['gait']['passed'] for x in details),
        'forward_diagnostic_source':'Reconstructed from complete saved COM trace; summary CSV has no forward_mm column.'}


def verify_retention(directory,frozen,means,gain,full):
    # These explicit old requested SWIM and HOLD paths bypass the new head.
    assert np.asarray(means).shape==(16,) and np.isfinite(means).all()
    # The archived scorer's eight-value argument is an unused gait-head input.
    # Its trace decoder is replaced below with the exact requested-SWIM branch;
    # sixteen steering means never enter either old requested branch.
    legacy_unused=np.zeros(8)
    original=retained.swim_latents
    original_case=retained.cpu_case
    retained.swim_latents=lambda coefficient,backbone:np.asarray(backbone,float)
    retained.cpu_case=requested_cpu_case
    try:
        result={'short':assess_requested_swim(directory/'short',frozen,legacy_unused,gain),
                'long60':assess_requested_swim(directory/'long60',frozen,legacy_unused,gain,True),
                'original-holds':retained.assess_holds(directory/'original-holds',frozen,legacy_unused,gain)}
        if full:
            result.update({'recovery':assess_requested_swim(directory/'recovery',frozen,legacy_unused,gain,False,True),
                'long300':assess_requested_swim(directory/'long300',frozen,legacy_unused,gain,True),
                'withheld-holds':retained.assess_holds(directory/'withheld-holds',frozen,legacy_unused,gain),
                'endurance-holds':retained.assess_holds(directory/'endurance-holds',frozen,legacy_unused,gain)})
    finally:
        retained.swim_latents=original;retained.cpu_case=original_case
    short=result['short'];long60=result['long60'];holds=result['original-holds']
    assert [(x['pose'],x['controls'],x['offset_m']) for x in short['cases']]==[(i,160,0) for i in range(4)]
    assert [(x['pose'],x['controls'],x['offset_m']) for x in long60['cases']]==[(0,1200,0),(3,1200,0)]
    near([x['offset_m'] for x in holds['cases']],[0,0,-.0005,-.0005,.0005,.0005],1e-10,1e-7)
    assert [x['pose'] for x in holds['cases']]==[0,3]*3 and all(x['controls']==1280 for x in holds['cases'])
    expected60={(p,s) for p in (0,3) for s in range(8,61,4)}
    keys60={tuple(x) for x in long60['passing_window_keys']}
    actual={'short_passes':short['movement_passes'],'original_holds':holds['passes'],
            'long60_windows':len(keys60),'long60_retained':len(keys60&expected60)}
    if full:
        recovery=result['recovery'];long300=result['long300'];withheld=result['withheld-holds'];endurance=result['endurance-holds']
        assert [x['pose'] for x in recovery['cases']]==[0,0,0] and all(x['controls']==1280 for x in recovery['cases'])
        near([x['offset_m'] for x in recovery['cases']],[0,-.0005,.0005],1e-10,1e-7)
        assert [(x['pose'],x['controls']) for x in long300['cases']]==[(0,6000),(3,6000)]
        assert [x['pose'] for x in withheld['cases']]==[1,2]*3 and all(x['controls']==1280 for x in withheld['cases'])
        near([x['offset_m'] for x in withheld['cases']],[0,0,-.00025,-.00025,.00025,.00025],1e-10,1e-7)
        assert [(x['pose'],x['controls'],x['offset_m']) for x in endurance['cases']]==[(0,6000,0),(3,6000,0)]
        expected300={(p,s) for p in (0,3) for s in range(8,301,4)};keys300={tuple(x) for x in long300['passing_window_keys']}
        actual.update(recovery_passes=recovery['recovery_passes'],long300_windows=len(keys300),
                      long300_retained=len(keys300&expected300),withheld_holds=withheld['passes'],endurance_holds=endurance['passes'])
    count=sum(r['controls'] for r in result.values());assert count==(46240 if full else 10720)
    native=js(directory/'retention.json');assert native['full']==full and native['controls']==count and native['substeps']==count*10
    for key,value in actual.items():assert native[key]==value
    passed=actual['short_passes']==4 and actual['original_holds']==6 and actual['long60_retained']==28
    if full:passed=passed and actual['recovery_passes']==3 and actual['long300_retained']==148 and actual['withheld_holds']==6 and actual['endurance_holds']==2
    return {'passed':passed,'controls':count,'substeps':count*10,'totals':actual,'details':result}


def verify_gate(root,protocol,frozen,gain):
    native=js(root/'gate-result.json');assert native['closed'] and native['matched_commands'] and not native['closed_loop_trajectory_parity_claimed']
    r=envelope(root/'gate-started/states.bin');version,stepbytes,worldbytes,infobytes=[r.value('I') for _ in range(4)]
    assert version==1 and worldbytes==696 and stepbytes==native['step_bytes'] and infobytes==native['step_info_bytes']
    csv=rows(root/'gate-started/states.csv');records={};actual_controls=actual_substeps=0;maxima=dict(position=0.,quaternion=0.,angle=0.,linear_velocity=0.,other_velocity=0.,torque=0.,energy=0.)
    world_dtype=np.dtype({'names':['world','energy'],'formats':[previous.WORLD_TYPE,('<f8',6)],'offsets':[0,640],'itemsize':stepbytes})
    bodykeys=['x','y','z','qx','qy','qz','qw']+[f'j{j}' for j in range(4)]+[f'v{j}' for j in range(10)]
    energykeys=['signed_work','positive_work','fluid_loss','numerical_loss','external_work','energy_error']
    for row in csv:
        key=tuple(int(row[k]) for k in ('group','case','backend','control'))
        assert tuple(r.value('i') for _ in range(4))==key and key not in records
        blob=r.data[r.at:r.at+stepbytes];r.at+=stepbytes
        record=np.frombuffer(blob,dtype=world_dtype,count=1)[0];world=record['world'];records[key]=record
        near([row[k] for k in bodykeys],world['body'][:21],0,0)
        near([row[k] for k in energykeys],record['energy'],0,0)
        assert row['attempted_substeps']==world['attempted'] and row['accepted_substeps']==world['substeps']
        assert world['failure']==world['invalid']==world['unsafe']==0 and np.all(abs(world['actual'])<=1)
        if key[3]:
            actual_controls+=1
            prior=records[key[:3]+(key[3]-1,)]['world'];actual_substeps+=int(world['attempted']-prior['attempted'])
            assert row['advanced']==1 and row['last_substeps']==10 and row['boundary_terminal']==0
            assert world['controls']==key[3] and world['substeps']==key[3]*10 and world['attempted']==key[3]*10
    r.finish();assert native['controls']==actual_controls and native['substeps']==actual_substeps
    for (group,case,backend,control),record in records.items():
        if backend==1 and group<2:
            old=records[(group,case,0,control)]
            for field in previous.WORLD_TYPE.names:assert np.array_equal(record['world'][field],old['world'][field]),(group,case,control,field)
            assert np.array_equal(record['energy'],old['energy'])
        if backend==2 and (group,case,1,control) in records:
            cpu=records[(group,case,1,control)];a,b=cpu['world']['body'],record['world']['body']
            for name,sl,tolerance in [('position',slice(0,3),2e-5),('quaternion',slice(3,7),2e-5),('angle',slice(7,11),2e-5),
                                      ('linear_velocity',slice(11,14),2e-7),('other_velocity',slice(14,21),2e-5)]:
                maxima[name]=max(maxima[name],near(b[sl],a[sl],tolerance,0))
            maxima['torque']=max(maxima['torque'],near(record['world']['actual'],cpu['world']['actual'],2e-5,0))
            e=abs(record['energy']-cpu['energy']);bound=1e-12+1e-4*np.maximum(abs(record['energy']),abs(cpu['energy']))
            assert np.all(e<=bound);maxima['energy']=max(maxima['energy'],float(e.max()))
    commands=rows(root/'gate-started/same-observation.csv');inference=[]
    for row in commands:
        obs=np.array([row[f'observation{j}'] for j in range(41)]);raw=np.array([row[f'raw{j}'] for j in range(8)])
        base=reference.shared.forward(frozen.astype(float),obs[None,:])[0]
        for backend in ('cpu','gpu'):
            recorded=np.array([row[f'{backend}_base{j}'] for j in range(4)])
            near(recorded,base,2e-5,2e-6)
            expected=recorded*gain if row['group']==1 else recorded
            near([row[f'{backend}_latent{j}'] for j in range(4)],expected,2e-7,2e-6)
            near([row[f'{backend}_command{j}'] for j in range(4)],np.tanh(expected),2e-7,2e-6)
        for name in ('base','latent','command'):
            near([row[f'cpu_{name}{j}'] for j in range(4)],[row[f'gpu_{name}{j}'] for j in range(4)],2e-5,0)
        inference.append(tuple(int(row[k]) for k in ('group','case','control')))
    assert len(inference)==len(set(inference))
    if native['passed']:
        assert actual_controls==2480 and actual_substeps==24800 and native['completed_cases']==26
        assert len(csv)==2542 and len(commands)==1040
        marker=envelope(root/'gate-passed.bin')
        assert [marker.value('I'),marker.value('I'),marker.value('I'),marker.value('Q'),marker.value('Q'),marker.value('Q')]==[0x31504754,1,1,2480,24800,protocol];marker.finish()
    else:
        assert not (root/'gate-passed.bin').exists() and actual_controls<=2480 and actual_substeps<=24800
        read(root/'gate-started/first-failure.bin')
    return {'passed':bool(native['passed']),'controls':actual_controls,'substeps':actual_substeps,'states':len(csv),
            'same_observation_rows':len(commands),'maximum_errors':maxima,'closed_loop_trajectory_parity_claimed':False}


def read_protocol(root):
    r=envelope(root/'protocol.bin');protocol=payload_checksum(root/'protocol.bin')
    assert [r.value('I'),r.value('I'),r.value('Q')]==[0x31525054,1,file_fnv(root/'approved-protocol.md')]
    parent,hold=r.value('Q'),r.value('Q')
    assert [r.value('i') for _ in range(26)]==[6,1,3,4,1,41,8,4,336,16,N,HORIZON,64,1024,4,1024,2480,10720,46240,16,1200,8,36000,CONTROL_CAP,SUBSTEP_CAP,900]
    near([r.value('f') for _ in range(22)],np.float32([.20,.75,.005,.5,.2,.01,.20,.12,.195,.115,.008,.001,10,.006,.003,.12,.06,.20,.12,4,16,6250]),0,0)
    assert [r.value('I') for _ in range(3)]==[0x4e23b197,0xa715c2d3,0x0316412d]
    assert [r.value('Q'),r.value('Q')]==[20*1024**3,8*1024**3]
    for withheld,count in ((False,16),(True,8)):
        for case in range(count):
            position=[r.value('f') for _ in range(3)];yaw=r.value('f');pose=r.value('i');identifier=r.value('i');flag=r.value('I')
            center,expected_yaw,expected_pose=case_spec(case,withheld)
            near(position,center,2e-8,2e-6);near(yaw,expected_yaw,1e-6,1e-6)
            assert pose==expected_pose and identifier==case and flag==int(withheld)
    r.finish();return protocol,parent,hold


def verify_preparation(root,frozen,gain,layout):
    prepared=js(root/'prepared.json');assert prepared['prepared'] and prepared['physical_controls']==prepared['optimizer_steps']==0
    assert prepared['worlds']==4096 and prepared['controls_cap']==CONTROL_CAP and prepared['substeps_cap']==SUBSTEP_CAP
    details={};_,tasktype=types(layout)
    for prefix in ('prepare','verified'):
        directory=root/(prefix+'-zero');r=envelope(directory/'initialization.bin')
        assert [r.value('I'),r.value('I'),r.value('i'),r.value('I')]==[0x314e5054,1,N,696]
        world=r.array(previous.WORLD_TYPE,N);assert r.value('I')==tasktype.itemsize;task=r.array(tasktype,N)
        obs=r.array(size=N*41).reshape(N,41);raw=r.array(size=N*8).reshape(N,8);base=r.array(size=N*4).reshape(N,4)
        latent=r.array(size=N*16).reshape(N,16);logp=r.array(size=N);r.finish()
        assert all(np.all(world[k]==0) for k in ('controls','substeps','attempted','failure','invalid','unsafe'))
        assert np.all(task['controls']==0) and np.all(task['maximum_controls']==320) and np.all(task['done']==0)
        context=np.concatenate((world['held'],world['rates'],world['last_torque']),axis=1)
        expected=observations(world['body'],context,task['fixed_depth_model'],0)
        observation_error=near(obs,expected,3e-5,3e-5);sensor_error=near(raw,raw_sensors(world['body']),3e-5,3e-5)
        nn_error=near(base,reference.shared.forward(frozen.astype(float),obs.astype(float)),2e-5,2e-6)
        rng_error=near(latent,np.float32(SIGMA)*previous.normal(0x4e23b197,0,np.arange(N*16,dtype=np.uint32).reshape(N,16)),2e-6,2e-6)
        probability_error=near(logp,log_probability(np.zeros(16),latent),1e-4,0)
        for case in range(16):
            actual=world['body'][case::16]
            near(actual,np.broadcast_to(expected_initial_body(case),actual.shape),3e-6,3e-6)
        bulk=js(directory/'bulk-policy-zero.json')
        assert bulk['passed'] and [bulk[k] for k in ('worlds','minibatch_episodes','loss_grid_blocks','checked_rows','checked_last_rows')]==[4096,1024,8,4096,4]
        assert bulk['physical_controls']==bulk['optimizer_steps']==0
        m=envelope(root/(prefix+'-math')/'gradient-zero.bin')
        assert [m.value('I'),m.value('I')]==[0x314d5054,1]
        means=m.array(size=16);u=m.array(size=1024*16).reshape(1024,16);oldp=m.array(size=1024);adv=m.array(size=1024)
        gradients=m.array(size=1024*16).reshape(1024,16);stats=m.array(size=1024*5).reshape(1024,5);m.finish()
        change=log_probability(means,u)-oldp;ratio=np.exp(change);clipped=np.clip(ratio,1-float(np.float32(.2)),1+float(np.float32(.2)))
        active=adv*ratio<=adv*clipped
        expected_gradient=np.where(active[:,None],-adv[:,None]*ratio[:,None]*(u-means)/SIGMA**2/1024,0)
        gradient_error=near(gradients,expected_gradient,2e-7,0)
        entropy=16*(math.log(SIGMA)+.5*(math.log(2*math.pi)+1))
        expected_stats=np.stack((-np.minimum(adv*ratio,adv*clipped),np.zeros(1024),np.full(1024,entropy),ratio-1-change,
                                 (abs(ratio-1)>float(np.float32(.2))).astype(float)),axis=-1)
        stats_error=near(stats,expected_stats,2e-4,0)
        d=envelope(root/(prefix+'-math')/'decode-zero.bin')
        assert [d.value('I'),d.value('I')]==[0x31445054,1] and d.value('f')==gain
        values=d.array(size=96*40).reshape(96,40);d.finish()
        decode_error=near(values[:,28:32],action_latents(values[:,:16],values[:,16:20],values[:,20:28]),2e-6,0)
        near(values[:,32:36],np.float32(gain)*values[:,16:20],0,0);near(values[:,36:40],values[:,16:20],0,0)
        report=js(root/(prefix+'-math')/'math-zero.json');diagnostic=js(root/(prefix+'-math')/'diagnostic.json')
        assert report['passed'] and report['physical_controls']==report['optimizer_steps']==0
        assert diagnostic['max_mean_loss_FD_error']<2e-7
        details[prefix]={'worlds':N,'observation_error':observation_error,'sensor_error':sensor_error,'NN_error':nn_error,
            'rng_error':rng_error,'probability_error':probability_error,'gradient_error':gradient_error,'stats_error':stats_error,
            'decode_error':decode_error,'bulk':bulk}
    read(root/'preflight-passed.txt');return details


def read_ready(root,protocol,phase,controls,substeps):
    name='baseline' if phase==1 else 'calibration'
    bindings=([root/'baseline-result.json',root/'baseline-retention/retention.json',root/'baseline-free/summary.csv',
               root/'baseline-free/manifest.json',root/'gate-passed.bin',root/'gate-result.json'] if phase==1 else
              [root/'calibration-result.json',root/'calibration-report.json',root/'fixed-scale.bin',
               root/'calibration-started/calibration-batch-0/manifest.json',root/'baseline-ready.bin'])
    r=envelope(root/(name+'-ready.bin'))
    assert [r.value('I'),r.value('I'),r.value('Q'),r.value('i')]==[0x31595054,1,protocol,phase]
    assert [r.value('i'),r.value('i')]==[controls,substeps]
    seconds=r.value('d');assert 0<seconds<900
    hashes=r.array('<u8',len(bindings));r.finish();assert list(hashes)==[file_fnv(path) for path in bindings]
    return {'controls':controls,'substeps':substeps,'active_seconds':seconds}


def verify_baseline(root,protocol,inherited,layout):
    frozen,holding,gain,*_=inherited
    retention=verify_retention(root/'baseline-retention',frozen,np.zeros(16),gain,False)
    assert retention['passed']
    batch,body=audit_batch(root/'baseline-free',protocol,layout,frozen,np.zeros(16))
    assert batch['purpose']==2 and batch['n']==16 and batch['horizon']==1200 and batch['index']==0
    scores=score_evaluation(root/'baseline-free',batch)
    controls=2480+retention['controls']+body['controls'];substeps=24800+retention['substeps']+body['substeps']
    native=js(root/'baseline-result.json')
    assert native['fault']=='' and native['controls']==controls and native['substeps']==substeps
    assert native['selection_passes']==scores['passes'] and native['PPO_updates']==native['actor_steps']==0
    ready=read_ready(root,protocol,1,controls,substeps)
    return {'retention':retention,'collection':body,'continuous':scores,'ready':ready}


def verify_calibration(root,protocol,inherited,layout,baseline):
    assert baseline['continuous']['passes']<16
    batch,body=audit_batch(root/'calibration-started/calibration-batch-0',protocol,layout,inherited[0],np.zeros(16))
    assert batch['purpose']==0 and batch['index']==0
    values=batch['returns'].astype(float);grouped=values.reshape(-1,16);means=grouped.mean(axis=0)
    scale=float(np.sqrt(np.mean((grouped-means)**2)))
    assert np.isfinite(scale) and scale>0
    r=envelope(root/'fixed-scale.bin')
    assert [r.value('I'),r.value('I'),r.value('Q')]==[0x31535054,1,protocol]
    native_scale=r.value('d');native_means=np.array([r.value('d') for _ in range(16)]);r.finish()
    near(native_scale,scale,1e-12,1e-12);near(native_means,means,1e-12,1e-12)
    report=js(root/'calibration-report.json');assert report['passed'] and report['episodes']==4096 and report['cases']==16 and report['episodes_per_case']==256
    near(report['fixed_global_residual_RMS'],scale,1e-12,1e-12)
    assert report['ordinary_boundary_terminals']==body['task_boundary_episodes'] and report['deadline_terminals']==body['deadline_episodes']
    controls=baseline['ready']['controls']+body['controls'];substeps=baseline['ready']['substeps']+body['substeps']
    native=js(root/'calibration-result.json');assert native['controls']==controls and native['substeps']==substeps and native['fault']==''
    assert native['PPO_updates']==native['actor_steps']==0 and not native['qualified']
    ready=read_ready(root,protocol,2,controls,substeps)
    return {'collection':body,'fixed_global_residual_RMS':native_scale,'independent_residual_RMS':scale,
            'case_means':means.tolist(),'ready':ready,
            'LOO_own_return_excluded':True,'episodes':4096,'training_started':False}


def write_calibration_release(root,protocol,report_path):
    path=root/'calibration-verified.bin';assert not path.exists()
    files=[report_path,root/'fixed-scale.bin',root/'calibration-ready.bin']
    payload=struct.pack('<IIQQQQ',0x31565054,1,protocol,*[file_fnv(p) for p in files])
    checksum=reference.shared.fnv(payload)
    path.write_bytes(struct.pack('<QQQ',0x31504b434c524443,len(payload),checksum)+payload)
    return digest(path)


def preserve_auditor(phase):
    base=Path('build/continuous-pond-preparation-2026-11-23/offline-audits')
    directory=base/('code-'+phase);attempt=1
    while directory.exists():
        directory=base/('code-'+phase+'-'+str(attempt));attempt+=1
    directory.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(reference.__file__),Path(previous.__file__),Path(retained.__file__),Path(reference.shared.__file__)]
    result=[]
    for source in sources:
        raw=read(source);destination=directory/source.name;destination.write_bytes(raw)
        result.append({'source':str(source.resolve()),'snapshot':str(destination.resolve()),'sha256':HASHES[str(source.resolve())]})
    (directory/'manifest.json').write_text(json_text(result,indent=2)+'\n',encoding='utf-8')
    return result


def sensor_reference_provenance():
    path=Path('build/continuous-pond-preparation-2026-11-23/offline-audits/frozen-study-device.ptx')
    raw=read(path)
    for token in (b'_ZN6cudarl5motor6motionERKNS_7physics9BodyStateE',b'0fB94D4153',b'0f3C0885E4',
                  b'0fBE2AAAA8',b'0f37CBAC00',b'0fBAB607ED',b'0f3D2AAABB',b'0fBEFFFFFF'):
        assert token in raw
    return {'path':str(path.resolve()),'sha256':HASHES[str(path.resolve())],
        'method':'Read-only cuobjdump of frozen EXE PTX; independent binary32/FMA COM and half-angle trig arithmetic; no kernel/body/optimizer execution.',
        'measurement_COM':'Independent double-precision articulated COM remains the acceptance reference.'}


def verify_calibration_marker(root,protocol):
    files=[root/'independent-calibration-verification.json',root/'fixed-scale.bin',root/'calibration-ready.bin']
    marker=envelope(root/'calibration-verified.bin')
    assert [marker.value('I'),marker.value('I'),marker.value('Q')]==[0x31565054,1,protocol]
    assert [marker.value('Q') for _ in range(3)]==[file_fnv(path) for path in files];marker.finish()
    proof=js(files[0]);assert proof['passed'] and not proof['qualified'] and proof['phase']=='calibration'
    assert proof['calibration']['training_started'] is False
    snapshots=[]
    for source in proof['auditor_sources']:
        assert digest(source['snapshot'])==source['sha256']
        snapshots.append({'snapshot':source['snapshot'],'sha256':source['sha256']})
    return {'passed':True,'proof_sha256':HASHES[str(files[0].resolve())],
        'marker_sha256':HASHES[str((root/'calibration-verified.bin').resolve())],'executed_auditor_snapshots':snapshots}


def verify_process_ledger(root,inherited_only):
    records=js(root/'process-execution.json')
    expected=['Gate','Baseline']+([] if inherited_only else ['Calibrate'])+['Train']
    assert [entry['phase'] for entry in records]==expected
    total=0.;details=[]
    for entry in records:
        assert entry['exit_code']==0 and entry['terminated_at_cap'] is False
        assert math.isfinite(entry['seconds']) and entry['seconds']>0
        total+=entry['seconds'];executable=Path(entry['executable'])
        assert executable.name==('turning_physics_gate.exe' if entry['phase']=='Gate' else 'ppo_turning_swim.exe')
        assert digest(executable)==entry['executable_sha256'].lower()
        read(entry['stdout']);read(entry['stderr'])
        details.append({'phase':entry['phase'],'seconds':entry['seconds'],'exit_code':0,
                        'executable_sha256':entry['executable_sha256'].lower()})
    assert total<900
    return {'passed':True,'phases':details,'combined_native_process_seconds':total,
            'maximum_seconds':900,'simulator_or_optimizer_calls_by_verifier':0}


def audit_worker(arguments):
    directory,protocol,layout,frozen,means=arguments
    HASHES.clear();VALIDATED.clear()
    VALIDATED.update((directory/(name+'.bin')).resolve() for name in ('meta','row-substeps')+tuple(name for name,_ in PARTS))
    batch,report=audit_batch(directory,protocol,layout,frozen,means)
    assert batch['purpose']==1
    return batch['returns'].astype(float).copy(),report,dict(HASHES)


def raw_audit_identity(root,protocol):
    functions=(fma32,cross32,rotate32,qmul32,axis32,sensor_com32,horizontal_forward,articulated_corners,
        boundary_values,directional_gap,raw_sensors,features,action_latents,observations,measured,rewards,
        log_probability,case_spec,expected_initial_body,dtype_from_layout,types,open_batch,audit_batch)
    identity={'schema':1,'protocol':protocol,'source':'\n'.join(inspect.getsource(fn) for fn in functions),
        'constants':[N,HORIZON,PARAMETERS,SIGMA,LIMIT,RX,RZ,IX,IZ,PI],
        'Python':platform.python_version(),'NumPy':np.__version__,
        'imports':{str(Path(module.__file__).resolve()):digest(module.__file__) for module in (reference,previous,reference.shared)},
        'layout_sha256':digest(root/'layout.json'),
        'frozen_inputs_manifest_sha256':digest(root/'frozen-inputs.json')}
    return hashlib.sha256(json_text(identity,sort_keys=True).encode()).hexdigest()


def read_batch_receipt(path,protocol,identity,means,update):
    receipt=js(path)
    assert receipt['schema']==1 and receipt['protocol']==protocol and receipt['raw_audit_identity']==identity
    assert receipt['all_actual_control_rows_reconstructed'] is True
    assert receipt['physical_controls_executed']==receipt['optimizer_steps_executed']==0
    assert receipt['update']==update and np.array_equal(np.float32(receipt['expected_means']),means)
    for source in receipt['executed_auditor_sources']:assert digest(source['snapshot'])==source['sha256']
    for source,value in receipt['input_sha256'].items():assert digest(source)==value
    report=receipt['report'];assert report['worlds']==4096 and report['horizon']==320 and report['purpose']==1 and report['index']==update
    assert np.array_equal(np.float32(report['means']),means)
    returns=np.asarray(receipt['returns'],float);assert returns.shape==(4096,) and np.isfinite(returns).all()
    return returns,report


def save_batch_receipt(path,protocol,identity,means,update,returns,report,hashes):
    assert ACTIVE_AUDITOR_SOURCES and not path.exists()
    value={'schema':1,'protocol':protocol,'raw_audit_identity':identity,'update':update,
        'all_actual_control_rows_reconstructed':True,'physical_controls_executed':0,'optimizer_steps_executed':0,
        'expected_means':means.tolist(),'returns':returns.tolist(),'report':report,
        'executed_auditor_sources':ACTIVE_AUDITOR_SOURCES,'input_sha256':dict(sorted(hashes.items()))}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(json_text(value,indent=2,sort_keys=True)+'\n')
    digest(path)


def training_batches(root,protocol,inherited,layout,updates):
    arguments=[];means,_=read_policy(root/'training-started/turning-policy-update-0.bin',protocol,inherited)
    for update in range(1,updates+1):
        directory=root/'training-started'/('training-batch-'+str(update))
        assert all((directory/(name+'.bin')).resolve() in VALIDATED for name in ('meta','row-substeps')+tuple(name for name,_ in PARTS))
        arguments.append((directory,protocol,layout,inherited[0],means.copy()))
        means,_=read_policy(root/'training-started'/('turning-policy-update-'+str(update)+'.bin'),protocol,inherited)
    ram=previous.available_ram_bytes();workers=min(4,os.cpu_count() or 1,updates,max(1,ram//(4*1024**3)))
    memory={'available_RAM_bytes':ram,'workers':workers,'reserve_per_worker_bytes':4*1024**3,'rows_per_chunk':4096}
    print(json_text({'offline_audit_memory_budget':memory}),flush=True)
    results={};identity=raw_audit_identity(root,protocol)
    cache=Path('build/continuous-pond-preparation-2026-11-23/offline-audits/batch-receipts')/(f'{protocol:016x}')/identity
    pending=[]
    for update,argument in enumerate(arguments,1):
        path=cache/('update-'+str(update)+'.json')
        if path.exists():
            results[update]=read_batch_receipt(path,protocol,identity,argument[4],update)
            print(json_text({'reused_hash_bound_full_row_receipt':update,'physical_controls':0,'optimizer_steps':0}),flush=True)
        else:pending.append((update,argument))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs=[(update,argument,executor.submit(audit_worker,argument)) for update,argument in pending]
        for update,argument,job in jobs:
            returns,report,hashes=job.result()
            for path,value in hashes.items():
                assert path not in HASHES or HASHES[path]==value
                HASHES[path]=value
            save_batch_receipt(cache/('update-'+str(update)+'.json'),protocol,identity,argument[4],update,returns,report,hashes)
            results[update]=(returns,report)
            print(json_text({'offline_batch_audit':update,'total':updates,'workers':workers,'physical_controls':0,'optimizer_steps':0}),flush=True)
    return results,memory


def verify_training(root,protocol,inherited,layout,baseline,calibration=None):
    native=js(root/'training-result.json')
    assert native['schema']==1 and native['study']=='continuous_pond_v1' and native['closed'] and native['phase_closed']
    assert native['fault']=='' and not native['runtime_adopted'] and native['retries']==0
    assert native['critic_steps']==native['critic_forward_calls']==0 and native['worlds']==4096
    inherited_only=baseline['continuous']['passes']==16
    assert native['inherited_only']==inherited_only
    updates=int(native['PPO_updates']);assert updates==(0 if inherited_only else updates) and 0<=updates<=64
    if not inherited_only:assert 1<=updates<=64 and calibration is not None
    prior=baseline['ready'] if inherited_only else calibration['ready']
    near(native['prior_active_seconds'],prior['active_seconds'],1e-10,1e-10)
    assert 0<native['active_seconds']+native['prior_active_seconds']<900
    entries=[json.loads(line) for line in read(root/'updates.jsonl').decode().splitlines()]
    assert len(entries)==updates
    means,policy=read_policy(root/'training-started/turning-policy-update-0.bin',protocol,inherited)
    assert np.all(means==0) and policy['update']==policy['Adam_steps']==0
    collections,memory=training_batches(root,protocol,inherited,layout,updates) if updates else ({},{'workers':0})
    controls,substeps=prior['controls'],prior['substeps'];details=[];selection=baseline['continuous']['passes']
    for update,entry in enumerate(entries,1):
        returns,collected=collections[update]
        a=envelope(root/'training-started'/('advantages-'+str(update)+'.bin'))
        assert [a.value('I'),a.value('I'),a.value('i'),a.value('i')]==[0x31415054,1,update,N]
        assert a.value('d')==calibration['fixed_global_residual_RMS']
        assert np.array_equal(a.array(size=N),returns.astype(np.float32))
        near(a.array(size=N),np.float32(leave_one_out(returns,calibration['fixed_global_residual_RMS'])),2e-6,2e-6);a.finish()
        current,current_policy=read_policy(root/'training-started'/('turning-policy-update-'+str(update)+'.bin'),protocol,inherited)
        assert current_policy['update']==update and 4<=current_policy['Adam_steps']-policy['Adam_steps']<=16
        assert (current_policy['Adam_steps']-policy['Adam_steps'])%4==0
        directory=root/'training-started'/('assessment-'+str(update))
        batch,assessment=audit_batch(directory,protocol,layout,inherited[0],current)
        assert batch['purpose']==2 and batch['n']==16 and batch['horizon']==1200
        scored=score_evaluation(directory,batch);selection=scored['passes']
        controls+=collected['controls']+assessment['controls'];substeps+=collected['substeps']+assessment['substeps']
        assert entry['update']==update and entry['worlds']==4096 and entry['minibatch_episodes']==1024
        assert entry['selection_passes']==selection and entry['selection_survivors']==sum(x['gates']['uninterrupted'] for x in scored['cases'])
        assert entry['actor_steps']==current_policy['Adam_steps'] and entry['controls']==controls and entry['substeps']==substeps
        near(entry['means'],current,0,0);assert math.isfinite(entry['KL']) and entry['KL']>=-1e-7
        near(entry['minimum_mean_speed_mm_s'],min(x['metrics']['mean_speed_mm_s'] for x in scored['cases']),2e-4,5e-5)
        near(entry['minimum_window_speed_mm_s'],min(x['metrics']['min_window_speed_mm_s'] for x in scored['cases']),2e-4,5e-5)
        if update<updates:assert selection<16
        details.append({'update':update,'collection':collected,'assessment_collection':assessment,'assessment':scored,
                        'policy':current_policy,'KL':entry['KL']})
        means,policy=current,current_policy
    assert native['actor_steps']==policy['Adam_steps'] and native['final_policy_update']==updates and native['selection_passes']==selection
    withheld=None
    if selection==16:
        directory=root/'withheld-validation';batch,collected=audit_batch(directory,protocol,layout,inherited[0],means)
        assert batch['purpose']==2 and batch['withheld'] and batch['n']==8 and batch['horizon']==36000
        withheld=score_evaluation(directory,batch);withheld['collection']=collected
        controls+=collected['controls'];substeps+=collected['substeps']
        assert native['withheld_continuous_passes']==withheld['passes'] and native['withheld_seconds']==1800
    else:
        assert updates==64 and not (root/'withheld-validation').exists() and native['withheld_continuous_passes'] is None
    retention=verify_retention(root/'final-retention',inherited[0],means,inherited[2],True)
    controls+=retention['controls'];substeps+=retention['substeps']
    rt=retention['totals'];qualification={'selection_passes':selection,'withheld_continuous_passes':withheld['passes'] if withheld else None,
        'withheld_seconds':1800,'short_swim_passes':rt['short_passes'],'long60_retained_windows':rt['long60_retained'],
        'recovery_passes':rt['recovery_passes'],'long300_retained_windows':rt['long300_retained'],
        'original_hold_passes':rt['original_holds'],'withheld_hold_passes':rt['withheld_holds'],'hold300_passes':rt['endurance_holds']}
    for key,value in qualification.items():assert native[key]==value
    qualified=selection==16 and withheld is not None and withheld['passes']==8 and retention['passed']
    assert native['qualified']==qualified
    ablation=None
    if qualified:
        mask=np.zeros(16,dtype=np.float32);mask[[0,8]]=means[[0,8]]
        directory=root/'constant-only-ablation';batch,collected=audit_batch(directory,protocol,layout,inherited[0],mask)
        assert batch['purpose']==4 and batch['n']==16 and batch['horizon']==1200
        ablation=score_evaluation(directory,batch);ablation['collection']=collected
        controls+=collected['controls'];substeps+=collected['substeps']
        assert native['ablation_passes']==ablation['passes']
    else:assert native['ablation_passes'] is None and not (root/'constant-only-ablation').exists()
    assert native['controls']==controls and native['substeps']==substeps and controls<=CONTROL_CAP and substeps<=SUBSTEP_CAP
    assert sorted(int(p.name.split('-')[-1]) for p in (root/'training-started').glob('training-batch-*'))==list(range(1,updates+1))
    return {'qualified':qualified,'controls':controls,'substeps':substeps,'PPO_updates':updates,'Adam_steps':policy['Adam_steps'],
        'final_policy_sha256':policy['sha256'],'assessment_sha256':HASHES[str((root/'training-result.json').resolve())],
        'qualification':qualification,'inherited_only':inherited_only,'offline_audit':memory,'updates':details,
        'retention':retention,'withheld':withheld,'ablation':ablation}


def audit_run(args):
    global ACTIVE_AUDITOR_SOURCES
    root=args.root.resolve();phase=args.phase
    assert root.is_dir()
    snapshot=preserve_auditor(phase)
    ACTIVE_AUDITOR_SOURCES=snapshot
    helper=args.helper.resolve();envelopes=validate_envelopes(root,helper,phase)
    protocol,_,_=read_protocol(root);inherited=reference.parents(root,protocol);layout=js(root/'layout.json')
    initial,initial_info=read_policy(root/'initial-policy/turning-policy-update-0.bin',protocol,inherited)
    assert np.all(initial==0)
    frozen=retained.verify_freeze(root,required=phase!='preparation')
    result={'schema':1,'study':'continuous_pond_v1','phase':phase,'passed':True,'qualified':False,
        'physical_controls_executed_by_verifier':0,'optimizer_steps_executed_by_verifier':0,
        'auditor_sources':snapshot,'envelopes':envelopes,'freeze':frozen,'initial_policy':initial_info}
    result['sensor_reference']=sensor_reference_provenance()
    result['preparation']=verify_preparation(root,inherited[0],inherited[2],layout)
    if phase!='preparation':
        result['gate']=verify_gate(root,protocol,inherited[0],inherited[2])
        if not result['gate']['passed']:
            result['closed']=True;result['outcome']='physics_gate_failed'
            assert phase=='gate','Later phases cannot follow a failed physical gate'
    if phase in ('baseline','calibration','training'):
        assert result['gate']['passed']
        baseline=verify_baseline(root,protocol,inherited,layout);result['baseline']=baseline
    if phase in ('calibration','training') and baseline['continuous']['passes']<16:
        calibration=verify_calibration(root,protocol,inherited,layout,baseline);result['calibration']=calibration
    if phase=='training':
        training=verify_training(root,protocol,inherited,layout,baseline,result.get('calibration'))
        result['training']=training
        for key in ('qualified','controls','substeps','PPO_updates','Adam_steps','final_policy_sha256','assessment_sha256','qualification'):
            result[key]=training[key]
        if not training['inherited_only']:result['calibration_release']=verify_calibration_marker(root,protocol)
        result['process_ledger']=verify_process_ledger(root,training['inherited_only'])
        result['closed']=True
    report_path=args.report.resolve() if args.report else root/('independent-verification.json' if phase=='training' else 'independent-'+phase+'-verification.json')
    assert not report_path.exists(),'Do not rewrite an already bound independent phase proof'
    result['input_sha256']=dict(sorted(HASHES.items()))
    report_path.write_text(json_text(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if phase=='calibration':
        assert report_path==root/'independent-calibration-verification.json'
        result['readiness_marker_sha256']=write_calibration_release(root,protocol,report_path)
    print(json_text({'passed':True,'phase':phase,'qualified':result['qualified'],'report':str(report_path),'sha256':digest(report_path),
                      'physical_controls':0,'optimizer_steps':0}))


def self_test():
    a=np.float32(1+2**-23);b=np.float32(1-2**-23)
    assert fma32(a,b,np.float32(-1))==np.float32(-2**-46)
    angles=np.linspace(-1,1,10001,dtype=np.float32);q=axis32([0,1,0],angles)
    assert np.max(abs(q[:,1]-np.sin(angles.astype(float)/2)))<1.2e-7
    assert np.max(abs(q[:,3]-np.cos(angles.astype(float)/2)))<1.2e-7
    sample={'nested':{'flag':np.bool_(True),'count':np.int64(7)},'array':np.array([1,2],dtype=np.int32)}
    assert json.loads(json_text(sample))=={'nested':{'flag':True,'count':7},'array':[1,2]}
    try:json_text(object())
    except TypeError:pass
    else:raise AssertionError('Unknown evidence JSON objects must be rejected')
    for withheld,count in ((False,16),(True,8)):
        body=np.stack([expected_initial_body(i,withheld) for i in range(count)])
        a,b,_=boundary_values(body);assert np.all(a<1) and np.all(b<1)
        context=np.zeros((count,12));context[:,:4]=body[:,7:11]
        depth=reference.motion(body)[0][:,1]/reference.SCALE
        obs=observations(body,context,depth,0);raw=raw_sensors(body)
        assert obs.shape==(count,41) and np.isfinite(obs).all() and np.isfinite(raw).all()
        base=np.arange(count*4).reshape(count,4)/10
        assert np.array_equal(action_latents(np.zeros((count,16)),base,raw),base)
        changed=action_latents(np.ones((count,16))*.2,base,raw)
        assert np.array_equal(changed[:,[1,3]],base[:,[1,3]]) and np.all(abs(changed-base)<=LIMIT)
        cpu_center,cpu_velocity=motion_cpu32(body)
        precise_center,precise_velocity=reference.motion(body)
        near(cpu_center*reference.SCALE,precise_center,5e-8,2e-6)
        near(cpu_velocity,precise_velocity,5e-8,2e-6)
    values=np.arange(N,dtype=float)/N;adv=leave_one_out(values,1)
    for case in range(16):assert abs(adv[case::16].sum())<1e-10
    still=np.broadcast_to(expected_initial_body(0),(400,23)).copy()
    assert turn_measurement(still)['positive_turn_rad']==0
    # Oscillation smaller than 45 degrees does not masquerade as a turn.
    still[:,3:7]=reference.axis_rotation([0,1,0],.1*np.sin(np.arange(400)))
    score=turn_measurement(still);assert score['positive_turn_rad']==score['negative_turn_rad']==0
    still[:,3:7]=reference.axis_rotation([0,1,0],np.linspace(0,4*np.pi,400))
    score=turn_measurement(still);assert score['positive_turn_rad']+abs(score['negative_turn_rad'])>=3.9*np.pi
    return {'passed':True,'physical_controls':0,'optimizer_steps':0,'initial_cases':24}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--phase',choices=('preparation','gate','baseline','calibration','training'),default='preparation')
    parser.add_argument('--helper',type=Path,default=Path('build/learned-amplitude-preparation-2026-11-22/check-evidence.exe'))
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.self_test:print(json_text(self_test()));return
    assert args.root is not None
    audit_run(args)


if __name__=='__main__':main()
