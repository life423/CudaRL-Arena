from pathlib import Path
import hashlib,json,shutil,struct,os,subprocess
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
local=Path(__file__).parent
pres=r/'build/review-preservation/2026-11-04-before-frozen-brake'
assert not pres.exists(),'Preparation identity exists'
pres.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
files=list((r/'assets').rglob('*'))+list((r/'build/ppo').rglob('*'))
files=[p for p in files if p.is_file()]
manifest=[{'path':str(p.relative_to(r)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
(pres/'prior-evidence-hashes.json').write_text(json.dumps(manifest,indent=2))
unrelated=[]
for h in json.loads((r/'build/review-preservation/2026-11-04-before-balanced-scale/manifest.json').read_text()):
 if h['path'].replace('\\','/').startswith(('live/','assets/ppo-smooth-exploration-v1')):
  assert sha(r/h['path'])==h['sha256'];unrelated.append(h)
(pres/'unrelated-hashes.json').write_text(json.dumps(unrelated,indent=2))
shutil.copy2(r/'CMakeLists.txt',pres/'CMakeLists.txt')
(pres/'preexisting-diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=r))
baseline=r/'assets/ppo-four-axis-checkpoints/update-336-episode64/ppo-update-336.bin'
assert sha(baseline)=='810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0'
B=r/'assets/ppo-four-axis-checkpoints/balanced-hold-scale-336to344/B'
assert sha(B/'ppo-append-update-344.bin')=='8D705FC82A2B12D465E4FDCD48BC8D881EA0710192B030B262E4A26A6C6AF2B2'
assert sha(B/'protocol.bin')=='B96BC59DD6F134946AA0FFCF5C371834347B54EAED7B6BB90322EDBCA48D8458'
class Reader:
 def __init__(self,b):self.b=b;self.at=0
 def v(self,t):x=struct.unpack_from('<'+t,self.b,self.at)[0];self.at+=struct.calcsize('<'+t);return x
 def array(self,dtype='<f4'):
  n=self.v('Q');dt=np.dtype(dtype);x=np.frombuffer(self.b,dtype=dt,count=n,offset=self.at);self.at+=n*dt.itemsize;return x
def payload(p):
 b=p.read_bytes();magic,n,c=struct.unpack_from('<QQQ',b);assert magic==0x31504B434C524443 and n==len(b)-24;return b[24:]
c=Reader(payload(B/'ppo-append-update-344.bin'))
assert c.v('I')==0x35415050 and c.v('I')==5;c.v('Q');assert c.v('I')==1247037243 and c.v('i')==344
for n in [6,6,3,9,4,41,4,2532,2433]:assert c.v('I')==n
c.v('f');c.array();params=c.array();assert len(params)==2532
batch=r/'build/ppo/balanced-hold-scale-336to344-v1/B/batch-336.bin'
evidence={x['path'].replace('\\','/'):x for x in json.loads((batch.parent.parent/'evidence-manifest.json').read_text())}
assert sha(batch)==evidence['B/batch-336.bin']['sha256']
b=Reader(payload(batch));assert b.v('I')==1;b=Reader(b.array('u1').tobytes());assert [b.v('I') for _ in range(4)]==[1,40960,41,4]
obs=b.array().reshape(40960,41);x=obs[640*32:641*32][np.array([(i//2)%2==1 for i in range(32)])].copy()
y=x.copy();at=0
for layer,(ni,no) in enumerate([(41,32),(32,32),(32,4)]):
 w=params[at:at+ni*no].reshape(no,ni);at+=ni*no;bias=params[at:at+no];at+=no;y=y@w.T+bias
 if layer<2:y=np.tanh(y)
np.savetxt(pres/'numerical-fixture.csv',np.c_[x,y],fmt='%.9g',delimiter=',')
p=r/'CMakeLists.txt';txt=p.read_text();needle='add_executable(motor_compare tools/motor_compare.cpp)';assert txt.count(needle)==1
txt=txt.replace(needle,'''add_executable(frozen_brake_probe tools/frozen_brake_probe.cpp)
target_link_libraries(frozen_brake_probe PRIVATE creature_physics)
target_include_directories(frozen_brake_probe PRIVATE "${CMAKE_SOURCE_DIR}/src/nn")
project_warnings(frozen_brake_probe)

'''+needle)
p.write_text(txt,newline='\n');shutil.copy2(local/'frozen_brake_probe.cpp',r/'tools/frozen_brake_probe.cpp')
print(json.dumps({'preserved_evidence_files':len(manifest),'preserved_evidence_bytes':sum(x['bytes'] for x in manifest),'unrelated_hashes_verified':len(unrelated),'source_changes':['CMakeLists.txt','tools/frozen_brake_probe.cpp'],'physical_controls':0,'optimizer_updates':0}))
