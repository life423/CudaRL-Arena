from pathlib import Path
import json,hashlib,shutil,subprocess,time
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
local=Path(__file__).parent;pres=r/'build/review-preservation/2026-11-04-before-frozen-brake'
run=r/'build/diagnostics/frozen-brake-B344-v1'
t=time.monotonic()
with (pres/'prepare.log').open('w') as f:
 p=subprocess.run([str(r/'build/Release/frozen_brake_probe.exe'),'--prepare',str(run),str(pres/'numerical-fixture.csv')],cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=60)
assert p.returncode==0,'Zero-step preparation failed; no physical attempt'
shutil.copy2(r/'build/Release/frozen_brake_probe.exe',run/'runner.exe')
shutil.copy2(local/'FROZEN_BRAKE_PROTOCOL.md',run/'frozen-protocol.md')
shutil.copy2(local/'execute_frozen_brake.ps1',r/'tools/run-frozen-brake-probe.ps1')
shutil.copytree(pres,run/'preservation')
sourcefiles=[p for p in (r/'src').rglob('*') if p.is_file() and p.suffix in ['.h','.cpp','.cuh','.cu'] and 'godot' not in p.parts]
sourcefiles += [r/x for x in ['CMakeLists.txt','AGENTS.md','tools/frozen_brake_probe.cpp','tools/run-frozen-brake-probe.ps1']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
manifest=[]
for p in sourcefiles:
 rel=p.relative_to(r);dst=run/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);manifest.append({'path':str(rel),'sha256':sha(p)})
(run/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
(run/'preparation.json').write_text(json.dumps({'seconds':time.monotonic()-t,'new_physical_controls':0,'optimizer_updates':0,'CPU_only':True},indent=2))
inputs=[run/x for x in ['runner.exe','frozen-protocol.md','frozen-checkpoint.bin','source-protocol.bin','protocol.bin','numerical-fixture.csv','numerical-preflight.json','readiness.json','source-manifest.json']]+sourcefiles
(run/'frozen-hashes.json').write_text(json.dumps([{'path':str(p),'sha256':sha(p)} for p in inputs],indent=2))
print(json.dumps({'prepared':str(run),'preflight':json.loads((run/'numerical-preflight.json').read_text()),'preparation_seconds':time.monotonic()-t},indent=2))
