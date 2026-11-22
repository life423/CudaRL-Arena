from pathlib import Path
import subprocess,os,json,hashlib,shutil,time
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
local=Path(__file__).parent
study=r/'build/ppo/balanced-hold-scale-336to344-v1'
pres=r/'build/review-preservation/2026-11-04-before-balanced-scale'
env=os.environ.copy();env['CUDARL_REQUIRE_GPU']='1'
t=time.monotonic();tests=[]
def run(args,log):
 left=60-(time.monotonic()-t)
 if left<=0:raise RuntimeError('60s zero-step preparation cap exhausted')
 with (pres/log).open('w') as f:
  p=subprocess.run(args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=left)
 if p.returncode:raise RuntimeError(f'{log} failed exit{p.returncode}; no physical run')
for k,flt in enumerate(['ppo_checkpoint:','position_hold','gaussian:','adam: rate override','adam: checkpoint fences','ppo_diagnostics:']):
 run([str(r/'build/Release/foundation_tests.exe'),flt],f'checks-{k}.log')
 txt=(pres/f'checks-{k}.log').read_text();assert 'SKIP' not in txt
 tests.append({'filter':flt,'log':f'checks-{k}.log','summary':txt.splitlines()[-1]})
run([str(r/'build/Release/ppo_hold_scale.exe'),'--prepare',str(study)],'prepare.log')
assert hashlib.sha256((study/'A/parent-checkpoint.bin').read_bytes()).hexdigest().upper()=='810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0'
assert (study/'A/parent-checkpoint.bin').read_bytes()==(study/'B/parent-checkpoint.bin').read_bytes()
shutil.copy2(r/'build/Release/ppo_hold_scale.exe',study/'runner.exe')
shutil.copy2(local/'execute_hold_study.ps1',r/'tools/run-hold-scale-study.ps1')
shutil.copy2(local/'HOLD_SCALE_PROTOCOL.md',r/'docs/HOLD_SCALE_2026-11-04.md')
shutil.copy2(local/'HOLD_SCALE_PROTOCOL.md',study/'frozen-protocol.md')
shutil.copytree(pres,study/'preservation')
sourcefiles=[p for p in (r/'src').rglob('*') if p.is_file() and p.suffix in ('.h','.cpp','.cu','.cuh') and 'godot' not in p.parts]
sourcefiles += [r/x for x in ['CMakeLists.txt','AGENTS.md','tools/ppo_motor.cu','tools/append_hold8.cu','tools/hold_scale_study.cu','tools/run-hold-scale-study.ps1','docs/HOLD_SCALE_2026-11-04.md']]
manifest=[]
for p in sourcefiles:
 rel=p.relative_to(r);dst=study/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
 manifest.append({'path':str(rel),'sha256':hashlib.sha256(p.read_bytes()).hexdigest().upper()})
(study/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
(study/'checks.json').write_text(json.dumps({'actual_CUDA_required':True,'checks':tests,'preparation_seconds':time.monotonic()-t,'physical_controls':0,'PPO_updates':0},indent=2))
inputs=[study/'runner.exe',study/'source-manifest.json',study/'frozen-protocol.md',study/'readiness.json',study/'checks.json']
for arm in ['A','B']:
 inputs += [study/arm/x for x in ['parent-checkpoint.bin','parent-protocol.bin','protocol.bin','migrated-initial.bin','migration-zero-checks.json','configuration-checks.json']]
inputs+=sourcefiles
hashes=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest().upper()} for p in inputs]
(study/'frozen-hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({'prepared':str(study),'zero_step_preparation_seconds':time.monotonic()-t,'checks':tests,'source_files':len(sourcefiles),'PPO_updates':0,'physical_controls':0},indent=2))
