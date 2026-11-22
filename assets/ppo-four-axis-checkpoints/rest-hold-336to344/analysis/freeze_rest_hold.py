from pathlib import Path
import json,hashlib,shutil,subprocess,os,time
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation');local=Path(__file__).parent
pres=r/'build/review-preservation/2026-11-04-before-rest-hold';run=r/'build/ppo/rest-hold-336to344-v1'
t=time.monotonic();env=os.environ.copy();env['CUDARL_REQUIRE_GPU']='1';checks=[]
def invoke(args,name):
 left=60-(time.monotonic()-t);assert left>0,'60s readiness cap exhausted'
 with (pres/name).open('w') as f:p=subprocess.run(args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=left)
 if p.returncode:raise RuntimeError(f'{name} failed exit{p.returncode}; no physical attempt')
for i,flt in enumerate(['ppo_checkpoint:','position_hold','gaussian:','ppo_diagnostics:']):
 invoke([str(r/'build/Release/foundation_tests.exe'),flt],f'checks-{i}.log');txt=(pres/f'checks-{i}.log').read_text();assert 'SKIP' not in txt
 checks.append({'filter':flt,'result':txt.splitlines()[-1]})
invoke([str(r/'build/Release/ppo_rest_hold.exe'),'--prepare',str(run)],'prepare.log')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
assert sha(run/'parent-checkpoint.bin')=='810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0'
shutil.copy2(r/'build/Release/ppo_rest_hold.exe',run/'runner.exe');shutil.copy2(local/'execute_rest_hold.ps1',r/'tools/run-rest-hold-study.ps1')
shutil.copy2(r/'assets/ppo-four-axis-checkpoints/rest-hold-proposal-v1/PROTOCOL.md',run/'approved-protocol.md')
(run/'approval-and-reporting.txt').write_text('Approved exact protocol including critic-only calibration. Report actual returns and critic validation before PPO; native pauses for reported-and-continue marker while120s wall caps continue. Proceed only if original gates pass. No retries/extensions/adoption/moving starts.\n')
shutil.copytree(pres,run/'preservation')
sources=[p for p in (r/'src').rglob('*') if p.is_file() and p.suffix in ['.h','.cpp','.cu','.cuh'] and 'godot' not in p.parts]
sources += [r/x for x in ['CMakeLists.txt','AGENTS.md','tools/ppo_motor.cu','tools/append_hold8.cu','tools/hold_scale_study.cu','tools/rest_hold_study.cu','tools/run-rest-hold-study.ps1']]
manifest=[]
for p in sources:
 rel=p.relative_to(r);dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest.append({'path':str(rel),'sha256':sha(p)})
(run/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
(run/'checks.json').write_text(json.dumps({'actual_CUDA_required':True,'checks':checks,'policy_optimizer_steps':0,'physical_controls':0,'readiness_seconds':time.monotonic()-t},indent=2))
inputs=[run/x for x in ['runner.exe','protocol.bin','parent-checkpoint.bin','parent-protocol.bin','migrated-initial.bin','zero-step-checks.json','readiness.json','checks.json','source-manifest.json','approved-protocol.md','approval-and-reporting.txt']]+sources
(run/'frozen-hashes.json').write_text(json.dumps([{'path':str(p),'sha256':sha(p)} for p in inputs],indent=2))
print(json.dumps({'prepared':str(run),'checks':checks,'zero_step':json.loads((run/'zero-step-checks.json').read_text()),'source_files':len(sources),'physical_controls':0,'policy_optimizer_steps':0},indent=2))
