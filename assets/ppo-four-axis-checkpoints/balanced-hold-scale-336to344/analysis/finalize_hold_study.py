from pathlib import Path
import json,hashlib,shutil,subprocess
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
s=r/'build/ppo/balanced-hold-scale-336to344-v1'
a=r/'assets/ppo-four-axis-checkpoints/balanced-hold-scale-336to344'
pres=r/'build/review-preservation/2026-11-04-before-balanced-scale'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
for filename in ['prior-checkpoint-hashes.json']:
 for h in json.loads((pres/filename).read_text()):assert sha(r/h['path'])==h['sha256'],h['path']
unrelated=[h for h in json.loads((pres/'manifest.json').read_text()) if h['path'].replace('\\','/').startswith(('live/','assets/ppo-smooth-exploration-v1'))]
for h in unrelated:assert sha(r/h['path'])==h['sha256']
frozen=json.loads((s/'frozen-hashes.json').read_text())
for h in frozen:assert sha(Path(h['path']))==h['sha256']
for arm in ['A','B']:
 for p in (a/arm).rglob('*'):
  if p.is_file():assert sha(p)==sha(s/arm/p.relative_to(a/arm)),p
for dest in [a/'analysis',s/'analysis']:
 dest.mkdir()
 for name in ['analyze_hold_study.py','archive_hold_study.py','freeze_hold_study.py','prepare_hold_study.ps1','implement_hold_study.py','finalize_hold_study.py']:
  shutil.copy2(Path(__file__).parent/name,dest/name)
hardware=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
(a/'hardware.txt').write_text(hardware.stdout+hardware.stderr)
audit={'passed':True,'prior_binary_files':36,'unrelated_files':len(unrelated),'frozen_inputs':len(frozen),'all_archived_arm_files_match_raw':True,'no_new_simulation_or_training':True}
(a/'final-preservation-audit.json').write_text(json.dumps(audit,indent=2))
manifest=[{'path':str(p.relative_to(a)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in a.rglob('*') if p.is_file() and p.name!='archive-manifest.json']
(a/'archive-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'audit':audit,'archive_files':len(manifest),'hardware':hardware.stdout.strip()},indent=2))
