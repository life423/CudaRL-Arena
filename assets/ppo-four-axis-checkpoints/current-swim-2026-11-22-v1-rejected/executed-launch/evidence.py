"""Protect earlier evidence and freeze this one current-study execution."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
PREP=Path(__file__).resolve().parent
ROOT=PREP.parents[1]
NAME='current-swim-2026-11-22-v1'
RUN=ROOT/'build/ppo'/NAME
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):value.update(part)
    return value.hexdigest().upper()
mode=sys.argv[1]
if mode=='capture':
    target=PREP/'protected-before.json'
    assert not target.exists(),'Protection identity already exists'
    paths={p for folder in ('assets','build/ppo','build/diagnostics') for p in (ROOT/folder).rglob('*')
           if p.is_file() and NAME not in p.relative_to(ROOT).parts}
    paths.update(p for p in (ROOT/'live/research_data').glob('*') if p.is_file())
    rows=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(paths)]
    target.write_text(json.dumps(rows,indent=2)+'\n')
    pinned=[s.strip() for s in (ROOT/'src/nn/motor/controller_contract.txt').read_text().splitlines() if s.strip()]
    (PREP/'pinned-contract-before.json').write_text(json.dumps([{ 'path':s,'sha256':sha(ROOT/s)} for s in pinned],indent=2)+'\n')
    print(json.dumps({'protected_prior_files':len(rows),'pinned_sources':len(pinned)}))
elif mode=='verify':
    rows=json.loads((PREP/'protected-before.json').read_text())
    pinned=json.loads((PREP/'pinned-contract-before.json').read_text())
    bad=[r['path'] for r in rows if not(ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    bad_pinned=[r['path'] for r in pinned if sha(ROOT/r['path'])!=r['sha256']]
    result={'protected_prior_files':len(rows),'unexpected_changes':bad,'pinned_source_changes':bad_pinned}
    (PREP/'preservation-after.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert not bad and not bad_pinned
elif mode=='freeze':
    target=RUN/'frozen-inputs.json';assert not target.exists(),'Execution already frozen'
    paths={p for folder in ('src','tools') for p in (ROOT/folder).rglob('*')
           if p.is_file() and p.suffix in ('.h','.cuh','.cu','.cpp','.inc','.txt')}
    paths.update(p for p in RUN.rglob('*') if p.is_file())
    paths.update({ROOT/'docs/CURRENT_SWIM_PROTOCOL_2026-11-22.md',ROOT/'CMakeLists.txt',Path(__file__),PREP/'launch.ps1'})
    paths.update(ROOT/'build/Release'/name for name in ('ppo_current_swim.exe','current_physics_gate.exe','creature_gpu.lib','creature_physics.lib','frozen_controller.lib'))
    rows=[]
    for path in sorted(paths):
        relative=path.relative_to(ROOT)
        rows.append({'path':relative.as_posix(),'sha256':sha(path)})
        destination=RUN/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    target.write_text(json.dumps(rows,indent=2)+'\n')
    (RUN/'git-head.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT))
    print(json.dumps({'frozen_inputs':len(rows)}))
elif mode=='check-frozen':
    rows=json.loads((RUN/'frozen-inputs.json').read_text())
    bad=[r['path'] for r in rows if sha(ROOT/r['path'])!=r['sha256']]
    print(json.dumps({'frozen_inputs':len(rows),'changed':bad}));assert not bad
else:raise SystemExit('capture / verify / freeze / check-frozen')
