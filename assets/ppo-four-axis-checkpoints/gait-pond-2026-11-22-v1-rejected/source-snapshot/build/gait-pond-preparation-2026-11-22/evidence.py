"""Preserve prior evidence and freeze the fresh gait run; no simulation."""
import hashlib,json,re,shutil,sys
from pathlib import Path
PREP=Path(__file__).resolve().parent
ROOT=PREP.parents[1]
RUN=ROOT/'build/ppo/gait-pond-2026-11-22-v1'
NATIVE=ROOT/'build/gait-pond-native-2026-11-22'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def closure(seeds):
    pending=list(seeds);found=set()
    while pending:
        p=pending.pop().resolve()
        if p in found:continue
        found.add(p)
        if p.suffix not in ('.cpp','.cu','.h','.cuh','.inl'):continue
        for include in re.findall(r'#\s*include\s*"([^"]+)"',p.read_text(errors='replace')):
            for candidate in (p.parent/include,ROOT/'src'/include,ROOT/'src/nn'/include,ROOT/'tools'/include,NATIVE/'generated'/include):
                if candidate.is_file():pending.append(candidate);break
    return found
mode=sys.argv[1]
if mode=='capture':
    dest=PREP/'protected-before.json';assert not dest.exists()
    paths={p for folder in ('assets','build/ppo','build/diagnostics') for p in (ROOT/folder).rglob('*') if p.is_file() and RUN not in p.parents}
    paths.update(p for p in (ROOT/'live/research_data').glob('*') if p.is_file() and p.name!='current-study.json')
    rows=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(paths)]
    dest.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(protected_files=len(rows))))
elif mode=='verify':
    rows=json.loads((PREP/'protected-before.json').read_text())
    bad=[r['path'] for r in rows if not(ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    result=dict(protected_files=len(rows),unexpected_changes=bad)
    (PREP/'preservation-after.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert not bad
elif mode=='freeze':
    dest=RUN/'frozen-inputs.json';assert not dest.exists()
    paths=closure([ROOT/'tools/gait_swim_study.cu'])
    paths.update(ROOT/'src'/p for p in ('core/checkpoint.cpp','simulation/world.cpp','nn/motor/frozen_controller.cpp'))
    paths.update(NATIVE/'Release'/p for p in ('ppo_gait_swim.exe','creature_gpu.lib','creature_physics.lib','frozen_controller.lib'))
    paths.update([ROOT/'CMakeLists.txt',NATIVE/'CMakeCache.txt',NATIVE/'generated/frozen_contract.h',ROOT/'docs/GAIT_POND_PROTOCOL_2026-11-22.md',Path(__file__),PREP/'launch.ps1'])
    paths.update(p for p in RUN.rglob('*') if p.is_file())
    rows=[]
    for p in sorted(paths):
        rel=p.relative_to(ROOT);rows.append(dict(path=rel.as_posix(),sha256=sha(p)))
        out=RUN/'source-snapshot'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    dest.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(frozen_inputs=len(rows))))
elif mode=='check':
    rows=json.loads((RUN/'frozen-inputs.json').read_text())
    bad=[r['path'] for r in rows if not(ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    print(json.dumps(dict(frozen_inputs=len(rows),changed=bad)));assert not bad
else:raise SystemExit('capture/verify/freeze/check')
