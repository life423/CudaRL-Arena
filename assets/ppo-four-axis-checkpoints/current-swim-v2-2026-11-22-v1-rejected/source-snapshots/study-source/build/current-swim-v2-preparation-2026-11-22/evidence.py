"""Phase-specific causal freezes for the approved GPU-sized v2 study."""
import hashlib,json,re,shutil,sys
from pathlib import Path
PREP=Path(__file__).resolve().parent;ROOT=PREP.parents[1]
RUN=ROOT/'build/ppo/current-swim-v2-2026-11-22-v1'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest().upper()
def include_closure(initial):
    pending=list(initial);found=set()
    while pending:
        p=pending.pop().resolve()
        if p in found:continue
        found.add(p)
        if p.suffix not in ('.h','.cuh','.cpp','.cu','.inc'):continue
        for name in re.findall(r'#\s*include\s*"([^"]+)"',p.read_text(errors='replace')):
            for candidate in (p.parent/name,ROOT/'src'/name,ROOT/'src/nn'/name,ROOT/'tools'/name):
                if candidate.is_file():pending.append(candidate);break
    return found
mode=sys.argv[1]
if mode=='capture':
    dest=PREP/'protected-before.json';assert not dest.exists()
    paths={p for folder in ('assets','build/ppo','build/diagnostics') for p in (ROOT/folder).rglob('*') if p.is_file() and RUN not in p.parents}
    paths.update(p for p in (ROOT/'live/research_data').glob('*') if p.is_file() and p.name!='current-study.json')
    rows=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(paths)]
    dest.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'protected_files':len(rows)}))
elif mode=='verify':
    rows=json.loads((PREP/'protected-before.json').read_text())
    bad=[r['path'] for r in rows if not(ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    out={'protected_files':len(rows),'unexpected_changes':bad};(PREP/'preservation-after.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out));assert not bad
elif mode in ('freeze-gate','freeze-study'):
    gate=mode=='freeze-gate';label='gate' if gate else 'study'
    target=RUN/(label+'-frozen-inputs.json');assert not target.exists()
    if gate:
        paths=include_closure([ROOT/'tools/current_physics_gate_v2.cu'])
        paths.add(ROOT/'build/current-gate-v2-preparation-2026-11-22/build/Release/current_physics_gate_v2.exe')
        paths.update(p for p in (RUN/'physics-gate').rglob('*') if p.is_file())
    else:
        paths=include_closure([ROOT/'tools/current_swim_v2_study.cu'])
        paths.add(ROOT/'build/Release/ppo_current_swim_v2.exe')
        paths.update(p for p in RUN.rglob('*') if p.is_file() and p.name not in ('status.json','process-execution.json','gpu-telemetry.csv') and not any(x in ('gate-source','study-source') for x in p.relative_to(RUN).parts))
    paths.update(ROOT/'build/Release'/n for n in ('creature_gpu.lib','creature_physics.lib','frozen_controller.lib'))
    paths.update(ROOT/'src'/n for n in ('core/checkpoint.cpp','simulation/world.cpp','nn/motor/frozen_controller.cpp'))
    paths.update({ROOT/'docs/CURRENT_SWIM_V2_PROTOCOL_2026-11-22.md',Path(__file__)})
    rows=[]
    for p in sorted(paths):
        rel=p.relative_to(ROOT);rows.append({'path':rel.as_posix(),'sha256':sha(p)})
        out=RUN/(label+'-source')/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    target.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'phase':label,'frozen_inputs':len(rows)}))
elif mode in ('check-gate','check-study'):
    label=mode.split('-')[1];rows=json.loads((RUN/(label+'-frozen-inputs.json')).read_text())
    bad=[r['path'] for r in rows if sha(ROOT/r['path'])!=r['sha256']]
    print(json.dumps({'phase':label,'frozen_inputs':len(rows),'changed':bad}));assert not bad
else:raise SystemExit('capture/verify/freeze-gate/freeze-study/check-gate/check-study')
