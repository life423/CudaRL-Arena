"""Archive the closed rejection's small causal/evidence files, preserving raw data.

Never simulates, trains, changes original evidence, or exports an actor.
"""
from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'build/ppo/continuous-pond-2026-11-23-v1'
    proof=json.loads((run/'independent-verification.json').read_text())
    result=json.loads((run/'training-result.json').read_text())
    assert proof['passed'] and not proof['qualified'] and result['closed'] and not result['qualified']
    assert proof['assessment_sha256']==hashlib.sha256((run/'training-result.json').read_bytes()).hexdigest()
    directory=root/'assets/ppo-four-axis-checkpoints/continuous-pond-2026-11-23-v1-rejected'
    assert not directory.exists()
    directory.mkdir()
    protection=json.loads((root/'build/continuous-pond-v2-preparation-2026-11-23/protected-before.json').read_text())
    prefix='build/ppo/continuous-pond-2026-11-23-v1/'
    raw_manifest=[dict(entry,path=entry['path'][len(prefix):]) for entry in protection if entry['path'].startswith(prefix)]
    assert raw_manifest
    included=[]
    output=directory/'evidence.zip'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or path.suffix=='.lock': continue
            relative=path.relative_to(run)
            if relative.name=='controls.csv': continue
            batch=any(part.startswith(('training-batch-','calibration-batch-','assessment-')) or part in ('baseline-free','withheld-validation','constant-only-ablation') for part in relative.parts)
            if batch and relative.name not in ('manifest.json','timing.json','summary.csv','motion-windows.csv'): continue
            archive.write(path,str(relative).replace('\\','/'))
            included.append({'path':str(relative).replace('\\','/'),'bytes':path.stat().st_size,
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with zipfile.ZipFile(output) as archive:
        for entry in included:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256']
    report={'schema':1,'closed':True,'qualified':False,'study':'continuous_pond_v1',
            'assessment_sha256':proof['assessment_sha256'],
            'verification_sha256':hashlib.sha256((run/'independent-verification.json').read_bytes()).hexdigest(),
            'archive_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'archived_files':included,'complete_local_evidence':raw_manifest,
            'full_raw_data_in_git':False,'original_data_unchanged':True,
            'raw_data_note':'Full control traces remain at the original local run. This compact archive does not substitute for their independent re-audit.',
            'physical_controls_executed':0,'optimizer_steps_executed':0}
    (directory/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'files':len(included),
                      'full_local_files':len(raw_manifest),'qualified':False,'originals_preserved':True}))


if __name__=='__main__':main()
