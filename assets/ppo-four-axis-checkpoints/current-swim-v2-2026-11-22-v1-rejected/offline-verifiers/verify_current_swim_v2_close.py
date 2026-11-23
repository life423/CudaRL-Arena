"""Seal the completed v2 row audit to its truthful closed-study summary.

Reads saved evidence only. No simulator, learning runtime or optimizer is used.
The complete row-audit report is preserved before its closure proof is added.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('build/ppo/current-swim-v2-2026-11-22-v1'))
    args=parser.parse_args();root=args.run.resolve()
    target=root/'independent-training-verification.json'
    report=json.loads(target.read_text());assert report['passed'] and report['phase']=='training'
    assert not report['qualified'] and 'closed_study_summary' not in report
    native_path=root/'training-result.json';aggregate_path=root/'study-result.json'
    native=json.loads(native_path.read_text());aggregate=json.loads(aggregate_path.read_text())
    assert sha(native_path)==report['input_sha256'][str(native_path)]
    keys=('schema','controls','substeps','PPO_updates','actor_steps','critic_steps','critic_forward_calls','fault','gate',
          'baseline_current_passes','final_current_passes','qualified','baseline_only','selected_worlds','phase_closed',
          'closed','runtime_adopted','retries','active_seconds','prior_active_seconds')
    assert all(aggregate[key]==native[key] for key in keys)
    assert aggregate['schema']==2 and aggregate['study']=='current_swim_v2' and aggregate['phase']=='bounded_updates_exhausted'
    assert aggregate['closed'] and not aggregate['qualified'] and not aggregate['runtime_adopted'] and not aggregate['fault']
    assert aggregate['training_started'] and aggregate['current_suites_run']
    assert aggregate['withheld_suites_run']==aggregate['endurance_suites_run']==False
    assert aggregate['withheld_passes'] is None and aggregate['endurance_passes'] is None
    assert native['withheld_passes']==native['endurance_passes']==0
    assert not (root/'reserved-withheld').exists() and not (root/'reserved-endurance').exists()
    source=aggregate['source_training_result']
    assert Path(source['path']).resolve()==native_path and source['sha256']==sha(native_path)
    trained=report['training'];n=aggregate['selected_worlds'];updates=aggregate['PPO_updates']
    assert n==16384 and updates==32 and aggregate['actor_steps']==trained['Adam_steps']==344
    assert aggregate['baseline_current_passes']==report['baseline']['current']['passes']==4
    assert aggregate['final_current_passes']==trained['updates'][-1]['current']['passes']==8
    assert trained['withheld'] is None and trained['endurance'] is None
    assert trained['calm']['short_passes']==4 and trained['calm']['hold_passes']==6 and trained['calm']['long_passes']==28
    expected={'physics_gate':3760,'baseline_calm':10720,'baseline_current':2560,'gpu_profile':2949120,
              'calibration':n*160,'ppo_collection':updates*n*160,'post_update_current_assessment':updates*2560,
              'final_calm':10720,'withheld_current':0,'current_endurance':0}
    assert aggregate['controls_by_phase']==expected
    assert sum(expected.values())==aggregate['controls']==trained['controls']==report['counted_controls']==89566320
    assert aggregate['substeps']==trained['substeps']==report['counted_substeps']==895663200
    assert aggregate['controls']<=89571440 and aggregate['substeps']<=895714400
    process_path=root/'process-execution.json';process=json.loads(process_path.read_text())
    assert process['cap_seconds']==aggregate['process_cap_seconds']==900
    assert process['active_process_seconds']==aggregate['active_process_seconds']<900
    labels=[entry['label'] for entry in process['steps']]
    assert labels==['Gate','Baseline','Profile','Calibrate','Train'] and len(set(labels))==5
    assert all(entry['status']=='completed' and entry['exit_code']==0 for entry in process['steps'])
    assert abs(sum(entry['seconds'] for entry in process['steps'])-process['active_process_seconds'])<1e-9
    checkpoint=root/'training-started/current-policy-update-32.bin'
    assert sha(checkpoint)==trained['final_policy_sha256']==report['input_sha256'][str(checkpoint)]
    archive=Path('build/current-swim-v2-preparation-2026-11-22/offline-audits/full-training-audit-before-close.json').resolve()
    assert not archive.exists(),'Preserve the existing one-shot closure identity'
    original=target.read_bytes();archive.write_bytes(original)
    inputs=report['input_sha256']
    for path in (aggregate_path,process_path,archive,Path(__file__).resolve(),
                 Path('build/current-swim-v2-preparation-2026-11-22/close_summary.py').resolve()):
        inputs[str(path)]=sha(path)
    report['closed_study_summary']={'passed':True,'aggregate_sha256':sha(aggregate_path),
        'source_full_row_audit_sha256':sha(archive),'controls_by_phase':expected,
        'qualified':False,'current_cases_before':4,'current_cases_after':8,
        'withheld_suites_run':False,'endurance_suites_run':False,
        'active_process_seconds':process['active_process_seconds'],'unused_control_allowance_closed':89571440-aggregate['controls'],
        'scope':'All saved row arithmetic was verified before this separate summary seal; partial improvement is not qualification',
        'physical_controls':0,'optimizer_steps':0}
    target.write_text(json.dumps(report,indent=2)+'\n')
    # An early CSV-format decoder failure used the preceding phase's filename.
    # That attempt is preserved separately. The completed full audit has now
    # independently rechecked the baseline, so restore its explicit proof.
    baseline_path=root/'independent-baseline-verification.json'
    if baseline_path.exists() and not json.loads(baseline_path.read_text())['passed']:
        history=Path('build/current-swim-v2-preparation-2026-11-22/offline-audits/profile-verifier-initial-formatting-failure.json')
        assert history.exists() and history.read_bytes()==baseline_path.read_bytes()
        baseline={'passed':True,'qualified':False,'phase':'baseline','scope':'Baseline reverified within the complete final row audit',
            'baseline':report['baseline'],'counted_controls':17040,'counted_substeps':170400,
            'source_complete_row_audit_sha256':sha(archive),'input_sha256':dict(inputs)}
        baseline_path.write_text(json.dumps(baseline,indent=2)+'\n')
    print(json.dumps({'passed':True,'qualified':False,'aggregate_sha256':sha(aggregate_path),'final_report_sha256':sha(target),
                      'controls':aggregate['controls'],'PPO_updates':32,'Adam_steps':344,'physical_controls':0,'optimizer_steps':0}))


if __name__=='__main__':main()
