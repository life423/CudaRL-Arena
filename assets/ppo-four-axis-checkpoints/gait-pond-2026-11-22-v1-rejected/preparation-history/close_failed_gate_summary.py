"""Derive a closed failed-gate summary without invoking native code."""
from pathlib import Path
import hashlib,json
run=Path('build/ppo/gait-pond-2026-11-22-v1').resolve()
source=run/'gate-result.json';native=json.loads(source.read_text())
assert native['closed'] and not native['qualified'] and native['fault']
assert native['controls']==960 and native['substeps']==9600 and native['PPO_updates']==native['actor_steps']==0
result=dict(native)
for key in ('short_movement_passes','short_gait_passes','long60_passing_windows','recovery_passes','long300_passing_windows','original_hold_passes','withheld_hold_passes','hold300_passes'):
    result[key]=None
result.update(phase='compatibility_gate_failed',training_started=False,behavioral_suites_run=False,
              calibration_started=False,viewer_fidelity_run=False,gate_passed=False,
              maximum_controls=85295280,maximum_substeps=852952800,closed_unused_controls=85295280-960,
              source_gate_result={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
              note='Closed before baseline/calibration/PPO. Strict independent closed-loop CPU/CUDA trajectory compatibility failed at control32; all saved physical step statuses were valid. No learned gait improvement or skill-retention result is claimed.')
process_path=run/'process-execution.json'
if process_path.exists():
    process=json.loads(process_path.read_text());result['active_process_seconds']=process['active_process_seconds'];result['process_cap_seconds']=900
target=run/'study-result.json'
if not target.exists():
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
else:
    existing=json.loads(target.read_text());assert existing['controls']==960 and not existing['training_started']
print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'physical_controls':0,'optimizer_steps':0}))
