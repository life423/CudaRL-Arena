"""Read-only native evidence aggregation; no simulator or optimizer."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
run = root / 'build/ppo/current-swim-v2-2026-11-22-v1'
source = run / 'training-result.json'
native = json.loads(source.read_text())
assert native['closed'] is True and native['phase_closed'] is True
assert native['qualified'] is False and native['PPO_updates'] == 32
assert native['gate'] == 'bounded_updates_exhausted_current_not_qualified'
assert native['baseline_current_passes'] == 4 and native['final_current_passes'] == 8
assert native['withheld_passes'] == 0 and native['endurance_passes'] == 0
process = json.loads((run / 'process-execution.json').read_text())
phases = dict(physics_gate=3760, baseline_calm=10720, baseline_current=2560,
              gpu_profile=2949120, calibration=2621440, ppo_collection=83886080,
              post_update_current_assessment=81920, final_calm=10720,
              withheld_current=0, current_endurance=0)
assert sum(phases.values()) == native['controls'] == 89566320
result = dict(native)
result.update(study='current_swim_v2', phase='bounded_updates_exhausted',
              training_started=True, current_suites_run=True,
              withheld_suites_run=False, endurance_suites_run=False,
              withheld_passes=None, endurance_passes=None,
              controls_by_phase=phases,
              active_process_seconds=process['active_process_seconds'],
              process_cap_seconds=process['cap_seconds'],
              source_training_result={'path':str(source), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
              note='The full 32-update budget is closed. Current swimming did not qualify. Unrun suites have null pass counts. The saved calm viewer policy is unchanged.')
(run / 'study-result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'summary':str(run / 'study-result.json'), 'controls':native['controls'],
                  'sha256':hashlib.sha256((run/'study-result.json').read_bytes()).hexdigest()}))
