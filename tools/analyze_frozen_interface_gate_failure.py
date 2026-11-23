"""Read-only numerical forensics for an already stopped interface comparison.

No simulation, GPU invocation, optimization, tolerance change or rerun occurs.
The original frozen preparation/result auditor remains unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import verify_frozen_interface_comparison as audit


def analyze(root):
    native = json.loads(audit.read(root/'result.json'))
    assert native['study'] == 'frozen_interface_comparison' and native['closed']
    assert native['hard_fault'] == 'Matched-command physics disagreement'
    assert native['PPO_updates'] == native['optimizer_steps'] == 0
    assert native['cases_started'] == native['probe_completed_cases'] == 0
    layout = json.loads(audit.read(root/'layout.json'))
    records, _ = audit.old.records(root, layout)
    sub = audit.old.table(root/'probe-started/substeps.csv')
    tolerance = np.array([2e-5]*11+[2e-7]*3+[2e-5]*7)
    rows = []
    for case in range(4):
        left = records[(records['group'] == 0) & (records['case'] == case) & (records['backend'] == 0)]
        right = records[(records['group'] == 0) & (records['case'] == case) & (records['backend'] == 1)]
        if len(left) == 0 or len(right) == 0:
            continue
        count = min(len(left), len(right))
        left, right = left[:count], right[:count]
        assert np.array_equal(left['tick'], right['tick'])
        assert np.array_equal(left['command'], right['command'])
        difference = abs(left['world']['body'][:, :21].astype(float)-right['world']['body'][:, :21].astype(float))
        violations = np.argwhere(difference > tolerance)
        first = int(violations[0, 0]) if len(violations) else None
        details = []
        if first is not None:
            assert np.all(difference[:first] <= tolerance)
            for component in np.flatnonzero(difference[first] > tolerance):
                cpu = float(left[first]['world']['body'][component])
                cuda = float(right[first]['world']['body'][component])
                details.append({'component': audit.BODY[component],
                                'state_index': int(component), 'cpu': cpu, 'cuda': cuda,
                                'absolute_difference': float(difference[first, component]),
                                'absolute_limit': float(tolerance[component]),
                                'relative_difference_to_cpu': float(difference[first, component]/abs(cpu)) if cpu else None})
        first_substep_difference = None
        first_substep_limit_crossing = None
        choose = (audit.old.col(sub, 'group') == 0) & (audit.old.col(sub, 'case') == case)
        ss = sub['data'][choose]
        cpu = ss[ss[:, sub['index']['backend']] == 0]
        cuda = ss[ss[:, sub['index']['backend']] == 1]
        assert len(cpu) == len(cuda)
        names = [sub['index'][x] for x in audit.BODY[:21]]
        dc = abs(cpu[:, names]-cuda[:, names])
        nonzero = np.argwhere(dc != 0)
        if len(nonzero):
            row, component = map(int, nonzero[0])
            first_substep_difference = {
                'control': int(cpu[row, sub['index']['control']]),
                'substep': int(cpu[row, sub['index']['substep']]),
                'component': audit.BODY[component],
                'absolute_difference': float(dc[row, component])}
        crossed = np.argwhere(dc > tolerance)
        if len(crossed):
            row, component = map(int, crossed[0])
            first_substep_limit_crossing = {
                'control': int(cpu[row, sub['index']['control']]),
                'substep': int(cpu[row, sub['index']['substep']]),
                'component': audit.BODY[component],
                'absolute_difference': float(dc[row, component]),
                'absolute_limit': float(tolerance[component])}
        rows.append({'gate_case': case, 'matched_control_endpoints': count-1,
                     'actual_commands_bit_identical': True,
                     'ordinary_terminations': [int(left[-1]['reason']), int(right[-1]['reason'])],
                     'native_step_failures': [int(left[-1]['failure']), int(right[-1]['failure'])],
                     'first_endpoint_limit_crossing_control': int(left[first]['tick']) if first is not None else None,
                     'first_endpoint_limit_crossing': details,
                     'maximum_position_difference_model': float(difference[:, :3].max(initial=0)),
                     'maximum_position_difference_m': float(difference[:, :3].max(initial=0))*audit.ref.SCALE,
                     'maximum_component_differences': {name: float(value) for name, value in zip(audit.BODY[:21], difference.max(axis=0))},
                     'first_recorded_substep_difference': first_substep_difference,
                     'first_recorded_substep_limit_crossing': first_substep_limit_crossing})
    assert rows and any(x['first_endpoint_limit_crossing_control'] is not None for x in rows)
    audit.source_dependencies()
    audit.read(__file__)
    return {'schema': 1, 'study': 'frozen_interface_comparison',
            'analysis': 'saved_matched_command_gate_failure', 'analysis_passed': True,
            'recorded_controls': native['controls'],
            'recorded_attempted_substeps': native['attempted_substeps'],
            'recorded_accepted_substeps': native['accepted_substeps'],
            'main_cases_attempted': 0, 'interface_ranking_available': False,
            'tolerance_changed': False, 'new_physical_controls': 0,
            'new_GPU_kernels': 0, 'optimizer_steps': 0, 'gate_cases': rows,
            'interpretation': 'Identical actual commands produced small backend state differences that exceeded a preregistered endpoint tolerance. The comparison stopped before either interface main grid. These recordings alone do not establish whether compiler rounding/FMA, accumulated numerical sensitivity, or a deeper backend implementation discrepancy is causal.',
            'input_sha256': dict(sorted(audit.prep.HASHES.items()))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    report = analyze(root)
    output = root/'independent-gate-failure-analysis.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'analysis_passed': True, 'gate_cases': report['gate_cases'],
                      'new_physical_controls': 0, 'GPU_kernels': 0,
                      'optimizer_steps': 0, 'proof_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
