"""Bind detailed existing gate-failure forensics to the full independent audit.

Original frozen sources and the first forensic tool/proof remain unchanged.
This analysis uses only existing binary/CSV records; no physical/GPU/Adam work.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import analyze_frozen_interface_gate_failure as initial

audit = initial.audit


def analyze(root):
    report = initial.analyze(root)
    full = json.loads(audit.read(root/'independent-result-verification.json'))
    native = json.loads(audit.read(root/'result.json'))
    process = json.loads(audit.read(root/'process-execution.json'))
    assert full['audit_passed'] and full['audit_status'] == 'verified_hard_stop'
    assert full['full_comparison_completed'] is False and full['native_result'] == native
    assert full['recorded_controls'] == native['controls'] == 214
    assert full['recorded_attempted_substeps'] == full['recorded_accepted_substeps'] == 2140
    assert full['matched_precheck_cases'] == 0 and full['cases'] == []
    assert full['weights_unchanged']
    assert process['exit_code'] == 3 and process['terminated_at_cap'] is False
    assert process['seconds'] <= 240 and process['PPO_updates'] == process['optimizer_steps'] == 0
    audit.old.load_weights(root)
    records, _ = audit.old.records(root, json.loads(audit.read(root/'layout.json')))
    assert np.all(records['world']['current'] == 0)
    assert np.isfinite(records['world']['energy']).all()
    left = records[(records['group'] == 0) & (records['case'] == 0) & (records['backend'] == 0)]
    right = records[(records['group'] == 0) & (records['case'] == 0) & (records['backend'] == 1)]
    assert len(left) == len(right) == 108
    tolerance = np.array([2e-5]*11+[2e-7]*3+[2e-5]*7)
    states = []
    for control in (106, 107):
        a, b = left[control], right[control]
        assert a['tick'] == b['tick'] == control
        body = []
        for j, name in enumerate(audit.BODY[:21]):
            cpu, cuda = float(a['world']['body'][j]), float(b['world']['body'][j])
            body.append({'component': name, 'cpu': cpu, 'cuda': cuda,
                         'absolute_difference': abs(cpu-cuda),
                         'absolute_limit': float(tolerance[j]),
                         'within_limit': bool(abs(cpu-cuda) <= tolerance[j])})
        motors = {}
        for name in ('held', 'rates', 'last_torque', 'actual', 'requested'):
            error = abs(a['world'][name].astype(float)-b['world'][name].astype(float))
            motors[name] = {'maximum_absolute_difference': float(error.max(initial=0)),
                            'absolute_limit': 2e-5, 'within_limit': bool(np.all(error <= 2e-5))}
        e0, e1 = a['world']['energy'], b['world']['energy']
        eerror = abs(e0-e1)
        elimit = 1e-12+1e-4*np.maximum(abs(e0), abs(e1))
        states.append({'control': control, 'body': body, 'motor': motors,
                       'energy_absolute_differences': eerror.tolist(),
                       'energy_limits': elimit.tolist(),
                       'energy_within_limits': bool(np.all(eerror <= elimit))})
    assert all(x['within_limit'] for x in states[0]['body'])
    assert [x['component'] for x in states[1]['body'] if not x['within_limit']] == ['v3']
    sub = audit.old.table(root/'probe-started/substeps.csv')
    data = sub['data']
    cpu = data[(audit.old.col(sub, 'group') == 0) & (audit.old.col(sub, 'case') == 0) & (audit.old.col(sub, 'backend') == 0)]
    cuda = data[(audit.old.col(sub, 'group') == 0) & (audit.old.col(sub, 'case') == 0) & (audit.old.col(sub, 'backend') == 1)]
    columns = [sub['index'][name] for name in audit.BODY[:21]]
    # Nine significant digits roundtrip each recorded native float32. Restore
    # that type before subtracting; do not treat printed decimals as exact f64.
    diff = abs(cpu[:, columns].astype(np.float32).astype(float)-cuda[:, columns].astype(np.float32).astype(float))
    nonzero = np.argwhere(diff != 0)
    crossed = np.argwhere(diff > tolerance)
    assert len(nonzero) and len(crossed)
    first_step = []
    for pairs in (nonzero, crossed):
        row, component = map(int, pairs[0])
        first_step.append({'control': int(cpu[row, sub['index']['control']]),
                           'substep': int(cpu[row, sub['index']['substep']]),
                           'component': audit.BODY[component],
                           'absolute_difference': float(diff[row, component]),
                           'absolute_limit': float(tolerance[component])})
    audit.read(initial.__file__)
    audit.read(__file__)
    report.update({'analysis': 'verified_saved_gate_failure_binary32',
                   'basis_full_audit_status': full['audit_status'],
                   'process_execution': process, 'native_result': native,
                   'endpoint_106_vs_107': states,
                   'first_recorded_substep_difference_binary32': first_step[0],
                   'first_recorded_substep_limit_crossing_binary32': first_step[1],
                   'applied_commands_bit_identical': True,
                   'all_recorded_currents_zero': True,
                   'energy_identity_maximum_error_J': full['maximum_errors']['energy_identity'],
                   'exact_parent_identity_verified': True,
                   'weights_unchanged': True,
                   'component_label': {'v3': 'root/body angular velocity about global X, rad/s'},
                   'supersedes_only_decimal_substep_differences': 'Earlier first forensic proof parsed printed decimal values in f64; binary32 values here are authoritative for substep differences. Both original analysis source and proof are preserved.',
                   'causal_limit': 'Small differences already exist at the first recorded substep and grow to the endpoint limit at107 under identical commands. This is compatible with accumulated rounding or arithmetic/FMA sensitivity, but the recordings do not isolate that explanation from an implementation discrepancy. No physical or inference tolerance is changed.',
                   'input_sha256': dict(sorted(audit.prep.HASHES.items()))})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    root = parser.parse_args().run.resolve()
    result = analyze(root)
    output = root/'independent-gate-failure-analysis-verified.json'
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    print(json.dumps({'analysis_passed': True,
                      'first_endpoint_violation': result['gate_cases'][0]['first_endpoint_limit_crossing'],
                      'first_substep_limit_crossing_binary32': result['first_recorded_substep_limit_crossing_binary32'],
                      'main_cases_attempted': 0, 'new_physical_controls': 0,
                      'GPU_kernels': 0, 'optimizer_steps': 0,
                      'proof_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
