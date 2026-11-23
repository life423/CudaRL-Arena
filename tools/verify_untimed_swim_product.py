"""Audit saved untimed product evidence without native code or integration."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(text):
    return tuple(float(x) for x in text.strip('()').split(','))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def rotate(q, v):
    a = cross(q[:3], v)
    b = cross(q[:3], tuple(a[i]+q[3]*v[i] for i in range(3)))
    return tuple(v[i]+2*b[i] for i in range(3))


def multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)


def ellipse(position, orientation, joints):
    """Geometry only, matching the conservative link boxes; no forces/steps."""
    center = tuple(.04*x for x in position)
    q = orientation
    links = [(center, q, .020, .0064, .008)]
    for k in range(2):
        offset = rotate(q, (-links[-1][2]*.5, 0, 0))
        anchor = tuple(center[i]+offset[i] for i in range(3))
        yaw, pitch = joints[2*k:2*k+2]
        q = multiply(multiply(q, (0, math.sin(yaw/2), 0, math.cos(yaw/2))),
                     (0, 0, math.sin(pitch/2), math.cos(pitch/2)))
        offset = rotate(q, (-.013, 0, 0))
        center = tuple(anchor[i]+offset[i] for i in range(3))
        links.append((center, q, .026, .0015 if k == 0 else .00635, .0015))
    peak = 0
    for center, q, length, height, width in links:
        for x, y, z in itertools.product((-1, 1), repeat=3):
            offset = rotate(q, (x*length/2, y*height, z*width))
            p = tuple(center[i]+offset[i] for i in range(3))
            peak = max(peak, (p[0]/.20)**2+(p[2]/.12)**2)
    return peak


def audit(root):
    inputs = json.loads((root/'verification-inputs.json').read_bytes())
    assert inputs['purpose'] == 'untimed_swim_product_check'
    assert inputs['maximum_attempted_controls'] == 800 and inputs['maximum_attempted_substeps'] == 8000
    assert inputs['maximum_simulated_seconds'] == 40 and inputs['maximum_process_seconds'] == 60
    for filename, digest in inputs['input_sha256'].items():
        assert sha(filename) == digest, filename
    preserve = json.loads((root/'preserved-before.json').read_bytes())
    for row in preserve['saved_files']:
        assert sha(row['snapshot']) == row['sha256']
    for filename, digest in preserve['policies'].items():
        assert sha(filename) == digest
    process = json.loads((root/'product-process.json').read_bytes())
    assert process['exit_code'] == 0 and process['terminated_at_cap'] is False
    assert 0 < process['seconds'] <= 60
    value = json.loads((root/'native-check.json').read_bytes())
    assert value['passed'] and value['failures'] == [] and all(x['passed'] for x in value['checks'])
    assert value['training_updates'] == value['optimizer_steps'] == value['research_interface_cases'] == 0
    rows = value['trace']
    assert len(rows) == 511 and value['attempted_controls'] == 510 and value['accepted_controls'] == 509
    assert value['attempted_substeps'] == 5100 and value['accepted_substeps'] == 5090
    assert value['attempted_controls'] <= 800 and value['attempted_substeps'] <= 8000
    first, last = rows[0], rows[-1]
    request_fields = ('request_origin', 'request_heading', 'request_position', 'terminal_velocity')
    maximum_com_jump = 0
    maximum_box_value = 0
    for i, row in enumerate(rows):
        accepted = min(i, 509)
        assert row['controls'] == row['task_controls'] == row['world_controls'] == accepted
        assert row['attempted_controls'] == i and row['substeps'] == i*10
        assert abs(row['time_s']-accepted*.05) < 1e-5
        assert row['untimed_swim'] is True and row['continuous'] is False and row['has_continuous'] is False
        assert row['finished'] is False and row['duration_s'] == row['maximum_controls'] == 0
        assert row['training'] is False and row['backbone_frozen'] is True and row['training_update'] == 336
        assert row['action_mapping'] == 3 and row['observation_schema'] == 4
        assert row['pond_domain_valid'] is True
        assert all(row[k] == first[k] for k in request_fields)
        for key, limit in (('held_targets', 1), ('reference_rates', 12), ('previous_torques', 1)):
            assert len(row[key]) == 4 and all(math.isfinite(x) and abs(x) <= limit for x in row[key])
        if i:
            jump = math.dist(vector(row['com_position']), vector(rows[i-1]['com_position']))*.04
            maximum_com_jump = max(maximum_com_jump, jump)
            assert jump < .002, ('COM discontinuity', i, jump)
            assert max(abs(a-b) for a, b in zip(row['held_targets'], rows[i-1]['held_targets'])) <= .600001
            assert max(abs(a-b) for a, b in zip(row['reference_rates'], rows[i-1]['reference_rates'])) <= 10.00001
        box_value = ellipse(tuple(row['head_'+x] for x in ('x', 'y', 'z')),
                            tuple(row[x] for x in ('qx', 'qy', 'qz', 'qw')), [row['j'+str(k)] for k in range(4)])
        assert box_value < 1
        maximum_box_value = max(maximum_box_value, box_value)
        if 0 < i < 510:
            assert row['paused'] is False and row['error'] == '' and row['failure_recorded'] is False
    assert last == value['final_state'] and last['paused'] is True and last['failure_recorded'] is True
    assert last['error'] == 'The creature reached the edge of the supported pond area'
    stable_fields = ('head_x', 'head_y', 'head_z', 'qx', 'qy', 'qz', 'qw',
                     'j0', 'j1', 'j2', 'j3', 'held_targets', 'reference_rates', 'previous_torques', 'time_s')
    assert all(last[k] == rows[-2][k] for k in stable_fields), 'Rejected candidate changed accepted body/history'
    candidate = last['failure_candidate']
    candidate_value = ellipse(vector(candidate['position']), vector(candidate['orientation']),
                              [candidate['j'+str(k)] for k in range(4)])
    assert candidate_value > 1.0001
    assert rows[160]['time_s'] == 8 and rows[161]['controls'] == 161 and rows[162]['controls'] == 162
    for i in (159, 160, 161, 162, 170, 171):
        assert rows[i]['held_targets'] != first['held_targets'] and rows[i]['reference_rates'] != first['reference_rates']
    checks = {item['check']: item['passed'] for item in value['checks']}
    assert all(checks[k] for k in ('Pause changes no movement state', 'Resume succeeds', 'Resume changes no movement state'))
    assert value['pause_resume_verified'] is True
    reference = {}
    with (root/'reference-poses.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['category'] == 'short' and row['case'] == '0':
                reference[int(row['control'])] = row
    columns = dict(zip(('head_x','head_y','head_z','qw','qx','qy','qz','j0','j1','j2','j3'),
                       ('position_x_model','position_y_model','position_z_model','quat_w','quat_x','quat_y','quat_z',
                        'angle0','angle1','angle2','angle3')))
    # get_state stores original float32 body fields as Godot numeric Variants;
    # JSON decimal formatting rounds the promoted doubles. Recover the source
    # binary32 values before comparing the in-memory check's exact summary.
    f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
    serialized_error = max(abs(rows[i][key]-float(reference[i][column])) for i in range(1, 161) for key, column in columns.items())
    error = max(abs(f32(rows[i][key])-float(reference[i][column])) for i in range(1, 161) for key, column in columns.items())
    assert error <= 2e-6 and abs(error-value['maximum_original_pose_error']) < 1e-16
    bound = {str(root/name): sha(root/name) for name in
             ('verification-inputs.json', 'native-check.json', 'product-process.json', 'native-check.json.started.json')}
    bound[str(Path(__file__).resolve())] = sha(__file__)
    return dict(schema=1, purpose='untimed_swim_product_saved_data_audit', passed=True,
                bound_inputs_verified=len(inputs['input_sha256']), accepted_controls=509, attempted_controls=510,
                accepted_substeps=5090, attempted_substeps=5100, accepted_simulated_seconds=509*.05,
                forward_distance_m=last['forward_m'], former8s_distance_m=rows[160]['forward_m'],
                maximum_original160_pose_error=error, maximum_serialized_original160_pose_error=serialized_error,
                original_request_persists_all_rows=True,
                counter_and_state_continuity_past160=True, maximum_endpoint_COM_jump_m=maximum_com_jump,
                maximum_accepted_endpoint_ellipse_box_value=maximum_box_value,
                rejected_candidate_ellipse_box_value=candidate_value,
                rejected_candidate_preserves_accepted_body_and_servo_history=True,
                pause_resume_assertions_confirmed_from_bound_checker=True,
                pause_resume_limitation='Raw paused/resumed snapshots were compared by the bound test but not saved separately; the saved trace and assertion outcomes corroborate continuity.',
                geometry_limitation='Offline double geometry only; candidate printed vectors are rounded. This corroborates the original boundary guard, not a force replay or continuous swept-volume claim.',
                no_autonomous_turning_qualification=True, new_physical_controls=0, new_substeps=0,
                new_GPU_kernels=0, new_optimizer_steps=0, input_sha256=bound)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    result = audit(root)
    output = root/'independent-product-verification.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}))
