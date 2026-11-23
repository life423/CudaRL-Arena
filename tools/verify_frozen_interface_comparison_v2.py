"""Independent saved-data audit for comparison v2, with explicit unfused arithmetic.

This reader never advances physics, launches CUDA, or updates any parameter.
It reconstructs sensors, actions, filters, recorded 5 ms guards and metrics;
it does not replay the force integrator.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import re
import xml.etree.ElementTree as ET

import numpy as np
import verify_frozen_requested_turn_result as old

prep, prior, ref, shared = old.prep, old.prior, old.ref, old.shared
read, envelope, near = old.read, old.envelope, old.near
BODY = old.BODY
ARMS = ('lane_following', 'heading_only')
PROFILE = 'cpu_compatible_no_implicit_fma_v1'
CONTROL_CAP, SUBSTEP_CAP, SECONDS_CAP, EVIDENCE_CAP = 127680, 1276800, 900, 2*1024**3
RADIUS_METADATA = dict(canonical_radius_version=1,
    fixed_geometry_radius_arguments_u32=[889811040,859599123,861293837],
    cpu_radius_bits_u32=[1006834286,996615495,997278936],unknown_radius_arguments_rejected=True)


def identity(value):
    assert value['schema'] == 1 and value['study'] == 'frozen_interface_comparison_v2'
    assert value['study_version'] == 2 and value['arithmetic_profile'] == PROFILE


def guard_causes_unfused(body, fixed_depth_model):
    """Versioned non-contracted binary32 guard; no FMA CUDA reconstruction."""
    return old.guard_causes(body, fixed_depth_model, 0)


def retained_cases():
    result = []
    def add(category, pose, controls, offset=0, holding=False):
        result.append(dict(case=len(result), category=category, pose=pose,
                           controls=controls, offset_m=float(np.float32(offset)), holding=holding,
                           arena_half_width_m=3. if category == 'long300' else float(np.float32(.6)),
                           arena_half_depth_m=float(np.float32(.4))))
    for pose in range(4): add('short', pose, 160)
    for pose in (0, 3): add('long60', pose, 1200)
    for offset in (0, -.0005, .0005): add('recovery', 0, 1280, offset)
    for pose in (0, 3): add('long300', pose, 6000)
    for category, poses, offsets in [('original-holds', (0,3), (0,-.0005,.0005)),
                                     ('withheld-holds', (1,2), (0,-.00025,.00025))]:
        for offset in offsets:
            for pose in poses: add(category, pose, 1280, offset, True)
    for pose in (0,3): add('endurance-holds', pose, 6000, 0, True)
    assert len(result) == 25 and sum(c['controls'] for c in result) == 46240
    return result


def retained_catalog(root):
    value = json.loads(read(root/'retention-cases.json'))
    identity(value)
    expected = retained_cases()
    assert len(value['cases']) == len(expected)
    for actual, wanted in zip(value['cases'], expected):
        for name, v in wanted.items():
            if isinstance(v, float): near(actual[name], v, 0, 0)
            else: assert actual[name] == v, (name, actual, wanted)
    h = envelope(root/'holding-parent.bin')
    h.at = 4+4+8+8+7*4+2*4
    h.array(size=2532); h.array(size=2)
    gain = h.value('f')
    near(value['holding_gain'], gain, 0, 0)
    return expected, gain


def source_dependencies():
    """Bind the executed reader and imported numerical reference implementations."""
    for module in (old, prep, prior, ref, shared, prior.previous, prior.retained,
                   old.cuda_reference, old.cuda_reference.contract):
        read(module.__file__)
    read(__file__)


def observer(world, task, heading_only):
    """World-space lane mask, then inverse rotation; no body/state modification."""
    body = np.asarray(world['body'], float)
    center, velocity = ref.motion(body)
    heading = np.asarray(task['request']['heading'], float)
    delta = center - np.asarray(task['start_com_model'], float) * ref.SCALE
    lane = heading * np.sum(delta * heading, axis=-1, keepdims=True) - delta
    lane = np.array(lane, copy=True)
    active = np.asarray(heading_only, bool)
    lane[..., 0] = np.where(active, 0, lane[..., 0])
    lane[..., 2] = np.where(active, 0, lane[..., 2])
    q = body[..., 3:7]
    out = np.zeros(body.shape[:-1] + (41,))
    out[..., :3] = ref.rotate(q, lane, True) / float(np.float32(.002))
    out[..., 3:6] = ref.rotate(q, task['request']['terminal_velocity'], True) / float(np.float32(.02))
    out[..., 6:9] = ref.rotate(q, heading, True)
    out[..., 9:12] = ref.rotate(q, [0, 1, 0], True)
    out[..., 12:15] = ref.rotate(q, velocity, True) / float(np.float32(.02))
    out[..., 15:18] = ref.rotate(q, body[..., 14:17], True) / 4
    out[..., 18:22] = body[..., 7:11]
    out[..., 22:26] = body[..., 17:21] / 12
    out[..., 26:30] = world['last_torque']
    out[..., 30:34] = world['held']
    out[..., 34] = np.maximum(0, 1 - task['controls'] / 40)
    out[..., 35] = 1
    out[..., 36:40] = world['rates'] / 12
    return out


def expected_heading(initial, offset_degrees):
    theta = math.radians(offset_degrees)
    x, y, z = map(float, initial)
    return np.array([math.cos(theta)*x + math.sin(theta)*z, y,
                     -math.sin(theta)*x + math.cos(theta)*z])


def task_snapshot(task, controls=None):
    result = np.array([task], dtype=task.dtype)
    if controls is not None:
        result['controls'][0] = controls
    return result


def compare_backend(left, right):
    assert len(left) == len(right)
    assert np.array_equal(left['tick'], right['tick'])
    for key in ('observation', 'base', 'command'):
        old.f32_equal(left[key], right[key])
    tolerance = np.array([2e-5]*11 + [2e-7]*3 + [2e-5]*7)
    difference = abs(left['world']['body'][:, :21].astype(float) -
                     right['world']['body'][:, :21].astype(float))
    assert np.all(difference <= tolerance)
    maximum_motor = 0.
    for field in ('held', 'rates', 'last_torque', 'actual', 'requested'):
        maximum_motor = max(maximum_motor, near(left['world'][field], right['world'][field], 2e-5, 0))
    a, b = left['world']['energy'], right['world']['energy']
    error = abs(a-b)
    assert np.all(error <= 1e-12 + 1e-4*np.maximum(abs(a), abs(b)))
    return {'position': float(difference[:, :3].max(initial=0)),
            'motor': maximum_motor, 'energy': float(error.max(initial=0))}


def geometry(body):
    """Independent articulated coordinates and conservative bank bounds."""
    body = np.asarray(body, float)
    center, velocity = ref.motion(body)
    active, inset, clearance = prior.boundary_values(body)
    up = ref.rotate(body[..., 3:7], [0, 1, 0])
    tilt = np.arccos(np.clip(up[..., 1], -1, 1))*180/float(np.float32(np.pi))
    return center, velocity, active, inset, clearance, tilt


def new_errors():
    return {key: 0. for key in ('observation', 'NN', 'command',
                              'filter_angle', 'filter_rate', 'energy_identity')}


def event_at(case, tick):
    return next((e for e in case['events'] if int(e['before_control']) == tick), None)


def active_mode(case, tick):
    return case['arm'] == 'heading_only' and tick > int(case['warm_controls'])


def canonical_case(case_id, gate=False):
    if gate:
        assert 0 <= case_id < 4
        fixture, arm = case_id//2, case_id%2
        pose, warm, scenario, maximum = (0, 80, 0, 120) if fixture == 0 else (3, 90, 1, 130)
    else:
        assert 0 <= case_id < 88
        fixture, scenario, arm = case_id//22, (case_id//2)%11, case_id%2
        pose, warm = ((0, 80), (3, 80), (1, 90), (2, 90))[fixture]
        post = 320 if scenario <= 4 else 260 if scenario <= 6 else 270 if scenario <= 8 else 340
        maximum = warm+post
    sign = 1 if scenario%2 else -1
    if scenario == 0:
        changes = [(0, 0)]
    elif scenario <= 4:
        changes = [(0, sign*(5 if scenario <= 2 else 15))]
    elif scenario <= 6:
        changes = [(20*i, sign*5*(i+1)) for i in range(6)]
    elif scenario <= 8:
        changes = [(10*i, sign*5*(i+1)) for i in range(12)]
    else:
        changes = [(20*i, sign*5*(i+1)) for i in range(6)]
        changes += [(160+20*i, sign*5*(5-i)) for i in range(6)]
    return {'case': case_id, 'fixture': fixture, 'pose': pose,
            'warm_controls': warm, 'scenario': scenario, 'arm': ARMS[arm],
            'maximum_controls': maximum,
            'events': [{'before_control': warm+1+t, 'offset_degrees': angle,
                        'rebase_lane': scenario != 0} for t, angle in changes]}


def canonical_cases():
    return {'schema': 1, 'study': 'frozen_interface_comparison_v2',
            'cases': [canonical_case(i) for i in range(88)],
            'gate_cases': [canonical_case(i, True) for i in range(4)]}


def validate_cases(metadata):
    wanted = canonical_cases()
    assert metadata['schema'] == 1 and metadata['study'] == wanted['study']
    for name in ('cases', 'gate_cases'):
        assert len(metadata[name]) == len(wanted[name])
        for actual, expected in zip(metadata[name], wanted[name]):
            for key, value in expected.items():
                assert actual[key] == value, (name, expected['case'], key, actual[key], value)
    assert sum(c['maximum_controls'] for c in wanted['cases']) == 34200
    assert 2*sum(c['maximum_controls'] for c in wanted['gate_cases']) == 1000
    return wanted


def case_metrics(rows, sub, case):
    """Preregistered functional gates and descriptive body-motion measurements."""
    warm, maximum = case['warm_controls'], case['maximum_controls']
    last = rows[-1]
    completed = int(last['tick']) == maximum and not last['reason'] and not last['failure']
    center, velocity = ref.motion(rows['world']['body'].astype(float))
    elapsed = rows['accepted'].astype(float)*float(np.float32(.005))
    arc = np.r_[0., np.linalg.norm(np.diff(center[:, [0, 2]], axis=0), axis=1)]
    subbody = sub['data'][:, [sub['index'][x] for x in BODY]].astype(np.float32)
    if len(subbody):
        com, _, active, inset, clearance, tilt = geometry(subbody)
        depth = abs(com[:, 1]-float(rows[0]['task']['start_com_model'][1])*ref.SCALE)
        depth_max, tilt_max = float(depth.max()), float(tilt.max())
        minimum_clearance = float(np.min(clearance))
        maximum_active, maximum_inset = float(active.max()), float(inset.max())
        _, gap = old.kinematics(subbody)
        minimum_arena_gap = float(gap.min())
    else:
        depth_max = tilt_max = minimum_clearance = maximum_active = maximum_inset = minimum_arena_gap = None
    min8 = []
    min2net = []
    for t in range(warm+160, len(rows)):
        first = t-159
        if np.all(rows['accepted'][first:t+1] == 10):
            min8.append(float(arc[first:t+1].sum()/elapsed[first:t+1].sum()))
    for t in range(warm+40, len(rows)):
        first = t-39
        if np.all(rows['accepted'][first:t+1] == 10):
            min2net.append(float(np.linalg.norm(center[t, [0, 2]]-center[first-1, [0, 2]])))
    # Boxcar COM-velocity courses: null below 2 mm/s. Once a gap occurs,
    # accumulated course cannot be credited by bridging an unobserved angle.
    means = np.full_like(velocity, np.nan)
    speed = np.full(len(rows), np.nan)
    phi = np.full(len(rows), np.nan)
    for t in range(20, len(rows)):
        if np.all(rows['accepted'][t-19:t+1] == 10):
            means[t] = velocity[t-19:t+1].mean(axis=0)
            speed[t] = np.linalg.norm(means[t, [0, 2]])
            if speed[t] >= .002:
                phi[t] = math.atan2(-means[t, 2], means[t, 0])
    progress = np.full(len(rows), np.nan)
    angles = np.full(len(rows), np.nan)
    baseline_valid = warm < len(rows) and np.isfinite(phi[warm])
    if baseline_valid:
        progress[warm] = 0
        angles[warm] = phi[warm]
        for t in range(warm+1, len(rows)):
            if not np.isfinite(phi[t]) or not np.isfinite(angles[t-1]):
                continue
            angles[t] = angles[t-1]+math.atan2(math.sin(phi[t]-phi[t-1]), math.cos(phi[t]-phi[t-1]))
            progress[t] = (angles[t]-angles[warm])*180/math.pi
    late = {'late_forward_m_s': None, 'late_horizontal_speed_m_s': None,
            'late_course_error_deg': None, 'late_vector_rmse_m_s': None,
            'late_course_from_original_deg': None, 'final_course_progress_deg': None}
    if completed:
        final_v = velocity[-80:]
        assert len(final_v) == 80 and np.all(rows['accepted'][-80:] == 10)
        desired = last['task']['request']['heading'].astype(float)
        mean = final_v.mean(axis=0)
        late['late_forward_m_s'] = float(np.dot(mean, desired))
        late['late_horizontal_speed_m_s'] = float(np.linalg.norm(mean[[0, 2]]))
        late['late_vector_rmse_m_s'] = float(np.sqrt(np.mean(np.sum((final_v-.008*desired)**2, axis=1))))
        if late['late_horizontal_speed_m_s'] >= .002:
            late['late_course_error_deg'] = float(old.course(desired, mean))
            late['late_course_from_original_deg'] = float(old.course(rows[0]['task']['request']['heading'], mean))
            if np.isfinite(angles[-1]):
                angle = math.atan2(-mean[2], mean[0])
                angle += round((angles[-1]-angle)/(2*math.pi))*2*math.pi
                late['final_course_progress_deg'] = float((angle-angles[warm])*180/math.pi)
    first_offset = next((e['offset_degrees'] for e in case['events'] if e['offset_degrees']), 0)
    response = None
    if first_offset:
        for t in range(warm+1, len(rows)):
            if np.isfinite(progress[t]) and np.sign(first_offset)*progress[t] >= abs(first_offset)/2:
                response = float(elapsed[warm+1:t+1].sum())
                break
    radii = []
    for t in range(warm+1, len(rows)):
        if np.isfinite(angles[t]) and np.isfinite(angles[t-1]):
            rate = (angles[t]-angles[t-1])/elapsed[t]
            if abs(rate) >= math.pi/180:
                radii.append(float(speed[t]/abs(rate)))
    reverse_progress = None
    if case['scenario'] >= 9 and warm+160 < len(rows) and np.isfinite(progress[warm+160]):
        reverse_progress = float(progress[warm+160])
    turn_ok = case['scenario'] == 0
    if 1 <= case['scenario'] <= 8:
        target = case['events'][-1]['offset_degrees']
        turn_ok = (late['final_course_progress_deg'] is not None and
                   np.sign(target)*late['final_course_progress_deg'] >= abs(target)/2)
    elif case['scenario'] >= 9:
        turn_ok = (reverse_progress is not None and np.sign(first_offset)*reverse_progress >= 15 and
                   late['late_course_from_original_deg'] is not None and abs(late['late_course_from_original_deg']) <= 5)
    functional = bool(completed and depth_max is not None and depth_max <= .001 and tilt_max <= 10 and
                      late['late_forward_m_s'] >= .006 and late['late_course_error_deg'] is not None and
                      abs(late['late_course_error_deg']) <= 5 and late['late_vector_rmse_m_s'] <= .0025 and
                      min8 and min(min8) >= .003 and turn_ok)
    accepted = rows['accepted'].astype(int)
    demanded = rows['world']['requested'].reshape(-1, 10, 4)
    selected = demanded[np.arange(10)[None, :] < accepted[:, None]]
    fraction = abs(selected.astype(float))/float(np.float32(1e-5))
    return {'case': case['case'], 'fixture': case['fixture'], 'pose': case['pose'],
            'warm_controls': warm, 'scenario': case['scenario'], 'arm': case['arm'],
            'maximum_controls': maximum, 'controls': int(last['tick']),
            'completed': bool(completed), 'reason': int(last['reason']),
            'hard_failure': int(last['failure']), 'functional_pass': functional,
            'turn_progress_pass': bool(turn_ok), 'maximum_depth_m': depth_max,
            'maximum_tilt_deg': tilt_max, 'minimum_bank_clearance_lower_bound_m': minimum_clearance,
            'maximum_active_radius_squared': maximum_active, 'maximum_inset_radius_squared': maximum_inset,
            'minimum_arena_gap_m': minimum_arena_gap,
            'postrequest_horizontal_arc_m': float(arc[warm+1:].sum()),
            'minimum_8s_postrequest_speed_m_s': min(min8) if min8 else None,
            'minimum_2s_postrequest_net_displacement_m': min(min2net) if min2net else None,
            'invalid_1s_course_windows': int(np.count_nonzero(~np.isfinite(phi[warm+1:]))),
            'contiguous_course_chain_valid': bool(baseline_valid and len(rows) > warm and np.isfinite(angles[-1])),
            'reverse_pre_return_course_progress_deg': reverse_progress,
            'first_half_initial_request_response_seconds': response,
            'median_local_radius_m': float(np.median(radii)) if radii else None,
            'local_radius_valid_samples': len(radii),
            'positive_work_J': float(last['world']['energy'][1]),
            'saturation_samples_per_axis': (fraction >= .999).sum(axis=0).tolist() if len(fraction) else [0]*4,
            'peak_requested_torque_fraction_per_axis': fraction.max(axis=0).tolist() if len(fraction) else [0.]*4,
            **late}


def summarize(cases):
    summary = []
    for arm in ARMS:
        items = [x for x in cases if x['arm'] == arm]
        nonzero = [x for x in items if x['scenario'] != 0]
        median = np.median([x['late_forward_m_s'] if x['completed'] else 0. for x in nonzero]) if nonzero else 0.
        summary.append({'arm': arm, 'cases': len(items),
                        'completed': sum(x['completed'] for x in items),
                        'zero_passes': sum(x['functional_pass'] for x in items if x['scenario'] == 0),
                        'nonzero_passes': sum(x['functional_pass'] for x in nonzero),
                        'repeated_return_passes': sum(x['functional_pass'] for x in items if x['scenario'] >= 5),
                        'tilt_stops': sum(x['reason'] == 5 for x in items),
                        'depth_stops': sum(x['reason'] == 4 for x in items),
                        'bank_stops': sum(x['reason'] == 1 for x in items),
                        'median_late_forward_m_s_incomplete_zero': float(median)})
    ranks = [(x['zero_passes'], x['repeated_return_passes'], x['nonzero_passes'],
              x['median_late_forward_m_s_incomplete_zero']) for x in summary]
    if ranks[0][:3] == ranks[1][:3] and abs(ranks[0][3]-ranks[1][3]) < 1e-9:
        preferred = 'tie'
    else:
        preferred = ARMS[1 if ranks[1] > ranks[0] else 0]
    adequate = []
    for arm in ARMS:
        items = [x for x in cases if x['arm'] == arm]
        required = [x for x in items if x['scenario'] in (0, 5, 6, 9, 10)]
        if len(required) == 20 and all(x['functional_pass'] for x in required):
            adequate.append(arm)
    return summary, {'status': 'relative_preference_only', 'preferred_arm': preferred,
                     'adequate_arms': adequate, 'navigation_qualified': False,
                     'rule': 'lexicographic zero/repeated-return/nonzero passes, then median late forward speed; incomplete=0; speed tie<1e-9'}


def protocol_binding(root):
    p = envelope(root/'protocol.bin')
    assert [p.value('I'), p.value('I')] == [0x32524946, 2]
    checksum = shared.fnv(p.data)
    assert p.value('Q') == shared.fnv(read(root/'approved-protocol.md'))
    assert p.value('Q') == shared.fnv(read(root/'cases.json'))
    assert p.value('Q') == shared.fnv(read(root/'retention-cases.json'))
    assert p.value('Q') == shared.fnv(read(root/'arithmetic-profile.json'))
    for name in ('parent-checkpoint.bin', 'parent-protocol.bin', 'holding-parent.bin'):
        assert p.value('Q') == shared.fnv(envelope(root/name).data)
    assert [p.value('i') for _ in range(11)] == [6, 4, 41, 4, 336, 88, 4, 430, CONTROL_CAP, SUBSTEP_CAP, SECONDS_CAP]
    near([p.value('f') for _ in range(7)], np.float32([.008, .005, 20, .20, .12, .195, .115]), 0, 0)
    assert p.value('Q') == EVIDENCE_CAP
    assert [p.value('I') for _ in range(3)] == [1, 25, 46240]
    p.finish()
    cases = validate_cases(json.loads(read(root/'cases.json')))
    return checksum, cases


def verify_freeze(root):
    freeze = json.loads(read(root/'frozen-inputs.json'))
    entries = freeze['files'] if isinstance(freeze, dict) else freeze
    assert entries
    for item in entries:
        assert hashlib.sha256(read(item['path'])).hexdigest().lower() == item['sha256'].lower()
        if 'snapshot' in item:
            assert hashlib.sha256(read(item['snapshot'])).hexdigest().lower() == item['sha256'].lower()
    return len(entries)


def frame_type(layout):
    frame, task = prep.types(layout)
    definitions = [(key, old.world_type() if key == 'world' else frame.fields[key][0]) for key in frame.names]
    definitions += [('first_before', ('<f4', 23)), ('candidate', ('<f4', 23))]
    context = prep.dtype(layout['case'], [(key, '<i4') for key in ('id', 'fixture', 'pose', 'warm', 'scenario', 'arm', 'maximum','retention_kind','retention_category')]+[(key,'<f4') for key in ('gain','offset')])
    definitions += [('context', context), ('request_active', '<i4'), ('offset_degrees', '<i4')]
    return prep.dtype(layout['frame'], definitions), task


def verify_arena(worlds, layout, width, depth):
    descriptor = np.dtype({'names':['arena'],'formats':[('<f4',2)],
                           'offsets':[layout['World']['arena']], 'itemsize':696})
    assert layout['Arena']==dict(size=8,half_width_m=0,half_depth_m=4)
    values=np.frombuffer(np.asarray(worlds).tobytes(),dtype=descriptor)['arena']
    old.f32_equal(values,np.broadcast_to(np.float32([width,depth]),values.shape))


def verify_context(frames, cases):
    assert len(frames) == len(cases)
    for frame, case in zip(frames, cases):
        expected = {'id': case['case'], 'fixture': case['fixture'], 'pose': case['pose'],
                    'warm': case['warm_controls'], 'scenario': case['scenario'],
                    'arm': ARMS.index(case['arm']), 'maximum': case['maximum_controls']}
        for key, value in expected.items():
            assert int(frame['context'][key]) == value, (case['case'], key)


def verify_latch(before, after, case, event):
    assert before['world'].tobytes() == after['world'].tobytes()
    assert before['context'].tobytes() == after['context'].tobytes()
    assert before['task']['controls'] == after['task']['controls']
    assert before['task']['last_world_control'] == after['task']['last_world_control']
    assert before['task']['start_com_model'][1] == after['task']['start_com_model'][1]
    assert after['request_active'] == 1
    assert after['offset_degrees'] == event['offset_degrees']
    if not event['rebase_lane']:
        assert before['task'].tobytes() == after['task'].tobytes()
    else:
        target = expected_heading(before['initial_heading'], event['offset_degrees'])
        near(after['task']['request']['heading'], target, 2e-7, 2e-6)
        near(after['task']['request']['terminal_velocity'], target*float(np.float32(.008)), 2e-8, 2e-6)
        origin = after['task']['start_com_model']
        near(origin[[0, 2]], ref.motion(before['world']['body'])[0][[0, 2]]/ref.SCALE, 3e-6, 3e-6)
        near(after['task']['request']['position'], origin+target*(float(np.float32(.072))/ref.SCALE), 3e-6, 3e-6)


def verify_preparation(root):
    assert not (root/'probe-started').exists() and not (root/'authorization.bin').exists()
    report = json.loads(read(root/'preparation.json'))
    identity(report)
    assert report['passed']
    assert report['physical_controls'] == report['physical_substeps'] == report['PPO_updates'] == report['optimizer_steps'] == 0
    assert report['physical_run_approved'] is False
    assert report['initial_fixtures'] == report['observer_fixtures'] == 88
    assert report['request_latch_fixtures'] == 520
    assert report['evidence_cap_bytes'] == EVIDENCE_CAP
    assert 0 < report['conservative_generated_evidence_bytes'] < report['evidence_cap_bytes']
    checksum, metadata = protocol_binding(root)
    arithmetic_sha = bind_arithmetic_preparation(root)
    retained, gain = retained_catalog(root)
    profile = json.loads(read(root/'arithmetic-profile.json'))
    assert profile == dict(schema=1,study_version=2,arithmetic_profile=PROFILE,arithmetic_profile_id=1,
                          cuda_implicit_fma=False,cuda_flag='--fmad=false',host_flags_changed=False,
                          NN_static_libraries_changed=False,force_equations_changed=False,weights_changed=False,**RADIUS_METADATA)
    assert report['explicit_CUDA_kernel_launches']==531 and report['NN_forward_calls']==3
    assert report['native_runs']==1 and report['NN_library_kernel_launches_measured'] is False
    weights = old.load_weights(root)
    layout = json.loads(read(root/'layout.json'))
    ftype, _ = frame_type(layout)
    fixture = envelope(root/'fixtures.bin')
    assert [fixture.value('I'), fixture.value('I'), fixture.value('Q'), fixture.value('I')] == [0x32464946, 2, checksum, ftype.itemsize]
    initial, before, after = [fixture.array(ftype, 88) for _ in range(3)]
    fixture.finish()
    errors = {'body': 0., 'observation': 0., 'NN': 0., 'command': 0.}
    for frames, synthetic in ((initial, False), (after, True)):
        verify_arena(frames['world'],layout,.6,.4)
        verify_context(frames, metadata['cases'])
        assert np.all(frames['valid'] == 1)
        for field in ('controls', 'substeps', 'attempted', 'time', 'failure', 'unsafe', 'invalid'):
            assert np.all(frames['world'][field] == 0)
        assert np.all(frames['task']['controls'] == 0) and np.all(frames['task']['last_world_control'] == 0)
        if not synthetic:
            expected = ref.initial_body(np.array([c['pose'] for c in metadata['cases']]))
            errors['body'] = near(frames['world']['body'][:, :21], expected, 2e-7, 2e-6)
            assert np.all(frames['request_active'] == 0)
            assert np.all(frames['offset_degrees'] == 0)
        enabled = (frames['context']['arm'] == 1) & (frames['request_active'] == 1)
        wanted = observer(frames['world'], frames['task'], enabled)
        errors['observation'] = max(errors['observation'], near(frames['observation'], wanted, 2e-5, 2e-5))
        z = shared.forward(weights, frames['observation'].astype(float))
        errors['NN'] = max(errors['NN'], near(frames['base'], z, 2e-5, 2e-6))
        errors['command'] = max(errors['command'], near(frames['command'], np.tanh(frames['base'].astype(float)), 2e-5, 0))
    for a, b in zip(before, after):
        assert a['world'].tobytes() == b['world'].tobytes()
        assert a['task'].tobytes() == b['task'].tobytes()
        assert a['context'].tobytes() == b['context'].tobytes()
    # Confirm the static probes actually exercise tilt and a nonzero depth signal.
    _, _, _, _, _, tilt = geometry(after['world']['body'])
    assert np.all(tilt > 1)
    bframes = after[after['context']['arm'] == 1]
    local = bframes['observation'][:, :3].astype(float)*float(np.float32(.002))
    lane = ref.rotate(bframes['world']['body'][:, 3:7].astype(float), local)
    near(lane[:, [0, 2]], np.zeros((44, 2)), 1e-7, 0)
    assert np.all(abs(lane[:, 1]) > 1e-5)
    latch = envelope(root/'pure-latches.bin')
    assert [latch.value('I'), latch.value('I'), latch.value('Q'), latch.value('I')] == [0x324C4946, 2, checksum, 520]
    count = 0
    for case in metadata['cases']:
        for event in case['events']:
            assert [latch.value('I'), latch.value('I')] == [case['case'], event['before_control']]
            a, b, c = [prep.raw_record(latch, ftype) for _ in range(3)]
            verify_latch(a, b, case, event)
            verify_latch(a, c, case, event)
            near(b['task']['request']['heading'], c['task']['request']['heading'], 2e-5, 0)
            count += 1
    latch.finish()
    assert count == 520
    guards = envelope(root/'pure-guards.bin')
    assert [guards.value('I'), guards.value('I'), guards.value('Q'), guards.value('I')] == [0x32474946, 2, checksum, 12]
    expected = [0, 0, 4, 0, 4, 0, 5, 0, 1, 104, 4, 1]
    for i, wanted in enumerate(expected):
        assert [guards.value('I'), guards.value('I')] == [i, wanted]
        frame = prep.raw_record(guards, ftype)
        assert [guards.value('I'), guards.value('I')] == [wanted, wanted]
        active, _, _ = prior.boundary_values(frame['world']['body'][None, :])
        for backend in (0, 1):
            actual = 104 if active[0] > 1 else int(guard_causes_unfused(frame['world']['body'][None, :], frame['task']['start_com_model'][1])[0])
            assert actual == wanted, (i, backend, actual, wanted)
    guards.finish()
    retention = envelope(root/'retention-fixtures.bin')
    assert [retention.value('I'),retention.value('I'),retention.value('Q'),retention.value('I')] == [0x32544946,2,checksum,ftype.itemsize]
    initial_retained, observed_retained = [retention.array(ftype,25) for _ in range(2)]
    retention.finish()
    for case,before,after in zip(retained,initial_retained,observed_retained):
        assert before['world'].tobytes()==after['world'].tobytes()
        # A hold observer may latch its request; initialization already did so.
        assert before['task'].tobytes()==after['task'].tobytes()
        verify_arena(after['world'],layout,case['arena_half_width_m'],case['arena_half_depth_m'])
        context=after['context']
        assert context['id']==case['case'] and context['pose']==case['pose'] and context['maximum']==case['controls']
        assert context['retention_kind']==(2 if case['holding'] else 1)
        assert context['retention_category']==['short','long60','recovery','long300','original-holds','withheld-holds','endurance-holds'].index(case['category'])
        near(context['gain'],gain if case['holding'] else 1,0,0)
        near(context['offset'],case['offset_m'],0,0)
        assert all(after['world'][field]==0 for field in ('controls','substeps','attempted','time','unsafe','invalid','failure'))
        expected=ref.initial_body(case['pose']).copy();expected[1]+=case['offset_m']/ref.SCALE
        near(after['world']['body'][:21],expected,2e-7,2e-6)
        wanted=observer(after['world'],after['task'],False)
        if case['holding']:
            center,_=ref.motion(after['world']['body'])
            anchor=center.copy();anchor[1]=after['task']['start_com_model'][1]*ref.SCALE
            wanted[40]=np.tanh(np.sum((anchor-center)*after['task']['request']['heading'])/float(np.float32(.02)))
        near(after['observation'],wanted,2e-5,2e-5)
        near(after['base'],shared.forward(weights,after['observation'][None,:].astype(float))[0],2e-5,2e-6)
        near(after['command'],np.tanh(after['base'].astype(float)*(gain if case['holding'] else 1)),2e-5,0)
    frozen_count = verify_freeze(root)
    source_dependencies()
    return {'schema': 1, 'study': 'frozen_interface_comparison_v2', 'phase': 'preparation',
            'study_version':2, 'arithmetic_profile':PROFILE, 'arithmetic_preparation_sha256':arithmetic_sha,
            'passed': True, 'physical_run_approved': False,
            'physical_controls': 0, 'physical_substeps': 0, 'optimizer_steps': 0,
            'PPO_updates': 0, 'initial_fixtures': 88, 'observer_fixtures': 88,
            'latched_request_fixtures_per_backend': 520, 'guard_fixtures_per_backend': 12,
            'retained_skill_observer_fixtures':25,
            'recorded_explicit_CUDA_kernel_launches':531,'recorded_NN_forward_calls':3,
            'NN_library_kernel_launches_measured':False,
            'maximum_errors': errors, 'frozen_inputs': frozen_count,
            'protocol_checksum': checksum, 'weights_unchanged': True,
            'input_sha256': dict(sorted(prep.HASHES.items()))}


def verify_transitions(rows, sub, weights, case, errors, cpu_rows=None):
    """Check all collection rows independently, including every accepted substep."""
    assert len(rows) and rows[0]['tick'] == 0
    key = tuple(int(rows[0][x]) for x in ('group', 'case', 'backend'))
    assert np.array_equal(rows['tick'], np.arange(len(rows)))
    first = rows[0]
    retained = key[0] == 2
    expected_body = ref.initial_body(case['pose']).copy()
    if retained: expected_body[1] += case['offset_m']/ref.SCALE
    near(first['world']['body'][:21], expected_body, 2e-7, 2e-6)
    assert all(first['world'][x] == 0 for x in ('controls', 'attempted', 'substeps'))
    assert first['task']['controls'] == first['task']['last_world_control'] == 0
    origin = first['task']['start_com_model'].copy()
    request = first['task']['request'].copy()
    initial_heading = request['heading'].copy()
    # Batch likelihood-free inference/observation checks across all rows. This
    # changes reader throughput only; every recorded row remains included.
    if len(rows) > 1:
        source_rows = cpu_rows if key[0] == 0 and key[2] == 1 else rows
        assert source_rows is not None and len(source_rows) >= len(rows)
        observation_task = rows['task'][1:].copy()
        observation_task['controls'] = source_rows['task'][:len(rows)-1]['controls']
        if source_rows is not rows:
            observation_task = source_rows['task'][1:len(rows)].copy()
            observation_task['controls'] = source_rows['task'][:len(rows)-1]['controls']
        wanted = observer(source_rows['world'][:len(rows)-1], observation_task,
                          False if retained else (case['arm'] == 'heading_only') & (rows['tick'][1:] > case['warm_controls']))
        gain = 1.
        if retained and case['holding']:
            center, _ = ref.motion(source_rows['world']['body'][:len(rows)-1])
            anchor = ref.motion(first['world']['body'])[0].copy()
            anchor[1] = origin[1]*ref.SCALE
            wanted[:, 40] = np.tanh(np.sum((anchor-center)*initial_heading, axis=-1)/float(np.float32(.02)))
            gain = case['gain']
        errors['observation'] = max(errors['observation'], near(rows['observation'][1:], wanted, 2e-5, 2e-5))
        z = shared.forward(weights, rows['observation'][1:].astype(float))
        errors['NN'] = max(errors['NN'], near(rows['base'][1:], z, 2e-5, 2e-6))
        errors['command'] = max(errors['command'], near(rows['command'][1:], np.tanh(gain*rows['base'][1:].astype(float)), 2e-5, 0))
    # Reconstruct expensive articulated geometry in batches, without sampling.
    all_body = old.cols(sub, BODY).astype(np.float32)
    all_ke, all_gap = old.kinematics(all_body)
    if retained:
        corners = prior.articulated_corners(all_body)
        all_gap = np.minimum.reduce((case['arena_half_width_m']-abs(corners[...,0]),
                                     case['arena_half_depth_m']-abs(corners[...,2]),
                                     corners[...,1]+float(np.float32(.16)), -corners[...,1])).min(axis=-1)
        all_active = np.zeros(len(all_body))
        all_causes = np.zeros(len(all_body), int)
    else:
        all_active = prior.boundary_values(all_body)[0]
        all_causes = guard_causes_unfused(all_body, origin[1])
    consumed = 0
    for j in range(1, len(rows)):
        before, after = rows[j-1], rows[j]
        w0, w1 = before['world'], after['world']
        tick = int(after['tick'])
        event = None if retained else event_at(case, tick)
        if event is not None:
            old_request = request.copy()
            target = expected_heading(initial_heading, event['offset_degrees'])
            request = after['task']['request'].copy()
            near(request['heading'], target, 2e-7, 2e-6)
            near(request['terminal_velocity'], target*float(np.float32(.008)), 2e-8, 2e-6)
            if event.get('rebase_lane', False):
                origin = after['task']['start_com_model'].copy()
                near(origin[[0, 2]], ref.motion(w0['body'])[0][[0, 2]]/ref.SCALE, 3e-6, 3e-6)
                assert origin[1] == first['task']['start_com_model'][1]
            else:
                assert request.tobytes() == old_request.tobytes()
            near(request['position'], origin + request['heading']*(float(np.float32(.072))/ref.SCALE), 3e-6, 3e-6)
        assert after['task']['request'].tobytes() == request.tobytes()
        old.f32_equal(after['task']['start_com_model'], origin)
        if not after['failure']:
            assert after['task']['controls'] == w1['controls']
            assert after['task']['last_world_control'] == w1['controls']
        else:
            assert j == len(rows)-1
            assert 0 <= int(w1['controls'])-int(after['task']['controls']) <= 1
        attempted, accepted = int(after['attempted']), int(after['accepted'])
        assert 0 <= accepted <= attempted <= 10
        assert int(w1['attempted'])-int(w0['attempted']) == attempted
        assert int(w1['substeps'])-int(w0['substeps']) == accepted
        if not after['failure']:
            assert accepted == attempted and 1 <= accepted <= 10
            assert w1['controls'] == w0['controls']+1
            assert accepted == 10 or after['reason'] in (1, 4, 5)
        assert np.isfinite(w1['actual']).all() and np.isfinite(w1['requested']).all()
        assert np.all(abs(w1['actual']) <= 1)
        if accepted:
            applied = w1['actual'][:accepted*4].astype(float)
            demanded = w1['requested'][:accepted*4].astype(float)
            decoded = np.clip(demanded, -float(np.float32(1e-5)), float(np.float32(1e-5)))/float(np.float32(1e-5))
            near(applied, decoded, 2e-7, 2e-6)
        if after['failure'] not in (101, 102, 103):
            assert np.all(w1['actual'][accepted*4:] == 0)
            assert np.all(w1['requested'][accepted*4:] == 0)
        angle, rate = w0['held'].astype(float).copy(), w0['rates'].astype(float).copy()
        h, omega = float(np.float32(.005)), float(np.float32(2*np.float32(np.pi)))
        for _ in range(accepted):
            rate += h*(omega*omega*(after['command']-angle)-2*omega*rate)
            angle += h*rate
        if not after['failure']:
            errors['filter_angle'] = max(errors['filter_angle'], near(w1['held'], angle, 2e-6, 2e-5))
            errors['filter_rate'] = max(errors['filter_rate'], near(w1['rates'], rate, 2e-5, 2e-5))
            old.f32_equal(w1['last_torque'], w1['actual'][(accepted-1)*4:accepted*4])
        segment = sub['data'][consumed:consumed+accepted]
        begin = consumed
        consumed += accepted
        index = sub['index']
        assert len(segment) == accepted
        if not accepted:
            continue
        assert np.array_equal(segment[:, index['control']], np.full(accepted, tick))
        assert np.array_equal(segment[:, index['substep']], np.arange(1, accepted+1))
        assert np.array_equal(segment[:, index['attempted_substeps']], int(w0['attempted'])+np.arange(1, accepted+1))
        assert np.array_equal(segment[:, index['accepted_substeps']], int(w0['substeps'])+np.arange(1, accepted+1))
        body = segment[:, [index[x] for x in BODY]].astype(np.float32)
        energy = segment[:, [index['energy'+str(i)] for i in range(6)]]
        assert np.isfinite(body).all() and np.isfinite(energy).all()
        if not after['failure']:
            old.f32_equal(body[-1], w1['body'])
            near(energy[-1], w1['energy'], 0, 0)
        assert np.all(abs(np.sum(body[:, 3:7].astype(float)**2, axis=1)-1) <= .01)
        assert np.all(abs(body[:, 7:11]) <= 1)
        ke, gap = all_ke[begin:consumed], all_gap[begin:consumed]
        active = all_active[begin:consumed]
        if not after['failure']:
            assert np.all(gap >= .01-2e-7) and np.all(active <= 1+2e-6)
        sequence = np.vstack((w0['energy'], energy))
        for field in (1, 2, 3):
            assert np.all(np.diff(sequence[:, field]) >= -1e-18)
        identity = ke-energy[:, 0]+energy[:, 2]+energy[:, 3]-energy[:, 4]
        errors['energy_identity'] = max(errors['energy_identity'], near(energy[:, 5], identity, 1e-12, 1e-4))
        if not after['failure']:
            assert np.all(abs(energy[:, 5]) <= .02*(energy[:, 1]+energy[:, 2])+2e-10)
            causes = all_causes[begin:consumed]
            assert np.all(causes[:-1] == 0)
            assert int(causes[-1]) == int(after['reason'])
    assert consumed == len(sub['data'])
    return {'group': key[0], 'case': key[1], 'backend': key[2],
            'physical_controls': len(rows)-1,
            'attempted_substeps': int(rows['attempted'][1:].sum()),
            'accepted_substeps': int(rows['accepted'][1:].sum()),
            'last_control': int(rows[-1]['tick']),
            'behavior_reason': int(rows[-1]['reason']),
            'hard_failure': int(rows[-1]['failure'])}


def retention_metrics(rows, sub, case):
    body = rows['world']['body'].astype(float)
    complete = len(rows) == case['controls']+1 and not rows[-1]['failure']
    values = ref.metrics(body, case['pose'])
    metric = {'case': case['case'], 'category': case['category'], 'pose': case['pose'],
              'controls': len(rows)-1, 'completed': complete, 'passed': False}
    if not complete:
        return metric
    score, values = ref.swim_metrics(body, case['pose'])
    movement = all(ref.swim_gates(score).values())
    metric.update(movement_passed=movement, movement=score)
    if case['holding']:
        center, velocity = ref.motion(body)
        anchor = center[0].copy()
        anchor[1] = rows[0]['task']['start_com_model'][1]*ref.SCALE
        forward = values['forward']
        hold = dict(max_speed_m_s=float(np.linalg.norm(velocity, axis=-1).max()),
                    max_anchor_m=float(np.linalg.norm(center-anchor, axis=-1).max()),
                    max_depth_m=float(values['depth'].max()), max_lane_m=float(values['lane'].max()),
                    max_tilt_deg=float(values['tilt'].max()),
                    final8_excursion_m=float(abs(forward[-161:]-forward[-161]).max()))
        metric['hold'] = hold
        limits = [.0005, .002, .001, .002, 10., .002]
        metric['passed'] = all(a <= b for a,b in zip(hold.values(), limits))
    elif case['category'] in ('long60','long300'):
        center, _ = ref.motion(body)
        heading = values['heading']
        windows = []
        for end in range(80, case['controls']+1, 80):
            start = 40 if end == 80 else end-80
            delta = center[end]-center[start]
            forward = float(np.sum(delta*heading))
            side = delta-heading*forward
            error = values['velocity'][start+1:end+1]-heading*float(np.float32(.008))
            window = dict(end_s=end//20, steady_mm_s=forward/((end-start)*float(np.float32(.05)))*1000,
                          RMSE_mm_s=float(np.sqrt(np.mean(np.sum(error*error, axis=-1))))*1000,
                          lateral_mm=float(values['lane'][1:end+1].max())*1000,
                          depth_mm=float(values['depth'][1:end+1].max())*1000,
                          tilt_deg=float(values['tilt'][1:end+1].max()),
                          path_deg=float(np.degrees(np.arctan2(np.linalg.norm(side[[0,2]]), forward))),
                          work_J_per_m=float((body[end,21]-body[start,21])/max(forward,1e-9)))
            window['passed'] = all(ref.swim_gates(window).values())
            windows.append(window)
        metric['windows'] = windows
        metric['passed'] = all(w['passed'] for w in windows if w['end_s'] >= 8)
    else:
        recovered = case['offset_m'] == 0 or bool(np.all(values['depth'][-160:] < abs(case['offset_m'])*.5))
        metric['recovery_passed'] = movement and recovered
        metric['passed'] = movement and (recovered if case['category'] == 'recovery' else True)
    subbody = old.cols(sub, BODY)
    dense = ref.metrics(subbody, case['pose'])
    metric['all_5ms_extrema'] = {'maximum_depth_m': float(dense['depth'].max()),
                                'maximum_lane_m': float(dense['lane'].max()),
                                'maximum_tilt_deg': float(dense['tilt'].max())}
    return metric


def retention_results(root, by_key, by_sub, catalog):
    native = json.loads(read(root/'retention.json')) if (root/'retention.json').exists() else None
    if native: identity(native)
    result = {}
    required = dict(short_passes=4, original_holds=6, long60_retained=28,
                    recovery_passes=3, long300_retained=148, withheld_holds=6, endurance_holds=2)
    for backend, name in enumerate(('cpu','cuda')):
        details = [retention_metrics(by_key[(2,c['case'],backend)], by_sub[(2,c['case'],backend)], c)
                   for c in catalog if (2,c['case'],backend) in by_key]
        totals = {key: 0 for key in required}
        keys = {'long60':[], 'long300':[]}
        for case in details:
            category = case['category']
            if category in keys:
                keys[category] += [[case['pose'],w['end_s']] for w in case.get('windows',[]) if w['passed']]
                totals[category+'_retained'] += sum(w['passed'] and w['end_s']>=8 for w in case.get('windows',[]))
            else:
                key = {'short':'short_passes','recovery':'recovery_passes','original-holds':'original_holds',
                       'withheld-holds':'withheld_holds','endurance-holds':'endurance_holds'}[category]
                totals[key] += bool(case['passed'])
        controls = sum(c['controls'] for c in details)
        substeps = sum(int(by_key[(2,c['case'],backend)]['accepted'].sum()) for c in details)
        complete = len(details)==25 and all(c['completed'] for c in details)
        passed = complete and totals == required
        result[name] = dict(passed=passed, completed=complete, controls=controls, substeps=substeps,
                            totals=totals, cases=details)
        if native:
            actual = native[name]
            assert actual['full'] and complete and actual['controls']==controls==46240 and actual['substeps']==substeps==462400
            for key,value in totals.items(): assert actual[key] == value, (name,key,actual[key],value)
            for category in keys:
                assert actual[category+'_windows'] == len(keys[category])
                assert sorted(actual[category+'_passing_keys']) == sorted(keys[category])
    both = all(v['passed'] for v in result.values())
    if native: assert native['both_passed'] == both
    # Bind native score tables and check every completed case and long window.
    if (root/'retention-started/summary.csv').exists():
        table = list(csv.DictReader(io.StringIO(read(root/'retention-started/summary.csv').decode())))
        completed = {(b,c['case']):c for b,name in enumerate(('cpu','cuda')) for c in result[name]['cases'] if c['completed']}
        assert len(table)==len(completed)
        for row in table:
            c = completed[(int(row['backend']),int(row['case']))]
            assert row['category']==c['category'] and int(row['pose'])==c['pose']
            assert int(row['controls'])==c['controls']
            if c['category'].endswith('holds'):
                assert bool(int(row['hold_passed']))==c['passed']
                for field,wanted in c['hold'].items():
                    native_field = {'max_speed_m_s':'max_hold_speed_m_s','max_anchor_m':'max_hold_anchor_m',
                                    'max_depth_m':'max_hold_depth_m','max_lane_m':'max_hold_lane_m',
                                    'max_tilt_deg':'max_hold_tilt_deg'}.get(field,field)
                    near(float(row[native_field]),wanted,2e-6 if field!='max_tilt_deg' else 2e-3,2e-4)
            else:
                assert bool(int(row['movement_passed']))==c['movement_passed']
                if c['category']=='recovery': assert bool(int(row['recovery_passed']))==c['passed']
                for field,scale in [('steady_m_s',1000),('RMSE_m_s',1000),('lateral_m',1000),('depth_m',1000),('tilt_deg',1),('path_deg',1),('work_J_per_m',1)]:
                    key = {'steady_m_s':'steady_mm_s','RMSE_m_s':'RMSE_mm_s','lateral_m':'lateral_mm','depth_m':'depth_mm'}.get(field,field)
                    near(float(row[field])*scale,c['movement'][key],2e-3 if scale==1000 or field in ('tilt_deg','path_deg') else 2e-6,2e-4)
        windows = list(csv.DictReader(io.StringIO(read(root/'retention-started/windows.csv').decode())))
        expected = {(b,c['case'],w['end_s']):w for (b,_),c in completed.items() for w in c.get('windows',[])}
        assert len(windows)==len(expected)
        for row in windows:
            w = expected[(int(row['backend']),int(row['case']),int(float(row['end_s'])))]
            assert bool(int(row['movement_passed']))==w['passed']
            for field,scale in [('steady_m_s',1000),('RMSE_m_s',1000),('lateral_m',1000),('depth_m',1000),('tilt_deg',1),('path_deg',1),('work_J_per_m',1)]:
                key={'steady_m_s':'steady_mm_s','RMSE_m_s':'RMSE_mm_s','lateral_m':'lateral_mm','depth_m':'depth_mm'}.get(field,field)
                near(float(row[field])*scale,w[key],2e-3 if scale==1000 or field in ('tilt_deg','path_deg') else 2e-6,2e-4)
    return result, both


ARITHMETIC_STAGES = [('filter_acceleration',4,'float32'),('filter_rate',4,'float32'),('filter_angle',4,'float32'),
    ('link0',81,'float32'),('link1',81,'float32'),('link2',81,'float32'),('buoyancy_force',3,'float32'),
    ('COM_model',3,'float32'),('COM_velocity_m_s',3,'float32'),('system_M',100,'float32'),
    ('system_R',100,'float32'),('system_C',10,'float32'),('system_force',10,'float32'),
    ('discrete_A',100,'float64'),('discrete_rhs',10,'float64'),('desired_joint_velocity',4,'float64'),
    ('root_rhs',6,'float64'),('root_velocity',6,'float64'),('requested_torque',4,'float32'),
    ('applied_torque',4,'float32'),('actuated_rhs',10,'float64'),('prospective_velocity',10,'float64')]
ARITHMETIC_INPUT = np.dtype([('body','<f4',23),('held','<f4',4),('rate','<f4',4),('command','<f4',4),
                             ('fluid','<f4',3),('original_control','<i4'),('source_backend','<i4')])
ARITHMETIC_OUTPUT = np.dtype([('value','<f8',638),('filter_valid','<i4'),('algebra_valid','<i4')])


def arithmetic_limits():
    """Field-unit absolute budgets, fixed before reading new arithmetic output."""
    absolute, relative = [], []
    for name,count,_ in ARITHMETIC_STAGES:
        a, r = np.full(count,1e-12), np.full(count,1e-4)
        if name.startswith('filter_'): a[:]=0; r[:]=0
        elif name.startswith('link'):
            a = np.r_[np.full(3,2e-8), np.full(4,2e-6), np.full(9,2e-5), np.full(3,1e-14),1e-10,1e-13,np.full(60,2e-6)]
            r[:]=2e-5
        elif name=='buoyancy_force': a[:]=1e-12
        elif name=='COM_model': a[:]=2e-6; r[:]=2e-6
        elif name=='COM_velocity_m_s': a[:]=2e-7; r[:]=0
        elif name in ('system_M','system_R','system_C','system_force'): r[:]=2e-5
        elif name=='desired_joint_velocity': a[:]=2e-5; r[:]=0
        elif name in ('root_velocity','prospective_velocity'):
            a=np.r_[np.full(3,2e-7),np.full(count-3,2e-5)]; r[:]=0
        elif name in ('requested_torque','applied_torque'): a[:]=1e-10
        absolute.extend(a); relative.extend(r)
    assert len(absolute)==638
    return np.asarray(absolute),np.asarray(relative)


def arithmetic_evidence(directory, inputs, input_checksum, profile):
    summary = json.loads(read(directory/'summary.json'))
    assert summary['schema']==1 and summary['study']=='frozen_interface_arithmetic_v2' and summary['study_version']==2
    assert summary['profile_id']==profile and summary['rows']==len(inputs)
    assert summary['arithmetic_profile']==(PROFILE if profile else 'historical_implicit_fma_reference_v1')
    assert summary['CPU_zero_time_algebra_evaluations']==summary['CUDA_zero_time_algebra_evaluations']==642
    assert summary['explicit_CUDA_kernel_launches']==summary['native_runs']==1
    assert summary['physical_controls']==summary['physical_substeps']==summary['integration_calls']==summary['PPO_updates']==summary['optimizer_steps']==0
    assert summary['body_inputs_unchanged'] and summary['finite_and_algebra_valid']
    layout = json.loads(read(directory/'trace-layout.json'))
    assert layout['input_bytes']==160 and layout['output_bytes']==5112 and layout['values']==638 and not layout['integration_performed']
    offset=0
    for stage, expected in zip(layout['stages'],ARITHMETIC_STAGES):
        name,count,precision=expected
        assert stage==dict(name=name,offset=offset,count=count,precision=precision)
        offset+=count
    assert len(layout['stages'])==len(ARITHMETIC_STAGES) and offset==638
    r=envelope(directory/'trace.bin')
    assert [r.value('I') for _ in range(5)]==[0x324F4946,2,len(inputs),5112,profile]
    assert r.value('Q')==input_checksum
    copy=r.array(ARITHMETIC_INPUT,len(inputs)); assert copy.tobytes()==inputs.tobytes()
    cpu,gpu=[r.array(ARITHMETIC_OUTPUT,len(inputs)) for _ in range(2)]
    r.finish()
    assert all(np.all(x['filter_valid']==1) and np.all(x['algebra_valid']==1) and np.isfinite(x['value']).all() for x in (cpu,gpu))
    exact_filter=cpu['value'][:,:12].tobytes()==gpu['value'][:,:12].tobytes()
    neutral=inputs['fluid'][:,0]==1000
    exact_neutral=bool(np.all(cpu['value'][neutral,255:258]==0) and np.all(gpu['value'][neutral,255:258]==0))
    assert summary['filter_bits_exact']==exact_filter and summary['neutral_force_exact_zero']==exact_neutral
    assert summary['neutral_density_rows']==int(neutral.sum())==214
    difference=abs(cpu['value']-gpu['value'])
    absolute,relative=arithmetic_limits()
    within=difference<=absolute+relative*np.maximum(abs(cpu['value']),abs(gpu['value']))
    stages=[];offset=0; first=None
    unequal_all=cpu['value']!=gpu['value']
    first_index=tuple(np.argwhere(unequal_all)[0]) if unequal_all.any() else None
    for name,count,precision in ARITHMETIC_STAGES:
        d=difference[:,offset:offset+count]
        unequal=cpu['value'][:,offset:offset+count]!=gpu['value'][:,offset:offset+count]
        stage={'name':name,'offset':offset,'count':count,'precision':precision,
               'maximum_absolute_difference':float(d.max()),'all_within_fixed_limits':bool(within[:,offset:offset+count].all()),
               'absolute_limits':absolute[offset:offset+count].tolist(),'relative_limits':relative[offset:offset+count].tolist(),
               'unequal_fields':int(unequal.sum())}
        if first_index is not None and offset<=first_index[1]<offset+count:
            row,field=first_index[0],first_index[1]-offset
            first=dict(stage=name,row=int(row),field=int(field),source_backend=int(inputs[row]['source_backend']),
                       original_control=int(inputs[row]['original_control']),density=float(inputs[row]['fluid'][0]),
                       cpu=float(cpu['value'][row,offset+field]),cuda=float(gpu['value'][row,offset+field]),
                       absolute_difference=float(d[row,field]))
        stages.append(stage);offset+=count
    return cpu,gpu,dict(profile_id=profile,arithmetic_profile=summary['arithmetic_profile'],
        rows=642,CPU_zero_time_algebra_evaluations=642,CUDA_zero_time_algebra_evaluations=642,
        CUDA_kernel_launches=1,neutral_force_exact_zero=exact_neutral,filter_bits_exact=exact_filter,
        all_fields_within_fixed_limits=bool(within.all()),first_unequal_operation=first,stages=stages)


def verify_arithmetic(fused_directory, unfused_directory, input_path, executables, attribution=None):
    r=envelope(input_path)
    assert [r.value('I') for _ in range(4)]==[0x32414946,2,642,160]
    checksum=shared.fnv(r.data)
    inputs=r.array(ARITHMETIC_INPUT,642);r.finish()
    assert np.isfinite(inputs['body']).all() and np.array_equal(inputs['source_backend'],np.repeat([0,1],321))
    assert np.array_equal(inputs['original_control'],np.tile(np.repeat(np.arange(1,108),3),2))
    near(inputs['fluid'],np.tile(np.array([[975,.001,9.81],[1000,.001,9.81],[1025,.001,9.81]],np.float32),(214,1)),0,0)
    old_run=Path('build/ppo/frozen-interface-comparison-2026-11-23-v1').resolve()
    source,_=old.records(old_run,json.loads(read(old_run/'layout.json')))
    for i, item in enumerate(inputs):
        rows=source[(source['group']==0)&(source['case']==0)&(source['backend']==item['source_backend'])]
        tick=int(item['original_control'])
        for field,stored in [('body','body'),('held','held'),('rate','rates')]: old.f32_equal(item[field],rows[tick-1]['world'][stored])
        old.f32_equal(item['command'],rows[tick]['command'])
    fused_cpu,fused_gpu,fused=arithmetic_evidence(fused_directory,inputs,checksum,0)
    cpu,gpu,unfused=arithmetic_evidence(unfused_directory,inputs,checksum,1)
    attribution_proof=None
    if (unfused_directory/'canonical-radius.json').exists():
        assert attribution is not None
        attribution_proof=verify_drag_arithmetic(attribution[0],input_path,attribution[1],attribution[2])
        canonical=json.loads(read(unfused_directory/'canonical-radius.json'))
        assert canonical==dict(schema=1,arithmetic_profile=PROFILE,**RADIUS_METADATA)
        host=[r for r in attribution_proof['radius_results'] if r['backend']==0]
        assert [int(np.float32(x['argument']).view(np.uint32)) for x in host]==RADIUS_METADATA['fixed_geometry_radius_arguments_u32']
        assert [int(np.float32(x['radius']).view(np.uint32)) for x in host]==RADIUS_METADATA['cpu_radius_bits_u32']
        assert all(x['all_rows_same_argument'] and x['all_rows_same_radius'] for x in host)
        for name in ('frozen_interface_arithmetic_profile_v2.h','frozen_interface_arithmetic_corrected_v2.cu'):
            read(Path(__file__).parent/name)
    assert fused_cpu.tobytes()==cpu.tobytes(), 'Host arithmetic changed between profiles'
    ready=unfused['filter_bits_exact'] and unfused['neutral_force_exact_zero'] and unfused['all_fields_within_fixed_limits']
    f=np.float32;omega=f(f(2)*f(np.pi));h=f(.005)
    acceleration=f(f(f(omega*omega)*f(inputs['command']-inputs['held']))-f(f(f(2)*omega)*inputs['rate']))
    rate=f(inputs['rate']+f(h*acceleration));angle=f(inputs['held']+f(h*rate))
    expected=np.concatenate((acceleration,rate,angle),axis=-1)
    for value in (cpu,gpu): old.f32_equal(value['value'][:,:12],expected)
    com,velocity=prior.motion_cpu32(inputs['body'])
    for value in (cpu,gpu):
        near(value['value'][:,258:261],com,2e-6,2e-6)
        near(value['value'][:,261:264],velocity,2e-7,0)
        buoyancy=value['value'][:,255:258]
        for link in range(3):
            volume=f(value['value'][:,12+link*81+20]);mass=f(value['value'][:,12+link*81+19])
            expected_force=f(f(f(inputs['fluid'][:,0]*volume)-mass)*inputs['fluid'][:,2])
            old.f32_equal(buoyancy[:,link],expected_force)
        assert np.all(buoyancy[inputs['fluid'][:,0]==975]<0)
        assert np.all(buoyancy[inputs['fluid'][:,0]==1025]>0)
    # Independently reconstruct the double discrete operator from recorded
    # binary32 tensors; no dynamics, state update, or native solve is invoked.
    for label,value in [('cpu',cpu),('cuda',gpu)]:
        v=value['value'];M=v[:,264:364].reshape(-1,10,10);R=v[:,364:464].reshape(-1,10,10)
        C,F=v[:,464:474],v[:,474:484]
        mass=(M+M.swapaxes(-1,-2))*.5
        resistance=(R+R.swapaxes(-1,-2))*.5
        A=mass+float(h)*resistance
        rhs=f(h*f(f(F)-f(C))).astype(float)
        for j in range(10): rhs+=mass[:,:,j]*inputs['body'][:,11+j,None]
        near(v[:,484:584].reshape(-1,10,10),A,1e-18,1e-12)
        near(v[:,584:594],rhs,1e-18,1e-12)
        # Check the two stored algebraic solves against their actual equations.
        root_rhs,root_velocity=v[:,598:604],v[:,604:610]
        root_product=np.einsum('nij,nj->ni',A[:,:6,:6],root_velocity)
        near(root_product,root_rhs,1e-18,1e-8)
        actuated_rhs,next_velocity=v[:,618:628],v[:,628:638]
        product=np.einsum('nij,nj->ni',A,next_velocity)
        near(product,actuated_rhs,1e-18,1e-8)
        neutral=inputs['fluid'][:,0]==1000
        assert np.all(v[neutral,474:484]==0), (label,'neutral generalized force')
    build_profiles=[]
    namespace={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
    for profile,path in enumerate(executables):
        read(path)
        project=path.parent.parent/(path.stem+'.vcxproj')
        tree=ET.fromstring(read(project))
        groups=[group for group in tree.findall('m:ItemDefinitionGroup',namespace) if 'Release|x64' in group.get('Condition','')]
        assert len(groups)==1
        cuda=groups[0].find('m:CudaCompile',namespace)
        options=cuda.find('m:AdditionalOptions',namespace).text
        defines=cuda.find('m:Defines',namespace).text
        expected_flag='--fmad=false' if profile else '--fmad=true'
        assert expected_flag in options and f'CUDARL_INTERFACE_UNFUSED={profile}' in defines
        assert '--use_fast_math' not in options
        read(project.parent/'CMakeCache.txt')
        compiler_files=list((project.parent/'CMakeFiles').glob('*/CMakeCUDACompiler.cmake'))
        assert len(compiler_files)==1
        compiler_text=read(compiler_files[0]).decode()
        match=re.search(r'set\(CMAKE_CUDA_COMPILER "([^"]+)"\)',compiler_text)
        assert match
        read(match.group(1))
        build_profiles.append(dict(executable=str(path),project=str(project),cuda_options=options,
                                   cuda_defines=defines,compiler=match.group(1)))
    assert build_profiles[0]['cuda_options'].replace('--fmad=true','--fmad=PROFILE')==build_profiles[1]['cuda_options'].replace('--fmad=false','--fmad=PROFILE')
    read(Path(__file__).parent/'frozen-interface-comparison-v2/CMakeLists.txt')
    source_dependencies()
    for name in ('frozen_interface_arithmetic_v2.cu','prepare_frozen_interface_arithmetic_v2.py'):
        read(Path(__file__).parent/name)
    return dict(schema=1,study='frozen_interface_comparison_v2',study_version=2,arithmetic_profile=PROFILE,
        phase='arithmetic-preparation',passed=bool(ready),host_arithmetic_unchanged=True,
        nonneutral_buoyancy_preserved=True,density_variant_force_signs={'975':'negative','1000':'exact_zero','1025':'positive'},
        physical_controls=0,physical_substeps=0,integration_calls=0,PPO_updates=0,optimizer_steps=0,
        CUDA_kernel_launches=2,original_PRE_inputs=214,density_variants=3,rows_per_profile=642,
        fused=fused,unfused=unfused,
        canonical_radius=RADIUS_METADATA if attribution_proof else None,
        drag_attribution=attribution_proof,
        preparation_evaluation_totals=dict(native_runs=4,CPU_zero_time_evaluations=2568,CUDA_zero_time_evaluations=2568,explicit_CUDA_kernel_launches=4) if attribution_proof else None,
        build_profiles=build_profiles,
        interpretation='First unequal operation is identified in ordered fixed-state operations. Later stage differences describe propagation on identical inputs; closed-loop causation and retained skills still require the fresh gated comparison.',
        input_sha256=dict(sorted(prep.HASHES.items())))


def bind_arithmetic_preparation(root):
    path=root/'independent-arithmetic-verification.json'
    proof=json.loads(read(path));identity(proof)
    assert proof['phase']=='arithmetic-preparation' and proof['passed'] and proof['host_arithmetic_unchanged']
    assert proof['physical_controls']==proof['physical_substeps']==proof['optimizer_steps']==0
    assert proof['unfused']['all_fields_within_fixed_limits'] and proof['unfused']['filter_bits_exact'] and proof['unfused']['neutral_force_exact_zero']
    assert proof['canonical_radius']==RADIUS_METADATA and proof['drag_attribution']['passed']
    for filename,digest in proof['input_sha256'].items():
        assert hashlib.sha256(read(filename)).hexdigest().lower()==digest.lower(), filename
    return hashlib.sha256(read(path)).hexdigest()


def verify_drag_arithmetic(directory, input_path, old_unfused, executable):
    """Attribute the remaining difference from saved scalar operands only."""
    ir=envelope(input_path);assert [ir.value('I') for _ in range(4)]==[0x32414946,2,642,160]
    checksum=shared.fnv(ir.data);inputs=ir.array(ARITHMETIC_INPUT,642);ir.finish()
    layout=json.loads(read(directory/'layout.json'));labels=layout['labels'];count=layout['values']
    assert layout['schema']==1 and layout['input_bytes']==160 and layout['output_bytes']==16392 and layout['capacity']==2048
    assert count==len(labels)==1256 and len(set(labels))==1256
    dtype=np.dtype({'names':['value','count'],'formats':[('<f8',2048),'<i4'],'offsets':[0,16384],'itemsize':16392})
    r=envelope(directory/'trace.bin');assert [r.value('I') for _ in range(4)]==[0x32444946,2,642,16392]
    assert r.value('Q')==checksum
    assert r.array(ARITHMETIC_INPUT,642).tobytes()==inputs.tobytes()
    cpu,gpu=[r.array(dtype,642) for _ in range(2)];r.finish()
    summary=json.loads(read(directory/'summary.json'))
    assert summary['study']=='frozen_interface_drag_arithmetic_v2' and summary['rows']==642
    assert summary['CPU_zero_time_drag_evaluations']==summary['CUDA_zero_time_drag_evaluations']==642
    assert summary['explicit_CUDA_kernel_launches']==summary['native_runs']==1 and summary['body_inputs_unchanged']
    for field in ('physical_controls','physical_substeps','integration_calls','PPO_updates','optimizer_steps'): assert summary[field]==0
    baseline=envelope(old_unfused/'trace.bin');assert [baseline.value('I') for _ in range(5)]==[0x324F4946,2,642,5112,1]
    assert baseline.value('Q')==checksum
    assert baseline.array(ARITHMETIC_INPUT,642).tobytes()==inputs.tobytes()
    original=[baseline.array(ARITHMETIC_OUTPUT,642) for _ in range(2)];baseline.finish()
    def get(values,label): return np.float32(values[:,labels.index(label)])
    f=np.float32;pi=f(np.pi);radius=[]
    for backend,(record,prior_record) in enumerate(zip((cpu,gpu),original)):
        assert np.all(record['count']==1256) and np.isfinite(record['value'][:,:1256]).all() and np.all(record['value'][:,1256:]==0)
        values=record['value']
        old.f32_equal(get(values,'final_R20'),prior_record['value'][:,384])
        old.f32_equal(get(values,'final_R02'),prior_record['value'][:,366])
        for b in range(3):
            link=f'link{b}.'
            volume=get(values,link+'volume');rad=get(values,link+'radius_cbrtf')
            old.f32_equal(volume,prior_record['value'][:,12+b*81+20])
            expected={'radius_numerator':f(f(3)*volume),'radius_denominator':np.full(642,f(f(4)*pi)),
                      'linear_6pi':np.full(642,f(f(6)*pi)),'angular_8pi':np.full(642,f(f(8)*pi))}
            expected['radius_argument']=f(expected['radius_numerator']/expected['radius_denominator'])
            expected['linear_viscosity']=f(expected['linear_6pi']*inputs['fluid'][:,1])
            expected['linear_radius']=f(expected['linear_viscosity']*rad)
            expected['angular_viscosity']=f(expected['angular_8pi']*inputs['fluid'][:,1])
            expected['angular_radius1']=f(expected['angular_viscosity']*rad)
            expected['angular_radius2']=f(expected['angular_radius1']*rad)
            expected['angular_radius3']=f(expected['angular_radius2']*rad)
            for name,wanted in expected.items(): old.f32_equal(get(values,link+name),wanted)
            radius.append(dict(backend=backend,link=b,argument=float(expected['radius_argument'][0]),
                               radius=float(rad[0]),all_rows_same_argument=bool(np.all(expected['radius_argument']==expected['radius_argument'][0])),
                               all_rows_same_radius=bool(np.all(rad==rad[0]))))
            for sample in range(3):
                prefix=link+f'sample{sample}.'
                J=np.stack([get(values,prefix+f'J{i}.{k}') for i in range(10) for k in range(3)],axis=-1).reshape(-1,10,3)
                velocity=np.stack([get(values,prefix+f'velocity{k}') for k in range(3)],axis=-1)
                for d in range(3):
                    p=prefix+f'direction{d}.'
                    axis=np.stack([get(values,link+f'axis{d}.{k}') for k in range(3)],axis=-1)
                    dot=lambda a,z:f(f(f(a[...,0]*z[...,0])+f(a[...,1]*z[...,1]))+f(a[...,2]*z[...,2]))
                    old.f32_equal(get(values,p+'velocity_projection'),dot(velocity,axis))
                    pressure0=f(f(.5)*inputs['fluid'][:,0]);pressure1=f(pressure0*get(values,link+f'Cd{d}'))
                    pressure2=f(pressure1*get(values,link+f'projected{d}'));pressure3=f(pressure2*abs(dot(velocity,axis)))
                    total=f(expected['linear_radius']+pressure3)
                    weight=f(f(5)/f(18)) if sample!=1 else f(f(4)/f(9))
                    coefficient=f(weight*total)
                    for field,wanted in [('pressure_density',pressure0),('pressure_Cd',pressure1),('pressure_projected',pressure2),('pressure_abs_velocity',pressure3),('total',total),('coefficient',coefficient)]: old.f32_equal(get(values,p+field),wanted)
                    projection=dot(J,axis[:,None,:])
                    for i in range(10): old.f32_equal(get(values,p+f'projection{i}'),projection[:,i])
                    for a,z,key in [(2,0,'20'),(0,2,'02')]:
                        first=f(coefficient*projection[:,a]);add=f(first*projection[:,z])
                        old.f32_equal(get(values,p+'first'+key),first);old.f32_equal(get(values,p+'add'+key),add)
                        old.f32_equal(get(values,p+'after'+key),f(get(values,p+'before'+key)+add))
    unequal=cpu['value'][:,:1256]!=gpu['value'][:,:1256]
    row,field=map(int,np.argwhere(unequal)[0])
    assert row==0 and labels[field]=='link0.radius_cbrtf'
    read(executable);project=executable.parent.parent/(executable.stem+'.vcxproj')
    project_text=read(project).decode();assert '--fmad=false' in project_text and '--use_fast_math' not in project_text
    read(Path(__file__).parent/'frozen_interface_drag_arithmetic_v2.cu');source_dependencies()
    return dict(schema=1,study='frozen_interface_comparison_v2',study_version=2,arithmetic_profile=PROFILE,
                phase='drag-arithmetic-attribution',passed=True,physical_controls=0,physical_substeps=0,integration_calls=0,
                PPO_updates=0,optimizer_steps=0,CUDA_kernel_launches=1,rows=642,
                original_unfused_R20_R02_bound_all_rows=True,unfused_scalar_products_reconstructed=True,
                first_unequal_operation=dict(row=row,label=labels[field],cpu=float(cpu['value'][row,field]),cuda=float(gpu['value'][row,field])),
                radius_results=radius,
                conclusion='Identical radius arguments yield different cbrtf float32 results. Their recorded viscosity coefficients and products explain the residual R20/R02 arithmetic differences without changing inputs or integrating a body.',
                input_sha256=dict(sorted(prep.HASHES.items())))


def verify_result(root):
    native = json.loads(read(root/'result.json'))
    identity(native)
    assert native['closed']
    assert native['PPO_updates'] == native['optimizer_steps'] == 0
    assert native['training'] is False and native['qualified'] is False and native['runtime_adopted'] is False
    assert native['descriptive_comparison_only'] is True
    assert 0 <= native['controls'] <= CONTROL_CAP and 0 <= native['accepted_substeps'] <= native['attempted_substeps'] <= SUBSTEP_CAP
    assert np.isfinite(native['process_seconds']) and native['process_seconds'] > 0
    if not native['hard_fault']:
        assert native['process_seconds'] <= SECONDS_CAP
        assert native['evidence_bytes_before_close_report']+16384 <= EVIDENCE_CAP
    process = json.loads(read(root/'process-execution.json'))
    assert process['schema'] == 1 and process['study'] == 'frozen_interface_comparison_v2'
    assert process['PPO_updates'] == process['optimizer_steps'] == 0
    assert np.isfinite(process['seconds']) and process['seconds'] > 0
    assert hashlib.sha256(read(process['executable'])).hexdigest().lower() == process['executable_sha256'].lower()
    if not native['hard_fault']:
        assert process['seconds'] <= SECONDS_CAP and process['exit_code'] in (0, 3) and process['terminated_at_cap'] is False
    checksum, metadata = protocol_binding(root)
    frozen_inputs = verify_freeze(root)
    preparation = json.loads(read(root/'independent-preparation-verification.json'))
    assert preparation['passed'] and preparation['phase'] == 'preparation'
    assert preparation['physical_controls'] == preparation['physical_substeps'] == preparation['optimizer_steps'] == 0
    assert preparation['protocol_checksum'] == checksum and preparation['weights_unchanged']
    identity(preparation)
    arithmetic_sha = bind_arithmetic_preparation(root)
    assert preparation['arithmetic_preparation_sha256'] == arithmetic_sha
    grant = envelope(root/'authorization.bin')
    assert [grant.value('I'), grant.value('I'), grant.value('Q')] == [0x32554946, 2, checksum]
    assert grant.value('Q') == shared.fnv(read(root/'frozen-inputs.json'))
    assert grant.value('Q') == shared.fnv(read(root/'independent-preparation-verification.json'))
    approval = bytes(grant.array('u1'))
    assert approval
    grant.finish()
    layout = json.loads(read(root/'layout.json'))
    rows, _ = old.records(root, layout)
    weights = old.load_weights(root)
    retained, hold_gain = retained_catalog(root)
    endpoints = old.table(root/'probe-started/controls.csv')
    substeps = old.table(root/'probe-started/substeps.csv')
    old.csv_endpoints(endpoints, rows)
    assert np.all(rows['world']['current'] == 0)
    near(rows['world']['time'], rows['world']['substeps'].astype(float)*float(np.float32(.005)), 0, 0)
    assert len(substeps['data']) == int(rows['accepted'].sum())
    errors = new_errors()
    groups, by_key, by_sub = [], {}, {}
    keys = list(dict.fromkeys(tuple(int(row[x]) for x in ('group', 'case', 'backend')) for row in rows))
    for key in keys:
        group, case_id, backend = key
        assert group in (0, 1, 2) and backend in (0, 1)
        assert 0 <= case_id < (4 if group == 0 else 25 if group == 2 else 88)
        assert group != 1 or backend == 1
        select = (rows['group'] == group) & (rows['case'] == case_id) & (rows['backend'] == backend)
        rr = rows[select]
        keep = (old.col(substeps, 'group') == group) & (old.col(substeps, 'case') == case_id) & (old.col(substeps, 'backend') == backend)
        ss = {'data': substeps['data'][keep], 'index': substeps['index']}
        context = {**retained[case_id], 'gain':hold_gain} if group == 2 else metadata['gate_cases' if group == 0 else 'cases'][case_id]
        verify_arena(rr['world'],layout,context['arena_half_width_m'] if group==2 else .6,.4)
        cpu_rows = by_key.get((0, case_id, 0)) if group == 0 and backend == 1 else None
        groups.append(verify_transitions(rr, ss, weights, context, errors, cpu_rows))
        by_key[key], by_sub[key] = rr, ss
    assert sum(x['physical_controls'] for x in groups) == native['controls']
    assert sum(x['attempted_substeps'] for x in groups) == native['attempted_substeps']
    assert sum(x['accepted_substeps'] for x in groups) == native['accepted_substeps']
    retention, skills_retained = retention_results(root, by_key, by_sub, retained)
    assert native['retention_passed'] == skills_retained
    if not skills_retained:
        assert native['probe_completed_cases'] == native['cases_started'] == 0
        assert not any(key[0] == 1 for key in by_key)
        if not native['hard_fault']:
            assert native['comparison_blocked_reason'] == 'retention_regression'
            assert all(v['completed'] for v in retention.values())
    backend_maxima = []
    for case in metadata['gate_cases']:
        a, b = by_key.get((0, case['case'], 0)), by_key.get((0, case['case'], 1))
        if a is None or b is None:
            continue
        if len(a) == len(b) == case['maximum_controls']+1 and not a[-1]['reason'] and not b[-1]['reason'] and not a[-1]['failure'] and not b[-1]['failure']:
            backend_maxima.append(compare_backend(a, b))
    assert len(backend_maxima) == native['precheck_completed_cases']
    pointwise = old.table(root/'probe-started/same-observation.csv')
    assert pointwise['data'].shape[1] == 59
    assert native['native_runs']==1 and native['NN_library_kernel_launches_measured'] is False
    if not native['hard_fault']:
        cuda_ret=sum(x['physical_controls'] for x in groups if x['group']==2 and x['backend']==1)
        main_controls=sum(x['physical_controls'] for x in groups if x['group']==1)
        gate_cuda=sum(x['physical_controls'] for x in groups if x['group']==0 and x['backend']==1)
        assert native['explicit_CUDA_kernel_launches']==gate_cuda+4*cuda_ret+5*main_controls
        assert native['NN_forward_calls']==len(pointwise['data'])+cuda_ret+main_controls
    for row in pointwise['data']:
        case, tick = map(int, row[:2])
        observation = row[2:43]
        cpu, gpu, command_cpu, command_gpu = np.split(row[43:], 4)
        near(cpu, shared.forward(weights, observation[None, :])[0], 2e-5, 2e-6)
        near(cpu, gpu, 2e-5, 0)
        near(command_cpu, np.tanh(cpu), 2e-5, 0)
        near(command_gpu, np.tanh(gpu), 2e-5, 0)
        for backend in (0, 1):
            rr = by_key.get((0, case, backend))
            if rr is not None and tick < len(rr):
                old.f32_equal(observation, rr[tick]['observation'])
    cases = []
    for context in metadata['cases']:
        key = (1, context['case'], 1)
        if key in by_key:
            cases.append(case_metrics(by_key[key], by_sub[key], context))
    # Arm differences are forbidden through the entire shared warmup.
    matched_warmups = 0
    for case_id in range(0, 88, 2):
        a, b = by_key.get((1, case_id, 1)), by_key.get((1, case_id+1, 1))
        if a is None or b is None:
            continue
        count = min(len(a), len(b), metadata['cases'][case_id]['warm_controls']+1)
        for key in ('observation', 'base', 'command'):
            old.f32_equal(a[key][:count], b[key][:count])
        for key in ('body', 'held', 'rates', 'last_torque', 'actual', 'requested', 'energy'):
            assert a['world'][key][:count].tobytes() == b['world'][key][:count].tobytes()
        assert a['task'][:count].tobytes() == b['task'][:count].tobytes()
        if count == metadata['cases'][case_id]['warm_controls']+1:
            matched_warmups += 1
    summaries = old.table(root/'probe-started/cases.csv')
    assert len(summaries['data']) == native['probe_completed_cases']
    for row in summaries['data']:
        case = int(row[summaries['index']['case']])
        context = metadata['cases'][case]
        rr = by_key[(1, case, 1)]
        for name in ('fixture', 'pose', 'warm_controls', 'scenario', 'maximum_controls'):
            assert row[summaries['index'][name]] == context[name]
        assert row[summaries['index']['arm']] == ARMS.index(context['arm'])
        assert row[summaries['index']['controls']] == rr[-1]['tick']
        assert row[summaries['index']['behavior_reason']] == rr[-1]['reason']
        near(row[summaries['index']['seconds']], rr[-1]['world']['time'], 0, 0)
    full = not native['hard_fault'] and skills_retained and len(cases) == 88
    if full:
        assert len(backend_maxima) == 4 and matched_warmups == 44
        assert native['cases_started'] == native['probe_completed_cases'] == 88
        assert all(x['completed'] or x['reason'] in (1, 4, 5) for x in cases)
    elif (root/'probe-started/first-failure.bin').exists():
        failed = envelope(root/'probe-started/first-failure.bin')
        assert failed.value('I') == 2
        group, case, tick = [failed.value('i') for _ in range(3)]
        ftype, _ = frame_type(layout)
        active, other = prep.raw_record(failed, ftype), prep.raw_record(failed, ftype)
        failed.finish()
        assert group in (0, 1, 2) and tick >= 0 and case >= 0
        assert native['hard_fault']
        # Preserve and bind the first PRE/candidate fields even if a hard fault
        # prevents the ordinary completed-case metrics from being meaningful.
        assert np.isfinite(active['first_before']).all()
        if active['failure']:
            assert active['failure'] != 0
    summary, selection = summarize(cases)
    if not full:
        selection = {'status': 'incomplete_comparison', 'preferred_arm': 'tie',
                     'adequate_arms': [], 'navigation_qualified': False}
    qualified = set(selection['adequate_arms']) if full else set()
    conclusion = ('both_qualify' if len(qualified)==2 else 'lane_following_qualifies' if 'lane_following' in qualified
                  else 'heading_only_qualifies' if 'heading_only' in qualified else 'neither_qualifies') if full else 'comparison_blocked'
    source_dependencies()
    return {'schema': 1, 'study': 'frozen_interface_comparison_v2',
            'study_version': 2, 'arithmetic_profile': PROFILE,
            'arithmetic_preparation_sha256': arithmetic_sha,
            'audit_status': 'passed' if full else 'verified_hard_stop' if native['hard_fault'] else 'verified_retention_stop', 'audit_passed': True,
            'retention': retention, 'original_swim_rest_retained': skills_retained, 'conclusion': conclusion,
            'full_comparison_completed': bool(full), 'descriptive_only': True,
            'qualified': False, 'weights_unchanged': True,
            'new_physical_controls_executed': 0, 'new_GPU_kernels_executed': 0, 'new_optimizer_steps': 0,
            'recorded_controls': native['controls'], 'recorded_attempted_substeps': native['attempted_substeps'],
            'recorded_accepted_substeps': native['accepted_substeps'], 'matched_precheck_cases': len(backend_maxima),
            'recorded_explicit_CUDA_kernel_launches':native['explicit_CUDA_kernel_launches'],
            'recorded_NN_forward_calls':native['NN_forward_calls'], 'NN_library_kernel_launches_measured':False,
            'identical_warmup_pairs': matched_warmups, 'cases': cases, 'arm_summary': summary,
            'selection': selection, 'unrun_cases': [i for i in range(88) if (1, i, 1) not in by_key],
            'maximum_errors': errors, 'matched_backend_maxima': backend_maxima,
            'groups': groups, 'native_result': native, 'protocol_checksum': checksum,
            'process_execution': process,
            'frozen_inputs': frozen_inputs,
            'evidence_scope': 'All recorded endpoints and accepted5ms poses, requests, sensors, NN outputs/actions, target filters, torque and energy ledgers are checked; force integration is not replayed. Geometry covers sampled5ms states, not continuous swept-volume proof.',
            'local_radius_definition': '1s boxcar COM-speed divided by absolute finite-difference course rate; speed>=.002m/s and rate>=1degree/s; descriptive only',
            'input_sha256': dict(sorted(prep.HASHES.items()))}


def self_test():
    old.self_test()
    world = np.zeros(2, dtype=old.world_type())
    world['body'][:, :21] = ref.initial_body(np.array([0, 3]))
    world['body'][:, 3:7] = ref.axis_rotation([1, 0, 0], np.array([.3, -.4]))
    request_type = np.dtype([(key, '<f4', (3,)) for key in ('position', 'terminal_velocity', 'heading')])
    task_type = np.dtype([('request', request_type), ('start_com_model', '<f4', (3,)), ('controls', '<i4')])
    task = np.zeros(2, dtype=task_type)
    center, _ = ref.motion(world['body'])
    task['start_com_model'] = center/ref.SCALE - np.array([.5, .025, -.3])
    task['request']['heading'] = [1, 0, 0]
    task['request']['terminal_velocity'] = [.008, 0, 0]
    a, b = observer(world, task, False), observer(world, task, True)
    assert not np.allclose(a[:, :3], b[:, :3])
    # Inverse-rotating only after masking preserves tilted depth components.
    world_lane = ref.rotate(world['body'][:, 3:7].astype(float), b[:, :3]*float(np.float32(.002)))
    near(world_lane[:, [0, 2]], np.zeros((2, 2)), 1e-10, 0)
    near(world_lane[:, 1], np.full(2, -.001), 1e-8, 0)
    near(a[:, 3:], b[:, 3:], 0, 0)
    for off in (0, 5, -5, 15, -15, 30, -30):
        near(expected_heading([1, 0, 0], off), [math.cos(math.radians(off)), 0, -math.sin(math.radians(off))], 1e-12, 0)
    cases = canonical_cases()
    validate_cases(cases)
    assert sum(len(x['events']) for x in cases['cases']) == 520
    fake = [{**x, 'completed': False, 'functional_pass': False, 'reason': 0,
             'late_forward_m_s': None} for x in cases['cases']]
    summary, selection = summarize(fake)
    assert selection['preferred_arm'] == 'tie' and selection['adequate_arms'] == []
    assert all(x['cases'] == 44 and x['completed'] == 0 for x in summary)
    # Synthetic saved poses exercise the acceptance rules without integration.
    rtype=np.dtype([('world',old.world_type()),('task',task_type),('failure','<i4')])
    for context in (retained_cases()[0],retained_cases()[4],retained_cases()[11]):
        count=context['controls'];r=np.zeros(count+1,dtype=rtype)
        r['world']['body'][:,:21]=ref.initial_body(context['pose'])
        if not context['holding']:
            heading=ref.rotate(r['world']['body'][0,3:7].astype(float),[1,0,0])
            r['world']['body'][:,:3]+=np.arange(count+1)[:,None]*.05*.008*heading/ref.SCALE
            r['world']['body'][:,11:14]=.008*heading
        center,_=ref.motion(r['world']['body'][0])
        r['task']['start_com_model']=center/ref.SCALE
        sub={'data':r['world']['body'].astype(float),'index':dict(zip(BODY,range(23)))}
        assert retention_metrics(r,sub,context)['passed']
        if context['holding']:
            r['world']['body'][:,11]=.0006
            assert not retention_metrics(r,sub,context)['passed']
    assert arithmetic_limits()[0].shape==(638,)
    return {'passed': True, 'world_mask_before_rotation': True,
            'depth_signal_preserved_under_tilt': True,
            'canonical_cases': 88, 'canonical_event_latches_per_backend': 520,
            'bounded_main_controls': 34200, 'bounded_gate_controls': 1000,
            'retention_cases_per_backend': len(retained_cases()), 'retention_controls_per_backend':46240,
            'control_cap':CONTROL_CAP, 'substep_cap':SUBSTEP_CAP,
            'new_physical_controls': 0, 'GPU_kernels': 0, 'optimizer_steps': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--preparation', action='store_true')
    parser.add_argument('--arithmetic', nargs=3, type=Path, metavar=('FUSED_DIRECTORY','UNFUSED_DIRECTORY','INPUT'))
    parser.add_argument('--arithmetic-executables', nargs=2, type=Path)
    parser.add_argument('--attribution',nargs=3,type=Path,metavar=('DRAG_DIRECTORY','ORIGINAL_UNFUSED_DIRECTORY','DRAG_EXECUTABLE'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.arithmetic:
        assert args.arithmetic_executables and args.output
        result=verify_arithmetic(*[p.resolve() for p in args.arithmetic], [p.resolve() for p in args.arithmetic_executables],
                                 [p.resolve() for p in args.attribution] if args.attribution else None)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({'passed':result['passed'],'phase':'arithmetic-preparation','proof_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))
        if not result['passed']: raise SystemExit(3)
    else:
        assert args.run is not None
        root = args.run.resolve()
        result = verify_preparation(root) if args.preparation else verify_result(root)
        output = root/('independent-preparation-verification.json' if args.preparation else 'independent-result-verification.json')
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'passed': result.get('passed', result.get('audit_passed')),
                          'phase': 'preparation' if args.preparation else 'result',
                          'physical_controls': 0, 'optimizer_steps': 0,
                          'proof_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
        if not args.preparation and not result['full_comparison_completed']:
            raise SystemExit(3)
