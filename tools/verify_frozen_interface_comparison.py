"""Independent saved-data audit for the frozen request-interface comparison.

This reader never advances physics, launches CUDA, or updates any parameter.
It reconstructs sensors, actions, filters, recorded 5 ms guards and metrics;
it does not replay the force integrator.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import verify_frozen_requested_turn_result as old

prep, prior, ref, shared = old.prep, old.prior, old.ref, old.shared
read, envelope, near = old.read, old.envelope, old.near
BODY = old.BODY
ARMS = ('lane_following', 'heading_only')


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
    return {'schema': 1, 'study': 'frozen_interface_comparison',
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
    assert [p.value('I'), p.value('I')] == [0x31524946, 1]
    checksum = shared.fnv(p.data)
    assert p.value('Q') == shared.fnv(read(root/'approved-protocol.md'))
    assert p.value('Q') == shared.fnv(read(root/'cases.json'))
    for name in ('parent-checkpoint.bin', 'parent-protocol.bin', 'holding-parent.bin'):
        assert p.value('Q') == shared.fnv(envelope(root/name).data)
    assert [p.value('i') for _ in range(11)] == [6, 4, 41, 4, 336, 88, 4, 430, 35200, 352000, 240]
    near([p.value('f') for _ in range(7)], np.float32([.008, .005, 20, .20, .12, .195, .115]), 0, 0)
    assert p.value('Q') == 512*1024**2
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
    context = prep.dtype(layout['case'], [(key, '<i4') for key in ('id', 'fixture', 'pose', 'warm', 'scenario', 'arm', 'maximum')])
    definitions += [('context', context), ('request_active', '<i4'), ('offset_degrees', '<i4')]
    return prep.dtype(layout['frame'], definitions), task


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
    assert report['study'] == 'frozen_interface_comparison' and report['passed']
    assert report['physical_controls'] == report['physical_substeps'] == report['PPO_updates'] == report['optimizer_steps'] == 0
    assert report['physical_run_approved'] is False
    assert report['initial_fixtures'] == report['observer_fixtures'] == 88
    assert report['request_latch_fixtures'] == 520
    assert report['evidence_cap_bytes'] == 512*1024**2
    assert 0 < report['conservative_generated_evidence_bytes'] < report['evidence_cap_bytes']
    checksum, metadata = protocol_binding(root)
    weights = old.load_weights(root)
    layout = json.loads(read(root/'layout.json'))
    ftype, _ = frame_type(layout)
    fixture = envelope(root/'fixtures.bin')
    assert [fixture.value('I'), fixture.value('I'), fixture.value('Q'), fixture.value('I')] == [0x31464946, 1, checksum, ftype.itemsize]
    initial, before, after = [fixture.array(ftype, 88) for _ in range(3)]
    fixture.finish()
    errors = {'body': 0., 'observation': 0., 'NN': 0., 'command': 0.}
    for frames, synthetic in ((initial, False), (after, True)):
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
    assert [latch.value('I'), latch.value('I'), latch.value('Q'), latch.value('I')] == [0x314C4946, 1, checksum, 520]
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
    assert [guards.value('I'), guards.value('I'), guards.value('Q'), guards.value('I')] == [0x31474946, 1, checksum, 12]
    expected = [0, 0, 4, 0, 4, 0, 5, 0, 1, 104, 4, 1]
    for i, wanted in enumerate(expected):
        assert [guards.value('I'), guards.value('I')] == [i, wanted]
        frame = prep.raw_record(guards, ftype)
        assert [guards.value('I'), guards.value('I')] == [wanted, wanted]
        active, _, _ = prior.boundary_values(frame['world']['body'][None, :])
        for backend in (0, 1):
            actual = 104 if active[0] > 1 else int(old.guard_causes(frame['world']['body'][None, :], frame['task']['start_com_model'][1], backend)[0])
            assert actual == wanted, (i, backend, actual, wanted)
    guards.finish()
    frozen_count = verify_freeze(root)
    source_dependencies()
    return {'schema': 1, 'study': 'frozen_interface_comparison', 'phase': 'preparation',
            'passed': True, 'physical_run_approved': False,
            'physical_controls': 0, 'physical_substeps': 0, 'optimizer_steps': 0,
            'PPO_updates': 0, 'initial_fixtures': 88, 'observer_fixtures': 88,
            'latched_request_fixtures_per_backend': 520, 'guard_fixtures_per_backend': 12,
            'maximum_errors': errors, 'frozen_inputs': frozen_count,
            'protocol_checksum': checksum, 'weights_unchanged': True,
            'input_sha256': dict(sorted(prep.HASHES.items()))}


def verify_transitions(rows, sub, weights, case, errors, cpu_rows=None):
    """Check all collection rows independently, including every accepted substep."""
    assert len(rows) and rows[0]['tick'] == 0
    key = tuple(int(rows[0][x]) for x in ('group', 'case', 'backend'))
    assert np.array_equal(rows['tick'], np.arange(len(rows)))
    first = rows[0]
    near(first['world']['body'][:21], ref.initial_body(case['pose']), 2e-7, 2e-6)
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
                          (case['arm'] == 'heading_only') & (rows['tick'][1:] > case['warm_controls']))
        errors['observation'] = max(errors['observation'], near(rows['observation'][1:], wanted, 2e-5, 2e-5))
        z = shared.forward(weights, rows['observation'][1:].astype(float))
        errors['NN'] = max(errors['NN'], near(rows['base'][1:], z, 2e-5, 2e-6))
        errors['command'] = max(errors['command'], near(rows['command'][1:], np.tanh(rows['base'][1:].astype(float)), 2e-5, 0))
    consumed = 0
    for j in range(1, len(rows)):
        before, after = rows[j-1], rows[j]
        w0, w1 = before['world'], after['world']
        tick = int(after['tick'])
        event = event_at(case, tick)
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
        ke, gap = old.kinematics(body)
        active, _, _ = prior.boundary_values(body)
        if not after['failure']:
            assert np.all(gap >= .01-2e-7) and np.all(active <= 1+2e-6)
        sequence = np.vstack((w0['energy'], energy))
        for field in (1, 2, 3):
            assert np.all(np.diff(sequence[:, field]) >= -1e-18)
        identity = ke-energy[:, 0]+energy[:, 2]+energy[:, 3]-energy[:, 4]
        errors['energy_identity'] = max(errors['energy_identity'], near(energy[:, 5], identity, 1e-12, 1e-4))
        if not after['failure']:
            assert np.all(abs(energy[:, 5]) <= .02*(energy[:, 1]+energy[:, 2])+2e-10)
            causes = old.guard_causes(body, origin[1], key[2])
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


def verify_result(root):
    native = json.loads(read(root/'result.json'))
    assert native['schema'] == 1 and native['study'] == 'frozen_interface_comparison' and native['closed']
    assert native['PPO_updates'] == native['optimizer_steps'] == 0
    assert native['training'] is False and native['qualified'] is False and native['runtime_adopted'] is False
    assert native['descriptive_comparison_only'] is True
    assert 0 <= native['controls'] <= 35200 and 0 <= native['accepted_substeps'] <= native['attempted_substeps'] <= 352000
    assert np.isfinite(native['process_seconds']) and native['process_seconds'] > 0
    if not native['hard_fault']:
        assert native['process_seconds'] <= 240
        assert native['evidence_bytes_before_close_report']+16384 <= 512*1024**2
    process = json.loads(read(root/'process-execution.json'))
    assert process['schema'] == 1 and process['study'] == 'frozen_interface_comparison'
    assert process['PPO_updates'] == process['optimizer_steps'] == 0
    assert np.isfinite(process['seconds']) and process['seconds'] > 0
    assert hashlib.sha256(read(process['executable'])).hexdigest().lower() == process['executable_sha256'].lower()
    if not native['hard_fault']:
        assert process['seconds'] <= 240 and process['exit_code'] == 0 and process['terminated_at_cap'] is False
    checksum, metadata = protocol_binding(root)
    frozen_inputs = verify_freeze(root)
    preparation = json.loads(read(root/'independent-preparation-verification.json'))
    assert preparation['passed'] and preparation['phase'] == 'preparation'
    assert preparation['physical_controls'] == preparation['physical_substeps'] == preparation['optimizer_steps'] == 0
    assert preparation['protocol_checksum'] == checksum and preparation['weights_unchanged']
    grant = envelope(root/'authorization.bin')
    assert [grant.value('I'), grant.value('I'), grant.value('Q')] == [0x31554946, 1, checksum]
    assert grant.value('Q') == shared.fnv(read(root/'frozen-inputs.json'))
    assert grant.value('Q') == shared.fnv(read(root/'independent-preparation-verification.json'))
    approval = bytes(grant.array('u1'))
    assert approval
    grant.finish()
    layout = json.loads(read(root/'layout.json'))
    rows, _ = old.records(root, layout)
    weights = old.load_weights(root)
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
        assert group in (0, 1) and backend in (0, 1)
        assert 0 <= case_id < (4 if group == 0 else 88)
        assert group == 0 or backend == 1
        select = (rows['group'] == group) & (rows['case'] == case_id) & (rows['backend'] == backend)
        rr = rows[select]
        keep = (old.col(substeps, 'group') == group) & (old.col(substeps, 'case') == case_id) & (old.col(substeps, 'backend') == backend)
        ss = {'data': substeps['data'][keep], 'index': substeps['index']}
        context = metadata['gate_cases' if group == 0 else 'cases'][case_id]
        cpu_rows = by_key.get((0, case_id, 0)) if group == 0 and backend == 1 else None
        groups.append(verify_transitions(rr, ss, weights, context, errors, cpu_rows))
        by_key[key], by_sub[key] = rr, ss
    assert sum(x['physical_controls'] for x in groups) == native['controls']
    assert sum(x['attempted_substeps'] for x in groups) == native['attempted_substeps']
    assert sum(x['accepted_substeps'] for x in groups) == native['accepted_substeps']
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
    full = not native['hard_fault'] and len(cases) == 88
    if full:
        assert len(backend_maxima) == 4 and matched_warmups == 44
        assert native['cases_started'] == native['probe_completed_cases'] == 88
        assert all(x['completed'] or x['reason'] in (1, 4, 5) for x in cases)
    elif (root/'probe-started/first-failure.bin').exists():
        failed = envelope(root/'probe-started/first-failure.bin')
        assert failed.value('I') == 1
        group, case, tick = [failed.value('i') for _ in range(3)]
        ftype, _ = frame_type(layout)
        active, other = prep.raw_record(failed, ftype), prep.raw_record(failed, ftype)
        failed.finish()
        assert group in (0, 1) and tick >= 0 and case >= 0
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
    source_dependencies()
    return {'schema': 1, 'study': 'frozen_interface_comparison',
            'audit_status': 'passed' if full else 'verified_hard_stop', 'audit_passed': True,
            'full_comparison_completed': bool(full), 'descriptive_only': True,
            'qualified': False, 'weights_unchanged': True,
            'new_physical_controls_executed': 0, 'new_GPU_kernels_executed': 0, 'new_optimizer_steps': 0,
            'recorded_controls': native['controls'], 'recorded_attempted_substeps': native['attempted_substeps'],
            'recorded_accepted_substeps': native['accepted_substeps'], 'matched_precheck_cases': len(backend_maxima),
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
    return {'passed': True, 'world_mask_before_rotation': True,
            'depth_signal_preserved_under_tilt': True,
            'canonical_cases': 88, 'canonical_event_latches_per_backend': 520,
            'bounded_main_controls': 34200, 'bounded_gate_controls': 1000,
            'new_physical_controls': 0, 'GPU_kernels': 0, 'optimizer_steps': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--preparation', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
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
