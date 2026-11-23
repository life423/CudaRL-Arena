"""Offline row audit for continuous-pond V2; never simulate or optimize.

The native learner is frozen separately. This module reads checksummed recorded
arrays and reconstructs geometry, policy actions, rewards and task accounting.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import math
from pathlib import Path
import struct

import numpy as np
import verify_continuous_pond_v2 as contract

v1 = contract.v1
reference, previous = contract.reference, contract.previous
near, envelope = contract.near, contract.envelope
PARTS = (('pre-filter', 12), ('post-body', 23), ('joint-latent', 4), ('reward', 1))
BEHAVIOR_REASONS = (1, 4, 5, 6)
CHUNK_ROWS = 4096


def cuda_com_model32(body):
    """Reconstruct the registered FMA COM sensor without an SI round trip."""
    body = np.asarray(body, np.float32)
    f = np.float32
    lengths = f([.020, .026, .026])
    centers, orientations = [body[..., :3] * f(.04)], [body[..., 3:7]]
    for link in (1, 2):
        parent = link - 1
        anchor = centers[parent] + v1.rotate32(orientations[parent], f([-lengths[parent] * f(.5), 0, 0]))
        yaw = v1.qmul32(orientations[parent], v1.axis32([0, 1, 0], body[..., 7 + parent * 2]))
        q = v1.qmul32(yaw, v1.axis32([0, 0, 1], body[..., 8 + parent * 2]))
        orientations.append(q)
        centers.append(anchor + v1.rotate32(q, f([-lengths[link] * f(.5), 0, 0])))
    pi, radius = f(math.pi), f(.0015)
    masses = [f(4) * pi * (lengths[0] * f(.5)) * f(.0064) * f(.008) / f(3) * f(1000),
              (pi * radius * radius * lengths[1]) * f(1000),
              (pi * radius * radius * lengths[2] + f(.0097) * lengths[2] * f(.0001)) * f(1000)]
    total, weighted = f(0), np.zeros_like(centers[0])
    for mass, center in zip(masses, centers):
        total += mass
        weighted = v1.fma32(center, mass, weighted)
    return weighted * (f(1) / (total * f(.04)))


def horizontal_arc32(before, after):
    """Float COM-model subtraction, horizontal norm and native SI scaling."""
    delta = cuda_com_model32(after) - cuda_com_model32(before)
    square = v1.fma32(delta[..., 2], delta[..., 2], delta[..., 0] * delta[..., 0])
    return np.sqrt(square) * np.float32(.04)


def rolling_speeds(arc, dt):
    """Recompute physical ring slots0..39 in the native double sum order."""
    arc, dt = np.asarray(arc, np.float32), np.asarray(dt, np.float32)
    assert arc.shape == dt.shape and arc.ndim == 2
    last = np.arange(len(arc))
    distance, elapsed = np.zeros(arc.shape), np.zeros(dt.shape)
    for slot in range(40):
        index = slot + 40 * ((last - slot) // 40)
        exists = index >= 0
        # Keep native slot order even for long captures and mixed-magnitude
        # increments. No cumulative-sum cancellation enters a strict predicate.
        distance[exists] += arc[index[exists]].astype(float)
        elapsed[exists] += dt[index[exists]].astype(float)
    return np.divide(distance, elapsed, out=np.zeros_like(elapsed), where=elapsed > 0).astype(np.float32)


def rewards(before, after, fixed_depth_model, substeps, behavior):
    values = v1.measured(after, fixed_depth_model)
    speed = np.linalg.norm(values['velocity'][..., [0, 2]], axis=-1)
    target = float(np.float32(.008))
    forward_error = (values['forward'] - target) / target
    side_error = values['side'] / target
    depth = values['depth'] / float(np.float32(.001))
    rate = np.clip(speed / target, 0, 1) - 2 * forward_error ** 2 - .25 * side_error ** 2
    rate -= .05 * depth ** 2 + .05 * (values['roll'] / 10) ** 2 + .01 * (values['pitch'] / 10) ** 2
    dt = np.asarray(substeps, np.float32) * np.float32(.005)
    work = np.asarray(after, float)[..., 21] - np.asarray(before, float)[..., 21]
    return dt * rate - 6250 * work - 4 * np.asarray(behavior)


def _counts_type(layout):
    return v1.dtype_from_layout(layout, 'counts', [
        ('controls', '<u8'), ('attempted_substeps', '<u8'), ('accepted_substeps', '<u8'),
        ('reward_sum', '<f8'), ('seconds', '<f8'), ('last_substeps', '<i4'), ('reason', '<i4'),
        ('done', 'u1'), ('truncated', 'u1')])


def open_batch(directory, protocol, layout, expected_means=None):
    directory = Path(directory)
    reader = envelope(directory / 'meta.bin')
    binding = contract.payload_checksum(directory / 'meta.bin')
    assert [reader.value('I'), reader.value('I'), reader.value('Q')] == [0x32425054, 2, protocol]
    n, horizon, ticks, purpose, index, cases, withheld, fault = [reader.value('i') for _ in range(8)]
    assert purpose in (0, 1, 2, 4) and withheld in (0, 1) and fault == 0
    assert n > 0 and 0 < ticks <= horizon and cases == (8 if withheld else 16)
    if purpose < 2:
        assert (n, horizon, cases, withheld) == (contract.N, contract.T, 16, 0)
        assert index == 0 if purpose == 0 else 1 <= index <= 24
    else:
        assert n == cases and horizon == (36000 if withheld else 1200) and index == 0
        assert purpose != 4 or not withheld
    means = reader.array(size=contract.P)
    latent = reader.array(size=n * contract.P).reshape(n, contract.P)
    logp, returns = reader.array(size=n), reader.array(size=n)
    assert all(np.isfinite(value).all() for value in (means, latent, logp, returns))
    if expected_means is not None:
        assert np.array_equal(means, expected_means)
    if purpose == 4:
        assert np.all(means[np.arange(contract.P) % 20 != 0] == 0)
    ctype, ttype = _counts_type(layout), contract.task_type(layout)
    assert reader.value('I') == ctype.itemsize
    counts = reader.array(ctype, n)
    assert reader.value('I') == layout['world_size'] == 696
    initial, final = reader.array(previous.WORLD_TYPE, n), reader.array(previous.WORLD_TYPE, n)
    assert reader.value('I') == ttype.itemsize
    initial_task, final_task = reader.array(ttype, n), reader.array(ttype, n)
    reader.finish()
    probability_error = near(logp, contract.log_probability(means, latent), 1e-4, 0)
    if purpose < 2:
        seed = 0x8e25a643 if purpose == 0 else 0x137be951
        expected = means + np.float32(contract.SIGMA) * previous.normal(
            seed, index, np.arange(n * contract.P, dtype=np.uint32).reshape(n, contract.P))
    else:
        expected = np.broadcast_to(means, (n, contract.P))
    rng_error = near(latent, expected, 2e-6, 2e-6)
    manifest = contract.js(directory / 'manifest.json')
    assert manifest['schema'] == 2 and manifest['meta_checksum'] == binding
    assert [manifest[key] for key in ('worlds', 'horizon', 'ticks', 'purpose')] == [n, horizon, ticks, purpose]
    assert len(manifest['parts']) == 4
    parts, mapped = {}, []
    for field, ((name, columns), entry) in enumerate(zip(PARTS, manifest['parts'])):
        path = directory / (name + '.bin')
        contract.digest(path)
        assert path.resolve() in contract.VALIDATED
        mm = np.memmap(path, dtype='u1', mode='r')
        mapped.append(mm)
        magic, size, checksum = struct.unpack_from('<QQQ', mm)
        assert magic == 0x31504B434C524443 and size == len(mm) - 24 and size <= 512 * 1024 ** 2
        assert entry == dict(field=field, name=path.name, columns=columns, payload_bytes=size, payload_checksum=checksum)
        part = reference.shared.Reader(memoryview(mm)[24:])
        assert [part.value('I'), part.value('I'), part.value('i'), part.value('Q'), part.value('i'), part.value('i'), part.value('i')] == [
            0x32425054, 2, field, binding, n, ticks, columns]
        parts[name] = part.array(size=n * ticks * columns).reshape(ticks, n, columns)
        part.finish()
    flags = envelope(directory / 'row-substeps.bin')
    assert [flags.value('I'), flags.value('I'), flags.value('Q'), flags.value('i'), flags.value('i')] == [
        0x32465054, 2, binding, n, ticks]
    assert contract.payload_checksum(directory / 'row-substeps.bin') == manifest['row_substeps_checksum']
    steps = flags.array('u1', n * ticks).reshape(ticks, n)
    flags.finish()
    assert np.all(steps <= 10)
    return dict(n=n, horizon=horizon, ticks=ticks, purpose=purpose, index=index, cases=cases,
                withheld=bool(withheld), means=means, latent=latent, logp=logp, returns=returns,
                counts=counts, initial=initial, final=final, initial_task=initial_task, task=final_task,
                parts=parts, steps=steps, mapped=mapped, probability_error=probability_error, rng_error=rng_error)


def _assert_prefix_and_terminals(batch):
    counts, steps, task = batch['counts'], batch['steps'], batch['task']
    n, ticks = batch['n'], batch['ticks']
    assert np.array_equal(np.sum(steps > 0, axis=0), counts['controls'])
    assert np.array_equal(steps.astype(np.uint64).sum(axis=0), counts['accepted_substeps'])
    assert np.array_equal(counts['attempted_substeps'], counts['accepted_substeps'])
    for world_key, count_key in (('substeps', 'accepted_substeps'), ('attempted', 'attempted_substeps'), ('controls', 'controls')):
        assert np.array_equal(batch['final'][world_key], counts[count_key])
    assert np.array_equal(task['controls'], counts['controls']) and np.array_equal(task['last_world_control'], counts['controls'])
    assert np.all((counts['controls'] > 0) & (counts['controls'] <= ticks))
    assert np.array_equal(steps > 0, np.arange(ticks)[:, None] < counts['controls'][None, :])
    last, episode = counts['controls'].astype(int) - 1, np.arange(n)
    assert np.array_equal(steps[last, episode], counts['last_substeps'])
    assert np.all(counts['accepted_substeps'] == (counts['controls'] - 1) * 10 + counts['last_substeps'])
    failed = np.isin(counts['reason'], BEHAVIOR_REASONS)
    if batch['purpose'] < 2:
        assert np.all(failed | (counts['reason'] == 2))
        assert np.all(counts['done'] == 1) and np.all(counts['truncated'] == 0) and np.all(task['done'] == 1)
        assert np.all(task['maximum_controls'] == contract.T)
        deadline = counts['reason'] == 2
        assert np.all(counts['controls'][deadline] == contract.T) and np.all(counts['last_substeps'][deadline] == 10)
    else:
        survived = counts['reason'] == 0
        assert np.all(failed | survived) and np.all(task['maximum_controls'] == 0)
        assert np.all(counts['truncated'][survived] == 1) and np.all(counts['done'][survived] == 0)
        assert np.all(counts['controls'][survived] == batch['horizon']) and np.all(counts['last_substeps'][survived] == 10)
        assert np.all(counts['done'][failed] == 1) and np.all(counts['truncated'][failed] == 0)
    assert np.array_equal(counts['reason'], task['reason']) and np.array_equal(counts['done'], task['done'])
    assert np.array_equal(task['boundary_terminal'], counts['reason'] == 1)
    assert np.array_equal(task['behavior_terminal'], failed)
    # Stalling is checked between full controls; partial controls are possible
    # only for the per-substep bank/depth/tilt ordinary terminals.
    assert np.all(counts['last_substeps'][counts['reason'] == 6] == 10)
    assert np.array_equal(batch['final']['body'], batch['parts']['post-body'][last, episode])
    return last, episode


def audit_batch(directory, protocol, layout, frozen, expected_means=None):
    directory = Path(directory)
    batch = open_batch(directory, protocol, layout, expected_means)
    n, ticks, counts, task = batch['n'], batch['ticks'], batch['counts'], batch['task']
    initial, final = batch['initial'], batch['final']
    for worlds in (initial, final):
        assert np.all(worlds['failure'] == 0) and np.all(worlds['unsafe'] == 0) and np.all(worlds['invalid'] == 0)
        assert np.all(worlds['current'] == 0) and np.all(abs(worlds['actual']) <= 1) and np.isfinite(worlds['requested']).all()
    assert np.all(initial['controls'] == 0) and np.all(initial['substeps'] == 0) and np.all(initial['attempted'] == 0)
    for key in ('controls', 'last_world_control', 'reason', 'elapsed_seconds', 'reward_sum', 'path_length_m',
                'done', 'boundary_terminal', 'behavior_terminal', 'maximum_depth_m', 'maximum_tilt_deg',
                'motion_window_arc', 'motion_window_dt', 'motion_window_next', 'motion_window_count', 'rolling_horizontal_speed_m_s'):
        assert np.all(batch['initial_task'][key] == 0), key
    assert np.all(batch['initial_task']['maximum_controls'] == (contract.T if batch['purpose'] < 2 else 0))
    for case in range(batch['cases']):
        actual = initial['body'][case::batch['cases']]
        near(actual, np.broadcast_to(v1.expected_initial_body(case, batch['withheld']), actual.shape), 3e-6, 3e-6)
    last, episode = _assert_prefix_and_terminals(batch)
    fixed_depth = batch['initial_task']['fixed_depth_model']
    near(fixed_depth, cuda_com_model32(initial['body'])[:, 1], 3e-6, 3e-6)
    near(batch['initial_task']['last_com_model'], cuda_com_model32(initial['body']), 3e-6, 3e-6)
    errors = {key: 0. for key in ('action', 'reward', 'filter_angle', 'filter_rate', 'initial_context')}
    peak_depth, peak_tilt = np.zeros(n), np.zeros(n)
    minimum_clearance = batch['initial_task']['minimum_clearance_m'].astype(float).copy()
    native_arc = np.zeros((ticks, n), np.float32)
    active_rows = np.flatnonzero(batch['steps'].reshape(-1))
    flat_parts = {key: value.reshape(-1, value.shape[-1]) for key, value in batch['parts'].items()}
    frozen = np.asarray(frozen, float)
    for first in range(0, len(active_rows), CHUNK_ROWS):
        ids = active_rows[first:first + CHUNK_ROWS]
        control, selected = ids // n, ids % n
        pre = batch['parts']['post-body'][np.maximum(0, control - 1), selected].copy()
        starts = control == 0
        pre[starts] = initial['body'][selected[starts]]
        post, context = flat_parts['post-body'][ids], flat_parts['pre-filter'][ids]
        action, reward = flat_parts['joint-latent'][ids], flat_parts['reward'][ids, 0]
        substeps = batch['steps'][control, selected]
        assert all(np.isfinite(value).all() for value in (pre, post, context, action, reward))
        assert np.all(abs(pre[:, 7:11]) <= 1) and np.all(abs(post[:, 7:11]) <= 1) and np.all(abs(context[:, 8:12]) <= 1)
        assert np.all(post[:, 21] >= pre[:, 21]) and np.all(post[:, 22] >= 0)
        if np.any(starts):
            e = selected[starts]
            expected = np.concatenate((initial['held'][e], initial['rates'][e], initial['last_torque'][e]), axis=1)
            errors['initial_context'] = max(errors['initial_context'], near(context[starts], expected, 0, 0))
        obs = v1.observations(pre, context, fixed_depth[selected], control)
        base = reference.shared.forward(frozen, obs)
        expected_action = contract.action_latents(batch['latent'][selected], base, contract.raw_sensors(pre, fixed_depth[selected]))
        errors['action'] = max(errors['action'], near(action, expected_action, 3e-5, 3e-6))
        angle, rate = context[:, :4].astype(float).copy(), context[:, 4:8].astype(float).copy()
        command = np.tanh(action.astype(float))
        h, omega = float(np.float32(.005)), float(np.float32(2 * np.float32(np.pi)))
        for substep in range(10):
            mask = substeps > substep
            rate[mask] += h * (omega * omega * (command[mask] - angle[mask]) - 2 * omega * rate[mask])
            angle[mask] += h * rate[mask]
        ending = counts['controls'][selected] == control + 1
        next_angle, next_rate = final['held'][selected].copy(), final['rates'][selected].copy()
        continuation = ~ending
        next_angle[continuation] = batch['parts']['pre-filter'][control[continuation] + 1, selected[continuation], :4]
        next_rate[continuation] = batch['parts']['pre-filter'][control[continuation] + 1, selected[continuation], 4:8]
        errors['filter_angle'] = max(errors['filter_angle'], near(next_angle, angle, 2e-6, 2e-5))
        errors['filter_rate'] = max(errors['filter_rate'], near(next_rate, rate, 2e-5, 2e-5))
        behavior = ending & np.isin(counts['reason'][selected], BEHAVIOR_REASONS)
        errors['reward'] = max(errors['reward'], near(reward, rewards(pre, post, fixed_depth[selected], substeps, behavior), 3e-5, 5e-5))
        active_radius, inset, clearance = v1.boundary_values(post)
        bank = ending & (counts['reason'][selected] == 1)
        assert np.all(active_radius <= 1 + 2e-6)
        assert np.all(inset[~bank] < 1 + 2e-6) and np.all(inset[bank] >= 1 - 2e-6)
        metrics = v1.measured(post, fixed_depth[selected])
        np.maximum.at(peak_depth, selected, abs(metrics['depth']))
        np.maximum.at(peak_tilt, selected, metrics['tilt'])
        np.minimum.at(minimum_clearance, selected, clearance)
        native_arc[control, selected] = horizontal_arc32(pre, post)
        depth32 = abs((cuda_com_model32(post)[:, 1] - fixed_depth[selected]) * np.float32(.04))
        # No saved accepted nonterminal state may pass a registered behavior
        # stop. Threshold reconstruction preserves binary32 COM arithmetic.
        depth_terminal = ending & (counts['reason'][selected] == 4)
        assert np.all(depth32[depth_terminal] > np.float32(.005))
        assert np.all(depth32[~bank & ~depth_terminal] <= np.float32(.005))
        tilt_radians32 = np.arccos(np.clip(v1.rotate32(post[:, 3:7], np.float32([0, 1, 0]))[:, 1], np.float32(-1), np.float32(1))).astype(np.float32)
        tilt32 = (tilt_radians32 * np.float32(180)) / np.float32(v1.PI)
        tilt_terminal = ending & (counts['reason'][selected] == 5)
        assert np.all(tilt32[tilt_terminal] > np.float32(20))
        assert np.all(tilt32[~bank & ~depth_terminal & ~tilt_terminal] <= np.float32(20))
    for first in range(0, ticks * n, 32768):
        ids = slice(first, min(ticks * n, first + 32768))
        inactive = batch['steps'].reshape(-1)[ids] == 0
        for value in flat_parts.values():
            assert np.all(value[ids][inactive] == 0)
    dt = batch['steps'].astype(np.float32) * np.float32(.005)
    rolling = rolling_speeds(native_arc, dt)
    live = batch['steps'] > 0
    at_check = np.arange(1, ticks + 1)[:, None] >= 80
    stall_expected = live & at_check & (rolling < np.float32(.002))
    terminal_rows = np.zeros_like(live)
    terminal_rows[last, episode] = True
    higher_priority = terminal_rows & np.isin(counts['reason'], (1, 4, 5))[None, :]
    actual_stall = terminal_rows & (counts['reason'] == 6)[None, :]
    assert np.array_equal(stall_expected & ~higher_priority, actual_stall)
    # Reconstruct every retained ring slot, not just the reported speed.
    expected_arc, expected_dt = np.zeros((n, 40), np.float32), np.zeros((n, 40), np.float32)
    total = counts['controls'].astype(int)
    for slot in range(40):
        index = slot + 40 * ((total - 1 - slot) // 40)
        exists = index >= 0
        expected_arc[exists, slot] = native_arc[index[exists], episode[exists]]
        expected_dt[exists, slot] = dt[index[exists], episode[exists]]
    near(task['motion_window_arc'], expected_arc, 2e-9, 2e-5)
    near(task['motion_window_dt'], expected_dt, 0, 0)
    assert np.array_equal(task['motion_window_next'], total % 40)
    assert np.array_equal(task['motion_window_count'], np.minimum(total, 40))
    near(task['rolling_horizontal_speed_m_s'], rolling[last, episode], 2e-8, 2e-5)
    near(task['last_com_model'], cuda_com_model32(final['body']), 3e-6, 3e-6)
    recorded_reward = batch['parts']['reward'][..., 0].sum(axis=0, dtype=float)
    seconds = dt.sum(axis=0, dtype=float)
    near(counts['reward_sum'], recorded_reward, 1e-7, 1e-7)
    near(task['reward_sum'], recorded_reward, 1e-7, 1e-7)
    near(counts['seconds'], seconds, 1e-12, 1e-12)
    near(task['elapsed_seconds'], seconds, 1e-12, 1e-12)
    near(batch['returns'], np.float32(recorded_reward / 60), 2e-6, 2e-6)
    near(task['maximum_depth_m'], peak_depth, 2e-7, 2e-5)
    near(task['maximum_tilt_deg'], peak_tilt, .003, 2e-5)
    near(task['minimum_clearance_m'], minimum_clearance, 2e-7, 2e-5)
    near(task['path_length_m'], native_arc.sum(axis=0, dtype=float), 1e-6, 2e-5)
    table = v1.rows(directory / 'episodes.csv')
    assert len(table) == n
    for i, row in enumerate(table):
        assert [row[key] for key in ('episode', 'case', 'withheld')] == [i, i % batch['cases'], int(batch['withheld'])]
        for key in ('actual_controls', 'attempted_substeps', 'accepted_substeps', 'last_substeps', 'reason', 'done', 'truncated'):
            assert row[key] == counts['controls' if key == 'actual_controls' else key][i]
        near([row['old_logp'], row['reward_sum'], row['return'], row['elapsed_seconds']],
             [batch['logp'][i], recorded_reward[i], batch['returns'][i], seconds[i]], 1e-7, 1e-10)
        near([row[f'latent{j}'] for j in range(contract.P)], batch['latent'][i], 1e-8, 1e-10)
    report = dict(controls=int(counts['controls'].sum()), substeps=int(counts['attempted_substeps'].sum()),
                  worlds=n, horizon=batch['horizon'], purpose=batch['purpose'], index=batch['index'],
                  means=batch['means'].tolist(), accepted_substeps=int(counts['accepted_substeps'].sum()),
                  task_boundary_episodes=int(np.sum(counts['reason'] == 1)),
                  deadline_episodes=int(np.sum(counts['reason'] == 2)),
                  task_depth_episodes=int(np.sum(counts['reason'] == 4)),
                  task_tilt_episodes=int(np.sum(counts['reason'] == 5)),
                  task_stalled_episodes=int(np.sum(counts['reason'] == 6)),
                  externally_truncated_episodes=int(np.sum(counts['truncated'])), stored_rows=n * ticks,
                  actual_rows=len(active_rows), probability_error=batch['probability_error'], rng_error=batch['rng_error'],
                  maximum_errors=errors, return_normalization_seconds=60,
                  learning_return_definition=batch['purpose'] < 2, optimizer_training_data=batch['purpose'] == 1)
    timing = contract.js(directory / 'timing.json')
    assert [timing[key] for key in ('worlds', 'horizon', 'stored_ticks', 'actual_controls', 'actual_substeps', 'fault', 'optimizer_steps')] == [
        n, batch['horizon'], ticks, report['controls'], report['substeps'], 0, 0]
    for key in ('collection_readback_validation_seconds', 'evidence_seconds'):
        assert math.isfinite(timing[key]) and timing[key] >= 0
    report['timing'] = timing
    report['evidence_scope'] = ('Every saved accepted control endpoint and terminal cause is independently checked. '
                                'Unrecorded intermediate5ms poses and first crossing rely on the frozen guarded '
                                'native wrapper and matched-command gate; this audit does not replay dynamics.')
    return batch, report


def score_evaluation(directory, batch):
    # Acceptance intentionally reuses the independently reviewed V1 movement,
    # coverage and actual-body45deg hysteresis gates; none is relaxed in V2.
    return v1.score_evaluation(Path(directory), batch)


def self_test():
    """Pure synthetic ring arithmetic only; no world advancement or file writes."""
    generator = np.random.default_rng(98123)
    arc = generator.uniform(0, .001, (121, 7)).astype(np.float32)
    dt = np.full_like(arc, np.float32(10) * np.float32(.005))
    dt[80:, 1] = 0
    arc[80:, 1] = 0
    actual = rolling_speeds(arc, dt)
    for end in range(1, len(arc) + 1):
        begin = max(0, end - 40)
        expected = np.divide(arc[begin:end].astype(float).sum(axis=0), dt[begin:end].astype(float).sum(axis=0),
                             out=np.zeros(7), where=dt[begin:end].astype(float).sum(axis=0) > 0).astype(np.float32)
        assert np.array_equal(actual[end - 1], expected)
    # Long mixed-magnitude fixture checks physical-slot order directly rather
    # than relying on chronological cumulative sums to round identically.
    long_arc = np.exp(generator.uniform(math.log(1e-14), math.log(.001), (36000, 2))).astype(np.float32)
    long_dt = (generator.integers(1, 11, (36000, 2)).astype(np.float32) * np.float32(.005))
    long_actual = rolling_speeds(long_arc, long_dt)
    arc_ring, dt_ring = np.zeros((40, 2), np.float32), np.zeros((40, 2), np.float32)
    for row in range(len(long_arc)):
        arc_ring[row % 40], dt_ring[row % 40] = long_arc[row], long_dt[row]
        distance, duration = np.zeros(2), np.zeros(2)
        for slot in range(min(row + 1, 40)):
            distance += arc_ring[slot].astype(float)
            duration += dt_ring[slot].astype(float)
        assert np.array_equal(long_actual[row], (distance / duration).astype(np.float32))
    stationary = rolling_speeds(np.zeros((80, 1), np.float32), np.full((80, 1), np.float32(.05)))
    assert stationary[-1, 0] < np.float32(.002)
    threshold_arc = np.float32(.002) * np.float32(.05)
    threshold = rolling_speeds(np.full((80, 1), threshold_arc), np.full((80, 1), np.float32(.05)))
    assert threshold[-1, 0] == np.float32(.002)
    assert not threshold[-1, 0] < np.float32(.002)
    return dict(passed=True, synthetic_rows=arc.size + long_arc.size, native_slot_order_rows=36000, stall_threshold_is_strict=True,
                physical_controls=0, optimizer_steps=0)
