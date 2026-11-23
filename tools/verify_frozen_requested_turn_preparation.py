"""Independently read zero-integration turn-probe fixtures; never step or train."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import verify_continuous_pond as prior

ref, shared = prior.reference, prior.reference.shared
HASHES = {}


def read(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def envelope(path):
    raw = read(path)
    magic, size, checksum = struct.unpack_from('<QQQ', raw)
    assert magic == 0x31504b434c524443 and size == len(raw) - 24
    assert shared.fnv(raw[24:]) == checksum
    return shared.Reader(raw[24:])


def raw_record(reader, dtype):
    value = np.frombuffer(reader.data, dtype=dtype, count=1, offset=reader.at)[0]
    reader.at += dtype.itemsize
    return value


def dtype(fields, entries):
    return np.dtype({'names': [name for name, _ in entries],
                     'formats': [kind for _, kind in entries],
                     'offsets': [fields[name] for name, _ in entries],
                     'itemsize': fields['size']})


def types(layout):
    layout = {key.lower(): value for key, value in layout.items()}
    task_layout = layout.get('task', layout.get('forwardtask'))
    request_layout = layout.get('request', {'size': 36, 'position': 0,
                                          'terminal_velocity': 12, 'heading': 24})
    request = dtype(request_layout, [(key, ('<f4', 3)) for key in
                                     ('position', 'terminal_velocity', 'heading')])
    task = dtype(task_layout, [('request', request), ('start_com_model', ('<f4', 3)),
                              ('controls', '<i4'), ('last_world_control', '<u8'),
                              ('outcome', '<i4')])
    frame = dtype(layout['frame'], [('world', prior.previous.WORLD_TYPE), ('task', task),
                    ('initial_heading', ('<f4', 3)), ('observation', ('<f4', 41)),
                    ('base', ('<f4', 4)), ('command', ('<f4', 4)), ('valid', '<i4'),
                    ('advanced', '<i4'), ('accepted', '<i4'), ('attempted', '<i4'),
                    ('reason', '<i4'), ('failure', '<i4')])
    assert layout['world']['size'] == 696 and layout['bodystate']['size'] == 92
    return frame, task


def fixed_observations(frame):
    world, task = frame['world'], frame['task']
    body = world['body'].astype(float)
    com, velocity = ref.motion(body)
    heading = task['request']['heading'].astype(float)
    displacement = com - task['start_com_model'].astype(float) * ref.SCALE
    lane = heading * np.sum(displacement * heading, axis=-1, keepdims=True) - displacement
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


def verify(root):
    assert not (root / 'authorization.bin').exists() and not (root / 'probe-started').exists()
    report = json.loads(read(root / 'preparation.json'))
    assert report['passed'] and report['physical_controls'] == report['physical_substeps'] == report['optimizer_steps'] == 0
    assert not report['physical_run_approved']
    protocol = envelope(root / 'protocol.bin')
    protocol_hash = shared.fnv(protocol.data)
    assert [protocol.value('I'), protocol.value('I')] == [0x31525446, 1]
    assert protocol.value('Q') == shared.fnv(read(root / 'approved-protocol.md'))
    parent, hold = envelope(root / 'parent-checkpoint.bin'), envelope(root / 'holding-parent.bin')
    assert protocol.value('Q') == shared.fnv(parent.data)
    assert protocol.value('Q') == shared.fnv(hold.data)
    assert hashlib.sha256(read(root / 'parent-checkpoint.bin')).hexdigest().upper() == ref.PARENT_SHA
    assert hashlib.sha256(read(root / 'holding-parent.bin')).hexdigest().upper() == ref.HOLD_SHA
    assert [protocol.value('i') for _ in range(12)] == [6, 4, 41, 4, 336, 22, 80, 320, 320, 7360, 73600, 120]
    ref.near([protocol.value('f') for _ in range(7)], np.float32([.008, .005, 20, .20, .12, .195, .115]), 0, 0)
    evidence_cap = protocol.value('Q')
    assert evidence_cap in (64 * 1024**2, 128 * 1024**2)
    offsets = [protocol.value('i') for _ in range(11)]
    assert offsets == [0, 5, -5, 15, -15, 30, -30, 60, -60, 90, -90]
    protocol.finish()

    assert [hold.value('I'), hold.value('I')] == [0x314d4148, 1]
    assert hold.value('Q') != 0  # Original holding protocol, distinct from its parent checkpoint.
    assert hold.value('Q') == shared.fnv(parent.data)
    assert [hold.value('I') for _ in range(7)] == [6, 6, 3, 4, 41, 4, 336]
    assert [hold.value('i'), hold.value('i')] == [7, 112]
    weights = hold.array(size=2532).copy()
    assert np.all(weights[:1312].reshape(32, 41)[:, 40] == 0)
    layout = json.loads(read(root / 'layout.json'))
    frame_dtype, task_dtype = types(layout)
    fixture = envelope(root / 'pure-inference.bin')
    assert [fixture.value('I'), fixture.value('I'), fixture.value('Q'), fixture.value('I')] == [0x315a5446, 1, protocol_hash, frame_dtype.itemsize]
    frames = fixture.array(frame_dtype, 26)
    fixture.finish()
    world, task = frames['world'], frames['task']
    assert all(np.all(world[name] == 0) for name in ('controls', 'substeps', 'time', 'attempted', 'failure', 'unsafe', 'invalid'))
    assert all(np.all(frames[name] == 0) for name in ('advanced', 'accepted', 'attempted', 'reason', 'failure'))
    assert np.all(frames['valid'] == 1) and np.all(task['controls'] == 0)
    poses = np.array([0, 1, 2, 3] + [0] * 11 + [3] * 11)
    expected = ref.initial_body(poses)
    body_error = ref.near(world['body'][..., :21], expected, 2e-7, 2e-6)
    observation_error = ref.near(frames['observation'], fixed_observations(frames), 2e-5, 0)
    expected_base = shared.forward(weights.astype(float), frames['observation'].astype(float))
    latent_error = ref.near(frames['base'], expected_base, 2e-5, 0)
    command_error = ref.near(frames['command'], np.tanh(expected_base), 2e-5, 0)

    for name, magic in (('pure-latches.bin', 0x314c5446), ('pure-gpu-latches.bin', 0x31475446)):
        latch = envelope(root / name)
        assert [latch.value('I'), latch.value('I'), latch.value('Q')] == [magic, 1, protocol_hash]
        for case in range(22):
            assert [latch.value('i'), latch.value('i')] == [case, offsets[case % 11]]
            before, after = raw_record(latch, task_dtype), raw_record(latch, task_dtype)
            assert before['controls'] == after['controls'] and before['last_world_control'] == after['last_world_control']
            assert before['start_com_model'][1] == after['start_com_model'][1]
            if offsets[case % 11] == 0:
                assert before.tobytes() == after.tobytes()
            else:
                angle = np.deg2rad(offsets[case % 11])
                x, y, z = before['request']['heading']
                wanted = [np.cos(angle) * x + np.sin(angle) * z, y,
                          -np.sin(angle) * x + np.cos(angle) * z]
                ref.near(after['request']['heading'], wanted, 2e-7, 2e-6)
                ref.near(after['request']['terminal_velocity'], np.array(wanted) * float(np.float32(.008)), 2e-8, 2e-6)
        latch.finish()
    freeze = json.loads(read(root / 'frozen-inputs.json'))
    entries = freeze['files'] if isinstance(freeze, dict) else freeze
    assert entries
    for entry in entries:
        assert hashlib.sha256(read(entry['path'])).hexdigest() == entry['sha256'].lower()
        if 'snapshot' in entry:
            assert hashlib.sha256(read(entry['snapshot'])).hexdigest() == entry['sha256'].lower()
    for module in (prior, ref, shared, prior.previous):
        read(module.__file__)
    read(__file__)
    result = {'schema': 1, 'phase': 'preparation', 'passed': True,
              'physical_run_approved': False, 'physical_controls': 0, 'optimizer_steps': 0,
              'initial_fixtures': 26, 'latched_request_fixtures_per_backend': 22,
              'maximum_body_error': body_error, 'maximum_observation_error': observation_error,
              'maximum_latent_error': latent_error, 'maximum_command_error': command_error,
              'frozen_inputs': len(entries), 'evidence_cap_bytes': evidence_cap,
              'input_sha256': dict(sorted(HASHES.items()))}
    output = root / 'independent-preparation-verification.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'physical_controls': 0, 'optimizer_steps': 0,
                      'proof_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    verify(parser.parse_args().run.resolve())
