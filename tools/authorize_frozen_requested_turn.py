"""Record a fresh explicit user grant for the small frozen-turn probe only.

This utility never steps physics, runs inference, trains, or adopts a policy.
Do not invoke it with generated, inferred, historical, or placeholder approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def fnv(raw):
    result = 14695981039346656037
    for byte in raw:
        result = ((result ^ byte) * 1099511628211) & 0xffffffffffffffff
    return result


def payload(path):
    raw = Path(path).read_bytes()
    magic, size, checksum = struct.unpack_from('<QQQ', raw)
    assert magic == 0x31504b434c524443 and size == len(raw) - 24
    assert fnv(raw[24:]) == checksum
    return raw[24:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--approval-message-file', type=Path, required=True)
    parser.add_argument('--confirm-explicit-user-grant', action='store_true', required=True)
    args = parser.parse_args()
    root, approval_path = args.run.resolve(), args.approval_message_file.resolve()
    assert args.confirm_explicit_user_grant and root.is_dir() and approval_path.is_file()
    assert not (root / 'design-hold.json').exists(), 'The proposed probe is held'
    for name in ('authorization.bin', 'authorization.json', 'probe-started', 'result.json'):
        assert not (root / name).exists(), 'Never replace an authorization or repeat a physical attempt'
    approval = approval_path.read_bytes()
    assert 0 < len(approval) <= 65536 and approval.decode('utf-8').strip()

    raw_protocol = payload(root / 'protocol.bin')
    assert struct.unpack_from('<II', raw_protocol) == (0x31525446, 1)
    assert struct.unpack_from('<Q', raw_protocol, 8)[0] == fnv((root / 'approved-protocol.md').read_bytes())
    assert struct.unpack_from('<12i', raw_protocol, 32) == (6, 4, 41, 4, 336, 22, 80, 320, 320, 7360, 73600, 120)
    assert struct.unpack_from('<Q', raw_protocol, 108)[0] == 128 * 1024**2

    proof_path = root / 'independent-preparation-verification.json'
    proof = json.loads(proof_path.read_bytes())
    assert proof['phase'] == 'preparation' and proof['passed'] and not proof['physical_run_approved']
    assert proof['physical_controls'] == proof['optimizer_steps'] == 0
    for name, digest in proof['input_sha256'].items():
        assert sha(name) == digest.lower(), name
    freeze_path = root / 'frozen-inputs.json'
    freeze = json.loads(freeze_path.read_bytes())
    entries = freeze['files'] if isinstance(freeze, dict) else freeze
    assert entries
    for entry in entries:
        assert sha(entry['path']) == entry['sha256'].lower(), entry['path']
        if 'snapshot' in entry:
            assert sha(entry['snapshot']) == entry['sha256'].lower()

    body = struct.pack('<IIQQQ', 0x31555446, 1, fnv(raw_protocol),
                       fnv(freeze_path.read_bytes()), len(approval)) + approval
    envelope = struct.pack('<QQQ', 0x31504b434c524443, len(body), fnv(body)) + body
    receipt = {'schema': 1, 'study': 'frozen_requested_turn_probe',
               'explicit_user_grant_recorded': True,
               'protocol_sha256': sha(root / 'protocol.bin'),
               'frozen_inputs_manifest_sha256': sha(freeze_path),
               'preparation_proof_sha256': sha(proof_path),
               'approval_message_file': str(approval_path),
               'approval_message_sha256': sha(approval_path),
               'authorization_sha256': hashlib.sha256(envelope).hexdigest(),
               'writer_sha256': sha(__file__), 'physical_controls': 0, 'optimizer_steps': 0}
    with (root / 'authorization.bin').open('xb') as stream:
        stream.write(envelope)
        stream.flush()
        os.fsync(stream.fileno())
    with (root / 'authorization.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'authorization_written': True, 'physical_controls': 0,
                      'optimizer_steps': 0, 'authorization_sha256': receipt['authorization_sha256']}))


if __name__ == '__main__':
    main()
