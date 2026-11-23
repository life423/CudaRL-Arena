"""Preserve, seal and authorize the frozen comparison; never simulate or train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

WORK = Path(__file__).resolve().parents[1]
assert WORK.name == 'minimal-foundation'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fnv(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte)*1099511628211) & 0xffffffffffffffff
    return value


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def protect(path):
    previous = WORK/'build/continuous-pond-v2-preparation-2026-11-23/protected-before.json'
    historic = {r['path']: r for r in json.loads(previous.read_bytes())}
    files = {WORK/name for name in historic}
    files |= {p for base in (WORK/'assets', WORK/'build/ppo') for p in base.rglob('*') if p.is_file()}
    entries = []
    for p in sorted(files):
        assert p.is_file(), p
        relative = p.relative_to(WORK).as_posix()
        digest = sha(p)
        if relative in historic:
            assert digest == historic[relative]['sha256'].lower(), f'Historical evidence changed: {relative}'
        entries.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': digest})
    save(path, entries)
    print(json.dumps({'protected_files': len(entries), 'bytes': sum(r['bytes'] for r in entries),
                      'manifest_sha256': sha(path), 'physical_controls': 0, 'optimizer_steps': 0}))


def verify_protected(before, after, run):
    rows = json.loads(Path(before).read_bytes())
    for row in rows:
        p = WORK/row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
    result = {'passed': True, 'study': 'frozen_interface_comparison',
              'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
              'before_sha256': sha(before), 'manifest_sha256': sha(before),
              'result_sha256': sha(run/'result.json'),
              'verification_sha256': sha(run/'independent-result-verification.json'),
              'physical_controls': 0, 'optimizer_steps': 0}
    save(after, result)
    print(json.dumps(result))


def seal(run, build, protected):
    run, build, protected = run.resolve(), build.resolve(), protected.resolve()
    assert run.parent == WORK/'build/ppo' and run.is_dir()
    for name in ('authorization.bin', 'probe-started', 'result.json', 'frozen-inputs.json'):
        assert not (run/name).exists(), name
    paths = {p for p in (WORK/'src').rglob('*') if p.is_file() and p.suffix in ('.h', '.cpp', '.cu', '.cuh', '.inc', '.in')}
    paths |= {p for p in (WORK/'tools').glob('*.py') if p.is_file()}
    paths |= {WORK/p for p in (
        'tools/frozen_interface_comparison.cu', 'tools/frozen_interface_authorization.h',
        'tools/frozen-interface-comparison/CMakeLists.txt', 'tools/run-frozen-interface-comparison.ps1',
        'docs/FROZEN_INTERFACE_COMPARISON_PROTOCOL_2026-11-23.md',
        'build/continuous-pond-native-2026-11-23/Release/creature_gpu.lib',
        'build/continuous-pond-native-2026-11-23/Release/creature_physics.lib',
        'build/continuous-pond-native-2026-11-23/CMakeCache.txt')}
    common = WORK/'tools/frozen_interface_comparison_common.inc'
    if common.is_file():
        paths.add(common)
    paths |= {build/'CMakeCache.txt', build/'Release/frozen_interface_comparison.exe', protected}
    paths |= {p for p in run.iterdir() if p.is_file()}
    snapshots = run/'frozen-causal-inputs'
    assert not snapshots.exists()
    rows = []
    for p in sorted(paths):
        assert p.is_file(), p
        copied = snapshots/p.relative_to(WORK)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, copied)
        digest = sha(p)
        assert sha(copied) == digest
        rows.append({'path': str(p), 'sha256': digest, 'snapshot': str(copied)})
    save(run/'frozen-inputs.json', rows)
    prep = json.loads((run/'preparation.json').read_bytes())
    existing = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    # Also include the original protection manifest outside the run root.
    assert existing + protected.stat().st_size + prep['conservative_generated_evidence_bytes'] + 1024**2 <= 512*1024**2
    assert shutil.disk_usage(run).free > 1024**3 + prep['conservative_generated_evidence_bytes']
    print(json.dumps({'frozen_files': len(rows), 'freeze_sha256': sha(run/'frozen-inputs.json'),
                      'prepared_bytes': existing, 'projected_generated_bytes': prep['conservative_generated_evidence_bytes'],
                      'physical_controls': 0, 'optimizer_steps': 0}))


def authorize(run, message):
    run, message = run.resolve(), message.resolve()
    assert run.parent == WORK/'build/ppo' and run.is_dir()
    for name in ('authorization.bin', 'authorization.json', 'probe-started', 'result.json', 'design-hold.json'):
        assert not (run/name).exists(), name
    approval = message.read_bytes()
    assert 0 < len(approval) <= 65536 and approval.decode('utf-8').strip()
    proof_path, freeze_path = run/'independent-preparation-verification.json', run/'frozen-inputs.json'
    proof = json.loads(proof_path.read_bytes())
    assert proof['phase'] == 'preparation' and proof['passed'] and not proof['physical_run_approved']
    assert proof['physical_controls'] == proof['optimizer_steps'] == 0
    for name, digest in proof['input_sha256'].items():
        assert sha(name) == digest.lower(), name
    for row in json.loads(freeze_path.read_bytes()):
        assert sha(row['path']) == sha(row['snapshot']) == row['sha256'], row['path']
    raw = (run/'protocol.bin').read_bytes()
    magic, size, checksum = struct.unpack_from('<QQQ', raw)
    assert magic == 0x31504b434c524443 and size == len(raw)-24 and fnv(raw[24:]) == checksum
    assert struct.unpack_from('<II', raw, 24) == (0x31524946, 1)
    body = struct.pack('<IIQQQQ', 0x31554946, 1, fnv(raw[24:]), fnv(freeze_path.read_bytes()),
                       fnv(proof_path.read_bytes()), len(approval)) + approval
    envelope = struct.pack('<QQQ', 0x31504b434c524443, len(body), fnv(body)) + body
    with (run/'authorization.bin').open('xb') as stream:
        stream.write(envelope)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {'schema': 1, 'study': 'frozen_interface_comparison', 'explicit_user_grant_recorded': True,
               'protocol_sha256': sha(run/'protocol.bin'), 'freeze_sha256': sha(freeze_path),
               'preparation_proof_sha256': sha(proof_path), 'approval_message_sha256': sha(message),
               'approval_message_file': str(message), 'authorization_sha256': sha(run/'authorization.bin'),
               'writer_sha256': sha(__file__), 'physical_controls': 0, 'optimizer_steps': 0}
    save(run/'authorization.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('protect'); p.add_argument('output', type=Path)
    p = sub.add_parser('verify-protected'); p.add_argument('before', type=Path); p.add_argument('after', type=Path); p.add_argument('--run', type=Path, required=True)
    p = sub.add_parser('freeze'); p.add_argument('run', type=Path); p.add_argument('build', type=Path); p.add_argument('protected', type=Path)
    p = sub.add_parser('authorize'); p.add_argument('run', type=Path); p.add_argument('message', type=Path)
    a = parser.parse_args()
    if a.command == 'protect': protect(a.output)
    elif a.command == 'verify-protected': verify_protected(a.before, a.after, a.run)
    elif a.command == 'freeze': seal(a.run, a.build, a.protected)
    else: authorize(a.run, a.message)
