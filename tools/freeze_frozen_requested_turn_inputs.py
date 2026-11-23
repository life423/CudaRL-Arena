"""Seal one prepared diagnostic's causal inputs, with no physical execution."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    assert root.name == 'minimal-foundation'
    run = root / 'build/ppo/frozen-request-turn-2026-11-23-v1'
    build = root / 'build/frozen-request-turn-native-2026-11-23'
    output, snapshots = run / 'frozen-inputs.json', run / 'frozen-causal-inputs'
    assert run.is_dir() and not output.exists() and not snapshots.exists()
    assert not (run / 'authorization.bin').exists() and not (run / 'probe-started').exists()
    required = [p for p in (root / 'src').rglob('*') if p.is_file()
                and p.suffix in ('.cpp', '.h', '.cu', '.cuh', '.inc', '.in')]
    required += [root / name for name in (
        'tools/frozen_requested_turn_probe.cu', 'tools/frozen_requested_turn_authorization.h',
        'tools/frozen-request-turn/CMakeLists.txt', 'tools/run-frozen-request-turn-probe.ps1',
        'tools/authorize_frozen_requested_turn.py', 'tools/freeze_frozen_requested_turn_inputs.py',
        'docs/FROZEN_REQUEST_TURN_PROBE_2026-11-23.md',
        'build/continuous-pond-native-2026-11-23/Release/creature_gpu.lib',
        'build/continuous-pond-native-2026-11-23/Release/creature_physics.lib',
        'build/continuous-pond-native-2026-11-23/CMakeCache.txt')]
    required += [build / 'CMakeCache.txt', build / 'Release/frozen_requested_turn_probe.exe']
    required += [p for p in run.iterdir() if p.is_file()]
    entries = []
    for path in sorted(set(required)):
        assert path.is_file(), path
        copy = snapshots / path.relative_to(root)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        digest = sha(path)
        assert sha(copy) == digest
        entries.append({'path': str(path), 'sha256': digest, 'snapshot': str(copy)})
    output.write_text(json.dumps(entries, indent=2) + '\n', encoding='utf-8')
    prep = json.loads((run / 'preparation.json').read_bytes())
    existing_bytes = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    assert existing_bytes + prep['conservative_generated_evidence_bytes'] + 1024**2 <= prep['evidence_cap_bytes']
    print(json.dumps({'frozen_files': len(entries), 'sha256': sha(output),
                      'prepared_bytes': existing_bytes,
                      'projected_generated_bytes': prep['conservative_generated_evidence_bytes'],
                      'physical_controls': 0, 'optimizer_steps': 0}))


if __name__ == '__main__':
    main()
