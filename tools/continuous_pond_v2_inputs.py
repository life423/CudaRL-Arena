"""Preserve inputs for the explicitly authorized finite food-free pond study.

This utility never starts simulation, training, or an optimizer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('protect', 'freeze', 'check-protected'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert root.name == 'minimal-foundation' and root.parent.name == '.worktrees'
    prep = root / 'build/continuous-pond-v2-preparation-2026-11-23'
    prep.mkdir(parents=True, exist_ok=True)
    protected = prep / 'protected-before.json'
    if args.phase == 'protect':
        assert not protected.exists(), 'Do not overwrite the original protection seal'
        files = sorted(set(
            p for folder in ('assets', 'build/ppo', 'live/research_data')
            for p in (root / folder).rglob('*') if p.is_file()
            and 'continuous-pond-v2-2026-11-23-v1' not in p.parts
        ))
        entries = [{'path': str(p.relative_to(root)).replace('\\', '/'),
                    'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
        old_manifest = root / 'build/continuous-pond-preparation-2026-11-23/protected-before.json'
        old_entries = json.loads(old_manifest.read_text(encoding='utf-8'))
        by_path = {entry['path']: entry for entry in entries}
        for original in old_entries:
            current = by_path[original['path']]
            assert current['bytes'] == original['bytes'] and current['sha256'] == original['sha256'], original['path']
        old_after = {'passed': True, 'files': len(old_entries), 'manifest_sha256': sha(old_manifest),
                     'physical_controls': 0, 'optimizer_steps': 0}
        (old_manifest.parent / 'protected-after.json').write_text(json.dumps(old_after, indent=2) + '\n', encoding='utf-8')
        protected.write_text(json.dumps(entries, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'protected_files': len(entries), 'bytes': sum(x['bytes'] for x in entries),
                          'manifest_sha256': sha(protected), 'physical_controls': 0}))
    elif args.phase == 'check-protected':
        entries = json.loads(protected.read_text(encoding='utf-8'))
        for item in entries:
            path = root / item['path']
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
        report = {'passed': True, 'files': len(entries), 'manifest_sha256': sha(protected),
                  'physical_controls': 0, 'optimizer_steps': 0}
        (prep / 'protected-after.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report))
    else:
        run = root / 'build/ppo/continuous-pond-v2-2026-11-23-v1'
        build = root / 'build/continuous-pond-v2-native-2026-11-23'
        output = run / 'frozen-inputs.json'
        assert run.is_dir() and not output.exists(), 'Prepare once before freezing once'
        required = [build / 'Release/ppo_turning_swim_v2.exe', build / 'Release/turning_physics_gate_v2.exe',
                    root / 'build/continuous-pond-native-2026-11-23/Release/creature_gpu.lib',
                    root / 'build/continuous-pond-native-2026-11-23/Release/creature_physics.lib',
                    root / 'build/continuous-pond-native-2026-11-23/Release/frozen_controller.lib',
                    root / 'build/continuous-pond-native-2026-11-23/generated/frozen_contract.h',
                    build / 'CMakeCache.txt', protected,
                    root / 'build/continuous-pond-v2-preparation-2026-11-23/cmake/CMakeLists.txt']
        required += [p for folder in ('src', 'tools', 'docs') for p in (root / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.h', '.cpp', '.cu', '.cuh', '.inc', '.txt', '.in', '.py', '.ps1')
                     and p.name != 'verify_continuous_pond_v2.py']
        required += [root / 'CMakeLists.txt', root / 'docs/CONTINUOUS_POND_V2_PROTOCOL_2026-11-23.md']
        required += [p for p in (root / 'live').rglob('*') if p.is_file()
                     and '.godot' not in p.parts and p.suffix in ('.gd', '.gdshader', '.godot', '.tscn')]
        required += [p for p in (build / 'generated').rglob('*') if p.is_file()]
        required += [p for p in run.rglob('*') if p.is_file() and p.suffix in ('.bin', '.md')]
        snapshots = run / 'frozen-causal-inputs'
        assert not snapshots.exists()
        entries = []
        for path in sorted(set(required)):
            assert path.is_file(), str(path)
            relative = path.relative_to(root)
            copy = snapshots / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            digest = sha(path)
            assert sha(copy) == digest
            entries.append({'path': str(path), 'sha256': digest, 'snapshot': str(copy)})
        output.write_text(json.dumps(entries, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'frozen_files': len(entries), 'manifest_sha256': sha(output),
                          'physical_controls': 0, 'optimizer_steps': 0}))


if __name__ == '__main__':
    main()
