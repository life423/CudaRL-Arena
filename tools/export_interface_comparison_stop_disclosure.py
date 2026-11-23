"""Present the verified CPU/CUDA prerequisite stop; no physics, policy or optimizer."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile

import export_interface_comparison_disclosure as frozen_export

STUDY = frozen_export.STUDY
require, count, sha = frozen_export.require, frozen_export.count, frozen_export.sha


def validate_stop(result, proof, process, protection, manifest_count, manifest_digest, result_digest, proof_digest):
    require(result.get('schema') == 1 and result.get('study') == STUDY and result.get('closed') is True, 'Wrong stopped diagnostic')
    require(result.get('hard_fault') == 'Matched-command physics disagreement', 'Not the recorded CPU/CUDA prerequisite stop')
    require(result.get('descriptive_comparison_only') is True and result.get('training') is False, 'Wrong diagnostic scope')
    require(result.get('qualified') is False and result.get('runtime_adopted') is False, 'Stopped result cannot adopt or qualify a motor')
    for name in ('precheck_completed_cases', 'cases_started', 'probe_completed_cases', 'PPO_updates', 'optimizer_steps'):
        require(count(result.get(name), 0), 'Stopped diagnostic cannot claim ' + name)
    require(result.get('controls') == 214 and result.get('attempted_substeps') == result.get('accepted_substeps') == 2140, 'Wrong recorded gate-stop spend')
    require(proof.get('schema') == 1 and proof.get('audit_passed') is True and proof.get('audit_status') == 'verified_hard_stop', 'Stopped data was not independently verified')
    require(proof.get('full_comparison_completed') is False and proof.get('descriptive_only') is True and proof.get('qualified') is False, 'Stopped audit cannot claim a complete comparison')
    require(proof.get('weights_unchanged') is True and proof.get('native_result') == result, 'Saved motor or native-result binding differs')
    require(proof.get('cases') == [] and proof.get('matched_precheck_cases') == proof.get('identical_warmup_pairs') == 0, 'No interface cases were performed')
    selection = proof.get('selection')
    require(isinstance(selection, dict) and selection.get('status') == 'incomplete_comparison' and selection.get('preferred_arm') == 'tie' and selection.get('adequate_arms') == [] and selection.get('navigation_qualified') is False, 'Stopped comparison cannot choose an interface')
    for target, source in [('recorded_controls', 'controls'), ('recorded_attempted_substeps', 'attempted_substeps'), ('recorded_accepted_substeps', 'accepted_substeps')]:
        require(proof.get(target) == result[source], 'Independent recorded count differs')
    require(process.get('schema') == 1 and process.get('study') == STUDY and process.get('exit_code') == 3, 'Wrong failed-process receipt')
    require(process.get('terminated_at_cap') is False, 'This is not a watchdog termination')
    seconds = process.get('seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 240, 'Invalid process time')
    for name in ('PPO_updates', 'optimizer_steps'):
        require(count(process.get(name), 0), 'Failed process cannot claim learning')
    require(protection.get('passed') is True and protection.get('study') == STUDY, 'Fresh protected-file verification did not pass')
    require(protection.get('files') == manifest_count and manifest_count > 0, 'Protected file count differs')
    require(str(protection.get('manifest_sha256', '')).lower() == manifest_digest, 'Protected manifest binding differs')
    require(str(protection.get('result_sha256', '')).lower() == result_digest and str(protection.get('verification_sha256', '')).lower() == proof_digest, 'Protection receipt is not bound to this stopped result')
    for name in ('physical_controls', 'optimizer_steps'):
        require(count(protection.get(name), 0), 'Protected verification performed unapproved work')


def self_test():
    result = dict(schema=1, study=STUDY, closed=True, hard_fault='Matched-command physics disagreement',
                  descriptive_comparison_only=True, training=False, qualified=False, runtime_adopted=False,
                  precheck_completed_cases=0, cases_started=0, probe_completed_cases=0, PPO_updates=0,
                  optimizer_steps=0, controls=214, attempted_substeps=2140, accepted_substeps=2140)
    proof = dict(schema=1, audit_passed=True, audit_status='verified_hard_stop', full_comparison_completed=False,
                 descriptive_only=True, qualified=False, weights_unchanged=True, native_result=result, cases=[],
                 matched_precheck_cases=0, identical_warmup_pairs=0, recorded_controls=214,
                 recorded_attempted_substeps=2140, recorded_accepted_substeps=2140,
                 selection=dict(status='incomplete_comparison', preferred_arm='tie', adequate_arms=[], navigation_qualified=False))
    process = dict(schema=1, study=STUDY, exit_code=3, terminated_at_cap=False, seconds=1, PPO_updates=0, optimizer_steps=0)
    protection = dict(passed=True, study=STUDY, files=1, manifest_sha256='a' * 64, result_sha256='b' * 64,
                      verification_sha256='c' * 64, physical_controls=0, optimizer_steps=0)
    documents = (result, proof, process, protection)
    def run(values):
        validate_stop(*values, 1, 'a' * 64, 'b' * 64, 'c' * 64)
    run(documents)
    mutations = [(0, 'PPO_updates', 1), (0, 'cases_started', 1), (0, 'runtime_adopted', True),
                 (1, 'audit_passed', False), (1, 'full_comparison_completed', True), (1, 'weights_unchanged', False),
                 (2, 'exit_code', 0), (2, 'terminated_at_cap', True), (3, 'passed', False),
                 (3, 'manifest_sha256', 'x'), (3, 'result_sha256', 'x'), (3, 'verification_sha256', 'x')]
    for index, field, value in mutations:
        changed = copy.deepcopy(documents)
        changed[index][field] = value
        try:
            run(changed)
        except ValueError:
            continue
        raise AssertionError('Accepted invalid TESTONLY mutation: ' + field)
    changed = copy.deepcopy(documents)
    changed[1]['selection']['preferred_arm'] = 'heading_only'
    try:
        run(changed)
    except ValueError:
        pass
    else:
        raise AssertionError('Stopped result selected an interface')
    print(json.dumps(dict(passed=True, assertions=14, physical_controls=0, GPU_kernels=0, optimizer_steps=0)))


def publish(args):
    root = args.run_root.resolve()
    result_path, proof_path, process_path = (root / name for name in ('result.json', 'independent-result-verification.json', 'process-execution.json'))
    result_bytes, proof_bytes, process_bytes = (path.read_bytes() for path in (result_path, proof_path, process_path))
    result, proof, process = (json.loads(value.decode('utf-8-sig')) for value in (result_bytes, proof_bytes, process_bytes))
    manifest_bytes = args.protected_manifest.read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    if isinstance(manifest, dict):
        manifest = manifest.get('files')
    require(isinstance(manifest, list), 'Protected manifest is not a file inventory')
    protection = json.loads(args.protected_after.read_text(encoding='utf-8-sig'))
    result_digest, proof_digest = sha(result_bytes), sha(proof_bytes)
    validate_stop(result, proof, process, protection, len(manifest), sha(manifest_bytes), result_digest, proof_digest)
    frozen_export.bound_input(proof, result_path, result_digest)
    frozen_export.bound_input(proof, process_path, sha(process_bytes))
    wrapper = dict(schema=1, source=dict(path='res://research_data/interface-comparison-study-result.json', sha256=result_digest,
                                        verification_input=str(result_path)),
                   verification=dict(path='res://research_data/interface-comparison-study-verification.json', sha256=proof_digest))
    outputs = {'interface-comparison-study-result.json': result_bytes,
               'interface-comparison-study-verification.json': proof_bytes,
               'interface-comparison-study.json': (json.dumps(wrapper, indent=2) + '\n').encode('utf-8')}
    destination = args.destination.resolve()
    for name, data in outputs.items():
        target = destination / name
        require(not target.exists() or target.read_bytes() == data, 'Refusing to replace a different disclosure')
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = destination / name
        if not target.exists():
            with tempfile.NamedTemporaryFile(prefix='verified-interface-stop-', suffix='.tmp', dir=destination, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(target)
        require(target.read_bytes() == data, 'Published bytes differ')
    print(json.dumps(dict(published=True, presentation_only=True, study=STUDY, interface_cases_attempted=0,
                          physical_controls=0, optimizer_steps=0,
                          exporter_sha256=sha(Path(__file__).read_bytes()),
                          frozen_export_helper_sha256=sha(Path(frozen_export.__file__).read_bytes()),
                          outputs={name: sha(data) for name, data in outputs.items()}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', nargs='?', type=Path)
    parser.add_argument('--protected-manifest', type=Path)
    parser.add_argument('--protected-after', type=Path)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1] / 'live/research_data')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not all((args.run_root, args.protected_manifest, args.protected_after)):
        parser.error('run_root, --protected-manifest and --protected-after are required')
    else:
        publish(args)
