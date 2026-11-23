"""Publish byte-bound, audited diagnostic disclosure; never load a motor or run physics."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

STUDY = 'frozen_interface_comparison'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    return str(path).replace('\\', '/').lower()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum and int(value) == value


def bound_input(proof, path, digest):
    inputs = proof.get('input_sha256')
    require(isinstance(inputs, dict), 'Independent audit input bindings are absent')
    matches = [str(value).lower() for key, value in inputs.items() if identity(key) == identity(path)]
    require(matches == [digest.lower()], 'Independent audit does not bind current ' + str(path))


def validate(result, proof, process, protection, manifest_count, manifest_digest, result_digest, proof_digest):
    require(result.get('schema') == 1 and result.get('study') == STUDY, 'Wrong native diagnostic schema/study')
    require(result.get('closed') is True and result.get('hard_fault') == '', 'Comparison did not close without a hard fault')
    require(result.get('descriptive_comparison_only') is True, 'Comparison scope is not descriptive')
    require(result.get('qualified') is False and result.get('runtime_adopted') is False and result.get('training') is False, 'Disclosure cannot qualify, train or adopt a motor')
    for name in ('PPO_updates', 'optimizer_steps'):
        require(count(result.get(name), 0) and count(process.get(name), 0), 'Frozen comparison recorded an optimizer update')
    require(result.get('precheck_completed_cases') == 4 and result.get('probe_completed_cases') == 88, 'Comparison grid did not finish')
    for name, maximum in [('controls', 35200), ('attempted_substeps', 352000), ('accepted_substeps', 352000)]:
        require(count(result.get(name), maximum), 'Invalid native count: ' + name)
    require(result['accepted_substeps'] <= result['attempted_substeps'] <= 10 * result['controls'], 'Impossible native substep counts')
    require(proof.get('schema') == 1 and proof.get('audit_passed') is True and proof.get('audit_status') == 'passed', 'Independent audit did not pass')
    require(proof.get('full_comparison_completed') is True and proof.get('descriptive_only') is True and proof.get('qualified') is False, 'Audit scope or completion is wrong')
    require(proof.get('weights_unchanged') is True, 'Saved motor identity was not verified')
    require(proof.get('native_result') == result, 'Independent audit native-result echo differs')
    for target, source in [('recorded_controls', 'controls'), ('recorded_attempted_substeps', 'attempted_substeps'), ('recorded_accepted_substeps', 'accepted_substeps')]:
        require(proof.get(target) == result[source], 'Independent audit count differs: ' + target)
    require(isinstance(proof.get('cases'), list) and len(proof['cases']) == 88, 'Independent audit has incomplete case data')
    require(isinstance(proof.get('arm_summary'), list) and len(proof['arm_summary']) == 2, 'Independent audit has incomplete arm data')
    require(process.get('schema') == 1 and process.get('study') == STUDY and process.get('exit_code') == 0, 'Physical process did not succeed')
    require(process.get('terminated_at_cap') is False, 'Physical process hit its watchdog')
    seconds = process.get('seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 240, 'Process exceeded its wall-time cap')
    require(protection.get('passed') is True and protection.get('study') == STUDY, 'Fresh protected-file verification did not pass')
    require(protection.get('files') == manifest_count and manifest_count > 0, 'Protected-file count differs')
    require(str(protection.get('manifest_sha256', '')).lower() == manifest_digest, 'Protection receipt is bound to another manifest')
    require(str(protection.get('result_sha256', '')).lower() == result_digest and str(protection.get('verification_sha256', '')).lower() == proof_digest, 'Protection receipt is not bound to this result and audit')
    for name in ('physical_controls', 'optimizer_steps'):
        require(count(protection.get(name), 0), 'Protected-file verification performed unapproved work')


def self_test():
    result = dict(schema=1, study=STUDY, closed=True, hard_fault='', descriptive_comparison_only=True, qualified=False,
                  runtime_adopted=False, training=False, PPO_updates=0, optimizer_steps=0, precheck_completed_cases=4,
                  probe_completed_cases=88, controls=35200, attempted_substeps=352000, accepted_substeps=352000)
    proof = dict(schema=1, audit_passed=True, audit_status='passed', full_comparison_completed=True,
                 descriptive_only=True, qualified=False, weights_unchanged=True, native_result=result,
                 recorded_controls=35200, recorded_attempted_substeps=352000, recorded_accepted_substeps=352000,
                 cases=[{'TESTONLY': True}] * 88, arm_summary=[{'TESTONLY': True}] * 2)
    process = dict(schema=1, study=STUDY, exit_code=0, terminated_at_cap=False, seconds=1, PPO_updates=0, optimizer_steps=0)
    protection = dict(passed=True, study=STUDY, files=1, manifest_sha256='a' * 64, result_sha256='b' * 64,
                      verification_sha256='c' * 64, physical_controls=0, optimizer_steps=0)
    def run(documents):
        validate(*documents, 1, 'a' * 64, 'b' * 64, 'c' * 64)
    documents = (result, proof, process, protection)
    run(documents)
    mutations = [(0, 'PPO_updates', 1), (0, 'runtime_adopted', True), (0, 'hard_fault', 'TESTONLY'),
                 (1, 'weights_unchanged', False), (1, 'audit_passed', False), (2, 'seconds', 241),
                 (2, 'terminated_at_cap', True), (3, 'passed', False), (3, 'manifest_sha256', 'x'),
                 (3, 'result_sha256', 'x'), (3, 'verification_sha256', 'x')]
    for index, field, value in mutations:
        changed = copy.deepcopy(documents)
        changed[index][field] = value
        try:
            run(changed)
        except ValueError:
            continue
        raise AssertionError('Accepted invalid TESTONLY mutation: ' + field)
    require(identity('C:\\TESTONLY\\result.json') == identity('c:/TESTONLY/result.json'), 'Portable audit identity differs')
    print(json.dumps(dict(passed=True, assertions=13, physical_controls=0, GPU_kernels=0, optimizer_steps=0)))


def publish(args):
    root = args.run_root.resolve()
    result_path = root / 'result.json'
    proof_path = root / 'independent-result-verification.json'
    process_path = root / 'process-execution.json'
    result_bytes, proof_bytes, process_bytes = (path.read_bytes() for path in (result_path, proof_path, process_path))
    result, proof, process = (json.loads(data.decode('utf-8-sig')) for data in (result_bytes, proof_bytes, process_bytes))
    manifest_bytes = args.protected_manifest.read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    if isinstance(manifest, dict):
        manifest = manifest.get('files')
    require(isinstance(manifest, list), 'Protected manifest is not a file inventory')
    protection = json.loads(args.protected_after.read_text(encoding='utf-8-sig'))
    result_digest, proof_digest = sha(result_bytes), sha(proof_bytes)
    validate(result, proof, process, protection, len(manifest), sha(manifest_bytes), result_digest, proof_digest)
    bound_input(proof, result_path, result_digest)
    bound_input(proof, process_path, sha(process_bytes))
    wrapper = dict(schema=1, source=dict(path='res://research_data/interface-comparison-study-result.json', sha256=result_digest,
                                        verification_input=str(result_path)),
                   verification=dict(path='res://research_data/interface-comparison-study-verification.json', sha256=proof_digest))
    outputs = {'interface-comparison-study-result.json': result_bytes,
               'interface-comparison-study-verification.json': proof_bytes,
               'interface-comparison-study.json': (json.dumps(wrapper, indent=2) + '\n').encode('utf-8')}
    destination = args.destination.resolve()
    for name, data in outputs.items():
        target = destination / name
        require(not target.exists() or target.read_bytes() == data, 'Refusing to replace different saved disclosure: ' + str(target))
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = destination / name
        if not target.exists():
            with tempfile.NamedTemporaryFile(prefix='TESTED-interface-', suffix='.tmp', dir=destination, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(target)
        require(target.read_bytes() == data, 'Published bytes differ')
    print(json.dumps(dict(published=True, study=STUDY, physical_controls=0, optimizer_steps=0,
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
    else:
        parser.error('run_root, --protected-manifest and --protected-after are required') if not all((args.run_root, args.protected_manifest, args.protected_after)) else publish(args)
