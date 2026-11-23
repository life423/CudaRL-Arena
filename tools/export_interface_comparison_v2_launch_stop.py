"""Publish a verified pre-dynamics launch failure; never create a physics result."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

STUDY = 'frozen_interface_comparison_v2'
PROFILE = 'cpu_compatible_no_implicit_fma_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def count(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and int(value) == value and 0 <= value <= maximum


def bound_input(proof, path, digest):
    inputs = proof.get('input_sha256')
    require(isinstance(inputs, dict), 'Missing independent input bindings')
    identity = lambda value: str(value).replace('\\', '/').lower()
    matches = [str(value).lower() for key, value in inputs.items() if identity(key) == identity(path)]
    require(matches == [digest.lower()], 'Independent proof does not bind ' + str(path))


def validate(proof, process, arithmetic, protection, manifest_count, manifest_digest, digests):
    for document in (proof, arithmetic):
        require(document.get('schema') == 1 and document.get('study') == STUDY and document.get('study_version') == 2,
                'Wrong v2 evidence identity')
        require(document.get('arithmetic_profile') == PROFILE, 'Wrong arithmetic profile')
    for field in ('audit_passed', 'launch_failed_before_dynamics', 'weights_unchanged', 'descriptive_only'):
        require(proof.get(field) is True, 'Missing verified launch-stop condition: ' + field)
    for field in ('qualified', 'runtime_adopted', 'full_comparison_completed'):
        require(proof.get(field) is False, 'Launch failure cannot claim ' + field)
    require(proof.get('audit_status') == 'verified_launch_stop' and proof.get('conclusion') == 'comparison_blocked',
            'Not a closed, independently verified launch stop')
    for field in ('recorded_controls', 'recorded_attempted_substeps', 'recorded_accepted_substeps',
                  'matched_precheck_cases', 'identical_warmup_pairs', 'PPO_updates', 'optimizer_steps'):
        require(count(proof.get(field), 0), 'Launch failure cannot claim ' + field)
    require('native_result' in proof and proof['native_result'] is None, 'Launch stop must not invent a physical result')
    require('original_swim_rest_retained' in proof and proof['original_swim_rest_retained'] is None,
            'Original-skill retention was not assessed')
    require(proof.get('cases') == [] and proof.get('arm_summary') == [] and proof.get('unrun_cases') == list(range(88)),
            'Every interface case must remain unrun')
    retention = proof.get('retention')
    require(isinstance(retention, dict), 'Missing explicit unrun retention')
    for backend in ('cpu', 'cuda'):
        kept = retention.get(backend)
        require(isinstance(kept, dict) and kept.get('status') == 'not_run' and 'passed' in kept and kept['passed'] is None
                and 'totals' in kept and kept['totals'] is None and kept.get('cases') == [], 'Retention must be unassessed')
        require(count(kept.get('controls'), 0) and count(kept.get('substeps'), 0), 'Retention must use zero physics')
    require(process.get('schema') == 1 and process.get('study') == STUDY and process.get('exit_code') == 2
            and process.get('terminated_at_cap') is False, 'Wrong failed-launch process receipt')
    require(proof.get('process_execution') == process, 'Process receipt differs from audited launch')
    require(str(proof.get('process_sha256', '')).lower() == digests['process'], 'Process receipt binding differs')
    seconds = process.get('seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 900, 'Invalid launch time')
    for field in ('PPO_updates', 'optimizer_steps'):
        require(count(process.get(field), 0), 'Launch performed learning')
    stop = proof.get('launch_stop')
    require(isinstance(stop, dict) and stop.get('phase') == 'pre_probe_inventory' and stop.get('native_exit_code') == 2
            and stop.get('reason') == 'filesystem_iterator_error' and isinstance(stop.get('stderr'), str)
            and 'recursive_directory_iterator' in stop['stderr'], 'Wrong pre-dynamics launch failure')
    require(arithmetic.get('phase') == 'arithmetic-preparation' and arithmetic.get('passed') is True,
            'Fixed-state arithmetic preparation did not pass')
    for field in ('physical_controls', 'physical_substeps', 'integration_calls', 'PPO_updates', 'optimizer_steps'):
        require(count(arithmetic.get(field), 0), 'Arithmetic preparation integrated or trained')
    require(str(proof.get('arithmetic_preparation_sha256', '')).lower() == digests['arithmetic'],
            'Arithmetic preparation binding differs')
    require(protection.get('passed') is True and protection.get('study') == STUDY, 'Protected files were not verified')
    require(protection.get('files') == manifest_count and manifest_count > 0, 'Protected inventory count differs')
    require(str(protection.get('manifest_sha256', '')).lower() == manifest_digest, 'Protected inventory binding differs')
    for field, key in (('launch_stop_verification_sha256', 'source'), ('process_execution_sha256', 'process'),
                       ('arithmetic_preparation_sha256', 'arithmetic')):
        require(str(protection.get(field, '')).lower() == digests[key], 'Protection receipt binding differs: ' + field)
    for field in ('physical_controls', 'optimizer_steps'):
        require(count(protection.get(field), 0), 'Protection verification performed unapproved work')


def self_test():
    arithmetic = dict(schema=1, study=STUDY, study_version=2, arithmetic_profile=PROFILE, phase='arithmetic-preparation',
                      passed=True, physical_controls=0, physical_substeps=0, integration_calls=0, PPO_updates=0, optimizer_steps=0)
    process = dict(schema=1, study=STUDY, exit_code=2, terminated_at_cap=False, seconds=.1, PPO_updates=0, optimizer_steps=0)
    not_run = dict(status='not_run', passed=None, controls=0, substeps=0, cases=[], totals=None)
    digests = dict(source='a'*64, process='b'*64, arithmetic='c'*64)
    proof = dict(schema=1, study=STUDY, study_version=2, arithmetic_profile=PROFILE, audit_passed=True,
                 audit_status='verified_launch_stop', launch_failed_before_dynamics=True, weights_unchanged=True,
                 descriptive_only=True, qualified=False, runtime_adopted=False, full_comparison_completed=False,
                 conclusion='comparison_blocked', recorded_controls=0, recorded_attempted_substeps=0,
                 recorded_accepted_substeps=0, matched_precheck_cases=0, identical_warmup_pairs=0, PPO_updates=0,
                 optimizer_steps=0, native_result=None, original_swim_rest_retained=None, cases=[], arm_summary=[],
                 unrun_cases=list(range(88)), retention=dict(cpu=copy.deepcopy(not_run), cuda=copy.deepcopy(not_run)),
                 process_execution=process, process_sha256=digests['process'], arithmetic_preparation_sha256=digests['arithmetic'],
                 launch_stop=dict(phase='pre_probe_inventory', native_exit_code=2, reason='filesystem_iterator_error',
                                  stderr='TESTONLY recursive_directory_iterator path failure'))
    protection = dict(passed=True, study=STUDY, files=1, manifest_sha256='d'*64,
                      launch_stop_verification_sha256=digests['source'], process_execution_sha256=digests['process'],
                      arithmetic_preparation_sha256=digests['arithmetic'], physical_controls=0, optimizer_steps=0)
    documents = (proof, process, arithmetic, protection)
    def run(values):
        validate(*values, 1, 'd'*64, digests)
    run(documents)
    mutations = [(0, 'audit_passed', False), (0, 'weights_unchanged', False), (0, 'conclusion', 'neither_qualifies'),
                 (0, 'native_result', {}), (0, 'original_swim_rest_retained', True), (0, 'recorded_controls', 1),
                 (0, 'PPO_updates', 1), (0, 'qualified', True), (0, 'runtime_adopted', True),
                 (0, 'unrun_cases', list(range(87))), (0, 'arithmetic_preparation_sha256', 'x'), (0, 'process_sha256', 'x'),
                 (1, 'exit_code', 0), (1, 'terminated_at_cap', True), (2, 'passed', False),
                 (2, 'integration_calls', 1), (2, 'arithmetic_profile', 'other'), (3, 'passed', False),
                 (3, 'launch_stop_verification_sha256', 'x'), (3, 'process_execution_sha256', 'x'),
                 (3, 'arithmetic_preparation_sha256', 'x'), (3, 'manifest_sha256', 'x')]
    for index, field, value in mutations:
        changed = copy.deepcopy(documents)
        changed[index][field] = value
        try:
            run(changed)
        except ValueError:
            continue
        raise AssertionError('Accepted invalid TESTONLY mutation: ' + field)
    changed = copy.deepcopy(documents)
    changed[0]['retention']['cuda']['passed'] = False
    try:
        run(changed)
    except ValueError:
        pass
    else:
        raise AssertionError('Unrun retention relabeled as measured failure')
    binding = dict(input_sha256={'C:\\TESTONLY\\process.json': digests['process']})
    bound_input(binding, 'c:/testonly/process.json', digests['process'])
    try:
        bound_input(binding, 'c:/testonly/process.json', 'x'*64)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted substituted process binding')
    print(json.dumps(dict(passed=True, assertions=26, physical_controls=0, GPU_kernels=0, optimizer_steps=0)))


def publish(args):
    root = args.run_root.resolve()
    paths = dict(source=root/'independent-launch-stop-verification.json', process=root/'process-execution.json',
                 arithmetic=root/'independent-arithmetic-verification.json', protection=args.protected_after.resolve())
    payloads = {key: path.read_bytes() for key, path in paths.items()}
    documents = {key: json.loads(value.decode('utf-8-sig')) for key, value in payloads.items()}
    digests = {key: sha(value) for key, value in payloads.items()}
    manifest_bytes = args.protected_manifest.read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    if isinstance(manifest, dict):
        manifest = manifest.get('files')
    require(isinstance(manifest, list), 'Protected inventory is not a file list')
    validate(*(documents[key] for key in ('source', 'process', 'arithmetic', 'protection')),
             len(manifest), sha(manifest_bytes), digests)
    for key in ('process', 'arithmetic'):
        bound_input(documents['source'], paths[key], digests[key])
    names = dict(source='interface-comparison-v2-launch-verification.json', process='interface-comparison-v2-launch-process.json',
                 arithmetic='interface-comparison-v2-launch-arithmetic.json', protection='interface-comparison-v2-launch-protection.json')
    wrapper = dict(schema=1, **{key: dict(path='res://research_data/'+names[key], sha256=digests[key],
                                        verification_input=str(paths[key])) for key in names})
    outputs = {names[key]: value for key, value in payloads.items()}
    outputs['interface-comparison-v2-launch-stop.json'] = (json.dumps(wrapper, indent=2)+'\n').encode('utf-8')
    destination = args.destination.resolve()
    for name, data in outputs.items():
        target = destination/name
        require(not target.exists() or target.read_bytes() == data, 'Refusing to replace different evidence: '+name)
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = destination/name
        if not target.exists():
            with tempfile.NamedTemporaryFile(prefix='verified-interface-launch-stop-', suffix='.tmp', dir=destination, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(target)
        require(target.read_bytes() == data, 'Published bytes differ: '+name)
    print(json.dumps(dict(published=True, presentation_only=True, study=STUDY, conclusion='comparison_blocked',
                          physical_controls=0, physical_substeps=0, optimizer_steps=0,
                          exporter_sha256=sha(Path(__file__).read_bytes()), outputs={name: sha(data) for name, data in outputs.items()}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', nargs='?', type=Path)
    parser.add_argument('--protected-manifest', type=Path)
    parser.add_argument('--protected-after', type=Path)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1]/'live/research_data')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not all((args.run_root, args.protected_manifest, args.protected_after)):
        parser.error('run_root, --protected-manifest and --protected-after are required')
    else:
        publish(args)
