"""Publish audited v2 comparison facts only; never run physics or load a motor."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

STUDY = 'frozen_interface_comparison_v2'
PROFILE = 'cpu_compatible_no_implicit_fma_v1'
CONCLUSIONS = ('lane_following_qualifies', 'heading_only_qualifies', 'both_qualify',
               'neither_qualifies', 'comparison_blocked')
TOTALS = dict(short_passes=4, original_holds=6, long60_retained=28,
              recovery_passes=3, long300_retained=148, withheld_holds=6, endurance_holds=2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def count(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and int(value) == value and 0 <= value <= maximum


def identity(path):
    return str(path).replace('\\', '/').lower()


def bound_input(proof, path, digest):
    inputs = proof.get('input_sha256')
    require(isinstance(inputs, dict), 'Missing independent input bindings')
    matches = [str(value).lower() for key, value in inputs.items() if identity(key) == identity(path)]
    require(matches == [digest.lower()], 'Independent proof does not bind ' + str(path))


def retained(value):
    require(isinstance(value, dict) and type(value.get('passed')) is bool, 'Malformed retained-skill result')
    require(count(value.get('controls'), 46240) and count(value.get('substeps'), 462400), 'Invalid retained-skill counts')
    require(value['substeps'] <= 10*value['controls'], 'Impossible retained-skill accounting')
    require(isinstance(value.get('cases'), list) and len(value['cases']) <= 25, 'Invalid retained-skill fixtures')
    totals = value.get('totals')
    require(isinstance(totals, dict), 'Missing retained-skill totals')
    for field, maximum in TOTALS.items():
        require(count(totals.get(field), maximum), 'Invalid retention total: ' + field)
        if value['passed']:
            require(totals[field] == maximum, 'Retained-skill total did not pass: ' + field)
    if value['passed']:
        require(value['controls'] == 46240 and value['substeps'] == 462400 and len(value['cases']) == 25,
                'Passed retention must complete its full suite')
    return value['passed']


def validate(result, proof, process, protection, manifest_count, manifest_digest, result_digest, proof_digest):
    for value in (result, proof):
        require(value.get('schema') == 1 and value.get('study') == STUDY and value.get('study_version') == 2,
                'Wrong v2 identity')
        require(value.get('arithmetic_profile') == PROFILE, 'Wrong arithmetic profile')
    require(result.get('closed') is True and isinstance(result.get('hard_fault'), str), 'Comparison is not closed')
    require(result.get('descriptive_comparison_only') is True, 'Not a descriptive comparison')
    for field in ('qualified', 'runtime_adopted', 'training'):
        require(result.get(field) is False, 'Disclosure cannot claim ' + field)
    for field in ('PPO_updates', 'optimizer_steps'):
        require(count(result.get(field), 0) and count(process.get(field), 0), 'No learning is permitted')
    for field, maximum in (('controls', 127680), ('attempted_substeps', 1276800), ('accepted_substeps', 1276800),
                           ('precheck_completed_cases', 4), ('cases_started', 88), ('probe_completed_cases', 88)):
        require(count(result.get(field), maximum), 'Invalid recorded ' + field)
    require(result['accepted_substeps'] <= result['attempted_substeps'] <= 10*result['controls'], 'Impossible substeps')
    require(result['probe_completed_cases'] <= result['cases_started'] <= min(88, result['probe_completed_cases']+1),
            'Impossible case accounting')
    require(proof.get('audit_passed') is True and proof.get('descriptive_only') is True and proof.get('qualified') is False,
            'No independently verified descriptive result')
    require(proof.get('weights_unchanged') is True and proof.get('native_result') == result,
            'Saved weights or exact native-result binding differs')
    arithmetic = proof.get('arithmetic_preparation_sha256')
    require(isinstance(arithmetic, str) and len(arithmetic) == 64 and all(c in '0123456789abcdefABCDEF' for c in arithmetic),
            'Missing arithmetic preparation identity')
    require(arithmetic.lower() in [str(x).lower() for x in proof.get('input_sha256', {}).values()],
            'Arithmetic preparation is not a bound input')
    retention = proof.get('retention')
    require(isinstance(retention, dict), 'Retained skills have not been checked')
    passed = [retained(retention.get(name)) for name in ('cpu', 'cuda')]
    require(proof.get('original_swim_rest_retained') is all(passed), 'Retained-skill claim differs from its suites')
    require(result.get('retention_passed') is all(passed), 'Native retention claim differs from its suites')
    require(isinstance(result.get('comparison_blocked_reason'), str), 'Missing native block reason')
    require(sum(retention[name]['controls'] for name in ('cpu', 'cuda')) <= result['controls'], 'Retention controls exceed actual work')
    require(sum(retention[name]['substeps'] for name in ('cpu', 'cuda')) <= result['attempted_substeps'], 'Retention substeps exceed actual work')
    for target, source in (('recorded_controls', 'controls'), ('recorded_attempted_substeps', 'attempted_substeps'),
                           ('recorded_accepted_substeps', 'accepted_substeps')):
        require(proof.get(target) == result[source], 'Independent counts differ')
    full = proof.get('full_comparison_completed')
    require(type(full) is bool, 'Missing completion scope')
    expected_full = result['hard_fault'] == '' and result['precheck_completed_cases'] == 4 and result['probe_completed_cases'] == 88
    require(full is expected_full, 'Native and audited completion disagree')
    conclusion = proof.get('conclusion')
    require(conclusion in CONCLUSIONS, 'Missing explicit conclusion')
    if full:
        require(proof.get('audit_status') == 'passed' and all(passed), 'Full comparison requires retained skills on both backends')
        require(isinstance(proof.get('cases'), list) and len(proof['cases']) == 88, 'Incomplete case evidence')
        require(proof.get('identical_warmup_pairs') == 44 and proof.get('matched_precheck_cases') == 4, 'Matched comparison checks are missing')
        selection = proof.get('selection')
        require(isinstance(selection, dict), 'Missing finite interface selection')
        adequate = selection.get('adequate_arms')
        require(isinstance(adequate, list) and len(set(adequate)) == len(adequate) and set(adequate) <= {'lane_following', 'heading_only'},
                'Malformed adequacy')
        expected = 'both_qualify' if len(adequate) == 2 else adequate[0]+'_qualifies' if adequate else 'neither_qualifies'
        require(conclusion == expected, 'Conclusion disagrees with finite interface criteria')
    else:
        if result['hard_fault']:
            require(proof.get('audit_status') == 'verified_hard_stop', 'Hard fault needs verified stopped data')
        else:
            require(proof.get('audit_status') == 'verified_retention_stop' and result['comparison_blocked_reason'] == 'retention_regression'
                    and not all(passed) and result['precheck_completed_cases'] == 4
                    and result['cases_started'] == result['probe_completed_cases'] == 0,
                    'Incomplete data needs a verified retention stop')
            require(all(len(retention[name]['cases']) == 25 and retention[name]['controls'] == 46240
                        and retention[name]['substeps'] == 462400 for name in ('cpu', 'cuda')),
                    'An ordinary retention stop must finish both suites')
        require(conclusion == 'comparison_blocked', 'An incomplete comparison cannot select a qualifying interface')
        require(isinstance(proof.get('cases'), list) and len(proof['cases']) <= result['cases_started'], 'Impossible stopped case evidence')
    require(process.get('schema') == 1 and process.get('study') == STUDY, 'Wrong physical-process receipt')
    require(process.get('exit_code') == (3 if result['hard_fault'] else 0), 'Physical process status differs')
    require(process.get('terminated_at_cap') is False, 'Watchdog-terminated evidence is not a closed result')
    require(type(process.get('seconds')) in (int, float) and math.isfinite(process['seconds']) and 0 < process['seconds'] <= 900,
            'Invalid process time')
    require(protection.get('passed') is True and protection.get('study') == STUDY, 'No fresh protected-file verification')
    require(protection.get('files') == manifest_count and manifest_count > 0, 'Protected file count differs')
    require(str(protection.get('manifest_sha256', '')).lower() == manifest_digest, 'Wrong protected manifest')
    require(str(protection.get('result_sha256', '')).lower() == result_digest and str(protection.get('verification_sha256', '')).lower() == proof_digest,
            'Protection receipt is not bound to this result/proof')
    for field in ('physical_controls', 'optimizer_steps'):
        require(count(protection.get(field), 0), 'Verification cannot perform extra physical/learning work')


def self_test():
    retention = dict(passed=True, controls=46240, substeps=462400, totals=TOTALS.copy(), cases=[{'TESTONLY': True}]*25)
    result = dict(schema=1, study=STUDY, study_version=2, arithmetic_profile=PROFILE, closed=True, hard_fault='',
                  descriptive_comparison_only=True, qualified=False, runtime_adopted=False, training=False,
                  retention_passed=True, comparison_blocked_reason='',
                  PPO_updates=0, optimizer_steps=0, precheck_completed_cases=4, cases_started=88, probe_completed_cases=88,
                  controls=127680, attempted_substeps=1276800, accepted_substeps=1276800)
    proof = dict(schema=1, study=STUDY, study_version=2, arithmetic_profile=PROFILE, arithmetic_preparation_sha256='a'*64,
                 input_sha256={'TESTONLY-arithmetic.json': 'a'*64}, audit_passed=True, audit_status='passed',
                 full_comparison_completed=True, descriptive_only=True, qualified=False, weights_unchanged=True,
                 native_result=result, retention=dict(cpu=copy.deepcopy(retention), cuda=copy.deepcopy(retention)),
                 original_swim_rest_retained=True, conclusion='neither_qualifies', recorded_controls=127680,
                 recorded_attempted_substeps=1276800, recorded_accepted_substeps=1276800,
                 cases=[{'TESTONLY': True}]*88, identical_warmup_pairs=44, matched_precheck_cases=4,
                 selection=dict(adequate_arms=[]))
    process = dict(schema=1, study=STUDY, exit_code=0, terminated_at_cap=False, seconds=1, PPO_updates=0, optimizer_steps=0)
    protection = dict(passed=True, study=STUDY, files=1, manifest_sha256='b'*64, result_sha256='c'*64,
                      verification_sha256='d'*64, physical_controls=0, optimizer_steps=0)
    documents = (result, proof, process, protection)
    def run(values):
        validate(*values, 1, 'b'*64, 'c'*64, 'd'*64)
    run(documents)
    mutations = [(0, 'PPO_updates', 1), (0, 'training', True), (0, 'arithmetic_profile', 'TESTONLY-wrong'),
                 (1, 'weights_unchanged', False), (1, 'arithmetic_preparation_sha256', 'e'*64),
                 (1, 'original_swim_rest_retained', False), (1, 'conclusion', 'both_qualify'),
                 (1, 'identical_warmup_pairs', 0), (2, 'seconds', 901), (2, 'terminated_at_cap', True),
                 (3, 'passed', False), (3, 'result_sha256', 'x')]
    for index, field, value in mutations:
        values = copy.deepcopy(documents)
        values[index][field] = value
        try:
            run(values)
        except ValueError:
            continue
        raise AssertionError('Accepted invalid TESTONLY mutation: ' + field)
    values = copy.deepcopy(documents)
    values[1]['retention']['cuda']['totals']['long300_retained'] = 147
    try:
        run(values)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted lost original long swim')
    # A verified prerequisite stop is publishable without pretending retention ran.
    values = copy.deepcopy(documents)
    values[0].update(hard_fault='TESTONLY prerequisite stop', precheck_completed_cases=0, cases_started=0,
                     probe_completed_cases=0, controls=2, attempted_substeps=20, accepted_substeps=20, retention_passed=False)
    values[1].update(full_comparison_completed=False, audit_status='verified_hard_stop', conclusion='comparison_blocked',
                     recorded_controls=2, recorded_attempted_substeps=20, recorded_accepted_substeps=20,
                     cases=[], original_swim_rest_retained=False)
    for value in values[1]['retention'].values():
        value.update(passed=False, controls=0, substeps=0, totals={k: 0 for k in TOTALS}, cases=[])
    values[2]['exit_code'] = 3
    run(values)
    values = copy.deepcopy(documents)
    values[1]['selection']['adequate_arms'] = ['lane_following', 'heading_only']
    values[1]['conclusion'] = 'both_qualify'
    run(values)
    values = copy.deepcopy(documents)
    values[0].update(retention_passed=False, comparison_blocked_reason='retention_regression', cases_started=0,
                     probe_completed_cases=0, controls=93480, attempted_substeps=934800, accepted_substeps=934800)
    values[1].update(full_comparison_completed=False, audit_status='verified_retention_stop', conclusion='comparison_blocked',
                     recorded_controls=93480, recorded_attempted_substeps=934800, recorded_accepted_substeps=934800,
                     cases=[], original_swim_rest_retained=False)
    values[1]['retention']['cpu']['passed'] = False
    values[1]['retention']['cpu']['totals']['short_passes'] = 3
    run(values)
    print(json.dumps(dict(passed=True, assertions=17, physical_controls=0, GPU_kernels=0, optimizer_steps=0)))


def publish(args):
    root = args.run_root.resolve()
    result_path, proof_path, process_path = [root/name for name in ('result.json', 'independent-result-verification.json', 'process-execution.json')]
    result_bytes, proof_bytes, process_bytes = [p.read_bytes() for p in (result_path, proof_path, process_path)]
    result, proof, process = [json.loads(raw.decode('utf-8-sig')) for raw in (result_bytes, proof_bytes, process_bytes)]
    manifest_bytes = args.protected_manifest.read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    if isinstance(manifest, dict):
        manifest = manifest.get('files')
    require(isinstance(manifest, list), 'Malformed protected-file inventory')
    protection = json.loads(args.protected_after.read_text(encoding='utf-8-sig'))
    result_digest, proof_digest = sha(result_bytes), sha(proof_bytes)
    validate(result, proof, process, protection, len(manifest), sha(manifest_bytes), result_digest, proof_digest)
    bound_input(proof, result_path, result_digest)
    bound_input(proof, process_path, sha(process_bytes))
    wrapper = dict(schema=1, source=dict(path='res://research_data/interface-comparison-v2-study-result.json',
                                        sha256=result_digest, verification_input=str(result_path)),
                   verification=dict(path='res://research_data/interface-comparison-v2-study-verification.json', sha256=proof_digest))
    outputs = {'interface-comparison-v2-study-result.json': result_bytes,
               'interface-comparison-v2-study-verification.json': proof_bytes,
               'interface-comparison-v2-study.json': (json.dumps(wrapper, indent=2)+'\n').encode('utf-8')}
    destination = args.destination.resolve()
    for name, data in outputs.items():
        target = destination/name
        require(not target.exists() or target.read_bytes() == data, 'Refusing to overwrite different v2 evidence')
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = destination/name
        if not target.exists():
            with tempfile.NamedTemporaryFile(prefix='verified-interface-v2-', suffix='.tmp', dir=destination, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(target)
        require(target.read_bytes() == data, 'Copied evidence differs')
    print(json.dumps(dict(published=True, study=STUDY, conclusion=proof['conclusion'],
                          original_swim_rest_retained=proof['original_swim_rest_retained'],
                          physical_controls=0, GPU_kernels=0, optimizer_steps=0,
                          exporter_sha256=sha(Path(__file__).read_bytes()),
                          outputs={name: sha(data) for name, data in outputs.items()}), indent=2))


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
