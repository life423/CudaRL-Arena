"""Close an evidenced pre-dynamics launch failure; never launch or simulate."""
import argparse
import hashlib
import json
from pathlib import Path

import verify_frozen_interface_comparison_v2 as v


def sha(path):
    return hashlib.sha256(v.read(path)).hexdigest()


def verify(root):
    absent = ['probe-started', 'retention-started', 'result.json', 'retention.json',
              'independent-result-verification.json']
    assert all(not (root/name).exists() for name in absent)
    process = json.loads(v.read(root/'process-execution.json'))
    assert process['schema']==1 and process['study']=='frozen_interface_comparison_v2'
    assert process['exit_code']==2 and process['terminated_at_cap'] is False
    assert 0 < process['seconds'] < 900
    assert process['PPO_updates']==process['optimizer_steps']==0
    executable = Path(process['executable'])
    assert sha(executable)==process['executable_sha256']
    stderr = v.read(root/'execution.stderr.log').decode().strip()
    stdout = v.read(root/'execution.stdout.log')
    assert not stdout
    assert stderr=='recursive_directory_iterator::operator++: The system cannot find the path specified.'

    checksum, catalog = v.protocol_binding(root)
    frozen = v.verify_freeze(root)
    assert frozen == 197
    arithmetic_sha = v.bind_arithmetic_preparation(root)
    arithmetic = json.loads(v.read(root/'independent-arithmetic-verification.json'))
    assert arithmetic['nonneutral_buoyancy_preserved'] is True
    preparation = json.loads(v.read(root/'independent-preparation-verification.json'))
    v.identity(preparation)
    assert preparation['phase']=='preparation' and preparation['passed']
    assert preparation['physical_run_approved'] is False
    assert preparation['physical_controls']==preparation['physical_substeps']==preparation['optimizer_steps']==preparation['PPO_updates']==0
    assert preparation['protocol_checksum']==checksum and preparation['frozen_inputs']==frozen
    assert preparation['weights_unchanged'] and preparation['arithmetic_preparation_sha256']==arithmetic_sha
    assert preparation['recorded_explicit_CUDA_kernel_launches']==531 and preparation['recorded_NN_forward_calls']==3
    assert preparation['NN_library_kernel_launches_measured'] is False
    for filename,digest in preparation['input_sha256'].items():
        assert sha(filename)==digest.lower(), filename

    grant = v.envelope(root/'authorization.bin')
    assert [grant.value('I'),grant.value('I'),grant.value('Q')]==[0x32554946,2,checksum]
    assert grant.value('Q')==v.shared.fnv(v.read(root/'frozen-inputs.json'))
    assert grant.value('Q')==v.shared.fnv(v.read(root/'independent-preparation-verification.json'))
    approved = bytes(grant.array('u1'));grant.finish()
    assert approved==v.read(root/'user-authorization.txt') and approved
    authorization=json.loads(v.read(root/'authorization.json'))
    assert authorization['schema']==1 and authorization['study']=='frozen_interface_comparison_v2'
    assert authorization['explicit_user_grant_recorded'] is True
    for field,filename in [('protocol_sha256','protocol.bin'),('freeze_sha256','frozen-inputs.json'),
                           ('preparation_proof_sha256','independent-preparation-verification.json'),
                           ('approval_message_sha256','user-authorization.txt'),('authorization_sha256','authorization.bin')]:
        assert authorization[field]==sha(root/filename)
    binding = v.envelope(root/'prepared.bin')
    assert [binding.value('I'),binding.value('I'),binding.value('Q')]==[0x32504946,2,checksum]
    for path in [root/name for name in ('approved-protocol.md','parent-checkpoint.bin','parent-protocol.bin','holding-parent.bin')]+[executable]:
        assert binding.value('Q')==v.shared.fnv(v.read(path))
    binding.finish()
    v.old.load_weights(root)

    # The thrown iterator is before creation of the one-shot probe directory,
    # before NN construction/inference, and before every integration loop.
    tools=Path(__file__).parent
    source=v.read(tools/'frozen_interface_comparison_v2.cu').decode().replace('\r\n','\n')
    run=source[source.index('int run(const fs::path& root'):source.index('\n}\nint main(')]
    inventory=run.index('fs::recursive_directory_iterator(root)')
    probe=run.index('fs::create_directory(root/"probe-started")')
    inference=run.index('nn::CpuMlp<float> cpu')
    dynamics=run.index('advance(active)')
    assert inventory < probe < inference < dynamics
    assert 'catch(const std::exception& e){std::cerr<<e.what()<<' in source
    assert 'return 2;' in source[source.index('int main('):]
    watchdog=v.read(tools/'run-frozen-interface-comparison-v2.ps1').decode()
    assert watchdog.count('Start-Process')==1 and 'process-execution.json' in watchdog
    assert '-ArgumentList @(\'"\'+$comparisonRun+\'"\')' in watchdog
    v.source_dependencies()
    v.read(__file__)
    # Recheck the absence after all reads; this verifier never creates an entry
    # marker or substitutes a native result for an unexecuted physical phase.
    assert all(not (root/name).exists() for name in absent)
    unrun=dict(status='not_run',passed=None,controls=0,substeps=0,cases=[],totals=None)
    return dict(schema=1,study='frozen_interface_comparison_v2',study_version=2,arithmetic_profile=v.PROFILE,
        audit_status='verified_launch_stop',audit_passed=True,launch_failed_before_dynamics=True,
        full_comparison_completed=False,conclusion='comparison_blocked',descriptive_only=True,
        qualified=False,runtime_adopted=False,training=False,PPO_updates=0,optimizer_steps=0,
        weights_unchanged=True,recorded_controls=0,recorded_attempted_substeps=0,recorded_accepted_substeps=0,
        new_physical_controls_executed=0,new_GPU_kernels_executed=0,new_optimizer_steps=0,
        native_physical_launch_attempts=1,physical_process_explicit_CUDA_kernel_launches=0,
        physical_process_NN_forward_calls=0,NN_library_kernel_launches_measured=False,
        prior_preparation=dict(fixed_state_CPU_evaluations=2568,fixed_state_CUDA_evaluations=2568,
                              fixed_state_explicit_CUDA_kernel_launches=4,
                              observer_explicit_CUDA_kernel_launches=531,observer_NN_forward_calls=3),
        matched_precheck_cases=0,identical_warmup_pairs=0,cases=[],arm_summary=[],unrun_cases=list(range(88)),
        original_swim_rest_retained=None,retention=dict(cpu=dict(unrun),cuda=dict(unrun)),
        selection=None,native_result=None,process_execution=process,
        arithmetic_preparation_sha256=arithmetic_sha,process_sha256=sha(root/'process-execution.json'),
        preparation_verification_sha256=sha(root/'independent-preparation-verification.json'),
        authorization_sha256=sha(root/'authorization.bin'),protocol_checksum=checksum,frozen_inputs=frozen,
        launch_stop=dict(phase='pre_probe_inventory',native_exit_code=2,reason='filesystem_iterator_error',stderr=stderr,
                         absent_paths=absent,source_order_verified=True),
        limitation='Fixed-state arithmetic passed. Fresh dynamic parity, original-skill retention on the corrected runtime and all88 interface cases were not run; no interface conclusion or navigation qualification follows.',
        input_sha256=dict(sorted(v.prep.HASHES.items())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    root=args.run.resolve();result=verify(root)
    output=root/'independent-launch-stop-verification.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(audit_passed=True,audit_status=result['audit_status'],
                          conclusion=result['conclusion'],proof_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
