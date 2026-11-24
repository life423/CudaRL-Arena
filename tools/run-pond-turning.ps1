#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$Protocol,
    [string]$ProtocolSha256,
    [ValidateSet(8,16,24,32)][int]$StopAfterUpdate=32,
    [switch]$Run,
    [switch]$ResumeVerifiedBaseline,
    [switch]$SelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$script:Stub=$null
$script:RunnerPath=$PSCommandPath

function Assert-Condition([bool]$Condition,[string]$Message) {
    if(-not $Condition){throw $Message}
}
function Field($Object,[string]$Name) {
    $property=$Object.PSObject.Properties[$Name]
    if($null -eq $property){throw "Missing required field: $Name"}
    return $property.Value
}
function Read-Json([string]$Path) { Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
function Write-Json([string]$Path,$Value) {
    [IO.File]::WriteAllText($Path,($Value|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
}
function File-Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Fnv64([byte[]]$Bytes) {
    $hash=[Numerics.BigInteger]::Parse('14695981039346656037')
    $prime=[Numerics.BigInteger]::Parse('1099511628211')
    $modulus=[Numerics.BigInteger]::Pow(2,64)
    foreach($value in $Bytes){$hash=[Numerics.BigInteger]::Remainder([Numerics.BigInteger]::Multiply([Numerics.BigInteger]::op_ExclusiveOr($hash,[Numerics.BigInteger]$value),$prime),$modulus)}
    return [uint64]$hash
}
function Read-Checkpoint([string]$Path,[switch]$VerifyPayload) {
    $bytes=[IO.File]::ReadAllBytes($Path)
    Assert-Condition ($bytes.Length -ge 24) "Truncated checkpoint: $Path"
    Assert-Condition ([BitConverter]::ToUInt64($bytes,0) -eq 0x31504B434C524443) "Wrong checkpoint wrapper: $Path"
    Assert-Condition ([BitConverter]::ToUInt64($bytes,8) -eq $bytes.Length-24) "Checkpoint length mismatch: $Path"
    $payload=[byte[]]::new($bytes.Length-24)
    [Array]::Copy($bytes,24,$payload,0,$payload.Length)
    $checksum=[BitConverter]::ToUInt64($bytes,16)
    if($VerifyPayload){Assert-Condition ((Fnv64 $payload) -eq $checksum) "Receipt checksum mismatch: $Path"}
    return @{Payload=$payload;Checksum=$checksum}
}
function Check-Retention($Result,[int]$Update,[string]$Candidate,[string]$Receipt) {
    foreach($name in @('passed','assessment_completed','training_admission_qualified','retention_passed')){
        Assert-Condition ((Field $Result $name) -is [bool] -and (Field $Result $name)) "Skill gate failed: $name"
    }
    Assert-Condition (-not (Field $Result 'preflight_only') -and -not (Field $Result 'hard_failure')) 'Preflight or hard fault cannot admit training'
    Assert-Condition ((Field $Result 'update') -eq $Update -and (Field $Result 'baseline_only') -eq ($Update -eq 0)) 'Assessment checkpoint/type mismatch'
    $historicalBaseline=$script:Plan.protocol_id -eq 'pond-motor-ppo-v5-half-second-2026-11-24' -and $Update -eq 0
    if($historicalBaseline){
        Assert-Condition ((Field $Result 'reused_archived_baseline') -eq $true -and (Field $Result 'attempted_controls') -eq 0 -and (Field $Result 'attempted_substeps') -eq 0) 'Baseline evidence reuse must execute zero physical controls'
    }
    $counts=if($historicalBaseline){'archived_retention_controls','archived_retention_substeps'}else{'retention_controls','retention_substeps'}
    foreach($entry in @(@('retention_cases_passed',20),@('retention_cases_completed',20),@('rest_cases_passed',14),@('short_swim_cases_passed',4),@('long_swim_cases_passed',2),@('retained_parent_long_window_passes',28),@('turn_cases_attempted',16),@($counts[0],30400),@($counts[1],304000))){
        Assert-Condition ((Field $Result $entry[0]) -eq $entry[1]) "Incomplete skill assessment: $($entry[0])"
    }
    $candidateChecksum=(Read-Checkpoint $Candidate).Checksum
    Assert-Condition ([string](Field $Result 'checkpoint_checksum') -eq [string]$candidateChecksum) 'Assessment tested another candidate'
    Assert-Condition ([string](Field $Result 'eval_source_identity') -eq [string]$script:Plan.retention_evaluator_source_fnv1a64) 'Evaluator identity mismatch'
    $receiptData=Read-Checkpoint $Receipt -VerifyPayload
    Assert-Condition ($receiptData.Payload.Length -eq 96) 'Wrong schema3 receipt payload length'
    $body=[byte[]]::new(88)
    [Array]::Copy($receiptData.Payload,0,$body,0,88)
    Assert-Condition ((Fnv64 $body) -eq [BitConverter]::ToUInt64($receiptData.Payload,88)) 'Native receipt inner seal mismatch'
    $reader=[IO.BinaryReader]::new([IO.MemoryStream]::new($body,$false))
    try {
        foreach($expected in @([uint32]0x31525044,3,4,2)){Assert-Condition ($reader.ReadUInt32() -eq $expected) 'Wrong typed receipt schema/runtime'}
        foreach($expected in @($candidateChecksum,$script:ParentChecksum,$script:ProtocolChecksum,[uint64]$script:Plan.retention_evaluator_source_fnv1a64)){
            Assert-Condition ($reader.ReadUInt64() -eq $expected) 'Typed receipt candidate/parent/protocol/evaluator mismatch'
        }
        foreach($expected in @(20,20,20,0)){Assert-Condition ($reader.ReadUInt32() -eq $expected) 'Typed receipt incomplete or physically failed'}
        Assert-Condition ($reader.ReadUInt64() -eq 30400 -and $reader.ReadUInt64() -eq 304000) 'Typed receipt control accounting mismatch'
        Assert-Condition ($reader.ReadUInt32() -eq 1 -and $reader.ReadUInt32() -eq [uint32]($Update -eq 0)) 'Failed or incorrectly typed receipt'
        Assert-Condition ($reader.BaseStream.Position -eq $reader.BaseStream.Length) 'Unexpected receipt bytes'
    } finally {$reader.Dispose()}
}
function Check-BaselineRecovery([string]$Root,[string]$PlanHash) {
    $old=Join-Path $Root 'runner/result.json';$oldResult=Read-Json $old
    $oldStart=Read-Json (Join-Path $Root 'runner/started.json')
    Assert-Condition ($oldResult.status -eq 'STOPPED' -and $oldResult.error -eq 'Unexpected receipt bytes' -and $oldResult.last_passed_training_chunk_update -eq 0 -and $null -eq $oldResult.native_training_accounting) 'Recovery is only for the verified baseline parser stop before training'
    Assert-Condition ($oldStart.protocol_sha256 -eq $PlanHash -and -not $oldStart.synthetic -and -not $oldResult.synthetic) 'Recovery must bind the real original invocation and unchanged protocol'
    Assert-Condition (@($oldResult.commands|Where-Object {$_.stage -like 'train-*'}).Count -eq 0) 'Training was already attempted; baseline-only recovery forbidden'
    $snapshot=Join-Path $Root 'runner-recovery-inputs/run-pond-turning-original.ps1'
    Assert-Condition ((File-Hash $snapshot) -eq $script:Plan.runner_script_sha256) 'Original sealed runner snapshot mismatch'
    Assert-Condition (-not (Test-Path -LiteralPath ($script:Plan.ledger_path_absolute+'.running')) -and -not (Test-Path -LiteralPath ($script:Plan.evaluation_ledger_path_absolute+'.assessment.lock'))) 'Native process state remains active/unresolved'
    $initial=Read-Json (Join-Path $Root 'init/result.json')
    Assert-Condition ($initial.initialized -and $initial.physical_controls -eq 0 -and $initial.physical_substeps -eq 0 -and $initial.optimizer_calls -eq 0) 'Recovery initialization was not zero-step'
    $candidate=Join-Path $Root 'init/candidate-0.bin';$receipt=Join-Path $Root 'baseline0/baseline-receipt.bin'
    $baseline=Read-Json (Join-Path $Root 'baseline0/result.json')
    Assert-Condition ($baseline.candidate_actor_adam_calls -eq 0 -and $baseline.candidate_critic_adam_calls -eq 0) 'Baseline candidate has optimizer updates'
    Check-Retention $baseline 0 $candidate $receipt
    $training=Read-Checkpoint $script:Plan.ledger_path_absolute -VerifyPayload
    $reader=[IO.BinaryReader]::new([IO.MemoryStream]::new($training.Payload,$false))
    try {
        Assert-Condition ($reader.ReadUInt32() -eq 0x314c5050 -and $reader.ReadUInt32() -eq 1) 'Wrong native training ledger'
        Assert-Condition ($reader.ReadUInt64() -eq $script:ProtocolChecksum -and $reader.ReadUInt64() -eq [uint64]$script:Plan.source_identity_fnv1a64) 'Training ledger identity mismatch'
        Assert-Condition ($reader.ReadUInt32() -eq 2 -and $reader.ReadUInt64() -eq (Read-Checkpoint $candidate).Checksum) 'Training ledger is not waiting at initialized candidate0'
        for($i=0;$i -lt 5;++$i){Assert-Condition ($reader.ReadUInt64() -eq 0) 'Native training/optimizer work already occurred'}
        $nativeSeconds=$reader.ReadDouble();$nativeBytes=$reader.ReadUInt64()
        Assert-Condition ($nativeSeconds -ge 0 -and $nativeSeconds -le $script:Plan.max_wall_seconds -and $nativeBytes -le $script:Plan.max_evidence_bytes) 'Training resource ledger invalid'
        Assert-Condition ($reader.ReadUInt32() -eq 1 -and $reader.ReadUInt64() -eq 0 -and $reader.ReadUInt64() -eq 0 -and $reader.BaseStream.Position -eq $reader.BaseStream.Length) 'Training ledger accounting unresolved/nonzero'
    } finally {$reader.Dispose()}
    $evaluation=Read-Checkpoint $script:Plan.evaluation_ledger_path_absolute -VerifyPayload
    $reader=[IO.BinaryReader]::new([IO.MemoryStream]::new($evaluation.Payload,$false))
    try {
        Assert-Condition ($reader.ReadUInt32() -eq 0x314c4550 -and $reader.ReadUInt32() -eq 2) 'Wrong native evaluation ledger'
        foreach($expected in @($script:ParentChecksum,$script:ProtocolChecksum,[uint64]$script:Plan.retention_evaluator_source_fnv1a64)){Assert-Condition ($reader.ReadUInt64() -eq $expected) 'Evaluation ledger identity mismatch'}
        Assert-Condition ($reader.ReadUInt64() -eq 1 -and $reader.ReadUInt64() -eq $baseline.attempted_controls -and $reader.ReadUInt64() -eq $baseline.attempted_substeps) 'Evaluation ledger is not exactly the completed baseline slot'
        [void]$reader.ReadUInt64();[void]$reader.ReadDouble()
        Assert-Condition ($reader.ReadUInt32() -eq 0 -and $reader.ReadUInt32() -eq 0 -and $reader.BaseStream.Position -eq $reader.BaseStream.Length) 'Evaluation ledger active/closed or malformed'
    } finally {$reader.Dispose()}
    return @{candidate=$candidate;receipt=$receipt;previous_runner_result_sha256=(File-Hash $old);original_runner_sha256=(File-Hash $snapshot);recovery_runner_sha256=(File-Hash $script:RunnerPath);training_ledger_sha256=(File-Hash $script:Plan.ledger_path_absolute);evaluation_ledger_sha256=(File-Hash $script:Plan.evaluation_ledger_path_absolute);candidate_sha256=(File-Hash $candidate);receipt_sha256=(File-Hash $receipt)}
}
function Invoke-Stage([string]$Name,[string]$Executable,[string[]]$Arguments,[scriptblock]$Validate) {
    $stdout=Join-Path $script:LogRoot "$Name.stdout.txt"
    $stderr=Join-Path $script:LogRoot "$Name.stderr.txt"
    $started=[DateTime]::UtcNow
    $gap=if($null -eq $script:LastExit){0}else{($started-$script:LastExit).TotalMilliseconds}
    $info=[Diagnostics.ProcessStartInfo]::new()
    $info.FileName=$Executable;$info.UseShellExecute=$false;$info.CreateNoWindow=$true
    $info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    foreach($argument in $Arguments){$info.ArgumentList.Add($argument)}
    if($null -ne $script:Stub){$info.FileName=$env:ComSpec;$info.ArgumentList.Clear();foreach($argument in @('/d','/c','exit','0')){$info.ArgumentList.Add($argument)}}
    $watch=[Diagnostics.Stopwatch]::StartNew()
    $process=[Diagnostics.Process]::Start($info)
    $outTask=$process.StandardOutput.ReadToEndAsync();$errTask=$process.StandardError.ReadToEndAsync()
    $process.WaitForExit();$exitCode=$process.ExitCode;$ended=[DateTime]::UtcNow;$watch.Stop()
    $io=[Diagnostics.Stopwatch]::StartNew()
    [IO.File]::WriteAllText($stdout,$outTask.GetAwaiter().GetResult())
    [IO.File]::WriteAllText($stderr,$errTask.GetAwaiter().GetResult())
    if($null -ne $script:Stub){$exitCode=Write-StubStage $Name $Arguments $stdout}
    $io.Stop();$process.Dispose();$script:LastExit=$ended
    $record=[ordered]@{stage=$Name;executable=$Executable;arguments=$Arguments;started_utc=$started.ToString('o');ended_utc=$ended.ToString('o');exit_code=$exitCode;process_wall_ms=$watch.Elapsed.TotalMilliseconds;runner_stream_io_ms=$io.Elapsed.TotalMilliseconds;validation_ms=0;handoff_gap_ms=$gap;synthetic=($null -ne $script:Stub);native_result=$null;native_result_read_error=$null;error=$null}
    $validation=[Diagnostics.Stopwatch]::StartNew()
    try {
        if($Name.StartsWith('train-')) {
            $nativeResultPath=Join-Path (Argument $Arguments '--out') 'result.json'
            if(Test-Path -LiteralPath $nativeResultPath) {
                try {$record.native_result=Read-Json $nativeResultPath;$script:LastTrainingResult=$record.native_result}
                catch {$record.native_result_read_error=$_.Exception.Message}
            }
        }
        Assert-Condition ($exitCode -eq 0) "Native stage $Name failed with exit $exitCode; see $stderr"
        return (& $Validate $stdout)
    } catch {$record.error=$_.Exception.Message;throw}
    finally {
        $validation.Stop();$record.validation_ms=$validation.Elapsed.TotalMilliseconds
        $script:Records.Add($record)
        Write-Json (Join-Path $script:LogRoot "$Name.command.json") $record
    }
}
function Run-Exercise([string]$PlanPath,[string]$ExpectedHash,[int]$Target,[bool]$Execute,[bool]$RecoverBaseline=$false) {
    Assert-Condition ($ExpectedHash -match '^[a-fA-F0-9]{64}$' -and (File-Hash $PlanPath) -eq $ExpectedHash.ToLowerInvariant()) 'Exact reviewed protocol SHA256 required'
    $script:Plan=Read-Json $PlanPath;$plan=$script:Plan
    Assert-Condition ($plan.status -eq 'SEALED_BEFORE_CANDIDATE_INITIALIZATION') 'Protocol is not sealed for initialization'
    Assert-Condition ($plan.protocol_id -in @('pond-motor-ppo-v2-2026-11-23','pond-motor-ppo-v3-curriculum-2026-11-23','pond-motor-ppo-v3-course-2026-11-23','pond-motor-ppo-v4-repeated-2026-11-24','pond-motor-ppo-v5-half-second-2026-11-24')) 'Unsupported study ID'
    $halfSecond=$plan.protocol_id -eq 'pond-motor-ppo-v5-half-second-2026-11-24'
    if($halfSecond){
        Assert-Condition ($Target -eq 8 -and $plan.max_updates -eq 8 -and $plan.study_program -eq 'half-second-course-v1') 'Half-second exercise requires exactly one eight-update allocation'
        Assert-Condition ((File-Hash $script:RunnerPath) -eq $plan.runner_script_sha256) 'Half-second runner differs from sealed protocol'
    }
    Assert-Condition ($plan.max_updates -le 32 -and $Target -le $plan.max_updates -and $plan.worlds -eq 512 -and $plan.rollout_controls -eq 64) 'Requested exercise exceeds sealed plan/settings'
    foreach($pair in @(@('trainer_executable_path_absolute','trainer_executable_sha256'),@('evaluator_executable_path_absolute','evaluator_executable_sha256'),@('parent_policy_path_absolute','parent_policy_sha256'),@('cpu_gate_path_absolute','cpu_gate_sha256'),@('cuda_gate_path_absolute','cuda_gate_sha256'),@('long_reference_path_absolute','long_reference_sha256'),@('human_protocol_path_absolute','human_protocol_sha256'))){
        $path=[string](Field $plan $pair[0]);Assert-Condition ([IO.Path]::IsPathRooted($path) -and (File-Hash $path) -eq [string](Field $plan $pair[1])) "Sealed input changed: $path"
    }
    $modeArguments=@()
    $property=$plan.PSObject.Properties['trainer_opt_in_arguments']
    if($null -ne $property){$modeArguments=@($property.Value)}
    Assert-Condition ($modeArguments.Count%2 -eq 0) 'Opt-in arguments must be flag/program pairs'
    for($i=0;$i -lt $modeArguments.Count;$i+=2){
        Assert-Condition ($modeArguments[$i] -in @('--turn-curriculum','--turn-reward','--reward-program','--turn-reward-program','--course-reward','--turn-schedule') -and $modeArguments[$i+1] -match '^[a-z0-9-]+$') 'Opt-in arguments cannot replace core flags or budgets'
    }
    if($plan.protocol_id -ne 'pond-motor-ppo-v2-2026-11-23'){Assert-Condition ($modeArguments.Count -gt 0) 'Future study requires its exact sealed native opt-in arguments'}
    $root=[IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($plan.ledger_path_absolute))
    Assert-Condition ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($plan.evaluation_ledger_path_absolute)) -eq $root) 'Training/evaluation ledgers must share the sealed study root'
    if(-not $Execute){return @{mode='inspect_only';protocol_id=$plan.protocol_id;stop_after_update=$Target;study_root=$root;physical_controls=0;optimizer_updates=0}}
    $script:ParentChecksum=(Read-Checkpoint $plan.parent_policy_path_absolute).Checksum
    $script:ProtocolChecksum=Fnv64 ([IO.File]::ReadAllBytes($PlanPath))
    $recovery=$null
    if($RecoverBaseline){$recovery=Check-BaselineRecovery $root $ExpectedHash}
    else{foreach($path in @($plan.ledger_path_absolute,$plan.evaluation_ledger_path_absolute,(Join-Path $root 'runner'),(Join-Path $root 'init'),(Join-Path $root 'baseline0'))){Assert-Condition (-not (Test-Path -LiteralPath $path)) "Existing allocation/output; no restart: $path"}}
    foreach($update in @(8,16,24,32)){Assert-Condition (-not (Test-Path -LiteralPath (Join-Path $root "eval$update")) -and -not (Test-Path -LiteralPath (Join-Path $root "updates-$($update-7)-$update"))) 'Existing checkpoint output; no overwrite'}
    $script:LogRoot=Join-Path $root $(if($RecoverBaseline){'runner-recovered-baseline'}else{'runner'})
    Assert-Condition (-not (Test-Path -LiteralPath $script:LogRoot)) 'Runner evidence already exists; no automatic recovery retry'
    New-Item -ItemType Directory -Path $script:LogRoot -ErrorAction Stop | Out-Null
    $script:Records=[Collections.Generic.List[object]]::new();$script:LastExit=$null;$script:LastTrainingResult=$null
    $clock=[Diagnostics.Stopwatch]::StartNew();$lastUpdate=0;$candidate=$null;$receipt=$null;$state='STOPPED';$failure=$null
    Write-Json (Join-Path $script:LogRoot 'started.json') @{protocol_sha256=$ExpectedHash;stop_after_update=$Target;started_utc=[DateTime]::UtcNow.ToString('o');synthetic=($null -ne $script:Stub);verified_baseline_recovery=$recovery}
    try {
        $trainer=Invoke-Stage 'describe-trainer' $plan.trainer_executable_path_absolute (@('--describe')+$modeArguments) {param($path) Read-Json $path}
        Assert-Condition ([string]$trainer.source_identity_fnv1a64 -eq [string]$plan.source_identity_fnv1a64 -and $trainer.qualified_physics_source_sha256 -eq $plan.qualified_physics_source_sha256) 'Trainer/source/physics binding mismatch'
        $nativeId=$trainer.PSObject.Properties['protocol_id'];if($null -ne $nativeId){Assert-Condition ($nativeId.Value -eq $plan.protocol_id) 'Trainer protocol mismatch'}else{Assert-Condition ($plan.protocol_id -eq 'pond-motor-ppo-v2-2026-11-23') 'Future trainer must identify its protocol'}
        $evaluator=Invoke-Stage 'describe-evaluator' $plan.evaluator_executable_path_absolute @('--describe','--protocol',$PlanPath) {param($path) Read-Json $path}
        Assert-Condition ([string]$evaluator.eval_source_identity_fnv1a64 -eq [string]$plan.retention_evaluator_source_fnv1a64 -and $evaluator.protocol_id -eq $plan.protocol_id) 'Evaluator/protocol binding mismatch'
        $common=@('--policy',$plan.parent_policy_path_absolute,'--protocol',$PlanPath,'--cpu-gate',$plan.cpu_gate_path_absolute,'--cuda-gate',$plan.cuda_gate_path_absolute,'--evaluator-source',[string]$plan.retention_evaluator_source_fnv1a64)+$modeArguments
        if($RecoverBaseline){$candidate=$recovery.candidate;$receipt=$recovery.receipt}
        else {
            $initial=Join-Path $root 'init'
            $candidate=Invoke-Stage 'initialize' $plan.trainer_executable_path_absolute (@('--initialize-only','--out',$initial,'--to-update','8')+$common) {
                param($ignored) $value=Read-Json (Join-Path $initial 'result.json');Assert-Condition ($value.initialized -and $value.physical_controls -eq 0 -and $value.optimizer_calls -eq 0) 'Initialization did not remain zero-step';Join-Path $initial 'candidate-0.bin'
            }
        }
        foreach($update in @(0,8,16,24,32)) {
            if($update -gt $Target){break}
            if($RecoverBaseline -and $update -eq 0){continue}
            if($update -gt 0) {
                $chunk=Join-Path $root "updates-$($update-7)-$update"
                $candidate=Invoke-Stage "train-$update" $plan.trainer_executable_path_absolute (@('--run','--out',$chunk,'--to-update',[string]$update,'--resume',$candidate,'--retention-receipt',$receipt)+$common) {
                    param($ignored) $value=Read-Json (Join-Path $chunk 'result.json')
                    Assert-Condition ($value.passed -and $value.physical_accounting_readback_complete -and $value.completed_updates -eq $update -and $value.attempted_controls -eq $update*32768 -and $value.attempted_substeps -eq $update*327680 -and $value.actor_Adam_calls -eq $update*32 -and $value.critic_Adam_calls -eq $update*32) 'Training chunk incomplete or accounting invalid'
                    Join-Path $chunk "candidate-$update.bin"
                }
                $lastUpdate=$update
            }
            $name=if($update -eq 0){'baseline0'}else{"eval$update"};$evaluation=Join-Path $root $name
            $evalArguments=@('--run','--parent',$plan.parent_policy_path_absolute,'--candidate',$candidate,'--protocol',$PlanPath,'--out',$evaluation,'--ledger',$plan.evaluation_ledger_path_absolute,'--long-reference',$plan.long_reference_path_absolute)
            if($update -eq 0){
                $evalArguments+='--baseline'
                if($halfSecond){$evalArguments+='--reuse-verified-baseline'}
            }
            $receipt=Invoke-Stage $name $plan.evaluator_executable_path_absolute $evalArguments {
                param($ignored) $path=Join-Path $evaluation $(if($update -eq 0){'baseline-receipt.bin'}else{'retention-receipt.bin'})
                Check-Retention (Read-Json (Join-Path $evaluation 'result.json')) $update $candidate $path
                return $path
            }
            Assert-Condition ((File-Hash $PlanPath) -eq $ExpectedHash.ToLowerInvariant() -and (File-Hash $plan.parent_policy_path_absolute) -eq $plan.parent_policy_sha256) 'Immutable protocol/parent changed during run'
        }
        $state=if($Target -eq $plan.max_updates){'COMPLETED_ALLOCATION'}else{'INTENTIONAL_CHECKPOINT_FINISH'}
    } catch {$failure=$_.Exception.Message;throw}
    finally {
        $clock.Stop()
        $processTotal=0.;$ioTotal=0.;$validationTotal=0.;$gapTotal=0.
        foreach($record in $script:Records){$processTotal+=$record['process_wall_ms'];$ioTotal+=$record['runner_stream_io_ms'];$validationTotal+=$record['validation_ms'];$gapTotal+=$record['handoff_gap_ms']}
        $result=[ordered]@{status=$state;protocol_id=$plan.protocol_id;stop_after_update=$Target;last_passed_training_chunk_update=$lastUpdate;last_candidate=$candidate;last_receipt=$receipt;native_training_accounting=$script:LastTrainingResult;error=$failure;total_host_wall_ms=$clock.Elapsed.TotalMilliseconds;native_process_wall_ms=$processTotal;runner_stream_io_ms=$ioTotal;validation_ms=$validationTotal;interstage_gaps_ms=$gapTotal;timing_note='Handoff gaps include validation, runner I/O and bookkeeping; overlapping categories must not be added. Native wall includes launch/wait; stream drainage and log writes are runner I/O. These are not isolated GPU compute times.';synthetic=($null -ne $script:Stub);commands=$script:Records.ToArray()}
        Write-Json (Join-Path $script:LogRoot 'result.json') $result
    }
    return $result
}

# Synthetic tests exercise this same orchestration and real child-process launch,
# while replacing the trainer/evaluator with cmd.exe exit0 and fixture outputs.
function Write-SyntheticCheckpoint([string]$Path,[byte[]]$Payload) {
    $memory=[IO.MemoryStream]::new();$writer=[IO.BinaryWriter]::new($memory)
    $writer.Write([uint64]0x31504B434C524443);$writer.Write([uint64]$Payload.Length);$writer.Write([uint64](Fnv64 $Payload));$writer.Write($Payload);$writer.Flush()
    [IO.File]::WriteAllBytes($Path,$memory.ToArray());$writer.Dispose()
}
function Argument([string[]]$Values,[string]$Name) {$index=[Array]::IndexOf($Values,$Name);Assert-Condition ($index -ge 0 -and $index+1 -lt $Values.Count) "Missing stub argument $Name";return $Values[$index+1]}
function Write-StubStage([string]$Name,[string[]]$Arguments,[string]$Stdout) {
    $script:Stub.Calls.Add($Name);$plan=$script:Plan
    if($Name -eq 'describe-trainer'){Write-Json $Stdout @{protocol_id=$plan.protocol_id;source_identity_fnv1a64=$plan.source_identity_fnv1a64;qualified_physics_source_sha256=$plan.qualified_physics_source_sha256};return 0}
    if($Name -eq 'describe-evaluator'){Write-Json $Stdout @{protocol_id=$plan.protocol_id;eval_source_identity_fnv1a64=$plan.retention_evaluator_source_fnv1a64};return 0}
    $out=Argument $Arguments '--out';New-Item -ItemType Directory -Path $out | Out-Null
    if($Name -eq 'initialize'){Write-SyntheticCheckpoint (Join-Path $out 'candidate-0.bin') ([byte[]]@(0));Write-Json (Join-Path $out 'result.json') @{initialized=$true;physical_controls=0;optimizer_calls=0};return 0}
    if($Name.StartsWith('train-')){
        $update=[int](Argument $Arguments '--to-update');$passed=-not ($script:Stub.Scenario -eq 'partial-training-failure' -and $update -eq 16)
        if(-not $passed){$update=12}
        Write-SyntheticCheckpoint (Join-Path $out "candidate-$update.bin") ([byte[]]@($update))
        Write-Json (Join-Path $out 'result.json') @{passed=$passed;physical_accounting_readback_complete=$true;completed_updates=$update;attempted_controls=$update*32768;attempted_substeps=$update*327680;actor_Adam_calls=$update*32;critic_Adam_calls=$update*32}
        return $(if($passed){0}else{2})
    }
    $update=if($Name -eq 'baseline0'){0}else{[int]$Name.Substring(4)}
    $passed=-not (($script:Stub.Scenario -eq 'baseline-failure' -and $update -eq 0) -or ($script:Stub.Scenario -eq 'retention-failure' -and $update -eq 8))
    $candidate=(Read-Checkpoint (Argument $Arguments '--candidate')).Checksum
    Write-Json (Join-Path $out 'result.json') @{passed=$passed;assessment_completed=$true;training_admission_qualified=$passed;retention_passed=$passed;preflight_only=$false;hard_failure=$false;update=$update;baseline_only=($update -eq 0);retention_cases_passed=20;retention_cases_completed=20;rest_cases_passed=14;short_swim_cases_passed=4;long_swim_cases_passed=2;retained_parent_long_window_passes=28;turn_cases_attempted=16;retention_controls=30400;retention_substeps=304000;checkpoint_checksum=[string]$candidate;eval_source_identity=$plan.retention_evaluator_source_fnv1a64}
    if(-not $passed){return 2}
    $memory=[IO.MemoryStream]::new();$writer=[IO.BinaryWriter]::new($memory)
    foreach($value in @([uint32]0x31525044,3,4,2)){$writer.Write([uint32]$value)}
    if($script:Stub.Scenario -eq 'bad-receipt'){$candidate=0}
    foreach($value in @($candidate,$script:ParentChecksum,$script:ProtocolChecksum,[uint64]$plan.retention_evaluator_source_fnv1a64)){$writer.Write([uint64]$value)}
    foreach($value in @(20,20,20,0)){$writer.Write([uint32]$value)}
    $writer.Write([uint64]30400);$writer.Write([uint64]304000);$writer.Write([uint32]1);$writer.Write([uint32]($update -eq 0));$writer.Flush()
    $inner=Fnv64 $memory.ToArray();if($script:Stub.Scenario -eq 'bad-inner-seal'){$inner=0};$writer.Write([uint64]$inner);$writer.Flush()
    Write-SyntheticCheckpoint (Join-Path $out $(if($update -eq 0){'baseline-receipt.bin'}else{'retention-receipt.bin'})) $memory.ToArray();$writer.Dispose();return 0
}
function Test-Runner {
    $temporary=Join-Path ([IO.Path]::GetTempPath()) ("pond-runner-tests-"+[guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temporary | Out-Null
    $tests=0
    foreach($scenario in @('success','short-finish','baseline-failure','bad-receipt','bad-inner-seal','retention-failure','partial-training-failure')) {
        $root=Join-Path $temporary $scenario;New-Item -ItemType Directory -Path $root | Out-Null
        $plan=[ordered]@{status='SEALED_BEFORE_CANDIDATE_INITIALIZATION';protocol_id='pond-motor-ppo-v4-repeated-2026-11-24';max_updates=32;worlds=512;rollout_controls=64;source_identity_fnv1a64='101';retention_evaluator_source_fnv1a64='202';qualified_physics_source_sha256=('a'*64);ledger_path_absolute=(Join-Path $root 'training-ledger.bin');evaluation_ledger_path_absolute=(Join-Path $root 'evaluation-ledger.bin');trainer_opt_in_arguments=@('--turn-schedule','repeated-headings-v1')}
        foreach($entry in @(@('trainer_executable','trainer'),@('evaluator_executable','evaluator'),@('parent_policy','parent'),@('cpu_gate','cpu'),@('cuda_gate','cuda'),@('long_reference','reference'),@('human_protocol','human'))){
            $path=Join-Path $root ($entry[1]+'.fixture');if($entry[0] -eq 'parent_policy'){Write-SyntheticCheckpoint $path ([byte[]]@(1,2,3))}else{[IO.File]::WriteAllText($path,'SYNTHETIC ONLY')}
            $plan[$entry[0]+'_path_absolute']=$path;$plan[$entry[0]+'_sha256']=File-Hash $path
        }
        $path=Join-Path $root 'protocol.json';Write-Json $path $plan;$hash=File-Hash $path
        $script:Stub=@{Scenario=$scenario;Calls=[Collections.Generic.List[string]]::new()}
        $failed=$false;$target=if($scenario -eq 'short-finish'){8}else{32}
        $caughtFailure=$null
        try{$result=Run-Exercise $path $hash $target $true}catch{$failed=$true;$caughtFailure=$_.Exception.Message}
        if($scenario -in @('success','short-finish')){Assert-Condition (-not $failed) "Synthetic $scenario failed: $caughtFailure";$expected=if($scenario -eq 'success'){'COMPLETED_ALLOCATION'}else{'INTENTIONAL_CHECKPOINT_FINISH'};Assert-Condition ($result.status -eq $expected) 'Wrong intentional/completed label'}
        else{Assert-Condition $failed "Synthetic $scenario unexpectedly admitted training"}
        if($scenario -in @('baseline-failure','bad-receipt','bad-inner-seal')){Assert-Condition (@($script:Stub.Calls|Where-Object {$_ -like 'train-*'}).Count -eq 0) 'Optimizer stage released without genuine baseline'}
        if($scenario -in @('retention-failure','short-finish')){Assert-Condition (-not $script:Stub.Calls.Contains('train-16')) 'Runner exceeded failed/intentional checkpoint'}
        if($scenario -eq 'partial-training-failure'){$record=Read-Json (Join-Path $root 'runner/result.json');Assert-Condition ($record.last_passed_training_chunk_update -eq 8 -and $record.native_training_accounting.completed_updates -eq 12 -and -not $script:Stub.Calls.Contains('eval16')) 'Partial native accounting was hidden or execution continued'}
        if($scenario -eq 'success'){$before=$script:Stub.Calls.Count;$rejected=$false;try{Run-Exercise $path $hash 32 $true|Out-Null}catch{$rejected=$true};Assert-Condition ($rejected -and $script:Stub.Calls.Count -eq $before) 'Existing study silently restarted'}
        ++$tests
    }
    $script:Stub=$null
    return @{passed=$true;scenarios=$tests;evidence_directory=$temporary;physical_controls=0;optimizer_updates=0;note='Synthetic fixture outputs plus real cmd.exe exit0 process launches; no native trainer/evaluator execution.'}
}
if($SelfTest){
    $result=Test-Runner
    if(-not [string]::IsNullOrWhiteSpace($Protocol)) {
        Assert-Condition ((File-Hash $Protocol) -eq $ProtocolSha256) 'Actual native fixture protocol mismatch'
        $script:Plan=Read-Json $Protocol;$script:ParentChecksum=(Read-Checkpoint $script:Plan.parent_policy_path_absolute).Checksum;$script:ProtocolChecksum=Fnv64 ([IO.File]::ReadAllBytes($Protocol))
        $root=[IO.Path]::GetDirectoryName($script:Plan.ledger_path_absolute)
        Check-Retention (Read-Json (Join-Path $root 'baseline0/result.json')) 0 (Join-Path $root 'init/candidate-0.bin') (Join-Path $root 'baseline0/baseline-receipt.bin')
        $result['actual_native_schema3_baseline_receipt_verified']=$true
        $ledger=Read-Checkpoint $script:Plan.ledger_path_absolute -VerifyPayload
        if([BitConverter]::ToUInt32($ledger.Payload,24) -eq 2) {
            $result['actual_fixture_proof']=Check-BaselineRecovery $root $ProtocolSha256
        } else {
            $rejected=$false;try{Check-BaselineRecovery $root $ProtocolSha256|Out-Null}catch{$rejected=$true}
            Assert-Condition $rejected 'Completed/active/trained ledger incorrectly allowed baseline recovery'
            $result['trained_ledger_baseline_recovery_correctly_rejected']=$true
        }
    }
    $result | ConvertTo-Json -Depth 5;return
}
Assert-Condition (-not [string]::IsNullOrWhiteSpace($Protocol)) 'Provide -Protocol and its reviewed -ProtocolSha256; use -Run only for an authorized sealed allocation'
Run-Exercise ([IO.Path]::GetFullPath($Protocol)) $ProtocolSha256 $StopAfterUpdate $Run.IsPresent $ResumeVerifiedBaseline.IsPresent | ConvertTo-Json -Depth 12
