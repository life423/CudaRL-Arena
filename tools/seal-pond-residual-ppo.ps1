#Requires -Version 7.0
param([ValidateSet('yaw','all')][string]$Axes='yaw')
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
function Local($name){Join-Path $repo $name}
function Digest($file){(Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant()}
function Need($ok,$why){if(!$ok){throw $why}}
function Json($file,$value){[IO.File]::WriteAllText($file,($value|ConvertTo-Json -Depth 12))}
$root=Local 'build/turn38-residual'
Need ($Axes -eq 'all' -or !(Test-Path -LiteralPath $root) -or @(Get-ChildItem -LiteralPath $root -Recurse -File).Count -eq 0) 'This preparation or allocation already contains evidence'
$trainer=Local 'build/residual-ppo/native/Release/pond_residual_ppo.exe'
$evaluator=Local 'build/pce-residual/native/Release/pond_candidate_check.exe'
$trainerBuild='build/residual-ppo/native'
$evaluatorBuild='build/pce-residual/native'
$sourceDirectory='tools/pond-residual-ppo'
if($Axes -eq 'all'){
    $root=Local 'build/turn39-residual-four'
    Need (!(Test-Path -LiteralPath $root)) 'This four-axis preparation or allocation already exists'
    $trainerBuild='build/residual-four-ppo/native'
    $evaluatorBuild='build/pce-residual-four/native'
    $trainer=Local "$trainerBuild/Release/pond_residual_ppo.exe"
    $evaluator=Local "$evaluatorBuild/Release/pond_candidate_check.exe"
    $sourceDirectory='tools/pond-residual-four-ppo'
    $previous=Local 'build/turn38-residual/runner/result.json'
    Need ((Get-Content $previous -Raw|ConvertFrom-Json).status -eq 'COMPLETED_ALLOCATION') 'Yaw experiment must complete before the comparison'
    foreach($update in @(8,16,24,32)){
        $file=Local "build/turn38-residual/eval$update/cases.csv"
        $turns=@(Import-Csv $file|Where-Object {$_.suite -eq 'turn' -and $_.passed -eq '1'})
        $left=@($turns|Where-Object {[double]$_.turn_degrees -gt 0}).Count
        $right=@($turns|Where-Object {[double]$_.turn_degrees -lt 0}).Count
        Need (!($left -ge 2 -and $right -ge 2)) 'Yaw steering already meets the declared improvement threshold'
    }
}
$desc=& $trainer --describe|ConvertFrom-Json
Need ($LASTEXITCODE -eq 0 -and $desc.protocol_id -in @('pond-motor-residual-ppo-v1-2026-11-24','pond-motor-residual-four-ppo-v1-2026-11-24')) 'Wrong residual trainer'
$prior=Get-Content (Local 'build/turn36/protocol.json') -Raw|ConvertFrom-Json -AsHashtable
$plan=[ordered]@{
    protocol_id=$desc.protocol_id
    status='SEALED_BEFORE_RESIDUAL_INITIALIZATION'
    max_updates=32
    worlds=512
    rollout_controls=64
    epochs=4
    minibatch=4096
    actor_learning_rate=0.0003
    critic_learning_rate=0.001
    max_training_controls=1048576
    max_training_substeps=10485760
    max_actor_Adam_calls=1024
    max_critic_Adam_calls=1024
    max_evaluation_controls=184000
    max_evaluation_substeps=1840000
    max_evaluation_assessments=5
    max_wall_seconds=1200
    max_evidence_bytes=4294967296
    source_sha256=$desc.source_sha256
    source_identity_fnv1a64=$desc.source_identity_fnv1a64
    qualified_physics_source_sha256=$desc.qualified_physics_source_sha256
    trainer_path_absolute=$trainer
    trainer_sha256=(Digest $trainer)
    evaluator_path_absolute=$evaluator
    evaluator_sha256=(Digest $evaluator)
    ledger_path_absolute=(Join-Path $root 'training-ledger.bin')
    evaluation_ledger_path_absolute=(Join-Path $root 'evaluation-ledger.bin')
    actor_inputs=42
    hidden_neurons=16
    actor_outputs=$(if($Axes -eq 'all'){4}else{2})
    trainable_actor_parameters=$desc.residual_parameters
    original_motor_frozen=$true
    residual_active_only_for_turn_requests=$true
    turning_check_every_updates=8
    minimum_turn_passes_for_full_retention=4
    minimum_left_passes_for_full_retention=2
    minimum_right_passes_for_full_retention=2
    full_retention_only_after_meaningful_turn_improvement=$true
    human_protocol_path_absolute=(Local 'docs/POND_RESIDUAL_PPO_PROTOCOL_2026-11-24.md')
    human_protocol_sha256=(Digest (Local 'docs/POND_RESIDUAL_PPO_PROTOCOL_2026-11-24.md'))
    runner_sha256=(Digest (Local 'tools/run-pond-residual-ppo.ps1'))
    sealed_utc=[DateTimeOffset]::UtcNow.ToString('o')
}
foreach($name in @('parent_policy','cpu_gate','cuda_gate','long_reference','mirror_result','previous_protocol','previous_candidate_zero','previous_baseline_receipt','previous_baseline_result')){
    $path=$prior[$name+'_path_absolute']
    Need ((Digest $path) -eq $prior[$name+'_sha256']) "Qualified input changed: $name"
    $plan[$name+'_path_absolute']=$path
    $plan[$name+'_sha256']=$prior[$name+'_sha256']
}
Need ((Get-Content $plan.mirror_result_path_absolute -Raw|ConvertFrom-Json).physical_symmetry_passed) 'Physics reflection failed'
$plan['critic_source_candidate_path_absolute']=Local 'build/turn36/updates-1-8/candidate-8.bin'
$plan['critic_source_candidate_sha256']=Digest $plan.critic_source_candidate_path_absolute
$plan['critic_source_protocol_path_absolute']=Local 'build/turn36/protocol.json'
$plan['critic_source_protocol_sha256']=Digest $plan.critic_source_protocol_path_absolute
$scratch=Local "$evaluatorBuild/../describe-plan.json"
Json $scratch $plan
$evaluation=& $evaluator --describe --protocol $scratch|ConvertFrom-Json
Need ($LASTEXITCODE -eq 0 -and $evaluation.protocol_id -eq $plan.protocol_id -and $evaluation.retention_cases -eq 20 -and $evaluation.turn_cases -eq 16) 'Wrong residual assessment'
$plan['retention_evaluator_source_fnv1a64']=$evaluation.eval_source_identity_fnv1a64
$plan['retention_evaluator_source_sha256']=$evaluation.eval_source_sha256
$snapshot=Join-Path $root 'source-snapshot'
New-Item -ItemType Directory -Path $snapshot -Force|Out-Null
$bindings=[ordered]@{}
foreach($manifest in @("$trainerBuild/pond-ppo-source-manifest.txt","$evaluatorBuild/source-manifest.txt")){
    foreach($line in Get-Content (Local $manifest)){
        $file,$expected=$line.Split(' ')
        Need ((Digest (Local $file)) -eq $expected) "Build source differs: $file"
        $bindings[$file]=$expected
    }
}
foreach($file in @("$sourceDirectory/train.cu","$sourceDirectory/contract.h",'tools/pond-residual-ppo/evaluator.h','tools/pond-residual-ppo/staged_schedule.h',"$sourceDirectory/CMakeLists.txt",'tools/run-pond-residual-ppo.ps1','tools/seal-pond-residual-ppo.ps1','docs/POND_RESIDUAL_PPO_PROTOCOL_2026-11-24.md')){
    $bindings[$file]=Digest (Local $file)
}
foreach($file in $bindings.Keys){
    $copy=Join-Path $snapshot $file
    New-Item -ItemType Directory -Path (Split-Path $copy) -Force|Out-Null
    Copy-Item -LiteralPath (Local $file) -Destination $copy
    Need ((Digest $copy) -eq $bindings[$file]) "Snapshot differs: $file"
}
Json (Join-Path $snapshot 'bindings.json') $bindings
Json (Join-Path $root 'trainer-descriptor.json') $desc
Json (Join-Path $root 'evaluator-descriptor.json') $evaluation
$protocol=Join-Path $root 'protocol.json'
Json $protocol $plan
$hash=Digest $protocol
[IO.File]::WriteAllText((Join-Path $root 'protocol.sha256'),$hash)
$hash
