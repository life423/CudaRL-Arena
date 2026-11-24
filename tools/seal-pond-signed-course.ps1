#Requires -Version 7.0
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
function Local([string]$name){Join-Path $repo $name}
function Digest([string]$file){(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()}
function Json([string]$path,$value){[IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 12)+[Environment]::NewLine)}
function Check([bool]$ok,[string]$why){if(!$ok){throw $why}}

$study=Local 'build/turn36'
Check (!(Test-Path -LiteralPath $study)) 'A preparation or allocation already exists'
$gate=Local 'assets/research-evidence/mirrored-turn-2026-11-24/result.json'
$mirror=Get-Content -LiteralPath $gate -Raw | ConvertFrom-Json
Check ($mirror.physical_symmetry_passed -is [bool] -and $mirror.physical_symmetry_passed) 'Physical symmetry gate failed'
$old=Local 'build/turn35/protocol.json'
Check ((Digest $old) -eq (Get-Content (Local 'assets/research-evidence/pond-turning-v5-half-second/protocol.sha256') -Raw).Trim()) 'V5 reference protocol differs'
$plan=Get-Content $old -Raw|ConvertFrom-Json -AsHashtable
$trainer=Local 'build/pt36/native/Release/pond_ppo.exe'
$evalExe=Local 'build/pce36/native/Release/pond_candidate_check.exe'
$mode='signed-half-second-v1'
$desc=& $trainer --describe --turn-schedule $mode | ConvertFrom-Json
Check ($LASTEXITCODE -eq 0 -and $desc.protocol_id -eq 'pond-motor-ppo-v6-signed-course-2026-11-24' -and $desc.observation_schema -eq 6 -and $desc.max_updates -eq 8 -and $desc.physical_controls -eq 0) 'Wrong trainer or schema'
Check ($desc.qualified_physics_source_sha256 -eq $plan.qualified_physics_source_sha256) 'Qualified physics changed'
$human=Local 'docs/POND_SIGNED_COURSE_EXPERIMENT.md'
$runner=Local 'tools/run-pond-turning.ps1'
$plan.protocol_id=$desc.protocol_id
$plan.study_program=$mode
$plan.study_program_id=$desc.study_program_id
$plan.trainer_opt_in_arguments=@('--turn-schedule',$mode)
$plan.turn_schedule_program='half-second-course-v1'
$plan.turn_reward_program='half-second-course-v1'
$plan.learning_changes='One appended signed preceding-half-second COM course-error actor input with zero-initialized weights'
$plan.baseline_reuse='Verify original v4 physical receipt, original 41 input coefficients in each row, new zero coefficients, gain and fixed scoring before any PPO'
$plan.observation_schema=$desc.observation_schema
$plan.task_version=$desc.task_version
$plan.task_stride=$desc.task_stride
$plan.base_task_stride=$desc.base_task_stride
$plan.schedule_stride=$desc.schedule_stride
$plan.course_history_stride=$desc.course_history_stride
$plan.transition_trace_stride=$desc.trace_stride
$plan.source_identity_fnv1a64=$desc.source_identity_fnv1a64
$plan.source_sha256=$desc.source_sha256
$plan.trainer_executable_path_absolute=$trainer
$plan.trainer_executable_sha256=Digest $trainer
$plan.evaluator_executable_path_absolute=$evalExe
$plan.evaluator_executable_sha256=Digest $evalExe
$plan.human_protocol_path_absolute=$human
$plan.human_protocol_sha256=Digest $human
$plan.runner_script_path_absolute=$runner
$plan.runner_script_sha256=Digest $runner
$plan.workflow_wrapper_sha256=Digest (Local 'tools/run-pond-gentle-turns.ps1')
$plan['mirror_result_path_absolute']=$gate
$plan['mirror_result_sha256']=Digest $gate
$plan['comparison_protocol_path_absolute']=$old
$plan['comparison_protocol_sha256']=Digest $old
$plan.transition_batch_version=3
$plan.new_physics_prerequisite_controls=$mirror.cpu_attempted_controls+$mirror.cuda_attempted_controls
$plan['new_physics_prerequisite_substeps']=$mirror.cpu_attempted_substeps+$mirror.cuda_attempted_substeps
$plan.ledger_path_absolute=Join-Path $study 'training-ledger.bin'
$plan.evaluation_ledger_path_absolute=Join-Path $study 'evaluation-ledger.bin'
foreach($name in @('cpu_gate','cuda_gate','parent_policy','physical_launch_proof','long_reference','previous_protocol','previous_candidate_zero','previous_baseline_receipt','previous_baseline_result')){
    Check ((Digest $plan[$name+'_path_absolute']) -eq $plan[$name+'_sha256']) "Sealed evidence changed: $name"
}
$all=[ordered]@{}
foreach($manifest in @('build/pt36/native/pond-ppo-source-manifest.txt','build/pce36/native/source-manifest.txt')){
    foreach($line in Get-Content (Local $manifest)){
        $relative,$expected=$line.Split(' ')
        Check ((Digest (Local $relative)) -eq $expected) "Build source differs: $relative"
        $all[$relative]=$expected
    }
}
foreach($relative in @('tools/seal-pond-signed-course.ps1','tools/run-pond-turning.ps1','tools/run-pond-gentle-turns.ps1','tools/publish-study-status.ps1','docs/POND_SIGNED_COURSE_EXPERIMENT.md','docs/MIRRORED_TURN_DIAGNOSTIC_2026-11-24.md')){
    $all[$relative]=Digest (Local $relative)
}
$descriptorInput=Local 'build/pce36/describe-signed-course.json'
Json $descriptorInput $plan
$evalDesc=& $evalExe --describe --protocol $descriptorInput | ConvertFrom-Json
Check ($LASTEXITCODE -eq 0 -and $evalDesc.protocol_id -eq $plan.protocol_id -and $evalDesc.task_stride -eq $desc.task_stride -and $evalDesc.retention_cases -eq 20 -and $evalDesc.turn_cases -eq 16) 'Unmatched evaluator'
$plan.retention_evaluator_source_sha256=$evalDesc.eval_source_sha256
$plan.retention_evaluator_source_fnv1a64=$evalDesc.eval_source_identity_fnv1a64
$snapshot=Join-Path $study 'source-snapshot'
New-Item -ItemType Directory -Path $snapshot -Force|Out-Null
foreach($relative in $all.Keys){
    $copy=Join-Path $snapshot $relative
    New-Item -ItemType Directory -Path (Split-Path $copy) -Force|Out-Null
    Copy-Item -LiteralPath (Local $relative) -Destination $copy
    Check ((Digest $copy) -eq $all[$relative]) "Snapshot mismatch: $relative"
}
$plan.source_snapshot_path_absolute=$snapshot
$plan.source_bindings_verified=$all.Count
$plan.source_snapshot_files=$all.Count
$plan.sealed_utc=[DateTimeOffset]::UtcNow.ToString('o')
$plan.base_commit=git -C $repo rev-parse HEAD
Json (Join-Path $snapshot 'bindings.json') $all
Json (Join-Path $study 'trainer-descriptor.json') $desc
Json (Join-Path $study 'evaluator-descriptor.json') $evalDesc
$file=Join-Path $study 'protocol.json'
Json $file $plan
$hash=Digest $file
[IO.File]::WriteAllText((Join-Path $study 'protocol.sha256'),$hash)
$hash
