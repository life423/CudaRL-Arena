$ErrorActionPreference='Stop'
$session=Split-Path -Parent $PSScriptRoot
$repository=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $session))
Add-Type -Path (Join-Path $PSScriptRoot 'Angle90SavedAudit.cs')
function Check($Condition,[string]$Message){if(!$Condition){throw $Message}}
$manifest=Get-Content (Join-Path $session 'manifest.json') -Raw|ConvertFrom-Json
$result=Get-Content (Join-Path $session 'training/result.json') -Raw|ConvertFrom-Json
Check ($result.stage -eq 'complete' -and $result.updates_completed -eq 128) 'Run incomplete'
$bad=@($manifest.sources+$manifest.inputs|Where-Object{(Get-FileHash -LiteralPath (Join-Path $repository $_.path)).Hash -ne $_.sha256})
Check ($bad.Count -eq 0) 'Bound source/input hash mismatch'
$basePath=Join-Path $repository 'build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin'
$base=[Angle90SavedAudit]::ReadActor($basePath)
$transferred=[float[]]$base.Clone();$transferred[0]=[float]($base[0]*[float]18)
$actor0Payload=[Angle90SavedAudit]::Payload((Join-Path $session 'training/actor-0.bin'))
$initialWeightsChecksum=[Angle90SavedAudit]::Hash([byte[]]$actor0Payload[8..($actor0Payload.Length-1)])
$sourceOffsetChecksum=[Angle90SavedAudit]::Hash([Angle90SavedAudit]::Payload($basePath))
$rows=@();$task=$null;$initial=$null;$final=$null
for($update=0;$update -le 128;$update++){
    $state=[Angle90SavedAudit]::ReadSession((Join-Path $session "training/checkpoints/session-$update.bin"))
    if($update -eq 0){$task=$state.task;$initial=$state}
    Check ($state.task -eq $task -and $state.update -eq $update) "Task/update at $update"
    Check ($state.worlds -eq 1024 -and $state.counters -eq 1024) "World/counter dimensions at $update"
    Check ($state.actor.Length -eq 32 -and $state.critic.Length -eq 1217) "Network dimensions at $update"
    Check ($state.config[0] -eq 1024 -and $state.config[1] -eq 64 -and $state.config[2] -eq 31 -and $state.config[3] -eq 36) "PPO dimensions at $update"
    Check ($state.config[4] -eq 4096 -and $state.config[5] -eq 4 -and $state.config[6] -eq 4 -and $state.config[7] -eq 128) "PPO schedule at $update"
    Check ($state.rng -eq [uint64]($update*64) -and $state.actions -eq [uint64]($update*1024*64)) "Sampling/RNG at $update"
    Check ($state.initialWeights -eq $initialWeightsChecksum -and $state.offset -eq $sourceOffsetChecksum) "Initial transfer identity at $update"
    for($i=0;$i -lt 4;$i++){Check ($state.initialCoefficients[$i] -eq $transferred[$i]) "Initial coefficient $i at $update"}
    for($i=4;$i -lt 32;$i++){Check ($state.actor[$i] -eq 0 -and $state.actorM[$i] -eq 0 -and $state.actorV[$i] -eq 0) "Inactive coefficient/moment $i at $update"}
    $export=[Angle90SavedAudit]::ReadActor((Join-Path $session "training/actor-$update.bin"))
    for($i=0;$i -lt 32;$i++){Check ($export[$i] -eq $state.actor[$i]) "Actor export $i at $update"}
    $metadata=Get-Content (Join-Path $session "training/actor-$update.metadata.json") -Raw|ConvertFrom-Json
    Check ($metadata.schema -eq 2 -and $metadata.feedback -eq 'whole-beat-90' -and $metadata.update -eq $update -and $metadata.task_identity -eq $task) "Mode metadata at $update"
    Check ($metadata.course_error_scale_deg -eq 90 -and $metadata.initial_course_weight_multiplier -eq 18 -and [string]$metadata.initial_actor_weights_checksum -eq [string]$initialWeightsChecksum) "Input/transfer metadata at $update"
    $rows+=[pscustomobject]@{update=$update;payload_checksum=$state.hash.ToString();actor_adam_steps=$state.actorSteps;critic_adam_steps=$state.criticSteps;sampled_actions=$state.actions;attempted_substeps=$state.attempted;accepted_substeps=$state.accepted;reliable_controls=$state.reliable;low_controls=$state.low;measured_periods=$state.measured;rejected_periods=$state.rejected}
    $final=$state
}
for($i=0;$i -lt 32;$i++){Check ($initial.actor[$i] -eq $transferred[$i]) "Initial actor transfer at slot $i"}
Check ([Angle90SavedAudit]::Equal([Angle90SavedAudit]::Payload((Join-Path $session 'training/latest.checkpoint')),[Angle90SavedAudit]::Payload((Join-Path $session 'training/checkpoints/session-128.bin')))) 'Latest/final payload'
Check ([Angle90SavedAudit]::Equal([Angle90SavedAudit]::Payload((Join-Path $session 'verification/data/continuous-2.bin')),[Angle90SavedAudit]::Payload((Join-Path $session 'verification/data/resumed-2.bin')))) 'Full resume payload'
$proof=Get-Content (Join-Path $session 'verification/data/verification.json') -Raw|ConvertFrom-Json
Check ($proof.pass -and $proof.feedback -eq 'whole-beat-90' -and $proof.physical_updates_executed -eq 3 -and $proof.actual_actor_steps -eq 192 -and $proof.actual_critic_steps -eq 192) 'Resume proof work'
Check ($proof.initial_actor_max_difference -le 1e-6 -and $proof.initial_critic_max_difference -le 1e-6 -and $proof.restored_actor_max_difference -le 1e-6 -and $proof.restored_critic_max_difference -le 1e-6) 'CPU/GPU observer parity'
Check ($final.actions -eq $result.sampled_actions -and $final.actorSteps -eq $result.optimizer_steps -and $final.attempted -eq $result.attempted_substeps -and $final.accepted -eq $result.accepted_substeps) 'Native JSON/serialized accounting'
$perWorld=128*64;$episodes=[math]::Floor($perWorld/188);$remaining=$perWorld-$episodes*188
$maximumSteps=[uint64](1024*16*($episodes*750+$remaining*4))
Check ($final.attempted -le $maximumSteps -and $final.accepted -le $final.attempted) 'Horizon physical bound'
Check ($final.reliable+$final.low -eq $final.accepted/16) 'Course validity accounting'
$protected=@{
'flexible-tail-preview/live_data/fitted-gait.bin'='2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6'
'flexible-tail-preview/live_data/yaw-offset-update-8.bin'='52BA55F110DD1227CBFEB1E47E6199E50DE74687E82E9DD651114CDA86FEC354'
'flexible-tail-preview/addons/flexible_tail_live/bin/flexible_tail_live.dll'='6B665439A243A923B03ABF614C4B0825AA7A272F7DB533EC8D44A48A48B51942'
'src/physics/flexible_tail.h'='B6878791EC1A3CC68112866DE8C105390A43AC143A8EDCBA9EA2BF2430CCB226'}
foreach($path in $protected.Keys){Check ((Get-FileHash -LiteralPath (Join-Path $repository $path)).Hash -eq $protected[$path]) "Protected live file $path"}
$out=[ordered]@{schema=3;pass=$true;read_only_saved_data_audit=$true;physics_steps=0;optimizer_updates=0;checkpoints_checked=129;actor_exports_checked=129;actor_metadata_checked=129;worlds=1024;feedback='whole-beat-90';course_error_scale_deg=90;initial_transfer_multiplier=18;initial_actor_exact_transferred_copy=$true;source_offset_payload_checksum=$sourceOffsetChecksum.ToString();initial_weights_checksum=$initialWeightsChecksum.ToString();inactive_actor_and_adam_exact_zero=$true;latest_exact_immutable_final=$true;full_resume_bytes_exact=$true;cpu_gpu_feedback_parity_pass=$true;source_input_bindings_pass=$true;original_live_bundle_unchanged=$true;verification=$proof;updates=128;transitions=$final.actions;accepted_substeps=$final.accepted;attempted_substeps=$final.attempted;actor_adam_steps=$final.actorSteps;critic_adam_steps=$final.criticSteps;left_starts=$final.left;right_starts=$final.right;clearance_stops=$final.clearance;posture_stops=$final.posture;reliable_controls=$final.reliable;low_controls=$final.low;measured_periods=$final.measured;rejected_periods=$final.rejected;training_wall_s=$result.total_session_wall_s;transitions_per_s=($final.actions/$result.total_session_wall_s);checkpoints=$rows}
$out|ConvertTo-Json -Depth 12|Set-Content (Join-Path $PSScriptRoot 'report.json') -Encoding utf8
[pscustomobject]@{pass=$true;updates=128;transitions=$final.actions;accepted_substeps=$final.accepted;actor_adam_steps=$final.actorSteps;wall_s=$result.total_session_wall_s;rate=$final.actions/$result.total_session_wall_s}|ConvertTo-Json
