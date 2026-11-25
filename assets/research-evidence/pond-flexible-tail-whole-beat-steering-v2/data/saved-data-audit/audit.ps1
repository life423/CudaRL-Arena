$ErrorActionPreference='Stop'
$session=Split-Path -Parent $PSScriptRoot
Add-Type -Path (Join-Path $PSScriptRoot 'BeatSteeringSavedAudit.cs')
function Assert-True($Condition,[string]$Message) { if (!$Condition) { throw $Message } }
$manifest=Get-Content (Join-Path $session 'manifest.json') -Raw | ConvertFrom-Json
$bindings=Get-Content (Join-Path $PSScriptRoot 'bindings-before-bank.json') -Raw | ConvertFrom-Json
Assert-True $bindings.pass 'Pre-bank source/input binding audit failed'
$result=Get-Content (Join-Path $session 'training/result.json') -Raw | ConvertFrom-Json
Assert-True ($result.stage -eq 'complete' -and $result.updates_completed -eq 128) 'Main study not complete'
$frozenOffset=@($manifest.inputs | Where-Object path -eq 'flexible-tail-preview/live_data/yaw-offset-update-8.bin')[0]
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $session))
$offset=Join-Path $root $frozenOffset.path
$original=[BeatSteeringSavedAudit]::ReadActor($offset)
$rows=@()
$task=$null
$initial=$null
$final=$null
for ($update=0;$update -le 128;$update++) {
    $path=Join-Path $session "training/checkpoints/session-$update.bin"
    $state=[BeatSteeringSavedAudit]::ReadSession($path)
    if ($update -eq 0) { $task=$state.task;$initial=$state }
    Assert-True ($state.task -eq $task -and $state.update -eq $update) "Checkpoint task/update mismatch at $update"
    Assert-True ($state.worlds -eq 1024 -and $state.counters -eq 1024) "World/counter count at $update"
    Assert-True ($state.actor.Length -eq 32 -and $state.critic.Length -eq 1217) "Parameter count at $update"
    Assert-True ($state.config[0] -eq 1024 -and $state.config[1] -eq 64 -and $state.config[2] -eq 31 -and $state.config[3] -eq 36) "PPO dimension at $update"
    Assert-True ($state.config[4] -eq 4096 -and $state.config[5] -eq 4 -and $state.config[6] -eq 4 -and $state.config[7] -eq 128) "PPO schedule at $update"
    Assert-True ($state.rng -eq [uint64]($update*64) -and $state.actions -eq [uint64]($update*1024*64)) "RNG/sample accounting at $update"
    for ($i=4;$i -lt 32;$i++) {
        Assert-True ($state.actor[$i] -eq 0 -and $state.actorM[$i] -eq 0 -and $state.actorV[$i] -eq 0) "Inactive coefficient/Adam moment changed at $update slot $i"
    }
    $export=[BeatSteeringSavedAudit]::ReadActor((Join-Path $session "training/actor-$update.bin"))
    Assert-True ($export.Length -eq 32) "Actor export dimension $update"
    for ($i=0;$i -lt 32;$i++) { Assert-True ($export[$i] -eq $state.actor[$i]) "Actor export mismatch at $update slot $i" }
    $metadata=Get-Content (Join-Path $session "training/actor-$update.metadata.json") -Raw | ConvertFrom-Json
    Assert-True ($metadata.feedback -eq 'whole-beat' -and $metadata.task_identity -eq $task -and $metadata.update -eq $update) "Missing observation identity at $update"
    $rows+=[pscustomobject]@{update=$update;payload_checksum=$state.hash.ToString();actor_adam_steps=$state.actorSteps;
        critic_adam_steps=$state.criticSteps;sampled_actions=$state.actions;attempted_substeps=$state.attempted;
        accepted_substeps=$state.accepted;reliable_controls=$state.reliable;low_controls=$state.low;
        measured_periods=$state.measured;rejected_periods=$state.rejected}
    $final=$state
}
Assert-True ([BeatSteeringSavedAudit]::Equal([BeatSteeringSavedAudit]::Payload((Join-Path $session 'training/actor-0.bin')),[BeatSteeringSavedAudit]::Payload($offset))) 'Initial actor differs from original'
Assert-True ([BeatSteeringSavedAudit]::Equal([BeatSteeringSavedAudit]::Payload((Join-Path $session 'training/latest.checkpoint')),[BeatSteeringSavedAudit]::Payload((Join-Path $session 'training/checkpoints/session-128.bin')))) 'Latest differs from final immutable checkpoint'
Assert-True ([BeatSteeringSavedAudit]::Equal([BeatSteeringSavedAudit]::Payload((Join-Path $session 'verification/data/continuous-2.bin')),[BeatSteeringSavedAudit]::Payload((Join-Path $session 'verification/data/resumed-2.bin')))) 'Exact resume state differs'
$verification=Get-Content (Join-Path $session 'verification/data/verification.json') -Raw | ConvertFrom-Json
Assert-True ($verification.pass -and $verification.physical_updates_executed -eq 3 -and $verification.actual_attempted_substeps -eq 12582912 -and $verification.actual_actor_steps -eq 192) 'Verification accounting'
Assert-True ($verification.initial_actor_max_difference -le 1e-6 -and $verification.initial_critic_max_difference -le 1e-6 -and $verification.restored_actor_max_difference -le 1e-6 -and $verification.restored_critic_max_difference -le 1e-6) 'CPU/GPU parity tolerance'
Assert-True ($final.actions -eq $result.sampled_actions -and $final.actorSteps -eq $result.optimizer_steps -and $final.attempted -eq $result.attempted_substeps -and $final.accepted -eq $result.accepted_substeps) 'Result/serialized counters disagree'
$actionsPerWorld=128*64
$episodes=[math]::Floor($actionsPerWorld/188)
$remaining=$actionsPerWorld-$episodes*188
$expectedSteps=[uint64](1024*16*($episodes*750+$remaining*4))
Assert-True ($final.attempted -eq $expectedSteps -and $final.accepted -eq $expectedSteps) 'Substeps differ from independent horizon calculation'
Assert-True ($final.reliable+$final.low -eq $expectedSteps/16) 'Reliable and low course controls do not cover physical work'
Assert-True ($final.left -eq $final.right -and $final.left -eq 22528 -and $final.clearance -eq 0 -and $final.posture -eq 0) 'Episode/result accounting'
$out=[ordered]@{schema=2;pass=$true;read_only_saved_data_audit=$true;physics_steps=0;optimizer_updates=0;
    checkpoints_checked=129;actor_exports_checked=129;actor_metadata_checked=129;worlds=1024;feedback='whole-beat';
    initial_actor_exact_original=$true;inactive_actor_and_adam_exact_zero=$true;latest_exact_immutable_final=$true;
    full_resume_bytes_exact=$true;cpu_gpu_feedback_parity_pass=$true;sources_and_inputs_before_bank_pass=$bindings.pass;
    pre_bank_bindings=$bindings;verification=$verification;updates=128;transitions=$final.actions;
    accepted_substeps=$final.accepted;attempted_substeps=$final.attempted;actor_adam_steps=$final.actorSteps;
    critic_adam_steps=$final.criticSteps;left_starts=$final.left;right_starts=$final.right;
    reliable_controls=$final.reliable;low_controls=$final.low;measured_periods=$final.measured;rejected_periods=$final.rejected;
    training_wall_s=$result.total_session_wall_s;transitions_per_s=($final.actions/$result.total_session_wall_s);checkpoints=$rows}
$out | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $PSScriptRoot 'report.json') -Encoding utf8
[pscustomobject]@{pass=$true;updates=128;transitions=$final.actions;accepted_substeps=$final.accepted;actor_adam_steps=$final.actorSteps;wall_s=$result.total_session_wall_s;rate=$final.actions/$result.total_session_wall_s} | ConvertTo-Json
