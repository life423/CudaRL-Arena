$ErrorActionPreference='Stop'
$session=Split-Path -Parent $PSScriptRoot
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $session))
Add-Type -Path (Join-Path $PSScriptRoot 'ChooserSavedAudit.cs')
function Assert-True($Condition,[string]$Message) { if (!$Condition) { throw $Message } }
$manifest=Get-Content (Join-Path $session 'manifest.json') -Raw | ConvertFrom-Json
$bad=@()
foreach ($binding in @($manifest.sources)+@($manifest.inputs)) {
    $path=Join-Path $root $binding.path
    if (!(Test-Path -LiteralPath $path) -or (Get-FileHash -LiteralPath $path).Hash -ne $binding.sha256) { $bad+=$binding.path }
}
Assert-True ($bad.Count -eq 0) ('Source/input binding differs: '+($bad -join ','))
$result=Get-Content (Join-Path $session 'training/result.json') -Raw | ConvertFrom-Json
Assert-True ($result.stage -eq 'complete' -and $result.updates_completed -eq 64) 'Main study not complete'
$seedPath=Join-Path $root 'assets/research-evidence/pond-flexible-tail-chooser-sustained-v3/data/training/actor-56.bin'
$motorPath=Join-Path $root 'build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin'
$rows=@();$task=$null;$initial=$null;$final=$null
for ($update=0;$update -le 64;$update++) {
    $path=Join-Path $session "training/checkpoints/session-$update.bin"
    $state=[ChooserSavedAudit]::ReadSession($path)
    if ($update -eq 0) { $task=$state.task;$initial=$state }
    Assert-True ($state.task -eq $task -and $state.update -eq $update) "Checkpoint task/update mismatch at $update"
    Assert-True ($state.worlds -eq 1024 -and $state.counters -eq 1024) "World/counter count at $update"
    Assert-True ($state.actor.Length -eq 1155 -and $state.critic.Length -eq 1089) "Parameter count at $update"
    Assert-True ($state.config[0] -eq 1024 -and $state.config[1] -eq 8 -and $state.config[2] -eq 1024 -and $state.config[3] -eq 4 -and $state.config[4] -eq 300) "PPO dimension at $update"
    Assert-True ([uint64]$state.keep+$state.left+$state.right -eq [uint64]($update*1024*8)) "Sample/RNG accounting at $update"
    $export=[ChooserSavedAudit]::ReadActor((Join-Path $session "training/actor-$update.bin"))
    Assert-True ($export.Length -eq 1155) "Actor export dimension $update"
    for ($i=0;$i -lt 1155;$i++) { Assert-True ($export[$i] -eq $state.actor[$i]) "Actor export mismatch at $update slot $i" }
    $metadata=Get-Content (Join-Path $session "training/actor-$update.metadata.json") -Raw | ConvertFrom-Json
    Assert-True ($metadata.motor_feedback -eq 'whole-beat' -and $metadata.horizon -eq 'long-horizon' -and $metadata.steering -eq 'held-heading' -and $metadata.task_identity -eq $task -and $metadata.update -eq $update) "Actor observation identity at $update"
    $rows+=[pscustomobject]@{update=$update;payload_checksum=$state.hash.ToString();actor_adam_steps=$state.actorSteps;
        critic_adam_steps=$state.criticSteps;sampled_decisions=([uint64]$state.keep+$state.left+$state.right);
        attempted_substeps=($state.attempted+$state.warmupAttempted);accepted_substeps=($state.accepted+$state.warmupAccepted);
        clearance_stops=$state.clearance;cue_timeouts=$state.cue;posture_stops=$state.posture;
        min_episode_rng_position=$state.minimumEpisode;max_episode_rng_position=$state.maximumEpisode}
    $final=$state
}
Assert-True ([ChooserSavedAudit]::Equal([ChooserSavedAudit]::Payload((Join-Path $session 'training/actor-0.bin')),[ChooserSavedAudit]::Payload($seedPath))) 'Initial actor differs from declared seed'
Assert-True ([ChooserSavedAudit]::Equal([ChooserSavedAudit]::Payload((Join-Path $session 'training/latest.checkpoint')),[ChooserSavedAudit]::Payload((Join-Path $session 'training/checkpoints/session-64.bin')))) 'Latest differs from final immutable checkpoint'
$verifyRoot=Join-Path $session 'verification/data'
if (!(Test-Path -LiteralPath $verifyRoot)) { $verifyRoot=Join-Path $session 'verification' }
$verification=Get-Content (Join-Path $verifyRoot 'verification.json') -Raw | ConvertFrom-Json
$one=[ChooserSavedAudit]::ReadSession((Join-Path $verifyRoot 'checkpoint-1.bin'))
$two=[ChooserSavedAudit]::ReadSession((Join-Path $verifyRoot 'continuous-2.bin'))
Assert-True ([ChooserSavedAudit]::Equal([ChooserSavedAudit]::Payload((Join-Path $verifyRoot 'continuous-2.bin')),[ChooserSavedAudit]::Payload((Join-Path $verifyRoot 'resumed-2.bin')))) 'Exact resume state differs'
$attemptedVerify=2*($two.attempted+$two.warmupAttempted)-($one.attempted+$one.warmupAttempted)
$acceptedVerify=2*($two.accepted+$two.warmupAccepted)-($one.accepted+$one.warmupAccepted)
$actorCallsVerify=2*$two.actorSteps-$one.actorSteps
Assert-True ($verification.pass -and $verification.physical_updates_executed -eq 3 -and $verification.actual_sampled_decisions -eq 24576) 'Verification sample accounting'
Assert-True ($verification.actual_attempted_substeps -eq $attemptedVerify -and $verification.actual_accepted_substeps -eq $acceptedVerify -and $verification.actual_actor_adam_steps -eq $actorCallsVerify -and $verification.actual_critic_adam_steps -eq $actorCallsVerify) 'Verification actual work differs from independent 2*update2-update1 calculation'
$attempted=$final.attempted+$final.warmupAttempted;$accepted=$final.accepted+$final.warmupAccepted
Assert-True ($attempted -eq $result.attempted_substeps -and $accepted -eq $result.accepted_substeps -and $final.actorSteps -eq $result.actor_adam_steps -and $final.criticSteps -eq $result.critic_adam_steps) 'Saved/runtime counters disagree'
Assert-True ($attempted -le 5033164800 -and $attemptedVerify -le 235929600) 'Physical budget exceeded'
Assert-True ($attempted-$accepted -eq $final.clearance -and $final.cue -eq 0 -and $final.posture -eq 0) 'Rejected substeps do not match counted clearance terminals'
Assert-True ($final.warmupAttempted -eq $final.warmupAccepted) 'Unrecorded warmup failure'
Assert-True ($final.offset -eq [ChooserSavedAudit]::Hash([ChooserSavedAudit]::Payload($motorPath))) 'Frozen motor checksum differs'
$evals=@()
for ($update=8;$update -le 64;$update+=8) {
    $directory=Join-Path $session ('evaluations/update-{0:D4}' -f $update)
    $assessment=Get-Content (Join-Path $directory 'assessment/summary.json') -Raw | ConvertFrom-Json
    $cases=@(Import-Csv (Join-Path $directory 'assessment/cases.csv'))
    Assert-True ($assessment.diagnostic_cases -eq 12 -and $cases.Count -eq 12) "Missing evaluation cases at $update"
    Assert-True ($assessment.endurance_attempts -eq 0 -and $assessment.endurance_passes -eq 0 -and $assessment.resets -eq 0 -and $assessment.optimizer_updates -eq 0) "Unexpected endurance, reset or optimizer spend at $update"
    $evals+=[pscustomobject]@{update=$update;passes=$assessment.diagnostic_passes;endurance_attempts=$assessment.endurance_attempts;
        attempted=$assessment.attempted_substeps;accepted=$assessment.accepted_substeps;coverage=$assessment.diagnostic_coverage_fraction}
}
$out=[ordered]@{schema=2;pass=$true;read_only_saved_data_audit=$true;physics_steps=0;optimizer_updates=0;
    checkpoints_checked=65;actor_exports_checked=65;actor_metadata_checked=65;worlds=1024;
    modes=@('held-heading','whole-beat','long-horizon');sources_and_inputs_unchanged=$true;
    initial_chooser_exact_seed=$true;frozen_motor56_unchanged=$true;latest_exact_immutable_final=$true;
    actor_critic_adam_and_world_rng_checked=$true;full_resume_bytes_exact=$true;proof_actual_work_independently_derived=$true;
    verification=$verification;updates=64;transitions=524288;attempted_substeps=$attempted;accepted_substeps=$accepted;
    actor_adam_steps=$final.actorSteps;critic_adam_steps=$final.criticSteps;clearance_terminals=$final.clearance;
    rejected_substeps_match_terminals=$true;cue_timeouts=0;posture_stops=0;endurance_attempts=0;
    training_wall_s=$result.invocation_wall_s;transitions_per_s=(524288/$result.invocation_wall_s);
    seed_chooser_sha256=(Get-FileHash $seedPath).Hash;frozen_motor56_sha256=(Get-FileHash $motorPath).Hash;
    proof_plus_main_attempted_substeps=($attempted+$attemptedVerify);proof_plus_main_accepted_substeps=($accepted+$acceptedVerify);
    evaluations=$evals;checkpoints=$rows}
$out | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $PSScriptRoot 'report.json') -Encoding utf8
[pscustomobject]@{pass=$true;updates=64;transitions=524288;attempted=$attempted;accepted=$accepted;actor_adam_steps=$final.actorSteps;wall_s=$result.invocation_wall_s;seed_motor_unchanged=$true} | ConvertTo-Json
