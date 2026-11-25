$ErrorActionPreference='Stop'
$repository=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../../..')).Path
Push-Location -LiteralPath $repository
try {
    $roots=@('seven-options-v6','seven-options-v6-corrected','seven-options-v6-bank-cap16','seven-options-v6-bank-resume') |
        ForEach-Object { (Resolve-Path -LiteralPath ('build/flexible-tail/'+$_)).Path }
    $old=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'report.json') -Raw | ConvertFrom-Json
    $new=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'report-seven-options-v6-bank-resume.json') -Raw | ConvertFrom-Json
    $stages=@($old.stages)+@($new.stages)
    function Total($items,[string]$property){return [long](($items | Measure-Object -Property $property -Sum).Sum)}
    $failed=Get-Content -LiteralPath (Join-Path $roots[0] 'stage-01-rhythm-baseline/verification/data/verification.json') -Raw | ConvertFrom-Json
    $motorMain=[ordered]@{ppo_updates=(Total $stages 'updates');actuator_decisions=(Total $stages 'samples');
        attempted_substeps=(Total $stages 'attempted_substeps');accepted_substeps=(Total $stages 'accepted_substeps');
        actor_adam_calls=(Total $stages 'new_actor_steps');critic_adam_calls=(Total $stages 'new_critic_steps')}
    $motorDiag=[ordered]@{ppo_updates=(Total $stages 'diagnostic_updates')+$failed.physical_updates_executed;
        actuator_decisions=(Total $stages 'diagnostic_samples')+$failed.sampled_actions_executed;
        attempted_substeps=(Total $stages 'diagnostic_attempted_substeps')+$failed.actual_attempted_substeps;
        accepted_substeps=(Total $stages 'diagnostic_accepted_substeps')+$failed.actual_accepted_substeps;
        actor_adam_calls=(Total $stages 'diagnostic_actor_steps')+$failed.actual_actor_steps;
        critic_adam_calls=(Total $stages 'diagnostic_critic_steps')+$failed.actual_critic_steps}
    $guard=Get-Content -LiteralPath (Join-Path $roots[1] 'stage-06-bank/verification/data/result.json') -Raw | ConvertFrom-Json
    $cd=Get-Content -LiteralPath (Join-Path $roots[2] 'stage-06-bank/verification/data/result.json') -Raw | ConvertFrom-Json
    $cm=Get-Content -LiteralPath (Join-Path $roots[3] 'stage-06-bank/training/result.json') -Raw | ConvertFrom-Json
    $chooser=[ordered]@{completed_main_updates=$cm.updates_completed;completed_main_decisions=$cm.executed_samples;
        completed_main_attempted_substeps=$cm.executed_attempted_substeps;completed_main_accepted_substeps=$cm.executed_accepted_substeps;
        completed_main_actor_adam_calls=$cm.executed_actor_steps;completed_main_critic_adam_calls=$cm.executed_critic_steps;
        diagnostic_updates=3;diagnostic_decisions=$cd.executed_samples;diagnostic_attempted_substeps=$cd.executed_attempted_substeps;
        diagnostic_accepted_substeps=$cd.executed_accepted_substeps;diagnostic_actor_adam_calls=$cd.executed_actor_steps;
        diagnostic_critic_adam_calls=$cd.executed_critic_steps;
        failed_action_guard_updates=0;failed_action_guard_attempted_substeps=$guard.attempted_substeps;
        failed_action_guard_accepted_substeps=$guard.accepted_substeps;
        killed_invocation_update_reservation=1;killed_invocation_completed_updates_known=0;
        killed_invocation_decisions_lower=0;killed_invocation_decisions_upper=8192;
        killed_invocation_substeps_lower=0;killed_invocation_substeps_upper=10485760;
        killed_invocation_adam_calls_each_lower=0;killed_invocation_adam_calls_each_upper=32}
    $seen=[Collections.Generic.HashSet[string]]::new();$processes=@();$evaluations=@();$material=@()
    foreach($root in $roots) {
        foreach($file in (Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.receipt.json')) {
            $hash=(Get-FileHash -LiteralPath $file.FullName).Hash
            if(!$seen.Add($hash)){continue} # Copied reuse receipts are not fresh execution.
            $receipt=Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
            $name=if($receipt.PSObject.Properties['name']){$receipt.name}else{$receipt.phase}
            $category=if($name -eq 'training'){'training'}elseif($name -eq 'verification'){'verification'}elseif($name -eq 'evaluation'){'cpu_evaluation'}elseif($name -match 'configure|compile'){'compilation'}else{$name}
            $processes += [pscustomobject]@{path=$file.FullName;category=$category;wall_s=$receipt.wall_s;exit=$receipt.exit;receipt_sha256=$hash}
            if($name -eq 'evaluation') {
                $assessment=Join-Path $file.Directory.FullName 'assessment'
                $resultPath=Join-Path $assessment 'result.json';$summaryPath=Join-Path $assessment 'summary.json'
                $result=if(Test-Path -LiteralPath $resultPath){Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json}elseif(Test-Path -LiteralPath $summaryPath){Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json}else{$null}
                if($null -eq $result){throw ('Missing physical assessment counts '+$file.FullName)}
                $attempted=[long]$result.attempted_substeps;$accepted=[long]$result.accepted_substeps
                $probe=Join-Path $assessment 'program-probes.json'
                if(Test-Path -LiteralPath $probe){$p=Get-Content -LiteralPath $probe -Raw | ConvertFrom-Json;$attempted+=$p.attempted_substeps;$accepted+=$p.accepted_substeps}
                $evaluations += [pscustomobject]@{receipt=$file.FullName;attempted_substeps=$attempted;accepted_substeps=$accepted;wall_s=$receipt.wall_s;exit=$receipt.exit}
            }
            if($name -eq 'qualification') {
                $r=Get-Content -LiteralPath (Join-Path $file.Directory.FullName 'result.json') -Raw | ConvertFrom-Json
                $material += [pscustomobject]@{receipt=$file.FullName;substeps=$r.total_substeps;wall_s=$receipt.wall_s}
            }
        }
    }
    $wall=@($processes | Group-Object category | ForEach-Object {[pscustomobject]@{category=$_.Name;processes=$_.Count;wall_s=($_.Group | Measure-Object wall_s -Sum).Sum}})
    $knownAttempted=[long]$motorMain.attempted_substeps+$motorDiag.attempted_substeps+
        $chooser.completed_main_attempted_substeps+$chooser.diagnostic_attempted_substeps+
        $chooser.failed_action_guard_attempted_substeps+(Total $evaluations 'attempted_substeps')+(Total $material 'substeps')
    $knownAccepted=[long]$motorMain.accepted_substeps+$motorDiag.accepted_substeps+
        $chooser.completed_main_accepted_substeps+$chooser.diagnostic_accepted_substeps+
        $chooser.failed_action_guard_accepted_substeps+(Total $evaluations 'accepted_substeps')+(Total $material 'substeps')
    $report=[ordered]@{schema=1;physics_steps_added_by_audit=0;optimizer_updates_added_by_audit=0;
        motor_stages=$stages;motor_main=$motorMain;motor_diagnostics_including_failed_first_attempt=$motorDiag;
        chooser=$chooser;cpu_assessments=$evaluations;material_qualification=$material;
        known_attempted_substeps=$knownAttempted;known_accepted_substeps=$knownAccepted;
        attempted_substeps_conservative_upper=$knownAttempted+10485760;
        accepted_substeps_conservative_upper=$knownAccepted+10485760;
        original_protocol_substep_ceiling=[long]4241983616;
        conservative_upper_within_protocol=($knownAttempted+10485760 -le 4241983616);
        process_wall_totals=$wall;process_wall_is_not_end_to_end_because_CPU_and_GPU_overlap=$true;
        interrupted_wall_charged_once_externally=1.0926767;
        wall_scope='Unique saved process receipts from the four versioned program roots; copied validation/reuse receipts deduplicated by exact SHA. Initial out-of-root compilation and interactive engineering time are excluded.';
        processes=$processes;
        uncertainty='Killed first chooser main has no completed update checkpoint or flushed work counters. Reserve its entire first8192-decision/10485760-substep/32-Adam-call update without claiming exact physical or optimizer spend.'}
    $report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ledger.json') -Encoding utf8
    [pscustomobject]@{motor_main=$motorMain;motor_diag=$motorDiag;chooser=$chooser;known_attempted_substeps=$knownAttempted;
        conservative_attempted_upper=$knownAttempted+10485760;within_protocol=$report.conservative_upper_within_protocol;
        wall=$wall;evaluation_invocations=$evaluations.Count;material_invocations=$material.Count} | ConvertTo-Json -Depth 7
} finally {Pop-Location}
