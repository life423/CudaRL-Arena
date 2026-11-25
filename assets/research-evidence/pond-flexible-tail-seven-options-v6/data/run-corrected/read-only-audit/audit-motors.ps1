param([string]$Root=(Split-Path -Parent $PSScriptRoot),[switch]$HistoricalBindings)
$ErrorActionPreference='Stop'
$repository=(Resolve-Path -LiteralPath (Join-Path $Root '../../..')).Path
Push-Location -LiteralPath $repository
try {
    Add-Type -Path (Join-Path $PSScriptRoot 'SevenStageSavedAudit.cs')
    function Assert-Equal($a,$b,[string]$message) {
        if(![SevenStageSavedAudit]::FloatEqual($a,$b)){throw $message}
    }
    function Allowed([int]$i,[uint32]$kind) {
        if($kind -eq 5){return $true}
        if($kind -eq 2){return $i -ge 4 -and $i -lt 13}
        if($kind -eq 3){return $i -ge 4 -and $i -lt 17}
        if($kind -eq 4) {
            if($i -lt 31){return $i -ge 4 -and $i -lt 17}
            if($i -lt 93){return ($i%31) -lt 17}
            return $i -gt 93
        }
        return $false
    }
    function Check-Model($a,$base,[string]$context) {
        if($a.material -gt 1){throw "Invalid material $context"}
        if($a.kind -ne 5) {
            Assert-Equal $a.weights[0..3] $base "Frozen actor prefix $context"
            for($i=4;$i -lt $a.weights.Length;$i++) {
                if(!(Allowed $i $a.kind) -and $a.weights[$i] -ne 0){throw "Unused actor coordinate $context/$i"}
            }
            if($a.skip.Length){throw "Unexpected skip $context"}
        } else {
            if($a.skip.Length -ne 96){throw "Missing frozen skip $context"}
            Assert-Equal $a.skip[0..3] $base "Frozen skip prefix $context"
            for($i=4;$i -lt 96;$i++) {
                if(!(Allowed $i 4) -and $a.skip[$i] -ne 0){throw "Unused frozen skip coordinate $context/$i"}
            }
        }
    }
    function Map-Moments($source,[uint32]$kind) {
        $n=if($kind -eq 5){563}elseif($kind -eq 4){96}else{32}
        $out=[float[]]::new($n)
        if($source.Length -eq $n){[Array]::Copy($source,$out,$n)}
        elseif($source.Length -eq 32 -and $n -eq 96){[Array]::Copy($source,$out,31);$out[93]=$source[31]}
        elseif(!($source.Length -eq 96 -and $n -eq 563)){throw 'Unsupported moment mapping'}
        return ,$out
    }
    $bindingCount=0;$historical=@()
    foreach($binding in (Get-Content -LiteralPath (Join-Path $Root 'bindings.json') -Raw | ConvertFrom-Json)) {
        $currentHash=(Get-FileHash -LiteralPath $binding.path).Hash
        if($currentHash -ne $binding.sha256){
            $archive=Join-Path (Join-Path $Root 'source') $binding.path
            if([IO.Path]::GetExtension($binding.path) -eq '.exe') {
                $archive=Join-Path (Join-Path $PSScriptRoot 'pre-categorical-repair-binaries') ([IO.Path]::GetFileName($binding.path))
            }
            if(!$HistoricalBindings -or !(Test-Path -LiteralPath $archive) -or
                (Get-FileHash -LiteralPath $archive).Hash -ne $binding.sha256){throw ('Frozen binding changed without preserved historical bytes: '+$binding.path)}
            $historical += [pscustomobject]@{path=$binding.path;historical_sha256=$binding.sha256;current_sha256=$currentHash;preserved=$archive}
        }
        $bindingCount++
    }
    $reports=@();$pending=@();$warnings=@()
    foreach($stage in (Get-ChildItem -LiteralPath $Root -Directory | Where-Object Name -match '^stage-')) {
        $resultPath=Join-Path $stage.FullName 'training/result.json'
        if(!(Test-Path -LiteralPath $resultPath)){continue}
        $result=Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
        if(!$result.PSObject.Properties['task_identity']){continue} # Rate chooser has a separate FTCR schema.
        if(!(Test-Path -LiteralPath (Join-Path $stage.FullName 'training/checkpoints/session-0.bin'))){continue}
        if($result.stage -in @('checkpoint','collection','setup','advantages','optimizer')) {
            $pending += [pscustomobject]@{stage=$stage.Name;updates=$result.updates_completed;status=$result.stage};continue
        }
        $parentInfo=Get-Content -LiteralPath (Join-Path $stage.FullName 'parent.json') -Raw | ConvertFrom-Json
        $parent=[SevenStageSavedAudit]::ReadSession($parentInfo.checkpoint)
        $source=[SevenStageSavedAudit]::ReadActor($parentInfo.actor)
        if($parent.task -ne $parentInfo.task -or $parent.update -ne $parentInfo.update){throw "Parent pointer mismatch $($stage.Name)"}
        Assert-Equal $source.weights $parent.actor "Parent actor mismatch $($stage.Name)"
        $updates=@(Import-Csv -LiteralPath (Join-Path $stage.FullName 'training/updates.csv'))
        $record=@();$identity=$null;$first=$null;$initialModel=$null;$newSteps=0
        for($u=0;$u -le $result.updates_completed;$u++) {
            $s=[SevenStageSavedAudit]::ReadSession((Join-Path $stage.FullName ('training/checkpoints/session-'+$u+'.bin')))
            $a=[SevenStageSavedAudit]::ReadActor((Join-Path $stage.FullName ('training/actor-'+$u+'.bin')))
            if($null -eq $first){$first=$s;$initialModel=$a;$identity=$s.identityHash}
            if($u -gt 0) {
                $row=$updates | Where-Object { [int]$_.update -eq $u }
                if(@($row).Count -ne 1){throw "Missing progress row $($stage.Name)/$u"}
                $newSteps += [int]$row.minibatches
            }
            if($s.version -ne 6 -or $s.update -ne $u -or $s.worlds -ne 1024 -or $s.counters -ne 1024 -or
                $s.actions -ne 65536*$u -or $s.actorSteps -ne $first.actorSteps+$newSteps -or
                $s.criticSteps -ne $first.criticSteps+$newSteps -or
                $s.rng -ne ($s.samplingOrigin+$u)*64 -or $s.samplingOrigin -ne $parent.rng/64 -or
                $s.parentPayload -ne $parent.hash -or $s.parentTask -ne $parent.task -or
                $s.parentUpdate -ne $parent.update -or $s.parentActorSteps -ne $parent.actorSteps -or
                $s.parentCriticSteps -ne $parent.criticSteps -or $s.identityHash -ne $identity -or
                $s.material -ne $a.material -or $s.study -ne $a.study -or $s.kind -ne $a.kind -or
                $s.initialWeights -ne [SevenStageSavedAudit]::WeightHash($first.actor) -or
                $s.skipHash -ne [SevenStageSavedAudit]::WeightHash($a.skip)) {
                throw "State accounting, mode or lineage mismatch $($stage.Name)/$u"
            }
            Assert-Equal $s.actor $a.weights "Actor export mismatch $($stage.Name)/$u"
            Assert-Equal $a.frozenBase $source.frozenBase "Frozen base binding $($stage.Name)/$u"
            Assert-Equal $a.skip $initialModel.skip "Frozen skip changed $($stage.Name)/$u"
            Check-Model $a $source.frozenBase "$($stage.Name)/$u"
            for($i=0;$i -lt $s.actorM.Length;$i++) {
                if(!(Allowed $i $s.kind) -and ($s.actorM[$i] -ne 0 -or $s.actorV[$i] -ne 0)) {
                    throw "Frozen moment changed $($stage.Name)/$u/$i"
                }
            }
            $record += [pscustomobject]@{update=$u;samples=$s.actions;attempted=$s.attempted;accepted=$s.accepted;
                actor_age=$s.actorSteps;critic_age=$s.criticSteps;new_actor_steps=$newSteps;new_critic_steps=$newSteps;
                next_sampling_index=$s.rng;payload_checksum=$s.hash}
            $last=$s;$lastActor=$a
        }
        if(![SevenStageSavedAudit]::Equal($first.nativeWorldBytes,$parent.nativeWorldBytes)) {throw "Native world reset at import $($stage.Name)"}
        Assert-Equal $first.critic $parent.critic "Critic weights changed at import $($stage.Name)"
        Assert-Equal $first.criticM $parent.criticM "Critic momentum changed at import $($stage.Name)"
        Assert-Equal $first.criticV $parent.criticV "Critic variance changed at import $($stage.Name)"
        Assert-Equal $first.actorM (Map-Moments $parent.actorM $first.kind) "Mapped actor momentum differs $($stage.Name)"
        Assert-Equal $first.actorV (Map-Moments $parent.actorV $first.kind) "Mapped actor variance differs $($stage.Name)"
        if($source.kind -eq 4 -and $first.kind -eq 5) {
            if($first.actorSteps -ne 0){throw 'New nonlinear optimizer age not zero'}
            Assert-Equal $initialModel.skip $source.weights 'Nonlinear skip was not the complete source readout'
            if(@($first.actor[512..562] | Where-Object {$_ -ne 0}).Count){throw 'Nonlinear correction readout not initially zero'}
        } else {
            if($first.actorSteps -ne $parent.actorSteps){throw 'Common actor optimizer age changed'}
            if($first.actor.Length -eq $parent.actor.Length) {Assert-Equal $first.actor $parent.actor 'Common actor weights changed'}
            elseif($first.kind -eq 4) {
                Assert-Equal $first.actor[0..30] $parent.actor[0..30] 'Linear source readout changed'
                if(![SevenStageSavedAudit]::BitsEqual($first.actor[93],$parent.actor[31])){throw 'Source intercept mapping differs'}
                if(@($first.actor[31..92] | Where-Object {$_ -ne 0}).Count -or $first.actor[94] -ne 0 -or $first.actor[95] -ne 0){throw 'New linear heads not zero'}
            }
        }
        if($first.criticSteps -ne $parent.criticSteps -or $first.actions -ne 0 -or $first.attempted -ne 0 -or
            $first.accepted -ne 0 -or $first.startsCount -ne 0){throw 'Initial age or diagnostic rebase differs'}
        $latest=[SevenStageSavedAudit]::ReadSession((Join-Path $stage.FullName 'training/latest.checkpoint'))
        if($latest.hash -ne $last.hash -or $last.actions -ne $result.sampled_actions -or $last.attempted -ne $result.attempted_substeps -or
            $last.accepted -ne $result.accepted_substeps -or $newSteps -ne $result.invocation_actor_steps -or
            $newSteps -ne $result.invocation_critic_steps){throw "Final result mismatch $($stage.Name)"}
        $fork=Get-Content -LiteralPath (Join-Path $stage.FullName 'training/fork-equivalence.json') -Raw | ConvertFrom-Json
        if(!$fork.pass -or $fork.physics_steps -ne 0 -or !$fork.native_worlds_bit_exact -or !$fork.optimizer_coordinate_transfer_exact){throw 'Import proof failure'}
        $diagnostic=Get-Content -LiteralPath (Join-Path $stage.FullName 'verification/data/verification.json') -Raw | ConvertFrom-Json
        $continuous=[SevenStageSavedAudit]::ReadSession((Join-Path $stage.FullName 'verification/data/continuous-2.bin'))
        $resumed=[SevenStageSavedAudit]::ReadSession((Join-Path $stage.FullName 'verification/data/resumed-2.bin'))
        if(!$diagnostic.pass -or $continuous.hash -ne $resumed.hash -or $diagnostic.physical_updates_executed -ne 3){throw 'Diagnostic resume mismatch'}
        $changed=@(0..($last.actor.Length-1) | Where-Object {![SevenStageSavedAudit]::BitsEqual($first.actor[$_],$last.actor[$_])})
        $warning=$null
        if($result.study_actor_steps -ne $newSteps){
            $warning='Raw study_actor_steps subtracts the source age even when nonlinear transfer resets the actor optimizer; true steps use session0 age and invocation counters.'
            $warnings += [pscustomobject]@{stage=$stage.Name;reported=$result.study_actor_steps;actual=$newSteps;reason=$warning}
        }
        $item=[ordered]@{stage=$stage.Name;pass=$true;status=$result.stage;updates=$last.update;checkpoints_verified=$record.Count;
            model_kind=$last.kind;parameters=$last.actor.Length;frozen_skip_parameters=$lastActor.skip.Length;study=$last.study;material=$last.material;
            parent_task=$parent.task;parent_update=$parent.update;parent_payload_checksum=$parent.hash;
            native_import_bit_exact=$true;common_critic_and_optimizer_bit_exact=$true;actor_coordinate_transfer_exact=$true;
            frozen_base_and_skip_exact=$true;mask_and_unused_moments_valid=$true;all_export_payload_checksums_valid=$true;
            exact_resume_pass=$true;diagnostic_updates=3;diagnostic_samples=$diagnostic.sampled_actions_executed;
            diagnostic_attempted_substeps=$diagnostic.actual_attempted_substeps;diagnostic_accepted_substeps=$diagnostic.actual_accepted_substeps;
            diagnostic_actor_steps=$diagnostic.actual_actor_steps;diagnostic_critic_steps=$diagnostic.actual_critic_steps;
            samples=$last.actions;attempted_substeps=$last.attempted;accepted_substeps=$last.accepted;
            imported_actor_age=$first.actorSteps;imported_critic_age=$first.criticSteps;new_actor_steps=$newSteps;new_critic_steps=$newSteps;
            final_actor_age=$last.actorSteps;final_critic_age=$last.criticSteps;sampling_origin=$last.samplingOrigin;next_sampling_index=$last.rng;
            changed_parameter_indices=$changed;reporting_warning=$warning;records=$record}
        $item | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot ($stage.Name+'.json')) -Encoding utf8
        $item.Remove('records');$item.Remove('changed_parameter_indices');$reports += [pscustomobject]$item
    }
    $failedPath='build/flexible-tail/seven-options-v6/stage-01-rhythm-baseline/verification/data/verification.json'
    $failed=Get-Content -LiteralPath $failedPath -Raw | ConvertFrom-Json
    $mainTotals=[ordered]@{ppo_updates=0;samples=0;attempted_substeps=0;accepted_substeps=0;actor_adam_calls=0;critic_adam_calls=0}
    $diagnosticTotals=[ordered]@{ppo_updates=0;samples=0;attempted_substeps=0;accepted_substeps=0;actor_adam_calls=0;critic_adam_calls=0}
    foreach($stageReport in $reports) {
        $mainTotals.ppo_updates += $stageReport.updates;$mainTotals.samples += $stageReport.samples
        $mainTotals.attempted_substeps += $stageReport.attempted_substeps;$mainTotals.accepted_substeps += $stageReport.accepted_substeps
        $mainTotals.actor_adam_calls += $stageReport.new_actor_steps;$mainTotals.critic_adam_calls += $stageReport.new_critic_steps
        $diagnosticTotals.ppo_updates += $stageReport.diagnostic_updates;$diagnosticTotals.samples += $stageReport.diagnostic_samples
        $diagnosticTotals.attempted_substeps += $stageReport.diagnostic_attempted_substeps;$diagnosticTotals.accepted_substeps += $stageReport.diagnostic_accepted_substeps
        $diagnosticTotals.actor_adam_calls += $stageReport.diagnostic_actor_steps;$diagnosticTotals.critic_adam_calls += $stageReport.diagnostic_critic_steps
    }
    $diagnosticTotals.ppo_updates += $failed.physical_updates_executed;$diagnosticTotals.samples += $failed.sampled_actions_executed
    $diagnosticTotals.attempted_substeps += $failed.actual_attempted_substeps;$diagnosticTotals.accepted_substeps += $failed.actual_accepted_substeps
    $diagnosticTotals.actor_adam_calls += $failed.actual_actor_steps;$diagnosticTotals.critic_adam_calls += $failed.actual_critic_steps
    $report=[ordered]@{schema=1;pass=$true;physics_steps=0;optimizer_updates=0;frozen_bindings_verified=$bindingCount;
        all_current_files_match_original_bindings=($historical.Count -eq 0);
        all_historical_source_executable_and_protected_asset_bindings_preserved=$true;
        historical_source_replacements=$historical;closed_motor_stages=$reports.Count;
        totals_scope='Closed motor stages only; chooser, material qualification and CPU physical assessments are accounted separately.';
        motor_main_totals=$mainTotals;motor_diagnostic_totals_including_preserved_failed_first_attempt=$diagnosticTotals;
        failed_first_diagnostic_updates=$failed.physical_updates_executed;failed_first_diagnostic_samples=$failed.sampled_actions_executed;
        failed_first_diagnostic_attempted_substeps=$failed.actual_attempted_substeps;failed_first_diagnostic_accepted_substeps=$failed.actual_accepted_substeps;
        failed_first_diagnostic_actor_steps=$failed.actual_actor_steps;failed_first_diagnostic_critic_steps=$failed.actual_critic_steps;
        warnings=$warnings;stages=$reports;pending=$pending}
    $name=if((Split-Path -Parent $PSScriptRoot) -eq $Root){'report.json'}else{'report-'+(Split-Path -Leaf $Root)+'.json'}
    $report | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $PSScriptRoot ($name+'.tmp')) -Encoding utf8
    [IO.File]::Move((Join-Path $PSScriptRoot ($name+'.tmp')),(Join-Path $PSScriptRoot $name),$true)
    [pscustomobject]@{pass=$true;closed_stages=$reports.Count;pending=$pending;warnings=$warnings;physics_steps=0;optimizer_updates=0} | ConvertTo-Json -Depth 5
} finally {Pop-Location}
