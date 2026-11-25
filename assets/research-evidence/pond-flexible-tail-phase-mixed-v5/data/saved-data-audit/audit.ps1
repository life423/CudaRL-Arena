$ErrorActionPreference='Stop'
$study=Split-Path -Parent $PSScriptRoot
$repository=(Resolve-Path -LiteralPath (Join-Path $study '../../..')).Path
Push-Location -LiteralPath $repository
try {
    Add-Type -Path (Join-Path $PSScriptRoot 'Phase5SavedAudit.cs')
    $source=[Phase5SavedAudit]::ReadActor((Resolve-Path -LiteralPath 'build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin').Path)
    $parent=[Phase5SavedAudit]::ReadSession((Resolve-Path -LiteralPath 'build/flexible-tail/phase-aware-v4/training/checkpoints/session-112.bin').Path)
    $parentActor=[Phase5SavedAudit]::ReadActor((Resolve-Path -LiteralPath 'build/flexible-tail/phase-aware-v4/training/actor-112.bin').Path)
    $records=@();$identity=$null;$first=$null
    for($u=0;$u -le 128;$u++) {
        $s=[Phase5SavedAudit]::ReadSession((Join-Path $study ('training/checkpoints/session-'+$u+'.bin')))
        $a=[Phase5SavedAudit]::ReadActor((Join-Path $study ('training/actor-'+$u+'.bin')))
        if($null -eq $identity){$identity=$s.identityHash;$first=$s}
        if($s.version -ne 5 -or $s.practice -ne 1 -or $s.update -ne $u -or
            $s.actorSteps -ne 7168+64*$u -or $s.criticSteps -ne 7168+64*$u -or
            $s.rng -ne (112+$u)*64 -or $s.actions -ne 65536*$u -or
            $s.worlds -ne 1024 -or $s.counters -ne 1024 -or $s.identityHash -ne $identity -or
            $s.critic.Length -ne 1217 -or $s.clearance -ne 0 -or $s.posture -ne 0 -or
            $s.attempted -ne $s.accepted -or $s.parentPayload -ne $parent.hash -or
            $s.parentTask -ne $parent.task -or $s.parentUpdate -ne 112 -or
            $s.parentActorSteps -ne 7168 -or $s.parentCriticSteps -ne 7168 -or
            $s.parentInitialWeights -ne $parent.initialWeights -or
            $s.initialWeights -ne [Phase5SavedAudit]::WeightHash($parent.actor)) {
            throw ('Session accounting, lineage or identity '+$u)
        }
        if(![Phase5SavedAudit]::FloatEqual($s.actor,$a.weights) -or
            ![Phase5SavedAudit]::FloatEqual($s.initialCoefficients,$source.frozenBase) -or
            ![Phase5SavedAudit]::FloatEqual($a.frozenBase,$source.frozenBase)) {
            throw ('Export or frozen-prefix mismatch '+$u)
        }
        for($i=0;$i -lt 32;$i++) {
            if($i -lt 4 -and ![Phase5SavedAudit]::BitsEqual($s.actor[$i],$source.weights[$i])) {throw ('Frozen weight '+$u)}
            if(($i -lt 4 -or $i -ge 13) -and ($s.actorM[$i] -ne 0 -or $s.actorV[$i] -ne 0 -or
                ($i -ge 13 -and $s.actor[$i] -ne 0))) {throw ('Frozen moment or unused parameter '+$u)}
        }
        $records += [pscustomobject]@{update=$u;actor_steps=$s.actorSteps;critic_steps=$s.criticSteps;
            new_actor_steps=$s.actorSteps-7168;new_critic_steps=$s.criticSteps-7168;
            next_sampling_index=$s.rng;samples=$s.actions;substeps=$s.attempted;checksum=$s.hash}
        $last=$s
    }
    if(![Phase5SavedAudit]::FloatEqual($first.actor,$parent.actor) -or
        ![Phase5SavedAudit]::FloatEqual($parentActor.weights,$first.actor) -or
        ![Phase5SavedAudit]::FloatEqual($first.critic,$parent.critic) -or
        ![Phase5SavedAudit]::Equal($first.nativeWorldBytes,$parent.nativeWorldBytes) -or
        ![Phase5SavedAudit]::Equal($first.optimizerBytes,$parent.optimizerBytes) -or
        $first.actions -ne 0 -or $first.attempted -ne 0 -or $first.startsCount -ne 0) {
        throw 'Fork initial state or counter rebase differs'
    }
    $latest=[Phase5SavedAudit]::ReadSession((Join-Path $study 'training/latest.checkpoint'))
    if($latest.hash -ne $last.hash){throw 'Latest checkpoint mismatch'}
    $manifest=Get-Content -LiteralPath (Join-Path $study 'manifest.json') -Raw | ConvertFrom-Json
    foreach($b in @($manifest.inputs)+@($manifest.sources)) {
        if((Get-FileHash -LiteralPath $b.path).Hash -ne $b.sha256){throw ('Bound file changed '+$b.path)}
    }
    if((Get-FileHash -LiteralPath 'flexible-tail-preview/addons/flexible_tail_live/bin/flexible_tail_live.dll').Hash -ne
        '6B665439A243A923B03ABF614C4B0825AA7A272F7DB533EC8D44A48A48B51942'){throw 'Live DLL changed'}
    $verification=Get-Content -LiteralPath (Join-Path $study 'verification/data/verification.json') -Raw | ConvertFrom-Json
    $fork=Get-Content -LiteralPath (Join-Path $study 'verification/data/fork-equivalence.json') -Raw | ConvertFrom-Json
    $continuous=[Phase5SavedAudit]::ReadSession((Join-Path $study 'verification/data/continuous-2.bin'))
    $resumed=[Phase5SavedAudit]::ReadSession((Join-Path $study 'verification/data/resumed-2.bin'))
    if(!$verification.pass -or !$fork.pass -or $continuous.hash -ne $resumed.hash -or
        $continuous.parentPayload -ne $parent.hash -or $continuous.rng -ne 114*64 -or
        $continuous.actorSteps -ne 7296 -or $verification.physical_updates_executed -ne 3 -or
        $verification.actual_actor_steps -ne 192 -or $verification.actual_critic_steps -ne 192) {
        throw 'Fork resume verification failure'
    }
    $result=Get-Content -LiteralPath (Join-Path $study 'training/result.json') -Raw | ConvertFrom-Json
    if($result.stage -ne 'complete' -or $result.updates_completed -ne 128 -or
        $result.sampled_actions -ne $last.actions -or $result.attempted_substeps -ne $last.attempted -or
        $result.invocation_actor_steps -ne 8192 -or $result.invocation_critic_steps -ne 8192) {
        throw 'Final result accounting failure'
    }
    # Lane suffix offsets follow the unchanged native layout: 3376-byte World,
    # episode:uint32, angle:float, needs_start:bool, then alignment padding.
    # Validate that layout against both saved requests before deriving starts.
    if($first.laneSize -ne 3392){throw 'Unrecognized native Lane layout for curriculum audit'}
    $angleCounts=@{left15=0;right15=0;left30=0;right30=0}
    for($lane=0;$lane -lt 1024;$lane++) {
        $offset=$lane*$first.laneSize+$first.laneSize-16
        $oldEpisode=[BitConverter]::ToUInt32($first.nativeWorldBytes,$offset)
        $newEpisode=[BitConverter]::ToUInt32($last.nativeWorldBytes,$offset)
        $initialAngle=[BitConverter]::ToSingle($first.nativeWorldBytes,$offset+4)
        if($oldEpisode -ne 39 -or [Math]::Abs([Math]::Abs($initialAngle)-[Math]::PI/6) -gt 1e-6 -or
            $newEpisode -lt $oldEpisode){throw 'Lane suffix validation failure'}
        $direction=if(($lane%2) -eq 0){'left'}else{'right'}
        for($e=$oldEpisode;$e -lt $newEpisode;$e++) {
            $angle=if((([int][Math]::Floor($lane/2)+$e)%2) -eq 1){30}else{15}
            $angleCounts[$direction+$angle]++
        }
    }
    if(($angleCounts.Values | Measure-Object -Sum).Sum -ne $last.startsCount -or
        $angleCounts.left15+$angleCounts.left30 -ne $last.left -or
        $angleCounts.right15+$angleCounts.right30 -ne $last.right) {throw 'Derived curriculum starts disagree with native counters'}
    $changed=@(0..31 | Where-Object {![Phase5SavedAudit]::BitsEqual($last.actor[$_],$parent.actor[$_])})
    $report=[ordered]@{schema=1;pass=$true;physics_steps=0;optimizer_updates=0;
        checkpoint_pairs_verified=$records.Count;phase_model_schema=2;session_schema=5;
        all_wrapped_checksums_valid=$true;all_lineage_and_source_bindings_valid=$true;
        parent_update=112;parent_payload_checksum=$parent.hash;parent_task_identity=$parent.task;
        initial_native_worlds_bit_exact=$true;initial_actor_bit_exact=$true;initial_critic_bit_exact=$true;
        initial_both_optimizers_bit_exact=$true;initial_diagnostic_counts_zero=$true;
        original_four_weights_bit_exact=$true;frozen_moments_zero=$true;unused_parameters_and_moments_zero=$true;
        changed_final_indices=$changed;latest_matches_final=$true;all_bound_inputs_and_sources_unchanged=$true;
        protected_live_library_unchanged=$true;exact_resume_pass=$verification.pass;
        resumed_payload_checksum=$resumed.hash;diagnostic_executed_updates=3;diagnostic_samples=196608;
        diagnostic_substeps=$verification.actual_attempted_substeps;diagnostic_actor_steps=192;diagnostic_critic_steps=192;
        new_updates=$last.update;new_samples=$last.actions;new_substeps=$last.attempted;
        imported_actor_steps=7168;imported_critic_steps=7168;new_actor_steps=8192;new_critic_steps=8192;
        final_actor_age=$last.actorSteps;final_critic_age=$last.criticSteps;next_sampling_index=$last.rng;
        new_starts=$last.startsCount;new_left_starts=$last.left;new_right_starts=$last.right;
        curriculum_starts_derived_from_saved_episode_positions=$angleCounts;records=$records}
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'report.json') -Encoding utf8
    $report.Remove('records');$report | ConvertTo-Json -Depth 6
} finally { Pop-Location }
