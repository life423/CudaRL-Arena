param([string]$Root=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference='Stop'
$curve=@();$selected=@();$angleMetrics=@();$caseOutcomes=@();$probes=@();$requestWindows=@()
function Average($rows,[string]$property){$values=@($rows|Where-Object {![string]::IsNullOrWhiteSpace($_.$property)}|ForEach-Object {[double]$_.$property});if($values.Count){return ($values|Measure-Object -Average).Average};return $null}
foreach($stage in Get-ChildItem $Root -Directory|Where-Object Name -Match '^stage-'){
    $assessments=@();$baseline=Join-Path $stage.FullName 'baseline/assessment'
    if(Test-Path (Join-Path $baseline 'result.json')){$assessments+=[pscustomobject]@{update=0;path=$baseline}}
    $eval=Join-Path $stage.FullName 'evaluations'
    if(Test-Path $eval){$assessments+=Get-ChildItem $eval -Directory|ForEach-Object {$where=Join-Path $_.FullName 'assessment';if(Test-Path (Join-Path $where 'result.json')){[pscustomobject]@{update=[int]($_.Name -replace 'update-','');path=$where}}}}
    foreach($a in $assessments){
        try{$report=Get-Content -Raw (Join-Path $a.path 'result.json')|ConvertFrom-Json}catch{continue}
        if($null -eq $report -or $null -eq $report.total_s){continue}
        $allWindows=Import-Csv (Join-Path $a.path 'windows.csv')
        $windows=$allWindows|Where-Object {$_.policy -eq 'candidate' -and $_.case -like 'phase_*'}
        $motion=@{15=0;30=0;60=0;90=0};$windowMap=@{};$referenceMap=@{}
        foreach($window in $allWindows){if($window.policy -eq 'candidate'){if(!$windowMap.ContainsKey($window.case)){$windowMap[$window.case]=@()};$windowMap[$window.case]+=$window}else{$referenceMap[$window.case+'|'+$window.window]=$window}}
        foreach($case in Import-Csv (Join-Path $a.path 'cases.csv')|Where-Object {$_.policy -eq 'candidate' -and $_.kind -eq 'turn'}){
            $angle=[int][Math]::Abs([double]$case.target_deg);$flags=[Collections.Generic.List[string]]::new();$strokeFlags=[Collections.Generic.List[string]]::new()
            if([int]$case.fault -ne 0){$flags.Add('fault_'+$case.fault)}
            if([double]$case.peak_tilt_deg -gt 10){$flags.Add('tilt')};if([double]$case.peak_depth_error_mm -gt 1){$flags.Add('depth')};if([double]$case.min_certified_clearance_mm -le 0){$flags.Add('clearance')}
            $caseWindows=@($windowMap[$case.case]);if($caseWindows.Count -ne 2){$flags.Add('missing_windows')}
            foreach($window in $caseWindows){
                if([int]$window.complete -ne 1 -or [string]::IsNullOrWhiteSpace($window.projected_mm_s)){$flags.Add('incomplete');continue}
                if([double]$window.course_error_deg -gt 5){$flags.Add('course')}
                if([double]$window.projected_mm_s -lt 6 -or [double]$window.min_rolling_mm_s -lt 6){$flags.Add('speed')}
                if([double]$window.achieved_deg*[double]$case.target_deg -le 0){$flags.Add('wrong_direction')}
                $reference=$referenceMap[$window.case+'|'+$window.window]
                foreach($axis in @('root','passive')){foreach($term in @('frequency_hz','swing_rad')){$name='main_'+$axis+'_'+$term;if([string]::IsNullOrWhiteSpace($window.$name) -or [double]$window.$name -lt .8*[double]$reference.$name){$strokeFlags.Add($axis+'_'+$term)}}}
            }
            if($flags.Count -eq 0){$motion[$angle]++}
            $caseOutcomes+=[pscustomobject]@{stage=$stage.Name;update=$a.update;case=$case.case;angle=$angle;direction=if([double]$case.target_deg -gt 0){'left'}else{'right'};motion_pass=($flags.Count -eq 0);legacy_pass=$case.passed;main_pass=$case.main_stroke_v1_passed;motion_failures=(@($flags|Sort-Object -Unique)-join '|');main_stroke_failures=(@($strokeFlags|Sort-Object -Unique)-join '|')}
        }
        $row=[pscustomobject]@{stage=$stage.Name;update=$a.update;main32=$report.candidate_main_stroke_turn_passes;main15=$report.candidate_main_15_passes;main30=$report.candidate_main_30_passes;main60=$report.candidate_main_60_passes;main90=$report.candidate_main_90_passes;legacy32=$report.candidate_turn_passes;legacy15=$report.candidate_15_passes;legacy30=$report.candidate_30_passes;legacy60=$report.candidate_60_passes;legacy90=$report.candidate_90_passes;swim=$report.candidate_straight_retention;from_rest=$report.candidate_from_rest_passes;all_inactive_commands_exact=$report.all_inactive_commands_exact;canonical_state_exact=$report.paired_retention_final_states_exact;raw_state_bytes_exact=$report.raw_navigation_bytes_exact;exact_pair_required=$report.retention_requires_exact_pair;canonical_schema=$report.navigation_state_comparison_schema;global_faults=$report.global_faults;mean_course_error_deg=(Average $windows 'course_error_deg');mean_projected_mm_s=(Average $windows 'projected_mm_s');eval_s=$report.total_s;assessment=$a.path}
        $curve+=$row
        foreach($angle in @(15,30,60,90)){$row|Add-Member NoteProperty ('motion'+$angle) $motion[$angle]}
        foreach($g in $windows|Group-Object case){
            $name=$g.Name;$parts=$name -split '_';$magnitude=[int]$parts[-1];$direction=$parts[-2]
            foreach($window in $g.Group){$angleMetrics+=[pscustomobject]@{stage=$stage.Name;update=$a.update;case=$name;angle=$magnitude;direction=$direction;window=$window.window;complete=$window.complete;error_deg=$window.course_error_deg;achieved_deg=$window.achieved_deg;projected_mm_s=$window.projected_mm_s;rolling_min_mm_s=$window.min_rolling_mm_s;root_hz=$window.root_frequency_hz;passive_legacy_hz=$window.passive_frequency_hz;main_root_hz=$window.main_root_frequency_hz;main_root_swing=$window.main_root_swing_rad;main_passive_hz=$window.main_passive_frequency_hz;main_passive_swing=$window.main_passive_swing_rad;legacy_stroke_pass=$window.stroke_pass;main_stroke_pass=$window.main_stroke_v1_pass}}
        }
    }
    $selection=Join-Path $stage.FullName 'selection.json'
    if(Test-Path $selection){
        $s=(Get-Content -Raw $selection|ConvertFrom-Json).selected
        if($s.PSObject.Properties.Name -contains 'study'){
            $training=Join-Path $stage.FullName 'training/result.json';$t=if(Test-Path $training){Get-Content -Raw $training|ConvertFrom-Json}else{$null}
            $selected+=[pscustomobject]@{stage=$stage.Name;study=$s.study;material=$s.material;selected_update=$s.update;main15=$s.gentle;main30=$s.thirty;main60=$s.sixty;main90=$s.ninety;main32=$s.main;legacy32=$s.legacy;retained_swim=$s.physical;mean_course_error_deg=$s.error;completed_updates=$t.updates_completed;training_actions=$t.sampled_actions;attempted_substeps=$t.attempted_substeps;accepted_substeps=$t.accepted_substeps;gpu_physics_ms=$t.physics_and_observation_gpu_ms;gpu_network_ms=$t.network_gpu_ms;gpu_optimizer_ms=$t.optimization_gpu_ms;evidence_ms=$t.evidence_writing_wall_ms;trainer_wall_s=$t.total_session_wall_s;actor=$s.actor;checkpoint=$s.checkpoint}
        }
    }
    $supplement=Join-Path $stage.FullName 'final/assessment/program-probes.json'
    if(Test-Path $supplement){
        try{$report=Get-Content -Raw $supplement|ConvertFrom-Json}catch{continue}
        if($null -eq $report -or $null -eq $report.physics_and_writing_s){continue}
        $probes+=[pscustomobject]@{stage=$stage.Name;scheduled_requests=$report.scheduled_requests;completed_requests=$report.completed_requests;motion_pass_requests=$report.motion_pass_requests;faulted_cases=$report.faulted_cases;attempted_substeps=$report.attempted_substeps;accepted_substeps=$report.accepted_substeps;wall_s=$report.physics_and_writing_s}
        $p=Join-Path $stage.FullName 'final/assessment/program-probe-windows.csv'
        if(Test-Path $p){foreach($row in Import-Csv $p){$requestWindows+=[pscustomobject]@{stage=$stage.Name;case=$row.case;leg=$row.leg;window=$row.window;requested_deg=$row.request_deg;course_error_deg=$row.error_deg;projected_mm_s=$row.projected_mm_s;achieved_deg=$row.achieved_deg;course_rate_deg_s=$row.measured_rate_deg_s;motion_pass=$row.motion_pass;main_root_hz=$row.main_root_hz;main_root_swing=$row.main_root_swing;main_passive_hz=$row.main_passive_hz;main_passive_swing=$row.main_passive_swing;archived_stroke_reference_pass=$row.archived_reference_main_pass}}}
    }
}
$curve|Export-Csv (Join-Path $PSScriptRoot 'motor-learning-curves.csv') -NoTypeInformation
$selected|Export-Csv (Join-Path $PSScriptRoot 'motor-selected.csv') -NoTypeInformation
$angleMetrics|Export-Csv (Join-Path $PSScriptRoot 'motor-angle-windows.csv') -NoTypeInformation
$caseOutcomes|Export-Csv (Join-Path $PSScriptRoot 'motor-case-outcomes.csv') -NoTypeInformation
$probes|Export-Csv (Join-Path $PSScriptRoot 'motor-supplements.csv') -NoTypeInformation
$requestWindows|Export-Csv (Join-Path $PSScriptRoot 'motor-request-windows.csv') -NoTypeInformation
[ordered]@{schema=1;source='existing native study outputs only';analysis_physics_steps=0;analysis_optimizer_updates=0;legacy_score_preserved=$true;mainstroke_schema=1;mainstroke_prominence_rad=.1;selected=$selected;supplements=$probes}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $PSScriptRoot 'summary.json')
$selected|ConvertTo-Json -Depth 4 -Compress
