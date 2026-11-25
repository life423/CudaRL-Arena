$ErrorActionPreference='Stop'
$session=Split-Path -Parent $PSScriptRoot
function Average($Rows,[string]$Field){
    $values=@($Rows | Where-Object {$_.$Field -ne ''} | ForEach-Object {[double]$_.$Field})
    if($values.Count){return ($values | Measure-Object -Average).Average};return $null
}
$sources=@([pscustomobject]@{update=0;path=(Join-Path $session 'baseline/assessment')})
foreach($directory in Get-ChildItem (Join-Path $session 'evaluations') -Directory | Sort-Object Name){
    if(Test-Path (Join-Path $directory.FullName 'assessment/result.json')){
        $sources+=[pscustomobject]@{update=[int]($directory.Name -replace '^update-','');path=(Join-Path $directory.FullName 'assessment')}
    }
}
$curve=@();$groups=@();$classification=@()
foreach($source in $sources){
    $summary=Get-Content (Join-Path $source.path 'result.json') -Raw | ConvertFrom-Json
    $curve+=[pscustomobject]@{update=$source.update;cases=$summary.turn_cases_per_arm;left=$summary.candidate_left_passes;right=$summary.candidate_right_passes;full=$summary.candidate_turn_passes;at15=$summary.candidate_15_passes;at30=$summary.candidate_30_passes;at60=$summary.candidate_60_passes;at90=$summary.candidate_90_passes;swim=$summary.candidate_straight_retention;from_rest=$summary.candidate_from_rest_passes;faults=$summary.faulted_trajectories;attempted=$summary.attempted_substeps;accepted=$summary.accepted_substeps}
    $cases=@(Import-Csv (Join-Path $source.path 'cases.csv'))
    $windows=@(Import-Csv (Join-Path $source.path 'windows.csv'))
    foreach($case in $cases | Where-Object {$_.kind -eq 'turn'}){
        $wins=@($windows | Where-Object {$_.case -eq $case.case -and $_.policy -eq $case.policy})
        $reference=@($windows | Where-Object {$_.case -eq $case.case -and $_.policy -eq 'wholebeat56_baseline'})
        $raw=[int]$case.fault -eq 0 -and [int]$case.controls -eq 850 -and [double]$case.peak_tilt_deg -le 10 -and [double]$case.peak_depth_error_mm -le 1 -and [double]$case.min_certified_clearance_mm -gt 0
        $course=$true;$speed=$true;$root_swing=$true;$root_frequency=$true;$passive_swing=$true;$passive_frequency=$true
        foreach($win in $wins){
            $base=@($reference | Where-Object window -eq $win.window)[0]
            $course=$course -and [int]$win.complete -eq 1 -and [double]$win.course_error_deg -le 5 -and ([double]$win.achieved_deg * [double]$case.target_deg) -gt 0
            $speed=$speed -and [double]$win.projected_mm_s -ge 6 -and [double]$win.min_rolling_mm_s -ge 6
            $root_swing=$root_swing -and [double]$win.root_swing_rad -ge .8*[double]$base.root_swing_rad
            $root_frequency=$root_frequency -and [double]$win.root_frequency_hz -ge .8*[double]$base.root_frequency_hz
            $passive_swing=$passive_swing -and [double]$win.passive_swing_rad -ge .8*[double]$base.passive_swing_rad
            $passive_frequency=$passive_frequency -and [double]$win.passive_frequency_hz -ge .8*[double]$base.passive_frequency_hz
        }
        $late=@($wins | Where-Object window -eq 'late_turn_or_previous')[0]
        $progress=[double]$late.achieved_deg * [Math]::Sign([double]$case.target_deg)
        $classification+=[pscustomobject]@{update=$source.update;policy=$case.policy;case=$case.case;target=[double]$case.target_deg;full=[int]$case.passed;motion_envelope=($raw -and $course -and $speed);course=$course;speed=$speed;root_swing=$root_swing;root_frequency=$root_frequency;passive_swing=$passive_swing;passive_frequency=$passive_frequency;achieved=$progress;course_error=[double]$late.course_error_deg;projected_speed=[double]$late.projected_mm_s;underturn=($progress -lt [Math]::Abs([double]$case.target_deg)-5);overshoot=($progress -gt [Math]::Abs([double]$case.target_deg)+5)}
    }
    foreach($policy in @('wholebeat56_baseline','candidate')){
      foreach($angle in @(15,30,60,90)){
       foreach($direction in @('both','left','right')){
        $selected=@($cases | Where-Object {$_.policy -eq $policy -and $_.kind -eq 'turn' -and [Math]::Abs([double]$_.target_deg) -eq $angle -and ($direction -eq 'both' -or ($direction -eq 'left' -and [double]$_.target_deg -gt 0) -or ($direction -eq 'right' -and [double]$_.target_deg -lt 0))})
        $names=@($selected | Select-Object -ExpandProperty case)
        $late=@($windows | Where-Object {$_.policy -eq $policy -and $_.case -in $names -and $_.window -eq 'late_turn_or_previous'})
        $onward=@($windows | Where-Object {$_.policy -eq $policy -and $_.case -in $names -and $_.window -eq 'onward_or_final'})
        $classified=@($classification | Where-Object {$_.update -eq $source.update -and $_.policy -eq $policy -and $_.case -in $names})
        $groups+=[pscustomobject]@{update=$source.update;policy=$policy;angle=$angle;direction=$direction;cases=$selected.Count;full_passes=(@($selected|Where-Object passed -eq '1').Count);motion_envelope_passes=(@($classified|Where-Object motion_envelope).Count);late_error_deg=(Average $late 'course_error_deg');onward_error_deg=(Average $onward 'course_error_deg');late_achieved_deg=(Average $classified 'achieved');late_speed_mm_s=(Average $late 'projected_mm_s');onward_speed_mm_s=(Average $onward 'projected_mm_s');root_frequency_hz=(Average $late 'root_frequency_hz');root_swing_rad=(Average $late 'root_swing_rad');passive_frequency_hz=(Average $late 'passive_frequency_hz');passive_swing_rad=(Average $late 'passive_swing_rad');response_s=(Average $selected 'response_s');effective_radius_mm=(Average $selected 'effective_complete_beat_radius_mm');target_time_records=(@($selected|Where-Object target_1s_begin_s -ne '').Count)}
       }
      }
    }
}
$curve|Export-Csv (Join-Path $PSScriptRoot 'curve.csv') -NoTypeInformation
$groups|Export-Csv (Join-Path $PSScriptRoot 'groups.csv') -NoTypeInformation
$classification|Export-Csv (Join-Path $PSScriptRoot 'case-classification.csv') -NoTypeInformation
$report=[ordered]@{schema=1;read_only=$true;new_physics_steps=0;new_optimizer_updates=0;assessments=$curve;groups=$groups;case_classification=$classification;source='Immutable evaluator CSV and JSON; movement-only is diagnostic and does not replace full gates.'}
$report|ConvertTo-Json -Depth 8|Set-Content (Join-Path $PSScriptRoot 'report.json') -Encoding utf8
$curve|Format-Table -AutoSize
