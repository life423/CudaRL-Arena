param([string]$MotorRoot=(Join-Path $PSScriptRoot '../seven-options-v6-corrected'),[string]$Continuation=(Join-Path $PSScriptRoot '../seven-options-v6-bank-resume'))
$ErrorActionPreference='Stop'
$stages=@();$chooser=@()
function Avg($rows,$property){$numbers=@($rows|Where-Object {![string]::IsNullOrWhiteSpace($_.$property)}|ForEach-Object {[double]$_.$property});if($numbers.Count){return ($numbers|Measure-Object -Average).Average};return $null}
function Motion-Pass($case,$windows){
    if([int]$case.fault -ne 0 -or [double]$case.peak_tilt_deg -gt 10 -or [double]$case.peak_depth_error_mm -gt 1 -or [double]$case.min_certified_clearance_mm -le 0 -or $windows.Count -ne 2){return $false}
    foreach($w in $windows){if([int]$w.complete -ne 1 -or [string]::IsNullOrWhiteSpace($w.projected_mm_s) -or [double]$w.projected_mm_s -lt 6 -or [double]$w.min_rolling_mm_s -lt 6 -or [double]$w.course_error_deg -gt 5 -or [double]$w.achieved_deg*[double]$case.target_deg -le 0){return $false}}
    return $true
}
$directories=@(Get-ChildItem $MotorRoot -Directory|Where-Object Name -Match '^stage-0[1-5]-')
if(Test-Path $Continuation){$directories+=Get-ChildItem $Continuation -Directory|Where-Object Name -Match '^stage-07-'}
foreach($directory in $directories){
    $selection=Join-Path $directory.FullName 'selection.json';if(!(Test-Path $selection)){continue}
    $s=(Get-Content -Raw $selection|ConvertFrom-Json).selected
    $assessment=$s.assessment;$summary=Get-Content -Raw (Join-Path $assessment result.json)|ConvertFrom-Json
    $cases=@(Import-Csv (Join-Path $assessment cases.csv)|Where-Object {$_.policy -eq 'candidate' -and $_.kind -eq 'turn'})
    $windows=@(Import-Csv (Join-Path $assessment windows.csv)|Where-Object {$_.policy -eq 'candidate' -and $_.case -like 'phase_*'})
    $trainingPath=Join-Path $directory.FullName training/result.json;$training=Get-Content -Raw $trainingPath|ConvertFrom-Json
    $angles=@()
    foreach($angle in @(15,30,60,90)){
        $ac=@($cases|Where-Object {[Math]::Abs([double]$_.target_deg) -eq $angle});$aw=@($windows|Where-Object {$_.case -match "_${angle}$"})
        $motion=0;$legacy=0;$main=0;$leftMotion=0;$rightMotion=0
        foreach($case in $ac){$cw=@($aw|Where-Object case -eq $case.case);if(Motion-Pass $case $cw){$motion++;if([double]$case.target_deg -gt 0){$leftMotion++}else{$rightMotion++}};$legacy+=[int]$case.passed;$main+=[int]$case.main_stroke_v1_passed}
        $angles+=[pscustomobject]@{degrees=$angle;cases=$ac.Count;motion_passes=$motion;left_motion=$leftMotion;right_motion=$rightMotion;main_stroke_passes=$main;legacy_passes=$legacy;mean_window_course_error_deg=(Avg $aw course_error_deg);mean_projected_speed_mm_s=(Avg $aw projected_mm_s);mean_initial_response_s=(Avg $ac response_s);mean_effective_turn_radius_mm=(Avg $ac effective_complete_beat_radius_mm);settled_1s_measurements=@($ac|Where-Object {![string]::IsNullOrWhiteSpace($_.target_1s_begin_s)}).Count;mean_main_root_hz=(Avg $aw main_root_frequency_hz);mean_main_passive_hz=(Avg $aw main_passive_frequency_hz);mean_main_root_swing_rad=(Avg $aw main_root_swing_rad);mean_main_passive_swing_rad=(Avg $aw main_passive_swing_rad)}
    }
    $supplementPath=Join-Path $directory.FullName final/assessment/program-probes.json
    $supplement=$null
    if(Test-Path $supplementPath){
        $proof=Get-Content -Raw $supplementPath|ConvertFrom-Json
        $pc=@(Import-Csv (Join-Path $directory.FullName final/assessment/program-probes.csv))
        $pw=@(Import-Csv (Join-Path $directory.FullName final/assessment/program-probe-windows.csv))
        $groups=@()
        foreach($kind in @('speed','rate','alternating','bank')){
            $group=@($pc|Where-Object {$_.case -like ($kind+'*')})
            $groups+=[pscustomobject]@{kind=$kind;cases=$group.Count;scheduled_requests=($group.scheduled_legs|ForEach-Object {[int]$_}|Measure-Object -Sum).Sum;completed_requests=($group.completed_legs|ForEach-Object {[int]$_}|Measure-Object -Sum).Sum;passed_requests=($group.passed_legs|ForEach-Object {[int]$_}|Measure-Object -Sum).Sum;clearance_guards=@($group|Where-Object fault -eq '2').Count;cases_detail=$group;windows_detail=@($pw|Where-Object {$_.case -like ($kind+'*')})}
        }
        $supplement=[pscustomobject]@{scheduled_requests=$proof.scheduled_requests;completed_requests=$proof.completed_requests;motion_pass_requests=$proof.motion_pass_requests;attempted_substeps=$proof.attempted_substeps;accepted_substeps=$proof.accepted_substeps;groups=$groups;qualification='motion criteria; archived stroke comparison is diagnostic and not an exact same-request reference'}
    }
    $stages+=[pscustomobject]@{stage=$directory.Name;study=$s.study;material=$s.material;completed_updates=$training.updates_completed;selected_update=$s.update;training_stage=$training.stage;training_actions=$training.sampled_actions;attempted_substeps=$training.attempted_substeps;trainer_wall_s=$training.total_session_wall_s;gpu_physics_ms=$training.physics_and_observation_gpu_ms;gpu_network_ms=$training.network_gpu_ms;gpu_ppo_ms=$training.optimization_gpu_ms;retained_swim=$summary.candidate_straight_retention;from_rest=$summary.candidate_from_rest_passes;holding_physically_tested=$summary.station_holding_tested;canonical_pair_required=$summary.retention_requires_exact_pair;canonical_pair_exact=$summary.paired_retention_final_states_exact;angles=$angles;supplement=$supplement;actor=$s.actor;checkpoint=$s.checkpoint;assessment=$assessment}
}
if(Test-Path $Continuation){
    $bank=Join-Path $Continuation stage-06-bank
    if(Test-Path (Join-Path $bank evaluations)){
        foreach($eval in Get-ChildItem (Join-Path $bank evaluations) -Directory){
            $path=Join-Path $eval.FullName assessment/summary.json;if(!(Test-Path $path)){continue}
            try{$r=Get-Content -Raw $path|ConvertFrom-Json}catch{continue};if($null -eq $r.internal_elapsed_s){continue}
            $rows=@(Import-Csv (Join-Path $eval.FullName assessment/cases.csv))
            $chooser+=[pscustomobject]@{update=[int]($eval.Name -replace 'update-','');pass=$r.pass;diagnostic_passes=$r.diagnostic_passes;diagnostic_cases=$r.diagnostic_cases;endurance_passes=$r.endurance_passes;endurance_attempts=$r.endurance_attempts;union_coverage_fraction=$r.diagnostic_coverage_fraction;visited_union_cells=$r.diagnostic_visited_cells;estimated_map_cells=$r.reachable_50mm_cells_estimate;guard_stops=$r.clearance_stops;other_native_faults=$r.other_native_faults;attempted_substeps=$r.attempted_substeps;accepted_substeps=$r.accepted_substeps;mean_case_speed_mm_s=(Avg $rows mean_post_launch_speed_mm_s);maximum_case_stall_s=($rows.max_stall_s|ForEach-Object {[double]$_}|Measure-Object -Maximum).Maximum;maximum_case_perimeter_laps=($rows.perimeter_net_revolutions|ForEach-Object {[double]$_}|Measure-Object -Maximum).Maximum;assessment=(Join-Path $eval.FullName assessment)}
        }
    }
}
$chooserSelection=if(Test-Path (Join-Path $Continuation stage-06-bank/selection.json)){(Get-Content -Raw (Join-Path $Continuation stage-06-bank/selection.json)|ConvertFrom-Json).selected}else{$null}
$chooserTraining=if(Test-Path (Join-Path $Continuation stage-06-bank/training/result.json)){Get-Content -Raw (Join-Path $Continuation stage-06-bank/training/result.json)|ConvertFrom-Json}else{$null}
$bodyPond=if(Test-Path (Join-Path $Continuation final-body-pond/assessment/summary.json)){Get-Content -Raw (Join-Path $Continuation final-body-pond/assessment/summary.json)|ConvertFrom-Json}else{$null}
$result=[ordered]@{schema=1;read_only_analysis=$true;analysis_physics_steps=0;analysis_optimizer_updates=0;main_stroke_schema=1;main_stroke_prominence_rad=.1;legacy_score_preserved=$true;stages=$stages;chooser_learning_curve=$chooser;chooser_selected=$chooserSelection;chooser_training=$chooserTraining;final_soft_body_pond=$bodyPond;limitations=@('Motion passage is distinct from the declared rhythm qualification.','Coverage fraction is a union over separate diagnostic starts, not single-creature continuous coverage.','Clearance guard stops are not measured collisions.','Station holding was not physically retested by this program.','Finite tests do not prove indefinite swimming.');sources=@($MotorRoot,$Continuation)}
$result|ConvertTo-Json -Depth 18|Set-Content (Join-Path $PSScriptRoot results.json)
$flat=foreach($s in $stages){foreach($angle in $s.angles){[pscustomobject]@{stage=$s.stage;completed_updates=$s.completed_updates;selected_update=$s.selected_update;degrees=$angle.degrees;motion_passes=$angle.motion_passes;main_stroke_passes=$angle.main_stroke_passes;legacy_passes=$angle.legacy_passes;mean_error_deg=$angle.mean_window_course_error_deg;mean_speed_mm_s=$angle.mean_projected_speed_mm_s;mean_initial_response_s=$angle.mean_initial_response_s;mean_effective_radius_mm=$angle.mean_effective_turn_radius_mm;swim=$s.retained_swim;trainer_wall_s=$s.trainer_wall_s}}}
$flat|Export-Csv (Join-Path $PSScriptRoot selected-motor-metrics.csv) -NoTypeInformation
$chooser|Export-Csv (Join-Path $PSScriptRoot chooser-learning-curve.csv) -NoTypeInformation
$lines=[Collections.Generic.List[string]]::new();$lines.Add('# Seven-option program: measured status');$lines.Add('');$lines.Add('Read-only aggregation; zero added physics or optimizer updates. Motion / main-stroke / legacy are separate counts, each out of 8 cases per angle.');$lines.Add('');$lines.Add('| Stage | Updates; selected | 15° motion/main/legacy | 30° | 60° | 90° | Swimming |');$lines.Add('|---|---|---|---|---|---|---|')
foreach($s in $stages){$cells=@($s.angles|ForEach-Object {"$($_.motion_passes)/$($_.main_stroke_passes)/$($_.legacy_passes)"});$lines.Add("|$($s.stage)|$($s.completed_updates);$($s.selected_update)|$($cells[0])|$($cells[1])|$($cells[2])|$($cells[3])|$($s.retained_swim)/12|")}
$lines.Add('');$lines.Add('The selected nonlinear motor completed all eight 60° movement cases while the strict matched passive-frequency criterion still failed. Its principal passive rhythm after alignment was about 1 Hz versus the still-underturning reference around 1.7 Hz; this score failure was preserved. This is actual moving-turn progress, not full rhythm qualification. No selected motor completed all 90° movement cases.');$lines.Add('');$lines.Add('| Stage | Speed | Rate | Alternating requests | Bank requests |');$lines.Add('|---|---|---|---|---|')
foreach($s in $stages){if($s.supplement){$groups=$s.supplement.groups;$cells=@($groups|ForEach-Object {"$($_.passed_requests)/$($_.scheduled_requests)"});$lines.Add("|$($s.stage)|$($cells[0])|$($cells[1])|$($cells[2])|$($cells[3])|")}}
$lines.Add('');$lines.Add('Alternating requests are two uninterrupted 30°/60°/90° sequences. Later guard stops remain in the denominator. Bank probes are motor request tests and do not establish autonomous navigation. Main-stroke comparisons for supplemental requests are archived-reference diagnostics.');$lines.Add('');$lines.Add('| Chooser update | Pond diagnostic | Endurance | Union coverage | Guard stops |');$lines.Add('|---|---|---|---|---|')
foreach($c in $chooser){$pct=([double]$c.union_coverage_fraction*100).ToString('F1');$lines.Add("|$($c.update)|$($c.diagnostic_passes)/12|$($c.endurance_passes)/$($c.endurance_attempts)|$pct%|$($c.guard_stops)|")}
$lines.Add('');if($bodyPond){$lines.Add("The selected soft-body motor retained 12/12 swimming cases, but the frozen chooser passed only $($bodyPond.diagnostic_passes)/12 pond cases with $($bodyPond.clearance_stops) guard stops, versus 8/12 and four stops on the baseline body. Endurance was not attempted. The body variant is not a navigation improvement under this comparison.")}
$lines.Add('');$lines.Add('Strongest selected baseline-body 60° movement: 8/8 cases, mean course error 1.08°, projected speed 13.82 mm/s, initial response 1.92 s and effective complete-beat radius 165.3 mm. The soft-body selection passed 8/8 main-stroke 30° cases at 13.31 mm/s, 1.34° error, 1.80 s initial response and 259.0 mm effective radius; it passed only 4/8 movement cases at 60°. Initial response is not settled completion time.');
$lines.Add('');$lines.Add('Speed matching remained partial: the selected baseline motor delivered about 8.40 / 9.23 / 10.64 mm/s for 6 / 8 / 12 mm/s requests, passing the 8 and 12 requests on both starts. The soft body delivered 12.75 / 13.06 / 13.52 mm/s, passing only the 12 request. Neither reliably matched the slowest request.');
$lines.Add('');$lines.Add('Holding was not physically retested. Endurance remains untested whenever all 12 pond diagnostics have not qualified. Coverage is the union of separate starts, not one animal exploring the full map. Original working Godot assets remain protected until the deployment gates pass.');$lines|Set-Content (Join-Path $PSScriptRoot README.md)
$stages|Select-Object stage,completed_updates,selected_update,retained_swim,trainer_wall_s|ConvertTo-Json -Compress
