param([Parameter(Mandatory=$true)][string]$Study)
$ErrorActionPreference='Stop'
$analysis=Join-Path $Study 'analysis'
$reports=@([pscustomobject]@{update=0; path=(Join-Path $Study 'baseline/assessment')})
$reports+=Get-ChildItem (Join-Path $Study 'evaluations') -Directory | ForEach-Object { $assessment=Join-Path $_.FullName 'assessment'; if(Test-Path (Join-Path $assessment 'result.json')){[pscustomobject]@{update=[int]($_.Name -replace 'update-','');path=$assessment}} }
$curve=@();$allcase=@();$grouped=@()
function Num($s){if([string]::IsNullOrWhiteSpace($s)){return $null};return [double]::Parse($s,[Globalization.CultureInfo]::InvariantCulture)}
function Ratio($value,$base){$b=Num $base;if($b -gt 0){return (Num $value)/$b};return $null}
function Mean($values){return ($values|Measure-Object -Average).Average}
foreach($entry in $reports){
  $report=Get-Content -Raw (Join-Path $entry.path 'result.json') | ConvertFrom-Json
  $cases=Import-Csv (Join-Path $entry.path 'cases.csv')
  $windows=Import-Csv (Join-Path $entry.path 'windows.csv')
  $baselineWindows=@{};foreach($row in $windows){if($row.policy -ne 'candidate'){$baselineWindows[($row.case+'|'+$row.window)]=$row}}
  $windowMap=@{};foreach($row in $windows){if($row.policy -eq 'candidate'){if(-not $windowMap.ContainsKey($row.case)){$windowMap[$row.case]=@()};$windowMap[$row.case]+=$row}}
  $metrics=@()
  foreach($case in @($cases|Where-Object {$_.policy -eq 'candidate' -and $_.kind -eq 'turn'})){
    $rows=@($windowMap[$case.case]);$flags=[Collections.Generic.List[string]]::new()
    if([int]$case.fault -ne 0){$flags.Add('fault_'+$case.fault)}
    if([double]$case.peak_tilt_deg -gt 10){$flags.Add('tilt')};if([double]$case.peak_depth_error_mm -gt 1){$flags.Add('depth')}
    if([double]$case.min_certified_clearance_mm -le 0){$flags.Add('clearance')}
    foreach($row in $rows){
      $base=$baselineWindows[($row.case+'|'+$row.window)]
      if([int]$row.complete -ne 1){$flags.Add('incomplete')}
      if([double]$row.course_error_deg -gt 5){$flags.Add('course')}
      if([double]$row.projected_mm_s -lt 6 -or [double]$row.min_rolling_mm_s -lt 6){$flags.Add('speed')}
      if(([double]$row.achieved_deg)*([double]$case.target_deg) -le 0){$flags.Add('wrong_direction')}
      $rf=Ratio $row.root_frequency_hz $base.root_frequency_hz;$rs=Ratio $row.root_swing_rad $base.root_swing_rad
      $pf=Ratio $row.passive_frequency_hz $base.passive_frequency_hz;$ps=Ratio $row.passive_swing_rad $base.passive_swing_rad
      if([double]$row.root_complete_beats -le 0 -or $rf -lt .8){$flags.Add('root_frequency')};if($rs -lt .8){$flags.Add('root_swing')}
      if([double]$row.passive_complete_beats -le 0 -or $pf -lt .8){$flags.Add('passive_frequency')};if($ps -lt .8){$flags.Add('passive_swing')}
      $metrics+=[pscustomobject]@{update=$entry.update;case=$case.case;angle=[Math]::Abs([double]$case.target_deg);direction=if([double]$case.target_deg -gt 0){'left'}else{'right'};window=$row.window;passed=[int]$case.passed;error_deg=(Num $row.course_error_deg);achieved_deg=(Num $row.achieved_deg);projected_mm_s=(Num $row.projected_mm_s);rolling_min_mm_s=(Num $row.min_rolling_mm_s);root_frequency_ratio=$rf;root_swing_ratio=$rs;passive_frequency_ratio=$pf;passive_swing_ratio=$ps;stroke_pass=[int]$row.stroke_pass}
    }
    $flags=@($flags|Select-Object -Unique)
    $allcase+=[pscustomobject]@{update=$entry.update;case=$case.case;angle=[Math]::Abs([double]$case.target_deg);direction=if([double]$case.target_deg -gt 0){'left'}else{'right'};passed=[int]$case.passed;motion_pass=[int](@($flags|Where-Object {$_ -notmatch 'frequency|swing'}).Count -eq 0);failure_flags=($flags -join '|');peak_tilt_deg=(Num $case.peak_tilt_deg);peak_depth_mm=(Num $case.peak_depth_error_mm);clearance_mm=(Num $case.min_certified_clearance_mm);response_s=(Num $case.response_s);target_1s_begin_s=(Num $case.target_1s_begin_s);radius_mm=(Num $case.effective_complete_beat_radius_mm)}
  }
  foreach($g in @($metrics|Group-Object angle,direction,window)){
    $rows=$g.Group;$first=$rows[0]
    $grouped+=[pscustomobject]@{update=$entry.update;angle=$first.angle;direction=$first.direction;window=$first.window;cases=$rows.Count;mean_error_deg=(Mean $rows.error_deg);max_error_deg=($rows.error_deg|Measure-Object -Maximum).Maximum;mean_achieved_deg=(Mean $rows.achieved_deg);mean_projected_mm_s=(Mean $rows.projected_mm_s);min_projected_mm_s=($rows.projected_mm_s|Measure-Object -Minimum).Minimum;min_rolling_mm_s=($rows.rolling_min_mm_s|Measure-Object -Minimum).Minimum;min_root_frequency_ratio=($rows.root_frequency_ratio|Measure-Object -Minimum).Minimum;min_root_swing_ratio=($rows.root_swing_ratio|Measure-Object -Minimum).Minimum;min_passive_frequency_ratio=($rows.passive_frequency_ratio|Measure-Object -Minimum).Minimum;min_passive_swing_ratio=($rows.passive_swing_ratio|Measure-Object -Minimum).Minimum}
  }
  $mine=@($allcase|Where-Object update -eq $entry.update)
  $curve+=[pscustomobject]@{update=$entry.update;full32=$report.candidate_turn_passes;gentle15=$report.candidate_15_passes;lesson30=$report.candidate_30_passes;untrained60=$report.candidate_60_passes;untrained90=$report.candidate_90_passes;left=$report.candidate_left_passes;right=$report.candidate_right_passes;motion15=@($mine|Where-Object {$_.angle -eq 15 -and $_.motion_pass -eq 1}).Count;motion30=@($mine|Where-Object {$_.angle -eq 30 -and $_.motion_pass -eq 1}).Count;swim=$report.candidate_straight_retention;from_rest=$report.candidate_from_rest_passes;max_tilt_deg=($mine.peak_tilt_deg|Measure-Object -Maximum).Maximum;max_depth_mm=($mine.peak_depth_mm|Measure-Object -Maximum).Maximum;min_clearance_mm=($mine.clearance_mm|Measure-Object -Minimum).Minimum;global_faults=$report.global_faults;eval_s=$report.total_s;qualified30=($report.candidate_30_passes -eq 8 -and $report.candidate_15_passes -eq 8 -and $report.candidate_straight_retention -eq 12 -and $report.global_faults -eq 0)}
}
$curve|Export-Csv (Join-Path $analysis 'learning-curve.csv') -NoTypeInformation
$allcase|Export-Csv (Join-Path $analysis 'case-failures.csv') -NoTypeInformation
$grouped|Export-Csv (Join-Path $analysis 'turn-metrics.csv') -NoTypeInformation
$summary=[ordered]@{schema=1;physics_steps=0;optimizer_updates=0;source='existing session outputs only';lesson30_gate='all8 fixed30 plus all8 fixed15,12straight retention and0globalfaults';qualified_updates=@($curve|Where-Object qualified30|ForEach-Object update);best30_updates=@($curve|Where-Object lesson30 -eq ($curve.lesson30|Measure-Object -Maximum).Maximum|ForEach-Object update);curve=$curve}
$summary|ConvertTo-Json -Depth 8|Set-Content (Join-Path $analysis 'turn-analysis.json')
$selected=@(0,8,16,112,($curve|Select-Object -Last 1).update)|Select-Object -Unique
$updates=Import-Csv (Join-Path $Study 'training/updates.csv')
$telemetry=@()
foreach($entry in @($reports|Where-Object {$_.update -in $selected})){
  $weights=$updates|Where-Object {[int]$_.update -eq $entry.update}|Select-Object -First 1
  $wPhase=if($entry.update -eq 0){0.}else{[double]$weights.w_root_phase_position};$wRate=if($entry.update -eq 0){0.}else{[double]$weights.w_root_phase_velocity}
  $rows=Import-Csv (Join-Path $entry.path 'controls.csv') | Where-Object {$_.policy -eq 'candidate' -and $_.case -match '_(15|30|60|90)$' -and (([int]$_.control-1)%4) -eq 0}
  foreach($g in @($rows|Group-Object case)){
    foreach($segment in @(@('early',0,2),@('middle',2,6),@('late_turn',11,13),@('onward',13,15))){
      $part=@($g.Group|Where-Object {[double]$_.time_s -gt $segment[1] -and [double]$_.time_s -le $segment[2]});if($part.Count -eq 0){continue}
      $baseSquared=0.;$learnedSquared=0.;$totalSquared=0.;$phaseSquared=0.;$velocitySquared=0.;$crossPhase=0.;$crossVelocity=0.;$baseSum=0.;$learnedSum=0.;$offset=0.;$directPhaseSquared=0.;$otherSquared=0.;$directPhaseSum=0.
      foreach($row in $part){$b=[double]$row.base_steering_latent;$l=[double]$row.learned_steering_latent;$t=[double]$row.total_steering_latent;$p=[double]$row.decision_root_phase_position;$v=[double]$row.decision_root_phase_velocity;$direct=$wPhase*$p+$wRate*$v;$directPhaseSquared+=$direct*$direct;$otherSquared+=($l-$direct)*($l-$direct);$directPhaseSum+=$direct;$baseSquared+=$b*$b;$learnedSquared+=$l*$l;$totalSquared+=$t*$t;$phaseSquared+=$p*$p;$velocitySquared+=$v*$v;$crossPhase+=$l*$p;$crossVelocity+=$l*$v;$baseSum+=$b;$learnedSum+=$l;$offset+=[double]$row.applied_yaw_offset}
      $n=$part.Count
      $telemetry+=[pscustomobject]@{update=$entry.update;case=$g.Name;window=$segment[0];decision_rows=$n;base_mean=$baseSum/$n;base_rms=[Math]::Sqrt($baseSquared/$n);learned_mean=$learnedSum/$n;learned_rms=[Math]::Sqrt($learnedSquared/$n);direct_phase_mean=$directPhaseSum/$n;direct_phase_rms=[Math]::Sqrt($directPhaseSquared/$n);other_learned_rms=[Math]::Sqrt($otherSquared/$n);total_rms=[Math]::Sqrt($totalSquared/$n);learned_to_base_rms=if($baseSquared -gt 0){[Math]::Sqrt($learnedSquared/$baseSquared)}else{$null};learned_phase_position_cosine=if($learnedSquared*$phaseSquared -gt 0){$crossPhase/[Math]::Sqrt($learnedSquared*$phaseSquared)}else{$null};learned_phase_velocity_cosine=if($learnedSquared*$velocitySquared -gt 0){$crossVelocity/[Math]::Sqrt($learnedSquared*$velocitySquared)}else{$null};mean_applied_offset=$offset/$n}
    }
  }
}
$telemetry|Export-Csv (Join-Path $analysis 'phase-contribution.csv') -NoTypeInformation
$updates|Where-Object {[int]$_.update -in $selected}|Select-Object update,w_course,w_heading,w_previous_offset,w_yaw_rate,w_full_error90,w_root_phase_position,w_root_phase_velocity,w_body_lateral_speed,w_error_speed,w_error_phase_cos2,w_error_phase_sin2,w_error2_phase_position,w_error2_phase_velocity|Export-Csv (Join-Path $analysis 'selected-weights.csv') -NoTypeInformation
$summary|ConvertTo-Json -Depth 5 -Compress
