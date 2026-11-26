param([string]$Output='build/flexible-tail/shoreline-warning-v1')
$ErrorActionPreference='Stop'
$study=(Resolve-Path -LiteralPath $Output).Path
$rows=@(Import-Csv -LiteralPath (Join-Path $study 'data/branches.csv'))
$prefixes=@(Import-Csv -LiteralPath (Join-Path $study 'data/prefixes.csv'))
$cues=@(Import-Csv -LiteralPath (Join-Path $study 'data/cues.csv'))
$summary=Get-Content -LiteralPath (Join-Path $study 'data/summary.json') -Raw|ConvertFrom-Json
if($rows.Count -ne 180 -or $prefixes.Count -ne 12 -or $cues.Count -ne 36 -or
    @($cues|Where-Object captured -ne '1').Count -or @($rows|Where-Object fault -notin @('0','2')).Count){throw 'Incomplete or invalid warning audit'}
$attempted=($rows|Measure-Object attempted_substeps -Sum).Sum+($prefixes|Measure-Object attempted -Sum).Sum
$accepted=($rows|Measure-Object accepted_substeps -Sum).Sum+($prefixes|Measure-Object accepted -Sum).Sum
if($attempted -ne $summary.attempted_substeps -or $accepted -ne $summary.accepted_substeps){throw 'Warning work ledger differs'}
$table=@(foreach($h in @(12,20,24)){foreach($rate in @(-4,4,-8,8,0)){
    $slice=@($rows|Where-Object {[int]$_.warning_horizon_s -eq $h -and [int]$_.rate_deg_s -eq $rate})
    [pscustomobject]@{warning_s=$h;request_deg_s=$rate;cases=$slice.Count;inward_before_stop=@($slice|Where-Object inward_s -ne '').Count;
        safe_40s=@($slice|Where-Object safe_40s -eq '1').Count;clearance_stops=@($slice|Where-Object fault -eq '2').Count;
        mean_actual_speed_mm_s=($slice|ForEach-Object {[double]$_.mean_COM_speed_mm_s}|Measure-Object -Average).Average}
}})
$table|Export-Csv -LiteralPath (Join-Path $study 'rate-summary.csv') -NoTypeInformation
$warning12=@($cues|Where-Object horizon_s -eq '12')
$turns=@($rows|Where-Object {$_.warning_horizon_s -eq '24' -and $_.rate_deg_s -in @('-4','4') -and $_.turn_90deg_s -ne ''})
$current=@($rows|Where-Object {$_.warning_horizon_s -eq '12' -and $_.rate_deg_s -in @('-4','4')})
$findings=[ordered]@{complete=$true;current_12s_moderate_turns_safe=@($current|Where-Object safe_40s -eq '1').Count;current_12s_moderate_turns_attempted=$current.Count;
    current_12s_moderate_turns_inward=@($current|Where-Object inward_s -ne '').Count;
    first_warning_actual_speed_mm_s_min=($warning12|Measure-Object first_speed_mm_s -Minimum).Minimum;
    first_warning_actual_speed_mm_s_max=($warning12|Measure-Object first_speed_mm_s -Maximum).Maximum;
    first_warning_range_mm_min=($warning12|Measure-Object first_range_mm -Minimum).Minimum;
    first_warning_range_mm_max=($warning12|Measure-Object first_range_mm -Maximum).Maximum;
    maximum_decision_delay_s=($warning12|Measure-Object decision_delay_s -Maximum).Maximum;
    measured_90deg_moderate_turn_time_s_min=($turns|Measure-Object turn_90deg_s -Minimum).Minimum;
    measured_90deg_moderate_turn_time_s_max=($turns|Measure-Object turn_90deg_s -Maximum).Maximum;
    measured_effective_radius_mm_min=($turns|Measure-Object effective_radius_mm -Minimum).Minimum;
    measured_effective_radius_mm_max=($turns|Measure-Object effective_radius_mm -Maximum).Maximum;
    attempted_substeps=$attempted;accepted_substeps=$accepted;optimizer_updates=0;Godot_updated=$false;mouse_keyboard_calls=0;
    interpretation='The12s soft penalty is too late for these outward capability probes.20/24s helps some turns but is not a reliable universal fix. Constant-shape clearance loss is not measured terrain contact.'}
$findings|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $study findings.json) -Encoding utf8NoBOM
$table|Format-Table -AutoSize
$findings|ConvertTo-Json -Depth 6
