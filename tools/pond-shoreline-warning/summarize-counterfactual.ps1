param([string]$Output='build/flexible-tail/chooser-counterfactual-v1')
$ErrorActionPreference='Stop'
$study=(Resolve-Path -LiteralPath $Output).Path
$rows=@(Import-Csv -LiteralPath (Join-Path $study 'data/branches.csv'))
$prefix=@(Import-Csv -LiteralPath (Join-Path $study 'data/prefix-parity.csv'))
$observations=@(Import-Csv -LiteralPath (Join-Path $study 'data/fork-observations.csv'))
$summary=Get-Content -LiteralPath (Join-Path $study 'data/summary.json') -Raw|ConvertFrom-Json
if($rows.Count -ne 234 -or $prefix.Count -ne 3 -or @($rows|Where-Object fault -notin @('0','2')).Count){throw 'Incomplete counterfactual data'}
$attempted=($rows|Measure-Object attempted -Sum).Sum+($prefix|Measure-Object attempted -Sum).Sum
$accepted=($rows|Measure-Object accepted -Sum).Sum+($prefix|Measure-Object accepted -Sum).Sum
if($attempted -ne $summary.attempted_substeps -or $accepted -ne $summary.accepted_substeps){throw 'Physical work accounting differs'}
if(@($rows|Where-Object useful_escape -eq '1').Count -ne $summary.useful_branches){throw 'Useful escape count differs'}
$table=@(foreach($fixture in 0..2){foreach($lead in @(20,12,4)){
    $slice=@($rows|Where-Object {[int]$_.fixture -eq $fixture -and [int]$_.lead_s -eq $lead})
    if($slice.Count -ne 26 -or @($slice.state_hash|Select-Object -Unique).Count -ne 1){throw 'Forks do not bind one exact state'}
    $source=$slice|Where-Object branch -eq 'source-policy';$installed=$slice|Where-Object branch -eq 'installed-policy'
    if($source.fault -ne '2'){throw 'Original policy continuation did not reproduce failure'}
    $winners=@($slice|Where-Object {$_.branch -notlike '*policy' -and $_.useful_escape -eq '1'})
    $best=$winners|Sort-Object {[double]$_.min_clearance_mm} -Descending|Select-Object -First 1
    $features=@($observations|Where-Object {[int]$_.fixture -eq $fixture -and [int]$_.lead_s -eq $lead});$values=@{}
    foreach($feature in $features){$values[[int]$feature.feature]=[double]$feature.value}
    [pscustomobject]@{fixture=$fixture;case=$source.case;approx_seconds_before_original_failure=$lead;state_hash=$source.state_hash;
        actual_time_before_failure_s=$source.elapsed_s;original_first_action=$source.first_action;original_escape=[int]$source.useful_escape;
        installed_first_action=$installed.first_action;installed_escape=[int]$installed.useful_escape;diagnostic_escapes=$winners.Count;diagnostic_attempts=24;
        largest_clearance_example=$best.branch;example_min_clearance_mm=$best.min_clearance_mm;
        cue_sensor_speed_mm_s=$values[4]*8;cue_straight_range_mm=$values[12]*500;cue_straight_TTC_s=$values[13]*30;cue_TTC_valid=$values[14]}
}})
$table|Export-Csv -LiteralPath (Join-Path $study 'state-comparison.csv') -NoTypeInformation
$findings=[ordered]@{complete=$true;exact_prefixes=3;exact_fork_states=9;branches=234;source_escape_cases=@($rows|Where-Object {$_.branch -eq 'source-policy' -and $_.useful_escape -eq '1'}).Count;
    installed_escape_cases=@($rows|Where-Object {$_.branch -eq 'installed-policy' -and $_.useful_escape -eq '1'}).Count;
    diagnostic_escape_branches=@($rows|Where-Object {$_.branch -notlike '*policy' -and $_.useful_escape -eq '1'}).Count;
    states_with_demonstrated_alternative=@($table|Where-Object diagnostic_escapes -gt 0).Count;
    safe_but_not_useful=@($rows|Where-Object {$_.safe_40s -eq '1' -and $_.useful_escape -eq '0'}).Count;
    attempted_substeps=$attempted;accepted_substeps=$accepted;optimizer_updates=0;mouse_keyboard_calls=0;Godot_updated=$false;
    warning='Lead labels are retrospective time before the original failure, not sensor TTC. Successful sequences are finite offline witnesses; an unsuccessful finite library does not prove irrecoverability.'}
$findings|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $study findings.json) -Encoding utf8NoBOM
$table|Format-Table fixture,approx_seconds_before_original_failure,original_escape,installed_escape,diagnostic_escapes,largest_clearance_example -AutoSize
$findings|ConvertTo-Json -Depth 6
