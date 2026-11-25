
param([Parameter(Mandatory)][string]$Study)
$ErrorActionPreference = 'Stop'
$previous = 'assets/research-evidence/pond-flexible-tail-chooser-v1/data/heldout'
$current = Join-Path $Study 'heldout'
$oldCases = @(Import-Csv "$previous/cases.csv" | Where-Object suite -eq 'diagnostic')
$newCases = @(Import-Csv "$current/cases.csv" | Where-Object suite -eq 'diagnostic')
if ($oldCases.Count -ne 12 -or $newCases.Count -ne 12) { throw 'Missing matched cases' }

function Read-Actions([string]$Trace) {
    $decisions = @(Import-Csv -LiteralPath $Trace | Where-Object event -eq 'decision')
    $turns = @($decisions | Where-Object { $_.action -in @('1', '2') })
    $first = $turns | Select-Object -First 1
    $longest = 0
    $run = 0
    $last = '0'
    foreach ($decision in $decisions) {
        if ($decision.action -eq '0') { $run = 0 }
        elseif ($decision.action -eq $last) { ++$run }
        else { $run = 1 }
        $longest = [math]::Max($longest, $run)
        $last = $decision.action
    }
    return [pscustomobject]@{
        first_s = if ($first) { [double]$first.time_s }
        first_gap_mm = if ($first) { [double]$first.clearance_bound_mm }
        longest_same_turn_run = $longest
    }
}

$paired = @(foreach ($case in $oldCases) {
    $next = @($newCases | Where-Object case -eq $case.case)
    if ($next.Count -ne 1) { throw "Unmatched fixture $($case.case)" }
    $next = $next[0]
    $old = Read-Actions "$previous/traces/$($case.case).csv"
    $new = Read-Actions "$current/traces/$($case.case).csv"
    [pscustomobject]@{
        case = $case.case
        first_run_seconds = [double]$case.simulated_s
        current_run_seconds = [double]$next.simulated_s
        added_seconds = [double]$next.simulated_s - [double]$case.simulated_s
        first_run_first_turn_s = $old.first_s
        current_first_turn_s = $new.first_s
        first_run_pre_turn_clearance_mm = $old.first_gap_mm
        current_pre_turn_clearance_mm = $new.first_gap_mm
        first_run_turns = [int]$case.left + [int]$case.right
        current_turns = [int]$next.left + [int]$next.right
        current_longest_same_turn_run = $new.longest_same_turn_run
        first_run_cells = [int]$case.unique_cells
        current_cells = [int]$next.unique_cells
        first_run_outcome = $case.outcome
        current_outcome = $next.outcome
    }
})
$paired | Export-Csv (Join-Path $Study 'matched-cases.csv') -NoTypeInformation
$before = @($paired.first_run_seconds | Sort-Object)
$after = @($paired.current_run_seconds | Sort-Object)
[pscustomobject]@{
    before_median_s = ($before[5] + $before[6]) / 2
    after_median_s = ($after[5] + $after[6]) / 2
    longer = @($paired | Where-Object added_seconds -gt 0).Count
    shorter = @($paired | Where-Object added_seconds -lt 0).Count
    before_keeps = ($oldCases | Measure-Object -Sum -Property keep).Sum
    before_decisions = ($oldCases | Measure-Object -Sum -Property decisions).Sum
    after_keeps = ($newCases | Measure-Object -Sum -Property keep).Sum
    after_decisions = ($newCases | Measure-Object -Sum -Property decisions).Sum
} | ConvertTo-Json | Tee-Object -FilePath (Join-Path $Study 'comparison.json')

