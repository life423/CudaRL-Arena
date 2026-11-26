param([string]$Run='build/flexible-tail/open-water-chooser-v1',
      [string]$Evidence='assets/research-evidence/pond-flexible-tail-open-water-chooser-v1',
      [switch]$MovingStarts)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repo
$root=(Resolve-Path $Run).Path
$out=Join-Path (Resolve-Path $Evidence) 'results'
function Read-Json($p){Get-Content -LiteralPath $p -Raw|ConvertFrom-Json}
function Copy-Files($source,$target,$names){
    $null=New-Item -ItemType Directory -Force -Path $target
    foreach($name in $names){Copy-Item -LiteralPath (Join-Path $source $name) -Destination $target}
}
function Sum-Column($rows,$name){($rows|Measure-Object -Property $name -Sum).Sum}
$selected=Read-Json "$root/selection/selected.json"
$update=[int]$selected.update
$actor="$root/training/actor-$update.bin"
if((Get-FileHash $actor).Hash -ne $selected.actor_sha256){throw 'Selected actor hash mismatch'}
foreach($b in (Read-Json "$Evidence/protected-inputs.json")){
    if((Get-FileHash -LiteralPath $b.path).Hash -ne $b.sha256){throw "Protected file differs: $($b.path)"}
}
$training=Read-Json "$root/training/result.json"
if($training.stage -ne 'complete' -or $training.updates_completed -ne 128){throw 'Training is incomplete'}
$ledger="$root/ledger-build/Release/depth_ledger.exe"
& $ledger "$root/training/checkpoints" --open-water "$root/cohorts.csv" | Out-File -Encoding utf8 "$root/depth-ledger.csv"
if($LASTEXITCODE){throw 'Native training ledger failed'}
$depth=Import-Csv "$root/depth-ledger.csv"; $groups=Import-Csv "$root/cohorts.csv"
if($groups.Count -ne 9 -or (Sum-Column $groups 'lanes') -ne 1024 -or
   (Sum-Column $groups 'time_limits') -ne $training.training_time_limits -or
   (Sum-Column $groups 'starts') -ne $training.starts -or
   (Sum-Column $groups 'clearance_stops') -ne $training.clearance_stops -or
   (Sum-Column $groups 'near_decisions') -ne $training.near_decisions){throw 'Cohort totals differ from native run'}
foreach($d in $training.depth_samples){
    $row=$depth|Where-Object {[int]$_.depth_mm -eq $d.mm}
    $atDepth=@($groups|Where-Object {[int]$_.depth_mm -eq $d.mm})
    if($row.Count -ne 1 -or $atDepth.Count -ne 3 -or [int]$row.unique_PPO_decisions -gt $d.transitions -or
       (Sum-Column $atDepth 'unique_PPO_decisions') -ne [int]$row.unique_PPO_decisions){throw 'Individual depth ledger mismatch'}
}
if((Sum-Column $groups 'minibatch_presentations') -ne $training.executed_actor_steps*1024){throw 'Optimizer count mismatch'}
$qual="$root/qualification"
$gate=Join-Path $PSScriptRoot publication.mjs
$gateArgs=@()
if($MovingStarts) {
    $gate=Join-Path $PSScriptRoot moving-publication.mjs
    $gateArgs=@("$root/selection/selected.json")
}
node $gate $qual $actor @gateArgs --report-only | Out-File -Encoding utf8 "$root/publication.json"
if($LASTEXITCODE){throw 'Invalid qualification evidence'}
Copy-Files $root $out @('bindings.json','training.receipt.json','preflight-bindings.json','replay-gate.json','depth-ledger.csv','cohorts.csv','publication.json')
Copy-Files "$root/preflight" "$out/preflight" @('curriculum-preflight.json','distant-parity.json','distant-senses.json','parent-import.json','time-limit-bootstrap-check.json')
Copy-Files "$root/inspect-process" "$out/preflight" @('inspection.receipt.json')
if($MovingStarts) {
    Copy-Files "$root/preflight" "$out/preflight" @('moving-geometry.json','moving-placements.csv')
    Copy-Files $root $out @('selection-preregistered.json','recovery-summary.json','training-bands.csv','band-audit-depths.csv','band-audit.json')
}
if(!$MovingStarts){Copy-Files "$root/placement-audit" "$out/placement-audit" @('placement.csv','audit.cpp','CMakeLists.txt')}
if(Test-Path "$root/placement-audit/validation-horizontal-ranges.csv"){
    Copy-Files "$root/placement-audit" "$out/placement-audit" @('validation_range.cpp','validation-horizontal-ranges.csv','training-horizontal-ranges.csv')
}
Copy-Files "$root/training" "$out/training" @('updates.csv','regions.csv','result.json',"actor-$update.bin",'actor-128.bin')
Copy-Files "$root/training/checkpoints" "$out/training" @("session-$update.bin")
$hashes=foreach($i in 0..128){
    [pscustomobject]@{update=$i;actor_sha256=(Get-FileHash "$root/training/actor-$i.bin").Hash
        full_state_sha256=(Get-FileHash "$root/training/checkpoints/session-$i.bin").Hash}
}
$hashes|Export-Csv "$out/training/checkpoint-hashes.csv" -NoTypeInformation
Copy-Files "$root/selection" "$out/selection" @('bindings.json','selected.json','scores.csv')
if(Test-Path "$root/selection/initial-motion.csv"){Copy-Files "$root/selection" "$out/selection" @('initial-motion.csv')}
if(Test-Path "$root/selection/order-check.json"){Copy-Files "$root/selection" "$out/selection" @('order-check.json')}
foreach($checkpoint in (Get-ChildItem "$root/selection" -Directory)){
    foreach($mm in 72,75,80){
        $from=Join-Path $checkpoint.FullName "depth-$mm"; $to="$out/selection/$($checkpoint.Name)/depth-$mm"
        Copy-Files $from $to @('validation.receipt.json')
        Copy-Files "$from/assessment" $to @('cases.csv','starts.csv')
    }
}
Copy-Files "$qual/candidate" "$out/qualification/candidate" @('bindings.json')
if(!(Test-Path "$out/qualification/installed")){Copy-Item -LiteralPath "$qual/installed" -Destination "$out/qualification/installed" -Recurse}
foreach($suite in 'shore-72','shore-75','shore-80','navigation','continuous'){
    $from="$qual/candidate/$suite";$to="$out/qualification/candidate/$suite"
    Copy-Files $from $to @("$suite.receipt.json")
    Copy-Files "$from/assessment" "$to/assessment" @((Get-ChildItem "$from/assessment" -File).Name)
    if($suite -eq 'shore-80'){
        Copy-Files "$from/assessment/traces" "$to/assessment/traces" @((Get-ChildItem "$from/assessment/traces" -File).Name)
    }else{Compress-Archive -LiteralPath "$from/assessment/traces" -DestinationPath "$to/assessment/traces.zip" -Force}
}
if($MovingStarts) {
    foreach($mm in 72,75,80){
        $suite="original-moving-$mm"
        $from="$qual/candidate/$suite"; $to="$out/qualification/candidate/$suite"
        Copy-Files $from $to @("$suite.receipt.json")
        Copy-Files "$from/assessment" "$to/assessment" @('starts.csv','cases.csv')
        Copy-Files "$from/assessment/traces" "$to/assessment/traces" @('moving_a.csv','moving_b.csv','moving_c.csv','moving_d.csv')
    }
    & node (Join-Path $PSScriptRoot compare-moving-decisions.mjs) 'build/flexible-tail/open-water-chooser-v1/selection/update-104' "$qual/candidate" "$out/original-moving-decisions.csv"
    if($LASTEXITCODE){throw 'Original moving decision prefixes differ before action divergence'}
    $gateArgs=@("$out/selection/selected.json")
}
$archived=node $gate "$out/qualification" "$out/training/actor-$update.bin" @gateArgs --report-only
if($LASTEXITCODE -or (($archived -join "`n") -ne ((Get-Content "$out/publication.json") -join "`n"))){throw 'Archived publication decision differs'}
$prior='assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/qualification'
$policies=[ordered]@{
    installed="$prior/installed"
    three_depth_128="$prior/candidate"
    north_bank_112='assets/research-evidence/pond-flexible-tail-north-bank-three-depth-v1/qualification/candidate'
    open_water_selected="$qual/candidate"
}
if($MovingStarts) {
    $policies.open_water_selected='assets/research-evidence/pond-flexible-tail-open-water-chooser-v1/results/qualification/candidate'
    $policies.Add('moving_start_selected',"$qual/candidate")
}
$shore=foreach($mm in 72,75,80){
    $reference=(Get-FileHash "$prior/installed/shore-$mm/assessment/starts.csv").Hash
    foreach($policy in $policies.GetEnumerator()){
        $suite="$($policy.Value)/shore-$mm/assessment"
        if((Get-FileHash "$suite/starts.csv").Hash -ne $reference){throw 'Held-out bank placements differ'}
        foreach($row in (Import-Csv "$suite/cases.csv")){
            [pscustomobject]@{depth_mm=$mm;chooser=$policy.Key;case=$row.case;completed_180s=([int]$row.accepted_substeps -eq 144000)
                metric_pass=$row.pass;fault=$row.fault;swim_s=$row.simulated_s;min_gap_mm=$row.min_clearance_mm
                max_stall_controls=$row.max_stall_controls;near_s=$row.close_time_s}
        }
    }
}
$shore|Export-Csv "$out/bank-swimmers.csv" -NoTypeInformation
$long=foreach($policy in $policies.GetEnumerator()){
    foreach($row in (Import-Csv "$($policy.Value)/continuous/assessment/cases.csv")){
        [pscustomobject]@{chooser=$policy.Key;case=$row.case;completed_1800s=$row.physical_pass;fault=$row.fault
            swim_s=$row.simulated_s;cells=$row.unique_cells;coverage=$row.coverage_fraction;min_gap_mm=$row.min_clearance_mm
            max_stall_controls=$row.max_stall_controls;mean_speed_mm_s=$row.mean_speed_mm_s}
    }
}
$long|Export-Csv "$out/long-swimmers.csv" -NoTypeInformation
Get-Content "$out/publication.json"
