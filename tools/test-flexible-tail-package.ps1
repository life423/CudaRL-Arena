# Verify the shipped native viewer and selected policy pair without physics/PPO.
param([string]$ProjectPath=(Join-Path $PSScriptRoot '../flexible-tail-preview'),
    [string]$CandidateActor, [string]$QualificationPath)
$ErrorActionPreference='Stop'
$project=(Resolve-Path -LiteralPath $ProjectPath).Path
$repository=Split-Path -Parent $project
$gate=Join-Path $PSScriptRoot 'pond-flexible-tail-qualification/publication.mjs'
if($CandidateActor -or $QualificationPath){
    if(!$CandidateActor -or !$QualificationPath){throw 'Candidate publication requires both its actor and matched native assessments.'}
    $study=Split-Path -Parent (Resolve-Path -LiteralPath $QualificationPath).Path
    if((Test-Path -LiteralPath "$study/selection-preregistered.json") -or
       (Test-Path -LiteralPath "$QualificationPath/candidate/original-moving-72")){
        $movingGate=Join-Path $PSScriptRoot 'pond-flexible-tail-qualification/moving-publication.mjs'
        & node $movingGate $QualificationPath $CandidateActor "$study/selection/selected.json"
    }else{
        & node $gate $QualificationPath $CandidateActor
    }
    if($LASTEXITCODE -ne 0){throw 'Candidate blocked by the native per-swimmer publication checks.'}
}
$manifest=Get-Content -LiteralPath (Join-Path $project 'live_data/exploration-deployment.json') -Raw|ConvertFrom-Json
if($manifest.schema -ne 1 -or $manifest.motor_update -ne 88 -or
    $manifest.diagnostic_passes -ne 12 -or $manifest.endurance_passes -ne 4 -or $manifest.clearance_stops -ne 0 -or
    $manifest.resets -ne 0 -or $manifest.training_active -or !$manifest.finite_test_only){throw 'Invalid qualified deployment manifest.'}
$activeChooser='exploration-chooser-88.bin'
$preservedChoosers=@()
if($manifest.chooser_study -eq 'appended-coverage-v3'){
    $activeChooser='exploration-chooser-appended-16.bin'
    $installedEvidence=Join-Path $PSScriptRoot '../assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/qualification'
    & node $gate $installedEvidence (Join-Path $project "live_data/$activeChooser") --installed-stalls
    if($LASTEXITCODE -ne 0){throw 'Installed chooser or per-swimmer native stall evidence differs.'}
    $preservedChoosers+= 'flexible-tail-preview/live_data/exploration-chooser-coverage-80.bin'
    if($manifest.chooser_update -ne 16 -or $manifest.active_chooser_file -ne $activeChooser -or
        $manifest.chooser_inputs -ne 35 -or $manifest.chooser_model_version -ne 5 -or
        $manifest.continuous_coverage_cases -ne 4 -or $manifest.continuous_horizon_s -ne 1800 -or
        $manifest.continuous_physical_passes -ne 4 -or $manifest.continuous_faults -ne 0 -or
        $manifest.minimum_individual_coverage -lt .3688405797 -or $manifest.mean_individual_coverage -lt .4289855072 -or
        $null -eq $manifest.maximum_slow_interval_s -or $manifest.maximum_slow_interval_s -gt 8.14){throw 'Invalid appended-sensing qualification.'}
}elseif($manifest.chooser_study -eq 'coverage-v1'){
    $activeChooser='exploration-chooser-coverage-80.bin'
    if($manifest.chooser_update -ne 80 -or $manifest.active_chooser_file -ne $activeChooser -or
        $manifest.continuous_coverage_cases -ne 4 -or $manifest.continuous_horizon_s -ne 1800 -or
        $manifest.continuous_physical_passes -ne 4 -or $manifest.continuous_faults -ne 0 -or
        $manifest.minimum_individual_coverage -lt .2644927536 -or $manifest.mean_individual_coverage -lt .2934782609){throw 'Invalid coverage qualification.'}
}elseif($manifest.chooser_study -ne 'reliable-chooser-v1-corrected' -or $manifest.chooser_update -ne 88){throw 'Unknown chooser qualification.'}
$required=@('flexible-tail-preview/live_data/fitted-gait.bin','flexible-tail-preview/live_data/yaw-offset-update-8.bin',
    'flexible-tail-preview/live_data/exploration-motor-88.bin','flexible-tail-preview/live_data/exploration-chooser-88.bin',
    'flexible-tail-preview/live_data/straight-evaluation-starts.bin','flexible-tail-preview/addons/flexible_tail_live/bin/flexible_tail_live.dll',
    'flexible-tail-preview/flexible_tail_preview.gd','flexible-tail-preview/flexible_tail_preview.tscn','flexible-tail-preview/habitat.json')
if($activeChooser -ne 'exploration-chooser-88.bin'){$required+=('flexible-tail-preview/live_data/'+$activeChooser)}
$required+=$preservedChoosers
if($manifest.files.Count -ne $required.Count){throw 'Incomplete deployment file list.'}
foreach($entry in $manifest.files){
    if($entry.path -notin $required -or $entry.sha256 -notmatch '^[0-9A-Fa-f]{64}$'){throw 'Unexpected package binding.'}
    $file=Join-Path $repository $entry.path
    if((Get-FileHash -LiteralPath $file).Hash -ne $entry.sha256){throw "Flexible-tail package hash mismatch: $($entry.path)"}
}
foreach($expected in $required){if(@($manifest.files|Where-Object path -eq $expected).Count -ne 1){throw 'Duplicate or missing package binding.'}}
$descriptor=Get-Content -LiteralPath (Join-Path $project 'addons/flexible_tail_live/flexible_tail_live.gdextension') -Raw
if($descriptor -notmatch 'res://addons/flexible_tail_live/bin/flexible_tail_live.dll'){throw 'Native extension library path differs.'}
Write-Output "Verified $($required.Count) flexible-tail bundle hashes, qualified policy pair, and preserved original motor; zero physics/PPO."
