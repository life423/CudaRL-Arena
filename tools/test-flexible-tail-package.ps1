# Verify the shipped native viewer and selected policy pair without physics/PPO.
param([string]$ProjectPath=(Join-Path $PSScriptRoot '../flexible-tail-preview'))
$ErrorActionPreference='Stop'
$project=(Resolve-Path -LiteralPath $ProjectPath).Path
$repository=Split-Path -Parent $project
$manifest=Get-Content -LiteralPath (Join-Path $project 'live_data/exploration-deployment.json') -Raw|ConvertFrom-Json
if($manifest.schema -ne 1 -or $manifest.motor_update -ne 88 -or
    $manifest.diagnostic_passes -ne 12 -or $manifest.endurance_passes -ne 4 -or $manifest.clearance_stops -ne 0 -or
    $manifest.resets -ne 0 -or $manifest.training_active -or !$manifest.finite_test_only){throw 'Invalid qualified deployment manifest.'}
$activeChooser='exploration-chooser-88.bin'
if($manifest.chooser_study -eq 'coverage-v1'){
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
