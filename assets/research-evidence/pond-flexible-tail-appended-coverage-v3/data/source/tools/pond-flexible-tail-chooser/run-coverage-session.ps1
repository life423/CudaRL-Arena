param([string]$Output='build/flexible-tail/coverage-v1',[switch]$Distant,[switch]$Appended)
if($Appended){$Distant=$true}
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
# Use the same bounded process, immutable receipt and assessment validators as
# the existing sustained chooser runner.
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin=$shared.IndexOf('$children=');$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
$script:root=[IO.Path]::GetFullPath($Output)
$build=Join-Path $root native-build
$rate=(Resolve-Path (Join-Path $build Release/pond_flexible_tail_rate_chooser.exe)).Path
$script:rateEval=(Resolve-Path (Join-Path $build Release/pond_flexible_tail_rate_evaluate.exe)).Path
$gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
$motor=(Resolve-Path flexible-tail-preview/live_data/exploration-motor-88.bin).Path
$starts=(Resolve-Path build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin).Path
$straight=(Resolve-Path flexible-tail-preview/live_data/straight-evaluation-starts.bin).Path
$prior=if($Distant){(Resolve-Path build/flexible-tail/coverage-v1/training).Path}else{(Resolve-Path build/flexible-tail/reliable-chooser-v1-corrected/stage-06-bank/training).Path}
$parent=Join-Path $prior $(if($Distant){'checkpoints/session-80.bin'}else{'checkpoints/session-88.bin'})
$parentTask=(Get-Content (Join-Path $prior result.json) -Raw|ConvertFrom-Json).task
$training=Join-Path $root training
if(Test-Path -LiteralPath $training){throw 'Coverage training is immutable; no automatic repeat.'}
if((Get-Content (Join-Path $root baseline/assessment/summary.json) -Raw|ConvertFrom-Json).physical_passes -ne 4){throw 'Missing completed baseline.'}
$paths=@(Get-ChildItem src -File -Recurse|Where-Object Extension -in @('.h','.cpp','.cu')|Select-Object -ExpandProperty FullName)
$paths+=@(Get-ChildItem $PSScriptRoot -File|Select-Object -ExpandProperty FullName)
$paths+=@($rate,$script:rateEval,$gait,$motor,$starts,$straight,$parent,(Join-Path $prior result.json),
    'tools/pond-flexible-tail-steering/run-seven-stages.ps1','flexible-tail-preview/live_data/exploration-chooser-88.bin','flexible-tail-preview/live_data/exploration-chooser-coverage-80.bin',
    'flexible-tail-preview/addons/flexible_tail_live/bin/flexible_tail_live.dll','flexible-tail-preview/habitat.json')
$script:bindings=@(Bind-Files $paths)
Json (Join-Path $root bindings.json) $bindings
foreach($binding in $bindings|Where-Object {$_.path -like 'src/*' -or $_.path -like 'tools/*'}){
    $copy=Join-Path $root ('source/'+$binding.path);$null=New-Item -Path (Split-Path -Parent $copy) -ItemType Directory -Force
    Copy-Item -LiteralPath $binding.path -Destination $copy
}
$all=@();$coverage=@();$evalJob=$null;$job=$null
try{
    $verify=Join-Path $root verification
    Wait-Native (Start-Native 'verification' $rate @('verify-fork',$gait,$motor,$starts,(Join-Path $verify data),'2','590','0',$parent,$parentTask) $verify 600)
    $proof=Get-Content (Join-Path $verify data/verification.json) -Raw|ConvertFrom-Json
    $import=Get-Content (Join-Path $verify data/parent-import.json) -Raw|ConvertFrom-Json
    if(!$proof.pass -or $proof.physical_updates_executed -ne 3 -or !$import.full_state_exact_before_counter_rebase){throw 'Full-state continuation gate failed.'}
    if($Distant -and !(Get-Content (Join-Path $verify data/distant-senses.json) -Raw|ConvertFrom-Json).pass){throw 'Distant sensory gate failed.'}
    if($Appended -and !(Get-Content (Join-Path $verify data/input-expansion.json) -Raw|ConvertFrom-Json).pass){throw 'Initial controller equivalence failed.'}
    Assert-Frozen
    $job=Start-Native 'training' $rate @('fork',$gait,$motor,$starts,$training,'128','1200','0',$parent,$parentTask) $root 1800
    $pending=[Collections.Generic.Queue[int]]::new();$scheduled=[Collections.Generic.HashSet[int]]::new()
    while(!$job.finished -or $evalJob -or $pending.Count){
        for($u=0;$u -le 128;$u+=8){if((Test-Path (Join-Path $training "actor-$u.bin")) -and $scheduled.Add($u)){$pending.Enqueue($u)}}
        if(!$evalJob -and $pending.Count){
            $u=$pending.Dequeue();$where=Join-Path $root ('evaluations/update-{0:D3}' -f $u)
            $evalJob=Start-Native 'evaluation' $script:rateEval @($gait,$motor,(Join-Path $training "actor-$u.bin"),$straight,(Join-Path $where assessment),'0') $where 30
            $evalJob|Add-Member NoteProperty update $u
        }
        if($evalJob -and ($evalJob.process.HasExited -or $evalJob.watch.Elapsed.TotalSeconds -ge $evalJob.limit)){
            Finish-Native $evalJob
            $r=Get-Content (Join-Path $evalJob.directory assessment/summary.json) -Raw|ConvertFrom-Json
            $all+=@{update=$evalJob.update;actor=(Join-Path $training "actor-$($evalJob.update).bin");results=$r;directory=$evalJob.directory}
            Write-Host ("coverage update{0}: safety {1}/12 + {2}/{3}, union {4:P1}" -f $evalJob.update,$r.diagnostic_passes,$r.endurance_passes,$r.endurance_attempts,$r.diagnostic_coverage_fraction)
            Json (Join-Path $root progress.json) $all
            $evalJob=$null
        }
        if(!$job.finished -and ($job.process.HasExited -or $job.watch.Elapsed.TotalSeconds -ge $job.limit)){Finish-Native $job}
        if(!$job.finished -or $evalJob -or $pending.Count){[Threading.Thread]::Sleep(100)}
    }
    $contenders=@($all|Where-Object {$_.update -gt 0 -and $_.results.pass}|Sort-Object @{Expression={$_.results.diagnostic_coverage_fraction};Descending=$true}|Select-Object -First 4)
    foreach($candidate in $contenders){
        $where=Join-Path $root ('continuous/update-{0:D3}' -f $candidate.update)
        Wait-Native (Start-Native 'coverage' $script:rateEval @($gait,$motor,$candidate.actor,$straight,(Join-Path $where assessment),'0','--continuous-coverage') $where 65)
        $r=Get-Content (Join-Path $where assessment/summary.json) -Raw|ConvertFrom-Json
        $coverage+=@{update=$candidate.update;actor=$candidate.actor;results=$r;directory=$where}
        Json (Join-Path $root continuous-results.json) $coverage
    }
    Assert-Frozen
    Json (Join-Path $root result.json) @{complete=$true;training=(Get-Content (Join-Path $training result.json) -Raw|ConvertFrom-Json);fixed=$all;continuous=$coverage;motor_frozen=$true;godot_adopted=$false}
}finally{
    foreach($owned in $children){if(!$owned.finished){
        if(!$owned.process.HasExited -and $owned.name -eq 'training'){
            $null=New-Item -Path (Join-Path $training STOP) -ItemType File -Force
            $null=$owned.process.WaitForExit(15000)
        }
        try{Finish-Native $owned}catch{Write-Warning $_}
    }}
}
