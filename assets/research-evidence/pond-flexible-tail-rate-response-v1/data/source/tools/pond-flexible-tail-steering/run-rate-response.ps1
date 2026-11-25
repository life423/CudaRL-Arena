param([string]$Output='build/flexible-tail/rate-response-v1')
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
# Reuse the shared process, transfer, evaluation and stop workflow without
# invoking its closed seven-stage entry point.
$shared=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'run-seven-stages.ps1') -Raw
$begin=$shared.IndexOf('$children=')
$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
try {
    $script:root=[IO.Path]::GetFullPath($Output)
    if(Test-Path -LiteralPath $root){throw 'New candidate output must be unused'}
    $null=New-Item -Path $root -ItemType Directory
    $script:reuse='';$script:rateEval=''
    $build=Join-Path $root native-build
    $compile=Join-Path $root compilation
    $cmake=(Get-Command cmake).Source
    Wait-Native (Start-Native 'configure' $cmake @('-S',$PSScriptRoot,'-B',$build,'-G','Visual Studio 17 2022','-A','x64') $compile 600)
    Wait-Native (Start-Native 'compile' $cmake @('--build',$build,'--config','Release','--parallel','4') $compile 600)
    $script:trainer=Join-Path $build Release/pond_flexible_tail_steering_ppo.exe
    $script:eval=Join-Path $build Release/pond_flexible_tail_steering_evaluate.exe
    $script:check=Join-Path $build Release/pond_flexible_tail_beat_feedback_check.exe
    $script:gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
    $script:baseOffset=(Resolve-Path build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin).Path
    $script:starts=(Resolve-Path build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin).Path
    $script:turns=(Resolve-Path build/flexible-tail/yaw-beat-v1/capture/eval-turn-worlds.bin).Path
    $script:straight=(Resolve-Path flexible-tail-preview/live_data/straight-evaluation-starts.bin).Path
    $parent=(Get-Content build/flexible-tail/seven-options-v6-corrected/stage-05-rate-baseline/selection.json -Raw|ConvertFrom-Json).selected
    $paths=@(Get-ChildItem src -File -Recurse|Where-Object Extension -in @('.h','.cpp','.cu')|Select-Object -ExpandProperty FullName)
    $paths+=@(Get-ChildItem $PSScriptRoot -File|Select-Object -ExpandProperty FullName)
    $paths+=@($trainer,$eval,$check,$gait,$baseOffset,$starts,$turns,$straight,$parent.actor,$parent.checkpoint,'flexible-tail-preview/live_data/yaw-offset-update-8.bin','flexible-tail-preview/habitat.json','flexible-tail-preview/addons/flexible_tail_live/bin/flexible_tail_live.dll')
    $script:bindings=@(Bind-Files $paths);Json (Join-Path $root bindings.json) $bindings
    foreach($b in $bindings|Where-Object {$_.path -like 'src/*' -or $_.path -like 'tools/*'}){$copy=Join-Path $root ('source/'+$b.path);$null=New-Item -Path (Split-Path -Parent $copy) -ItemType Directory -Force;Copy-Item -LiteralPath $b.path -Destination $copy}
    $selected=Run-Motor 1 'rate-response' 'baseline' $parent $true
    Json (Join-Path $root selected.json) $selected
    # Compile only the shared CPU pond evaluator, with this motor schema.
    $rateBuild=Join-Path $root bank-eval-build
    Wait-Native (Start-Native 'configure-bank' $cmake @('-S','tools/pond-flexible-tail-chooser','-B',$rateBuild,'-G','Visual Studio 17 2022','-A','x64') $compile 600)
    Wait-Native (Start-Native 'compile-bank' $cmake @('--build',$rateBuild,'--config','Release','--target','pond_flexible_tail_rate_evaluate','--parallel','4') $compile 600)
    $script:rateEval=Join-Path $rateBuild Release/pond_flexible_tail_rate_evaluate.exe
    $chooser=(Get-Content build/flexible-tail/seven-options-v6-bank-resume/stage-06-bank/selection.json -Raw|ConvertFrom-Json).selected.actor
    $where=Join-Path $root bank
    Wait-Native (Start-Native 'evaluation' $rateEval @($gait,$selected.actor,$chooser,$straight,(Join-Path $where assessment),'0') $where 30)
    Assert-Frozen
    Json (Join-Path $root result.json) @{completed=$true;selected=$selected;bank=(Get-Content (Join-Path $where assessment/summary.json) -Raw|ConvertFrom-Json);godot_adopted=$false}
} finally {
    foreach($j in $children){if(!$j.finished){if(!$j.process.HasExited -and $j.name -eq 'training'){$null=New-Item -Path (Join-Path $j.arguments[4] STOP) -ItemType File -Force;$null=$j.process.WaitForExit(15000)};try{Finish-Native $j}catch{}}}
}
