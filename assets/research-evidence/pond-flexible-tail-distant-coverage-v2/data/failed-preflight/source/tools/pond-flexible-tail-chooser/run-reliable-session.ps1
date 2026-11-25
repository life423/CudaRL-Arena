param([string]$Output='build/flexible-tail/reliable-chooser-v1-corrected')
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin=$shared.IndexOf('$children=');$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
try {
    $script:root=[IO.Path]::GetFullPath($Output)
    if(Test-Path -LiteralPath $root){throw 'Reliable chooser output must be unused'}
    $null=New-Item -Path $root -ItemType Directory
    $script:chooserResume='';$script:rateEval=''
    $cmake=(Get-Command cmake).Source;$build=Join-Path $root native-build;$compile=Join-Path $root compilation
    Wait-Native (Start-Native 'configure' $cmake @('-S',$PSScriptRoot,'-B',$build,'-G','Visual Studio 17 2022','-A','x64','-DCUDARL_RELIABLE_CHOOSER=ON') $compile 600)
    Wait-Native (Start-Native 'compile' $cmake @('--build',$build,'--config','Release','--target','pond_flexible_tail_rate_chooser','pond_flexible_tail_rate_evaluate','--parallel','4') $compile 600)
    $capacity=Join-Path $root capacity-build
    Wait-Native (Start-Native 'configure-capacity' $cmake @('-S','tools/pond-ppo-capacity-check','-B',$capacity,'-G','Visual Studio 17 2022','-A','x64') $compile 600)
    Wait-Native (Start-Native 'compile-capacity' $cmake @('--build',$capacity,'--config','Release','--parallel','4') $compile 600)
    $where=Join-Path $root capacity
    Wait-Native (Start-Native 'capacity' (Join-Path $capacity Release/pond_ppo_capacity_check.exe) @((Join-Path $where result.json)) $where 20)
    $script:rate=Join-Path $build Release/pond_flexible_tail_rate_chooser.exe
    $script:rateEval=Join-Path $build Release/pond_flexible_tail_rate_evaluate.exe
    $script:gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
    $script:starts=(Resolve-Path build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin).Path
    $script:straight=(Resolve-Path flexible-tail-preview/live_data/straight-evaluation-starts.bin).Path
    $motor=Get-Content build/flexible-tail/rate-response-v1/selected.json -Raw|ConvertFrom-Json
    $paths=@(Get-ChildItem src -File -Recurse|Where-Object Extension -in @('.h','.cpp','.cu')|Select-Object -ExpandProperty FullName)
    $paths+=@(Get-ChildItem $PSScriptRoot -File|Select-Object -ExpandProperty FullName)
    $paths+=@($rate,$rateEval,$gait,$starts,$straight,$motor.actor,$motor.checkpoint,'tools/pond-flexible-tail-steering/run-seven-stages.ps1','flexible-tail-preview/live_data/yaw-offset-update-8.bin','flexible-tail-preview/habitat.json','flexible-tail-preview/addons/flexible_tail_live/bin/flexible_tail_live.dll')
    $script:bindings=@(Bind-Files $paths);Json (Join-Path $root bindings.json) $bindings
    $failed='build/flexible-tail/reliable-chooser-v1/stage-06-bank/verification/data/failure.txt'
    if(Test-Path -LiteralPath $failed){Json (Join-Path $root previous-preflight.json) @{root='build/flexible-tail/reliable-chooser-v1';failure=(Get-Content $failed -Raw);physics_steps=0;optimizer_updates=0;correction='initial placement moves inland only until unchanged static warmup certificate passes';reserved_geometry_correction_used=$true}}
    foreach($b in $bindings|Where-Object {$_.path -like 'src/*' -or $_.path -like 'tools/*'}){$copy=Join-Path $root ('source/'+$b.path);$null=New-Item -Path (Split-Path -Parent $copy) -ItemType Directory -Force;Copy-Item -LiteralPath $b.path -Destination $copy}
    $selected=Run-Chooser $motor 128
    Assert-Frozen
    Json (Join-Path $root result.json) @{completed=$true;selected=$selected;motor_frozen=$true;godot_adopted=$false}
} finally {
    foreach($j in $children){if(!$j.finished){if(!$j.process.HasExited -and $j.name -eq 'training'){$null=New-Item -Path (Join-Path $j.arguments[4] STOP) -ItemType File -Force;$null=$j.process.WaitForExit(15000)};try{Finish-Native $j}catch{}}}
}
