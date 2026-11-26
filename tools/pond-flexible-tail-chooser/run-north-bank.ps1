param([string]$Output='build/flexible-tail/north-bank-three-depth-v1')
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin=$shared.IndexOf('$children=');$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
$root=(Resolve-Path -LiteralPath $Output).Path
$rate=(Resolve-Path "$root/native-build/Release/pond_flexible_tail_rate_chooser.exe").Path
$gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
$motor=(Resolve-Path flexible-tail-preview/live_data/exploration-motor-88.bin).Path
$starts=(Resolve-Path build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin).Path
$parentRoot='build/flexible-tail/three-depth-exploration-v1/training'
$parent=(Resolve-Path "$parentRoot/checkpoints/session-128.bin").Path
$portable=(Resolve-Path "$parentRoot/actor-128.bin").Path
$parentTask=(Get-Content "$parentRoot/result.json" -Raw|ConvertFrom-Json).task
if((Get-FileHash $parent).Hash -ne '6362831e113966b6b63a036b4219ed0dcfa89c0350988485504cc5c9ab87d8cc' -or
    (Get-FileHash $portable).Hash -ne '7A16B85CDF19E98D150510F0D24713B592A2EB7DC5ADB124880B0E4EB30473A4' -or
    (Get-FileHash $motor).Hash -ne '01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E'){
    throw 'Incorrect parent checkpoint or frozen motor'
}
if((Test-Path "$root/verification") -or (Test-Path "$root/training")){throw 'Study output already exists'}
$script:bindings=@(Bind-Files @($gait,$motor,$starts,$parent,$portable,$rate,"$parentRoot/result.json",
    (Join-Path $PSScriptRoot 'train.cu'),(Join-Path $PSScriptRoot 'rate_session.cuh'),(Join-Path $PSScriptRoot 'bank_training_starts.h'),
    'src/rl/flexible_tail_three_depth_map.h','flexible-tail-preview/live_data/exploration-chooser-appended-16.bin',
    'flexible-tail-preview/habitat.json','assets/research-evidence/pond-flexible-tail-north-bank-three-depth-v1/protocol.md'))
Json "$root/bindings.json" $bindings
try {
    Wait-Native (Start-Native 'verification' $rate @('verify-fork',$gait,$motor,$starts,"$root/verification",'2','590','0',$parent,$parentTask,'35') "$root/verify-process" 600)
    foreach($name in @('verification','curriculum-preflight','time-limit-bootstrap-check','distant-senses','parent-import')){
        $gate=Get-Content "$root/verification/$name.json" -Raw|ConvertFrom-Json
        if(!$gate.pass){throw "Failed $name"}
    }
    if(!$gate.full_state_exact_before_counter_rebase -or $gate.parent_update -ne 128 -or
        (Get-FileHash "$root/verification/actor-0.bin").Hash -ne (Get-FileHash $portable).Hash){throw 'Fork differs from parent'}
    Assert-Frozen
    Wait-Native (Start-Native 'training' $rate @('fork',$gait,$motor,$starts,"$root/training",'128','1200','0',$parent,$parentTask,'35') $root 1800)
    Assert-Frozen
    $r=Get-Content "$root/training/result.json" -Raw|ConvertFrom-Json
    if($r.stage -ne 'complete' -or $r.updates_completed -ne 128 -or $r.executed_samples -ne 1048576){throw 'Incomplete north-bank session'}
} finally {
    foreach($owned in $children){if(!$owned.finished){Finish-Native $owned}}
}
