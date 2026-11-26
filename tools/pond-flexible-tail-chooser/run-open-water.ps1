param(
    [switch]$Train,
    [switch]$MovingStarts,
    [string]$Output,
    [string]$ReplayReceipt
)
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
if (!$Output) { $Output=if($MovingStarts){'build/flexible-tail/moving-start-chooser-v1'}else{'build/flexible-tail/open-water-chooser-v1'} }
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin=$shared.IndexOf('$children=')
$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
$root=(Resolve-Path -LiteralPath $Output).Path
$native=(Resolve-Path "$root/trainer-build/Release/pond_flexible_tail_rate_chooser.exe").Path
$parentDir='assets/research-evidence/pond-flexible-tail-north-bank-three-depth-v1/training'
$parent=(Resolve-Path "$parentDir/session-112.bin").Path
$parentActor=(Resolve-Path "$parentDir/actor-112.bin").Path
$parentTask=(Get-Content "$parentDir/result.json" -Raw|ConvertFrom-Json).task
$motor=(Resolve-Path 'flexible-tail-preview/live_data/exploration-motor-88.bin').Path
$gait=(Resolve-Path 'flexible-tail-preview/live_data/fitted-gait.bin').Path
$starts=(Resolve-Path 'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin').Path
$grandparent='build/flexible-tail/three-depth-exploration-v1/training/checkpoints/session-128.bin'
$grandparentActor='build/flexible-tail/three-depth-exploration-v1/training/actor-128.bin'
$expected=@{
    $parent='6797224633C5E08F53C5937ECF1D1DF226805CCD9DBCBE2DB5A36AF0DFFA891E'
    $parentActor='F8B2E7D62E4BFA5412F166C70F7A89EE960ECAFE7FBE939D3083ED5DFFB4E10C'
    $motor='01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E'
    $grandparent='6362831E113966B6B63A036B4219ED0DCFA89C0350988485504CC5C9AB87D8CC'
    $grandparentActor='7A16B85CDF19E98D150510F0D24713B592A2EB7DC5ADB124880B0E4EB30473A4'
}
if($MovingStarts){
    $previous='build/flexible-tail/open-water-chooser-v1/training'
    $expected["$previous/actor-104.bin"]='932CCF8E942C25A6901CB07B69E0B87243A3BA571FC559BEE94E189AA365FA93'
    $expected["$previous/checkpoints/session-104.bin"]='76BAA03292340D370444A24DE790925B0A56D0213F30D8A22DF37FE76FC32291'
}
foreach($file in $expected.Keys){if((Get-FileHash -LiteralPath $file).Hash -ne $expected[$file]){throw "Wrong checkpoint: $file"}}
$files=@($gait,$motor,$starts,$parent,$parentActor,$native,$grandparent,$grandparentActor,"$parentDir/result.json",
    "$PSScriptRoot/train.cu","$PSScriptRoot/rate_session.cuh","$PSScriptRoot/bank_training_starts.h",
    $PSCommandPath,'src/rl/flexible_tail_three_depth_map.h','flexible-tail-preview/live_data/exploration-chooser-appended-16.bin',
    'flexible-tail-preview/habitat.json')
if($MovingStarts){
    $files+=@("$PSScriptRoot/moving_placement_audit.cuh","$PSScriptRoot/CMakeLists.txt",
             "$previous/actor-104.bin","$previous/checkpoints/session-104.bin")
}
$proof="$root/preflight"
$script:bindings=@(Bind-Files $files)
try {
    if(!$Train){
        if(Test-Path "$root/preflight-bindings.json"){throw 'Inspection already attempted; preserve its evidence'}
        Json "$root/preflight-bindings.json" $bindings
        Wait-Native (Start-Native 'inspection' $native @('inspect-fork',$gait,$motor,$starts,$proof,'1','60','0',$parent,$parentTask,'35') "$root/inspect-process" 75)
    }else{
        if((Test-Path "$root/training") -or (Test-Path "$root/bindings.json")){throw 'Training already attempted'}
        $before=Get-Content "$root/preflight-bindings.json" -Raw|ConvertFrom-Json
        foreach($bound in $before){if((Get-FileHash -LiteralPath $bound.path).Hash -ne $bound.sha256){throw "Preflight input changed: $($bound.path)"}}
    }
    $inspectResult=Get-Content "$root/inspect-process/inspection.receipt.json" -Raw|ConvertFrom-Json
    if($inspectResult.exit -ne 0 -or $inspectResult.timeout -or $inspectResult.executable -ne $native){throw 'Invalid input inspection'}
    foreach($name in 'curriculum-preflight','parent-import','distant-senses','time-limit-bootstrap-check'){
        $check=Get-Content "$proof/$name.json" -Raw|ConvertFrom-Json
        if(!$check.pass){throw "Input check failed: $name"}
    }
    if($MovingStarts){
        $geometry=Get-Content "$proof/moving-geometry.json" -Raw|ConvertFrom-Json
        if(!$geometry.pass -or $geometry.paired_placements -ne 1536 -or $geometry.optimizer_updates -ne 0){throw 'Unverified stratified placement build'}
    }
    $import=Get-Content "$proof/parent-import.json" -Raw|ConvertFrom-Json
    if(!$import.full_state_exact_before_counter_rebase -or $import.parent_update -ne 112 -or
        $import.optimizer_updates -ne 0 -or
        (Get-FileHash -LiteralPath "$proof/parent-actor.bin").Hash -ne $expected[$parentActor]){throw 'Inexact +112 import'}
    Assert-Frozen
    if(!$Train){return}
    if(!$ReplayReceipt){$ReplayReceipt="$root/replay-gate.json"}
    $gate=Get-Content -LiteralPath $ReplayReceipt -Raw|ConvertFrom-Json
    if($gate.schema -ne 1 -or $gate.training_unlocked -isnot [bool] -or !$gate.training_unlocked -or
        $gate.chooser_sha256 -ne $expected[$parentActor] -or $gate.motor_sha256 -ne $expected[$motor] -or
        (Get-FileHash -LiteralPath $gate.replay_evidence).Hash -ne $gate.replay_sha256){throw 'Missing confirmed recoverable frozen continuation'}
    $protocol='assets/research-evidence/pond-flexible-tail-open-water-chooser-v1/protocol.md'
    if($MovingStarts){
        $protocol='assets/research-evidence/pond-flexible-tail-moving-start-chooser-v1/protocol.md'
        $files+=@('assets/research-evidence/pond-flexible-tail-moving-start-chooser-v1/selection-protocol.md',
                 "$proof/moving-geometry.json","$proof/moving-placements.csv")
    }
    $script:bindings=@(Bind-Files ($files+@($protocol,$ReplayReceipt,$gate.replay_evidence)))
    Json "$root/bindings.json" $bindings
    Wait-Native (Start-Native 'training' $native @('fork',$gait,$motor,$starts,"$root/training",'128','1200','0',$parent,$parentTask,'35') $root 1220)
    Assert-Frozen
    $study=Get-Content "$root/training/result.json" -Raw|ConvertFrom-Json
    if($study.stage -ne 'complete' -or $study.updates_completed -ne 128 -or $study.executed_samples -ne 1048576){throw 'Incomplete bounded training session'}
}finally{
    foreach($owned in $children){if(!$owned.finished){Finish-Native $owned}}
}
