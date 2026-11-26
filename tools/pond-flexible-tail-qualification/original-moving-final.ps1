param([Parameter(Mandatory)][string]$Evaluator,[Parameter(Mandatory)][string]$Actor,[Parameter(Mandatory)][string]$Output)
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin=$shared.IndexOf('$children=')
$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
$root=(Resolve-Path -LiteralPath $Output).Path
$native=(Resolve-Path -LiteralPath $Evaluator).Path
$actorPath=(Resolve-Path -LiteralPath $Actor).Path
$selected=Get-Content -Raw (Join-Path $root '../../selection/selected.json')|ConvertFrom-Json
if((Get-FileHash $actorPath).Hash -ne $selected.actor_sha256){throw 'Final moving actor differs from frozen choice'}
$fixtures=Resolve-Path assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/qualification/fixtures
$archives=Resolve-Path assets/research-evidence/pond-flexible-tail-open-water-chooser-v1/results/selection/update-104
$gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
$motor=(Resolve-Path flexible-tail-preview/live_data/exploration-motor-88.bin).Path
$script:bindings=Get-Content -Raw (Join-Path $root bindings.json)|ConvertFrom-Json
Assert-Frozen
try{
    $jobs=foreach($mm in 72,75,80){
        $name="original-moving-$mm"
        $where=Join-Path $root $name
        if(Test-Path -LiteralPath $where){throw "Final moving suite already exists: $name"}
        $fixture=if($mm -eq 80){(Resolve-Path flexible-tail-preview/live_data/straight-evaluation-starts.bin).Path}else{Join-Path $fixtures "shore-$mm.bin"}
        Start-Native $name $native @($gait,$motor,$actorPath,$fixture,(Join-Path $where assessment),'0','--open-validation') $where 45
    }
    foreach($job in $jobs){Wait-Native $job}
    foreach($mm in 72,75,80){
        $where=Join-Path $root "original-moving-$mm/assessment"
        if((Get-FileHash "$where/starts.csv").Hash -ne (Get-FileHash "$archives/depth-$mm/starts.csv").Hash){throw "Original fixture identity changed at $mm"}
        $cases=@(Import-Csv "$where/cases.csv"|Where-Object suite -eq moving)
        if($cases.Count -ne 4 -or ($cases.case -join ',') -ne 'moving_a,moving_b,moving_c,moving_d'){throw "Incomplete original final cases at $mm"}
    }
    Assert-Frozen
}finally{
    foreach($job in $children){if(!$job.finished){Finish-Native $job}}
}
