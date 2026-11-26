param([string]$Output='build/flexible-tail/open-water-chooser-v1',[switch]$MovingStarts)
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin=$shared.IndexOf('$children=')
$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
$root=(Resolve-Path -LiteralPath $Output).Path
$selection=Join-Path $root selection
if((Test-Path $selection) -or (Test-Path "$root/qualification/candidate")){throw 'Selection or final evidence already exists'}
$null=New-Item $selection -ItemType Directory
$native=(Resolve-Path "$root/selection-build/Release/depth_eval.exe").Path
$gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
$motor=(Resolve-Path flexible-tail-preview/live_data/exploration-motor-88.bin).Path
$fixtureDirectory=(Resolve-Path assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/qualification/fixtures).Path
$parent=(Resolve-Path assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/data/training/actor-128.bin).Path
$forkSource=(Resolve-Path assets/research-evidence/pond-flexible-tail-north-bank-three-depth-v1/training/actor-112.bin).Path
$study=if($MovingStarts){'moving-start-chooser-v1'}else{'open-water-chooser-v1'}
$protocol=(Resolve-Path "assets/research-evidence/pond-flexible-tail-$study/selection-protocol.md").Path
$validationFlag=if($MovingStarts){'--moving-start-validation'}else{'--open-validation'}
$movingCount=if($MovingStarts){18}else{12}
$limit=if($MovingStarts){60}else{45}
$fixtures=@{72="$fixtureDirectory/shore-72.bin";75="$fixtureDirectory/shore-75.bin";80=(Resolve-Path flexible-tail-preview/live_data/straight-evaluation-starts.bin).Path}
if((Get-FileHash $parent).Hash -ne '7A16B85CDF19E98D150510F0D24713B592A2EB7DC5ADB124880B0E4EB30473A4' -or
    (Get-FileHash $forkSource).Hash -ne 'F8B2E7D62E4BFA5412F166C70F7A89EE960ECAFE7FBE939D3083ED5DFFB4E10C' -or
    (Get-FileHash $motor).Hash -ne '01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E'){throw 'Frozen parent, fork or motor differs'}
$files=@($native,$gait,$motor,$fixtures[72],$fixtures[75],$fixtures[80],$parent,$forkSource,$protocol,
    'tools/pond-flexible-tail-chooser/open-water-validation.h',$PSCommandPath)
if($MovingStarts){
    $files+=@('tools/pond-flexible-tail-chooser/moving-start-validation.h',
        'tools/pond-flexible-tail-qualification/assess.ps1','tools/pond-flexible-tail-qualification/original-moving-final.ps1',
        'tools/pond-flexible-tail-steering/run-seven-stages.ps1')
}
$script:bindings=@(Bind-Files $files)
Json "$selection/bindings.json" $bindings
$clock=[Diagnostics.Stopwatch]::StartNew()
$parentCells=@{}
$startHashes=@{}
$scores=@()
function Wait-Actor([int]$update) {
    $actor="$root/training/actor-$update.bin"
    while(!(Test-Path -LiteralPath $actor)){
        if($clock.Elapsed.TotalSeconds -gt 1800 -or (Test-Path "$root/training/failure.txt") -or (Test-Path "$root/training.receipt.json")){
            throw "Training closed before checkpoint $update"
        }
        Start-Sleep -Seconds 1
    }
    return $actor
}
function Is-Safe($row) { return ($row.pass -eq '1' -and $row.fault -eq '0' -and [int]$row.max_stall_controls -le 407) }

try{
    # The ancestor supplies per-case coverage references once. Fork +0 is also
    # evaluated, but neither preserved actor can be selected as the new choice.
    foreach($u in (@(-1,0)+@(1..16|ForEach-Object {8*$_}))){
        $actor=if($u -lt 0){$parent}elseif($u -eq 0){$forkSource}else{Wait-Actor $u}
        $folder=if($u -lt 0){'ancestor-128'}else{'update-{0:D3}' -f $u}
        $where=Join-Path $selection $folder
        $hash=(Get-FileHash $actor).Hash
        $jobs=foreach($mm in 72,75,80){
            $out="$where/depth-$mm"
            Start-Native 'validation' $native @($gait,$motor,$actor,$fixtures[$mm],"$out/assessment",'0',$validationFlag) $out $limit
        }
        foreach($job in $jobs){Wait-Native $job}
        if((Get-FileHash $actor).Hash -ne $hash){throw 'Actor changed during validation'}
        $rows=foreach($mm in 72,75,80) {
            $path="$where/depth-$mm/assessment"
            $startHash=(Get-FileHash "$path/starts.csv").Hash
            if($u -lt 0){$startHashes[$mm]=$startHash}
            if($startHashes[$mm] -ne $startHash){throw "Validation starts changed at $mm mm"}
            $cases=@(Import-Csv "$path/cases.csv")
            if($cases.Count -ne ($movingCount/3+4) -or @($cases|Where-Object depth_mm -ne $mm).Count){throw 'Incomplete validation cohort'}
            $cases
        }
        $moving=@($rows|Where-Object suite -eq moving)
        $tangent=@($rows|Where-Object suite -eq tangent)
        $long=@($rows|Where-Object suite -eq long)
        if($moving.Count -ne $movingCount -or $tangent.Count -ne 6 -or $long.Count -ne 6){throw 'Incomplete validation suites'}
        $movingPass=@($moving|Where-Object {Is-Safe $_}).Count
        $tangentPass=@($tangent|Where-Object {Is-Safe $_}).Count
        $longPass=@($long|Where-Object {Is-Safe $_}).Count
        $coveragePass=0
        $worstRatio=[double]::PositiveInfinity
        $sumCells=0
        foreach($row in $long){
            $key="$($row.depth_mm)-$($row.case)"
            if($u -lt 0){$parentCells.Add($key,[int]$row.unique_cells)}
            $safe=Is-Safe $row
            $cells=if($safe){[int]$row.unique_cells}else{0}
            if($safe -and $cells -ge $parentCells[$key]){++$coveragePass}
            $worstRatio=[Math]::Min($worstRatio,$cells/[math]::Max(1,$parentCells[$key]))
            $sumCells+=$cells
        }
        $scores+=[pscustomobject]@{
            update=$u;actor_sha256=$hash;moving=$movingPass;tangent=$tangentPass;long=$longPass
            worst_fraction=[Math]::Min($movingPass/$movingCount,[Math]::Min($tangentPass/6,$longPass/6))
            total=$movingPass+$tangentPass+$longPass;long_retained=$coveragePass;minimum_coverage_ratio=$worstRatio
            sustained=@($tangent|Where-Object {(Is-Safe $_) -and [double]$_.close_s -ge 15}).Count
            long_cells=$sumCells
        }
        $scores|Export-Csv "$selection/scores.csv" -NoTypeInformation
    }
    $order=@('worst_fraction')
    if($MovingStarts){$order+='moving'}
    $order+=@('total','long_retained','minimum_coverage_ratio','sustained','long_cells')
    $fields=foreach($name in $order){
        @{Expression=[string]$name;Descending=$true}
    }
    $winner=$scores|Where-Object update -gt 0|Sort-Object -Property ($fields+@('update'))|Select-Object -First 1
    $actor=Wait-Actor $winner.update
    if((Get-FileHash $actor).Hash -ne $winner.actor_sha256 -or (Get-FileHash (Wait-Actor 0)).Hash -ne (Get-FileHash $forkSource).Hash){
        throw 'Fork or selected checkpoint mismatch'
    }
    Assert-Frozen
    Json "$selection/selected.json" $winner
    & (Join-Path $PSScriptRoot assess.ps1) -Evaluator $native -Actor $actor -Fixtures $fixtureDirectory -Output "$root/qualification/candidate"
    if($MovingStarts){
        & (Join-Path $PSScriptRoot original-moving-final.ps1) -Evaluator $native -Actor $actor -Output "$root/qualification/candidate"
    }
}finally{
    foreach($owned in $children){if(!$owned.finished){Finish-Native $owned}}
}
