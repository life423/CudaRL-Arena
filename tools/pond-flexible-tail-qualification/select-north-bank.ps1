param([string]$Output='build/flexible-tail/north-bank-three-depth-v1',[switch]$UseSavedValidation)
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin=$shared.IndexOf('$children=');$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
$root=(Resolve-Path -LiteralPath $Output).Path
$selection=Join-Path $root 'selection'
if($UseSavedValidation){
    if((Test-Path "$selection/selected.json") -or (Test-Path "$root/qualification/candidate")){throw 'A held-out selection already exists'}
    $previous=@(Import-Csv "$selection/scores.csv")
    if($previous.Count -ne 17){throw 'Incomplete saved validation'}
}else{
    if(Test-Path $selection){throw 'Selection evidence exists'}
    $null=New-Item $selection -ItemType Directory
}
$native=(Resolve-Path "$root/native-build/Release/pond_flexible_tail_rate_evaluate.exe").Path
$gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
$motor=(Resolve-Path flexible-tail-preview/live_data/exploration-motor-88.bin).Path
$fixtures=(Resolve-Path assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/qualification/fixtures).Path
$validation=Join-Path $fixtures 'shore-75.bin'
$script:bindings=@(Bind-Files @($native,$gait,$motor,$validation,(Join-Path $PSScriptRoot '../pond-flexible-tail-chooser/rate-evaluate.cpp'),$PSCommandPath))
if($UseSavedValidation){
    foreach($bound in (Get-Content "$selection/bindings.json" -Raw|ConvertFrom-Json)){
        if((Resolve-Path $bound.path).Path -ne $PSCommandPath -and (Get-FileHash $bound.path).Hash -ne $bound.sha256){throw 'Validation input differs'}
    }
    Json "$selection/ranking-bindings.json" $bindings
}else{Json "$selection/bindings.json" $bindings}
$scores=@()
$clock=[Diagnostics.Stopwatch]::StartNew()
try{
    foreach($u in 0..16|ForEach-Object {8*$_}){
        $actor=Join-Path $root "training/actor-$u.bin"
        while(!(Test-Path -LiteralPath $actor)){
            if($clock.Elapsed.TotalSeconds -gt 1800 -or (Test-Path "$root/training/failure.txt") -or (Test-Path "$root/training.receipt.json")){
                throw "Training closed before checkpoint $u"
            }
            Start-Sleep -Seconds 1
        }
        $where=Join-Path $selection ('update-{0:D3}' -f $u)
        $arguments=@($gait,$motor,$actor,$validation,"$where/assessment",'0','--north-validation')
        if($UseSavedValidation){
            $receipt=Get-Content "$where/selection.receipt.json" -Raw|ConvertFrom-Json
            if($receipt.timeout -or $receipt.exit -ne 0 -or $receipt.executable -ne $native -or
                (Compare-Object $receipt.arguments $arguments -SyncWindow 0) -or
                $previous[$u/8].actor_sha256 -ne (Get-FileHash $actor).Hash){throw "Invalid saved evaluation at update $u"}
        }else{Wait-Native (Start-Native 'selection' $native $arguments $where 45)}
        $rows=Import-Csv "$where/assessment/cases.csv"
        $bank=@($rows|Where-Object suite -eq bank);$open=@($rows|Where-Object suite -eq open)
        if($bank.Count -ne 8 -or $open.Count -ne 4){throw 'Incomplete validation suites'}
        $passed={ $_.pass -eq '1' -and [int]$_.max_stall_controls -le 407 }
        $safeBank=@($bank|Where-Object $passed)
        $safeOpen=@($open|Where-Object $passed)
        $scores+=[pscustomobject]@{
            update=$u; actor_sha256=(Get-FileHash $actor).Hash
            bank=$safeBank.Count; interior=$safeOpen.Count
            worst_fraction=[Math]::Min($safeBank.Count/8,$safeOpen.Count/4);total=$safeBank.Count+$safeOpen.Count
            sustained=@($safeBank|Where-Object {[double]$_.close_s -ge 15}).Count
            min_bank_substeps=($bank|Measure-Object accepted_substeps -Minimum).Minimum
            interior_cells=($open|Measure-Object unique_cells -Sum).Sum
        }
        if(!$UseSavedValidation){$scores|Export-Csv "$selection/scores.csv" -NoTypeInformation}
    }
    $fields=@('worst_fraction','total','sustained','min_bank_substeps','interior_cells')|ForEach-Object {@{Expression=[string]$_;Descending=$true}}
    $winner=$scores|Where-Object update -gt 0|Sort-Object -Property ($fields+@('update'))|Select-Object -First 1
    Assert-Frozen
    Json "$selection/selected.json" $winner
    $actor=Join-Path $root "training/actor-$($winner.update).bin"
    if((Get-FileHash $actor).Hash -ne $winner.actor_sha256){throw 'Selected actor differs'}
    & (Join-Path $PSScriptRoot assess.ps1) -Evaluator $native -Actor $actor -Fixtures $fixtures -Output "$root/qualification/candidate"
}finally{
    foreach($owned in $children){if(!$owned.finished){Finish-Native $owned}}
}
