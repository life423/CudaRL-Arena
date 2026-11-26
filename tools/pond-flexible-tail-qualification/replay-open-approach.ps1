param([Parameter(Mandatory)][string]$Evaluator,[Parameter(Mandatory)][string]$Output,[switch]$Refine,[switch]$ConfirmLong)
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
$root=[IO.Path]::GetFullPath($Output)
if(!(Test-Path $root)){$null=New-Item $root -ItemType Directory}
if(Test-Path "$root/assessment"){throw 'Replay output already exists'}
$shared=Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$start=$shared.IndexOf('$children=')
$end=[regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($start,$end-$start)))
$native=(Resolve-Path -LiteralPath $Evaluator).Path
$g=(Resolve-Path 'flexible-tail-preview/live_data/fitted-gait.bin').Path
$m=(Resolve-Path 'flexible-tail-preview/live_data/exploration-motor-88.bin').Path
$c=(Resolve-Path 'assets/research-evidence/pond-flexible-tail-north-bank-three-depth-v1/training/actor-112.bin').Path
$p=(Resolve-Path 'assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/data/training/actor-128.bin').Path
$w=(Resolve-Path 'flexible-tail-preview/live_data/straight-evaluation-starts.bin').Path
$evidence='assets/research-evidence/pond-flexible-tail-open-water-replay-v1'
$t=(Resolve-Path "$evidence/godot-start.csv").Path
if((Get-FileHash $c).Hash -ne 'F8B2E7D62E4BFA5412F166C70F7A89EE960ECAFE7FBE939D3083ED5DFFB4E10C' -or
    (Get-FileHash $p).Hash -ne '7A16B85CDF19E98D150510F0D24713B592A2EB7DC5ADB124880B0E4EB30473A4' -or
    (Get-FileHash $m).Hash -ne '01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E'){throw 'Unexpected frozen actor'}
$mode=if($ConfirmLong){'--open-replay-long'}elseif($Refine){'--open-replay-refine'}else{'--open-replay'}
$protocol=if($ConfirmLong){'confirmation.md'}elseif($Refine){'refinement.md'}else{'protocol.md'}
$extra=@()
if($ConfirmLong){
    $extra+= (Resolve-Path "$evidence/refinement/forks.csv").Path
    if((Get-FileHash $extra[0]).Hash -ne '375F98FF30AAD8F2B72340B38FF6F6F7EDB21B77C83FD87969554B577B1DD829'){throw 'Unexpected short-run evidence'}
}
if((Get-FileHash $t).Hash -ne 'F71D2B246787C5D43C26FC680C572E56DD78A7B212821B5BA1F1947D4128EFD5'){throw 'Original Godot trace differs'}
$script:bindings=@(Bind-Files @($native,$g,$m,$c,$p,$w,$t,$PSCommandPath,'tools/pond-flexible-tail-chooser/open_approach_audit.h',
    "$evidence/$protocol") ) + @(Bind-Files $extra)
Json "$root/bindings.json" $bindings
$arguments=@($g,$m,$c,$w,"$root/assessment",'0',$mode,$p,$t)+$extra
Wait-Native (Start-Native replay $native $arguments $root 150)
Assert-Frozen
