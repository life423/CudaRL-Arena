$ErrorActionPreference='Stop'
$repo=(Resolve-Path "$PSScriptRoot/../../..").Path
Set-Location $repo
$root='build/flexible-tail/moving-start-chooser-v1'
$selectedFile="$root/selection/selected.json"
$selected=Get-Content $selectedFile -Raw|ConvertFrom-Json
$archive='assets/research-evidence/pond-flexible-tail-moving-start-chooser-v1/upstream'
$base='assets/research-evidence/pond-flexible-tail-moving-start-recovery-v1/results'
$actor="$root/training/actor-$($selected.update).bin"
$gait='flexible-tail-preview/live_data/fitted-gait.bin'
$motor='flexible-tail-preview/live_data/exploration-motor-88.bin'
$reference104='assets/research-evidence/pond-flexible-tail-open-water-chooser-v1/results/training/actor-104.bin'
$reference112='assets/research-evidence/pond-flexible-tail-north-bank-three-depth-v1/training/actor-112.bin'
$reference128='build/flexible-tail/three-depth-exploration-v1/training/actor-128.bin'
$native="$root/upstream-build/Release/upstream.exe"
if((Get-FileHash $actor).Hash -ne $selected.actor_sha256){throw 'Selection hash differs'}
$old=Get-Content "$base/bindings.json" -Raw|ConvertFrom-Json
$raw="$root/upstream"
if(Test-Path $raw){throw 'Preserve existing upstream run'}
$inputFiles=@($native,$actor,$selectedFile,$gait,$motor,$reference104,$reference112,$reference128,"$archive/protocol.md","$PSScriptRoot/upstream.cpp")
foreach($bound in $old) {
  if($bound.path -in @($gait,$motor,$reference104,$reference112,$reference128) -and (Get-FileHash $bound.path).Hash -ne $bound.sha256){throw "Frozen input changed: $($bound.path)"}
}
$null=New-Item -ItemType Directory $raw
foreach ($depth in 72,75,80){
    $fixture=if($depth -eq 80){'flexible-tail-preview/live_data/straight-evaluation-starts.bin'}else{"assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1/qualification/fixtures/shore-$depth.bin"}
    $expected=@($old|Where-Object path -eq $fixture)
    if($expected.Count -ne 1 -or (Get-FileHash $fixture).Hash -ne $expected[0].sha256){throw 'Frozen fixture hash differs'}
    $inputFiles+=$fixture
    $argsList=@($gait,$motor,$actor,$fixture,$reference104,$reference112,$reference128,"$raw/depth-$depth")
    $timer=[Diagnostics.Stopwatch]::StartNew()
    $proc=Start-Process -FilePath $native -WindowStyle Hidden -ArgumentList $argsList -PassThru -RedirectStandardOutput "$raw/depth-$depth.stdout" -RedirectStandardError "$raw/depth-$depth.stderr"
    $timeout=-not $proc.WaitForExit(50000)
    if($timeout){$proc.Kill();$proc.WaitForExit()}
    $timer.Stop()
    [pscustomobject]@{executable=$native;args=$argsList;exit=$proc.ExitCode;timeout=$timeout;wall_s=$timer.Elapsed.TotalSeconds}|ConvertTo-Json|Set-Content -Encoding utf8 "$raw/receipt-$depth.json"
    if($timeout -or $proc.ExitCode -ne 0){throw (Get-Content "$raw/depth-$depth.stderr" -Raw)}
    Copy-Item -LiteralPath "$raw/depth-$depth/candidate.csv" -Destination "$archive/candidate-$depth.csv"
    Copy-Item -LiteralPath "$raw/receipt-$depth.json" -Destination $archive
}
if((Get-FileHash $actor).Hash -ne $selected.actor_sha256){throw 'Chooser changed during evaluation'}
@(foreach($item in $inputFiles){[pscustomobject]@{path=(Resolve-Path -Relative $item);sha256=(Get-FileHash $item).Hash}})|ConvertTo-Json|Set-Content -Encoding utf8 "$archive/bindings.json"
node "$PSScriptRoot/compare.mjs" --archive $archive --raw $raw --baseline $base --selected $selectedFile --actor $actor
if($LASTEXITCODE -ne 0){throw 'Comparison failed'}
