param(
    [Parameter(Mandatory)][string]$Evaluator,
    [Parameter(Mandatory)][string]$Actor,
    [Parameter(Mandatory)][string]$Fixtures,
    [Parameter(Mandatory)][string]$Output,
    [switch]$RegressionOnly
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
# Share the existing timeouts, receipts, and checks for legitimate negative results.
$shared = Get-Content tools/pond-flexible-tail-steering/run-seven-stages.ps1 -Raw
$begin = $shared.IndexOf('$children=')
$end = [regex]::Match($shared,'(?m)^try\{').Index
. ([scriptblock]::Create($shared.Substring($begin,$end-$begin)))
$script:rateEval = (Resolve-Path -LiteralPath $Evaluator).Path
$actorPath = (Resolve-Path -LiteralPath $Actor).Path
$fixtureDir = (Resolve-Path -LiteralPath $Fixtures).Path
$root = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $root) { throw 'Assessment output already exists' }
$null = New-Item -ItemType Directory -Path $root
$gait = (Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
$motor = (Resolve-Path flexible-tail-preview/live_data/exploration-motor-88.bin).Path
$straight = (Resolve-Path flexible-tail-preview/live_data/straight-evaluation-starts.bin).Path
$files = @($rateEval,$actorPath,$gait,$motor,$straight)
if (!$RegressionOnly) { $files += @(72,75 | ForEach-Object { Join-Path $fixtureDir "shore-$_.bin" }) }
$script:bindings = @(Bind-Files $files)
Json (Join-Path $root 'bindings.json') $script:bindings
function Assess($name,$starts,$mode) {
    $where = Join-Path $root $name
    $argsList = @($gait,$motor,$actorPath,$starts,(Join-Path $where assessment),'0')
    if ($mode) { $argsList += $mode }
    Wait-Native (Start-Native $name $script:rateEval $argsList $where 65)
}
if (!$RegressionOnly) {
    foreach ($mm in @(72,75,80)) {
        $fixture = if ($mm -eq 80) { $straight } else { Join-Path $fixtureDir "shore-$mm.bin" }
        Assess "shore-$mm" $fixture '--close-shore'
    }
    Assess 'navigation' $straight ''
}
Assess 'continuous' $straight '--continuous-coverage'
Assert-Frozen
