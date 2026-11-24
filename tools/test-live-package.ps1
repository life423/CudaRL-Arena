# Check the bytes that Godot Project Manager will load. No simulation or build required.
param([string]$ProjectPath = (Join-Path $PSScriptRoot '../live'))
$ErrorActionPreference = 'Stop'
$watch = [Diagnostics.Stopwatch]::StartNew()
$project = (Resolve-Path -LiteralPath $ProjectPath).Path

function Confirm-Resource([string]$Resource, [string]$Digest) {
    if ($Resource -cnotmatch '^res://[a-zA-Z0-9_./-]+$' -or $Resource.Contains('..') -or $Digest -notmatch '^[0-9a-fA-F]{64}$') {
        throw "Invalid verified resource or digest: $Resource"
    }
    $file = Join-Path $project $Resource.Substring(6)
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $Digest) {
        throw "Godot package digest mismatch: $Resource"
    }
    return $file
}

$descriptor = Get-Content -Raw -LiteralPath (Join-Path $project 'addons/frozen_swimmer/frozen_swimmer.gdextension')
$library = [regex]::Match($descriptor, '(?m)^\s*windows\.x86_64\s*=\s*"([^"]+)"\s*$')
$binding = Get-Content -Raw -LiteralPath (Join-Path $project 'research_data/shoreline-runtime.json') | ConvertFrom-Json
if (!$library.Success -or $library.Groups[1].Value -cne $binding.dll_path) {
    throw 'The Windows Godot extension and the verified shoreline select different native libraries.'
}
$null = Confirm-Resource $binding.dll_path $binding.dll_sha256
foreach ($receipt in @($binding.cpu_source, $binding.gpu_source, $binding.bridge_equivalence, $binding.candidate_source)) {
    $null = Confirm-Resource $receipt.path $receipt.sha256
}
$study = Get-Content -LiteralPath (Join-Path $project $binding.candidate_source.path.Substring(6)) -Raw | ConvertFrom-Json
$steering = Get-Content -LiteralPath (Join-Path $project $binding.bridge_equivalence.path.Substring(6)) -Raw | ConvertFrom-Json
if ($steering.DLL_sha256 -ne $binding.dll_sha256 -or $steering.original_motor_sha256 -ne $study.playing_package_sha256) {
    throw 'The steering check refers to a different native library or motor.'
}
$null = Confirm-Resource 'res://research_data/learned-amplitude-v1.bin' $study.playing_package_sha256
foreach ($item in $study.sources.PSObject.Properties) {
    $null = Confirm-Resource $item.Value.path $item.Value.sha256
}
Write-Output ('Verified the Godot Windows library, saved motor, and recorded study package in {0:N3} s.' -f $watch.Elapsed.TotalSeconds)
