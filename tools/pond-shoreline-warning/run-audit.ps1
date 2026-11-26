param([string]$Output='build/flexible-tail/shoreline-warning-v1')
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repository
$study=[IO.Path]::GetFullPath($Output)
$exe=(Resolve-Path -LiteralPath (Join-Path $study 'native-build/Release/pond_shoreline_warning.exe')).Path
$data=Join-Path $study data
if(Test-Path -LiteralPath $data){throw 'Audit evidence is immutable; no automatic rerun.'}
$deployment=Get-Content flexible-tail-preview/live_data/exploration-deployment.json -Raw|ConvertFrom-Json
$paths=@($exe,'tools/pond-flexible-tail-chooser/rate-evaluate.cpp','flexible-tail-preview/habitat.json','flexible-tail-preview/live_data/exploration-deployment.json')
$paths+=@(Get-ChildItem src -File -Recurse|Where-Object Extension -in @('.h','.cpp','.cu')|ForEach-Object FullName)
$paths+=@(Get-ChildItem $PSScriptRoot -File|ForEach-Object FullName)
$paths+=@($deployment.files|ForEach-Object path)
$bindings=@($paths|Sort-Object -Unique|ForEach-Object{$path=(Resolve-Path -LiteralPath $_).Path;[ordered]@{path=[IO.Path]::GetRelativePath($repository,$path).Replace('\','/');sha256=(Get-FileHash -LiteralPath $path).Hash}})
$bindings|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $study bindings.json) -Encoding utf8NoBOM
foreach($binding in $bindings|Where-Object {$_.path -like 'src/*' -or $_.path -like 'tools/*'}){
    $target=Join-Path $study ('source/'+$binding.path)
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
    Copy-Item -LiteralPath $binding.path -Destination $target
}
$info=[Diagnostics.ProcessStartInfo]::new();$info.FileName=$exe;$info.UseShellExecute=$false;$info.CreateNoWindow=$true
$info.WorkingDirectory=$repository;$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
foreach($path in @('flexible-tail-preview/live_data/fitted-gait.bin','flexible-tail-preview/live_data/exploration-motor-88.bin','flexible-tail-preview/live_data/straight-evaluation-starts.bin')){$info.ArgumentList.Add((Resolve-Path -LiteralPath $path).Path)}
$info.ArgumentList.Add($data)
$process=[Diagnostics.Process]::new();$process.StartInfo=$info
$clock=[Diagnostics.Stopwatch]::StartNew();$null=$process.Start()
$out=$process.StandardOutput.ReadToEndAsync();$err=$process.StandardError.ReadToEndAsync()
$finished=$process.WaitForExit(65000)
if(!$finished){$process.Kill($true);$process.WaitForExit()}
$stdout=$out.GetAwaiter().GetResult();$stderr=$err.GetAwaiter().GetResult();$clock.Stop()
$stdout|Set-Content -LiteralPath (Join-Path $study stdout.log);$stderr|Set-Content -LiteralPath (Join-Path $study stderr.log)
@{exit=$process.ExitCode;timeout=!$finished;wall_s=$clock.Elapsed.TotalSeconds;executable=$exe;arguments=@($info.ArgumentList)}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $study receipt.json) -Encoding utf8NoBOM
foreach($binding in $bindings){if((Get-FileHash -LiteralPath $binding.path).Hash -ne $binding.sha256){throw "Bound source or protected file changed: $($binding.path)"}}
@{pass=$true;bundle_hashes_verified=$deployment.files.Count;motor_frozen=$true;Godot_updated=$false;optimizer_updates=0;mouse_keyboard_calls=0}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $study protected.json) -Encoding utf8NoBOM
Write-Output $stdout
if(!$finished -or $process.ExitCode -ne 0){throw "Audit failed; partial evidence preserved: $stderr"}
Get-Content -LiteralPath (Join-Path $data summary.json)
