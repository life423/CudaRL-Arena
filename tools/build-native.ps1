param([Parameter(Mandatory)][ValidateSet('frozen_swimmer')][string]$Target)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$build=Join-Path $repo 'build'
$cache=Join-Path $build 'CMakeCache.txt'
$stagedOutput=Join-Path $build 'godot-staged'
$required=@{CUDARL_BUILD_GODOT='ON'}
$configure=-not(Test-Path -LiteralPath $cache)
foreach($entry in $required.GetEnumerator()){
    if(-not $configure -and -not(Select-String -LiteralPath $cache -Pattern ('^'+$entry.Key+':BOOL='+$entry.Value+'$') -Quiet)){$configure=$true}
}
if(-not $configure){
    $output=Select-String -LiteralPath $cache -Pattern '^CUDARL_GODOT_OUTPUT_DIR:PATH=(.*)$'|Select-Object -First 1
    if(-not $output -or [IO.Path]::GetFullPath($output.Matches[0].Groups[1].Value) -ne [IO.Path]::GetFullPath($stagedOutput)){$configure=$true}
}
$configureSeconds=0.0
if($configure){
    $flags=@($required.GetEnumerator()|ForEach-Object {'-D'+$_.Key+'='+$_.Value})
    $flags+=('-DCUDARL_GODOT_OUTPUT_DIR='+$stagedOutput)
    $watch=[Diagnostics.Stopwatch]::StartNew()
    & cmake -S $repo -B $build @flags | Out-Host
    if($LASTEXITCODE -ne 0){throw 'Native configuration failed'}
    $configureSeconds=$watch.Elapsed.TotalSeconds
}
# CMake/MSBuild already regenerate changed build inputs and compile only affected
# dependencies. A cached launch must not explicitly configure a second time.
$watch=[Diagnostics.Stopwatch]::StartNew()
& cmake --build $build --config Release --target $Target --parallel 2 | Out-Host
if($LASTEXITCODE -ne 0){throw ('Native build failed: '+$Target)}
[pscustomobject]@{target=$Target;explicit_configure=$configure;configure_seconds=$configureSeconds;build_seconds=$watch.Elapsed.TotalSeconds;staged_dll=(Join-Path $stagedOutput 'frozen_swimmer-steering-v1.dll');live_library_unchanged=$true}
