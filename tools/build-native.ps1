param([Parameter(Mandatory)][ValidateSet('frozen_swimmer','ppo_motor')][string]$Target)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$build=Join-Path $repo 'build'
$cache=Join-Path $build 'CMakeCache.txt'
$required=if($Target -eq 'frozen_swimmer'){@{CUDARL_BUILD_GODOT='ON'}}else{@{CUDARL_BUILD_PPO='ON';CUDARL_ENABLE_CUDA='ON'}}
$configure=-not(Test-Path -LiteralPath $cache)
foreach($entry in $required.GetEnumerator()){
    if(-not $configure -and -not(Select-String -LiteralPath $cache -Pattern ('^'+$entry.Key+':BOOL='+$entry.Value+'$') -Quiet)){$configure=$true}
}
$configureSeconds=0.0
if($configure){
    $flags=@($required.GetEnumerator()|ForEach-Object {'-D'+$_.Key+'='+$_.Value})
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
[pscustomobject]@{target=$Target;explicit_configure=$configure;configure_seconds=$configureSeconds;build_seconds=$watch.Elapsed.TotalSeconds}
