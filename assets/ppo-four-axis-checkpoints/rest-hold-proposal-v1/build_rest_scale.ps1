$ErrorActionPreference='Stop'
$repo='C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation'
$analysis=Join-Path $repo 'build/analysis/rest-hold-proposal-v1'
if(Test-Path -LiteralPath $analysis){throw 'Proposal analysis identity exists'}
New-Item -ItemType Directory -Path $analysis|Out-Null
Copy-Item -LiteralPath 'C:\Users\aiand\Documents\Codex\2026-11-04\task\rest_scale_probe.cpp' -Destination (Join-Path $analysis 'rest_scale_probe.cpp')
@'
cmake_minimum_required(VERSION 3.24)
project(RestScaleProbe LANGUAGES CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(REPO "C:/Users/aiand/Documents/programming/CudaRL-Arena/.worktrees/minimal-foundation")
add_executable(rest_scale_probe rest_scale_probe.cpp "${REPO}/src/core/checkpoint.cpp")
target_include_directories(rest_scale_probe PRIVATE "${REPO}/src" "${REPO}/src/nn")
target_compile_features(rest_scale_probe PRIVATE cxx_std_17)
target_compile_options(rest_scale_probe PRIVATE /W4 /WX /permissive-)
'@ | Set-Content -LiteralPath (Join-Path $analysis 'CMakeLists.txt')
$cmake=(Get-Command cmake).Source
$clock=[Diagnostics.Stopwatch]::StartNew()
foreach($phase in @('configure','build')){
 $args=if($phase -eq 'configure'){@('-S',('"'+$analysis+'"'),'-B',('"'+(Join-Path $analysis 'build')+'"'),'-G','"Visual Studio 17 2022"','-A','x64')}else{@('--build',('"'+(Join-Path $analysis 'build')+'"'),'--config','Release','--parallel','2')}
 $native=Start-Process -FilePath $cmake -ArgumentList $args -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $analysis ($phase+'.log')) -RedirectStandardError (Join-Path $analysis ($phase+'.err'))
 while(-not $native.WaitForExit(100)){if($clock.Elapsed.TotalSeconds -ge 60){$native.Kill($true);throw '60s nonphysical build cap exhausted'}}
 $native.Refresh();if($native.ExitCode -ne 0){Get-Content (Join-Path $analysis ($phase+'.log')) -Tail 25;Get-Content (Join-Path $analysis ($phase+'.err'));throw 'Nonphysical probe build failed'}
}
$exe=Join-Path $analysis 'build/Release/rest_scale_probe.exe'
$native=Start-Process -FilePath $exe -ArgumentList @(('"'+$repo+'"'),('"'+(Join-Path $analysis 'results')+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $analysis 'probe.log') -RedirectStandardError (Join-Path $analysis 'probe.err')
if(-not $native.WaitForExit(10000)){$native.Kill($true);throw '10s static arithmetic cap exhausted'}
$native.Refresh();Get-Content (Join-Path $analysis 'probe.log');Get-Content (Join-Path $analysis 'probe.err')
if($native.ExitCode -ne 0){throw 'Static probe failed; no physics or training attempted'}
@{elapsed_seconds=$clock.Elapsed.TotalSeconds;physical_controls=0;optimizer_steps=0;exit_code=$native.ExitCode}|ConvertTo-Json|Set-Content (Join-Path $analysis 'execution.json')
Get-Content (Join-Path $analysis 'results/rest-starts.csv');Get-Content (Join-Path $analysis 'results/instantaneous-reward-grid.csv')
