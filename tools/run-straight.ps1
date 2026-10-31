param(
    [ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Name=(Get-Date -AsUTC -Format 'yyyyMMdd-HHmmss'),
    [ValidateRange(1,600)][int]$FitSteps=600,
    [ValidateRange(1,20)][int]$WallSeconds=20,
    [switch]$SkipBuild
)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'run-status.psm1') -Force
$watch=[Diagnostics.Stopwatch]::StartNew()
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base=Join-Path $repo 'build/speed-motor'
$output=Join-Path $base $Name
if(Test-Path -LiteralPath $output){throw 'Run identity exists; no retry/overwrite'}
New-Item -ItemType Directory -Path $base -Force|Out-Null
$latest=Join-Path $base 'latest-status.json'
$script:currentPhase='preparing';$own=$null;$nativePid=$null;$nativeStart=$null;$lastProgress=@{}
function Status([string]$Phase,[hashtable]$Extra=@{}){
    $script:currentPhase=$Phase
    $data=@{phase=$Phase;pid=$nativePid;launcher_pid=$PID;started_utc=$nativeStart;heartbeat_utc=(Get-Date -AsUTC -Format o);utc=(Get-Date -AsUTC -Format o);run=$output;control_cap=1632;substep_cap=16320;wall_cap_seconds=$WallSeconds}
    foreach($key in $Extra.Keys){$data[$key]=$Extra[$key]}
    Write-RunStatus -Path $latest -Value $data
}
try {
Status 'preparing'
Status 'building'
if(-not $SkipBuild){
    & cmake -S $repo -B (Join-Path $repo 'build') -DCUDARL_BUILD_STROKE_MOTOR=ON
    if($LASTEXITCODE -ne 0){throw 'Configuration failed; no physical run'}
    & cmake --build (Join-Path $repo 'build') --config Release --target stroke_motor_demo --parallel 2
    if($LASTEXITCODE -ne 0){throw 'Build failed; no physical run'}
}
Status 'checking'
$exe=Get-Item -LiteralPath (Join-Path $repo 'build/Release/stroke_motor_demo.exe')
$sources=@(Get-ChildItem (Join-Path $repo 'src') -Recurse -File)+@(Get-Item (Join-Path $repo 'tools/stroke_motor_demo.cu'),$PSCommandPath,(Join-Path $repo 'CMakeLists.txt'),(Join-Path $PSScriptRoot 'run-status.psm1'))
$compiled=@($sources|Where-Object {$_.FullName -notmatch '[\\/]godot[\\/]' -and $_.Extension -in '.h','.cpp','.cu'})
if(@($compiled|Where-Object LastWriteTimeUtc -gt $exe.LastWriteTimeUtc).Count){throw 'Executable older than native sources; build first'}
New-Item -ItemType Directory -Path (Join-Path $output 'source') -Force|Out-Null
$manifest=@()
foreach($file in $sources){
    $relative=[IO.Path]::GetRelativePath($repo,$file.FullName);$target=Join-Path (Join-Path $output 'source') $relative
    New-Item -ItemType Directory -Path (Split-Path $target) -Force|Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $target
    $manifest+=@{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName).Hash}
}
$manifest|ConvertTo-Json|Set-Content (Join-Path $output 'source-manifest.json')
$actor=Join-Path $output 'initial-controller.bin';Copy-Item -LiteralPath (Join-Path $repo 'live/assets/frozen-controller.bin') -Destination $actor
$reference=Join-Path $output 'reference-8mm.csv';Copy-Item -LiteralPath (Join-Path $repo 'assets/steady-speed-v1/baseline-reference.csv') -Destination $reference
$exampleSource=Join-Path $repo 'assets/steady-speed-v1'
$proof=Get-Content (Join-Path $exampleSource 'example-provenance.json') -Raw|ConvertFrom-Json
if($proof.source_trial_accepted -or $proof.accepted_rows -ne 960 -or ($proof.accepted_profiles -join ',') -ne '0,1,2,3,4,5'){throw 'Accepted subset proof invalid'}
if((Get-FileHash (Join-Path $exampleSource 'teacher-gates.json')).Hash -ne $proof.curated_teacher_gates_sha256){throw 'Curated source gate evidence changed'}
if((Get-FileHash $reference).Hash -ne $proof.baseline_reference_sha256){throw 'Curated baseline reference changed'}
$gates=Get-Content (Join-Path $exampleSource 'teacher-gates.json') -Raw|ConvertFrom-Json
foreach($id in 0..5){if($gates[$id].profile -ne $id -or -not $gates[$id].passed){throw 'Source constant profile not accepted'}}
if((Get-FileHash (Join-Path $exampleSource 'accepted-examples.bin')).Hash -ne $proof.accepted_examples_sha256){throw 'Accepted examples changed'}
$examples=Join-Path $output 'accepted-examples.bin';Copy-Item -LiteralPath (Join-Path $exampleSource 'accepted-examples.bin') -Destination $examples
Copy-Item -LiteralPath (Join-Path $exampleSource 'example-provenance.json') -Destination (Join-Path $output 'example-provenance.json')
$frozenExe=Join-Path $output 'stroke_motor_demo.exe';Copy-Item -LiteralPath $exe.FullName -Destination $frozenExe
$argsList=@('--examples',$examples,'--actor',$actor,'--reference',$reference,'--fit-steps',"$FitSteps",'--wall-seconds',"$WallSeconds")
& $frozenExe --prepare $output @argsList
if($LASTEXITCODE -ne 0){throw 'Registration failed; no physical run'}
Get-FileHash $actor,$reference,$examples,(Join-Path $output 'example-provenance.json'),$frozenExe,(Join-Path $output 'protocol.bin'),(Join-Path $output 'source-manifest.json')|Select-Object Path,Hash|ConvertTo-Json|Set-Content (Join-Path $output 'frozen-hashes.json')
$setup=$watch.Elapsed.TotalSeconds
Status 'ready' @{setup_seconds=$setup;result_path=(Join-Path $output 'result.json')}
$processWatch=[Diagnostics.Stopwatch]::StartNew()
$processArgs=@('--run',('"'+$output+'"'),'--examples',('"'+$examples+'"'),'--actor',('"'+$actor+'"'),'--reference',('"'+$reference+'"'),'--fit-steps',"$FitSteps",'--wall-seconds',"$WallSeconds")
$own=Start-Process -FilePath $frozenExe -ArgumentList $processArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'native.log') -RedirectStandardError (Join-Path $output 'native.err')
$nativePid=$own.Id;$nativeStart=(Get-Date -AsUTC -Format o)
Status 'running' @{setup_seconds=$setup;process_seconds=0}
while(-not $own.WaitForExit(100)){
    if($processWatch.Elapsed.TotalSeconds -gt $WallSeconds+5){Stop-Process -Id $own.Id;Status 'wrapper_timeout';throw 'Own bounded process stopped; no retry'}
    $native=Join-Path $output 'status.json';$state=$null
    if(Test-Path $native){try{$state=Get-Content $native -Raw|ConvertFrom-Json}catch{}}
    if($null -ne $state){$lastProgress=@{setup_seconds=$setup;process_seconds=$processWatch.Elapsed.TotalSeconds;completed_updates=$state.completed_updates;controls=$state.controls;physical_substeps=$state.physical_substeps;result_path=(Join-Path $output 'result.json')};Status $state.phase $lastProgress}
}
$summary=Join-Path $output 'summary.json'
if(-not(Test-Path $summary)){Status 'no_summary';throw 'Run ended without summary; inspect evidence, no retry'}
$result=Get-Content $summary -Raw|ConvertFrom-Json
$result|Add-Member setup_seconds $setup
$result|Add-Member process_seconds $processWatch.Elapsed.TotalSeconds
$result|Add-Member run_path $output
$result|ConvertTo-Json -Depth 8|Set-Content (Join-Path $output 'result.json')
$finalPhase=if([string]::IsNullOrEmpty($result.fault)){'done'}else{'failed'}
Status $finalPhase @{setup_seconds=$setup;native_seconds=$result.wall_seconds;completed_updates=$result.supervised_steps;controls=$result.charged_controls;physical_substeps=$result.actual_native_substeps;result_path=(Join-Path $output 'result.json');selected_candidate=$result.selected_candidate;fault=$result.fault}
$result|ConvertTo-Json -Depth 8 -Compress|Write-Output
exit $own.ExitCode
} catch {
    $failure=$_
    if($null -ne $own -and -not $own.HasExited){Stop-Process -Id $own.Id}
    Status 'failed' @{failed_phase=$script:currentPhase;error=$failure.Exception.Message;last_progress=$lastProgress;result_path=(Join-Path $output 'result.json')}
    Write-Error $failure
    exit 1
}
