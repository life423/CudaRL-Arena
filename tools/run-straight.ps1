param(
    [ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Name='signed-transitions-v1',
    [switch]$Run,
    [switch]$SkipBuild
)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'run-status.psm1') -Force
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base=Join-Path $repo 'build/speed-motor'
$output=Join-Path $base $Name
$latest=Join-Path $base 'latest-status.json'
$prepareWatch=[Diagnostics.Stopwatch]::StartNew()
$timing=[pscustomobject]@{explicit_configure=$false;configure_seconds=0.;build_seconds=0.}
$nativePid=$null;$started=$null;$phase='preparing';$own=$null
function Status([string]$Next,[hashtable]$Extra=@{}){
    $script:phase=$Next
    $value=@{phase=$Next;run=$output;pid=$nativePid;started_utc=$started;utc=[DateTime]::UtcNow.ToString('o');control_cap=3680;substep_cap=36800;optimizer_cap=600;native_wall_seconds=20;process_wall_seconds=25}
    foreach($key in $Extra.Keys){$value[$key]=$Extra[$key]}
    Write-RunStatus -Path $latest -Value $value
}
try {
    New-Item -ItemType Directory -Force $base|Out-Null
    if(-not $Run){
        if(Test-Path -LiteralPath $output){throw 'Prepared identity exists; choose a new name, no overwrite'}
        Status 'building'
        if(-not $SkipBuild){$timing=& (Join-Path $PSScriptRoot 'build-native.ps1') -Target stroke_motor_demo}
        Status 'checking'
        $asset=Join-Path $repo 'assets/steady-speed-v1'
        $proof=Get-Content (Join-Path $asset 'example-provenance.json') -Raw|ConvertFrom-Json
        if($proof.source_trial_accepted -or $proof.accepted_rows -ne 960 -or ($proof.accepted_profiles -join ',') -ne '0,1,2,3,4,5'){throw 'Invalid accepted-source lineage'}
        foreach($pair in @(@('accepted-examples.bin','accepted_examples_sha256'),@('teacher-gates.json','curated_teacher_gates_sha256'))){if((Get-FileHash (Join-Path $asset $pair[0])).Hash -ne $proof.($pair[1])){throw ('Source identity changed: '+$pair[0])}}
        $gates=Get-Content (Join-Path $asset 'teacher-gates.json') -Raw|ConvertFrom-Json
        foreach($id in 0..5){if($gates[$id].profile -ne $id -or -not $gates[$id].passed){throw 'Source constant gate is not accepted'}}
        $exe=Get-Item (Join-Path $repo 'build/Release/stroke_motor_demo.exe')
        $nativeSources=@(Get-ChildItem (Join-Path $repo 'src') -Recurse -File|Where-Object {$_.FullName -notmatch '[\\/]godot[\\/]' -and $_.Extension -in '.h','.cpp','.cu'})+@(Get-Item (Join-Path $PSScriptRoot 'stroke_motor_demo.cu'),(Join-Path $PSScriptRoot 'transition_motor.h'))
        if(@($nativeSources|Where-Object LastWriteTimeUtc -gt $exe.LastWriteTimeUtc).Count){throw 'Native runner is stale; build first'}
        New-Item -ItemType Directory -Path $output|Out-Null
        Copy-Item $exe.FullName (Join-Path $output 'runner.exe')
        Copy-Item (Join-Path $repo 'live/assets/frozen-controller.bin') (Join-Path $output 'initial-controller.bin')
        Copy-Item (Join-Path $asset 'accepted-examples.bin') (Join-Path $output 'source-examples.bin')
        Copy-Item (Join-Path $asset 'example-provenance.json') (Join-Path $output 'source-provenance.json')
        $manifest=@()
        foreach($file in $nativeSources){$relative=[IO.Path]::GetRelativePath($repo,$file.FullName);$destination=Join-Path (Join-Path $output 'source') $relative;New-Item -ItemType Directory -Force (Split-Path $destination)|Out-Null;Copy-Item $file.FullName $destination;$manifest+=@{path=$relative;sha256=(Get-FileHash $file.FullName).Hash}}
        $manifest|ConvertTo-Json|Set-Content (Join-Path $output 'source-manifest.json')
        & (Join-Path $output 'runner.exe') --prepare-transitions $output --actor (Join-Path $output 'initial-controller.bin') --examples (Join-Path $output 'source-examples.bin')
        if($LASTEXITCODE){throw 'Nonphysical transition registration failed'}
        Get-FileHash (Join-Path $output 'runner.exe'),(Join-Path $output 'initial-controller.bin'),(Join-Path $output 'source-examples.bin'),(Join-Path $output 'source-provenance.json'),(Join-Path $output 'source-manifest.json'),(Join-Path $output 'transition-protocol.bin')|Select-Object Path,Hash|ConvertTo-Json|Set-Content (Join-Path $output 'frozen-hashes.json')
        $timing|Add-Member preparation_seconds ([Math]::Max(0.0,$prepareWatch.Elapsed.TotalSeconds-$timing.configure_seconds-$timing.build_seconds))
        $timing|ConvertTo-Json|Set-Content (Join-Path $output 'preparation-timing.json')
        Status 'ready' @{physical_substeps=0;optimizer_steps=0;preparation_seconds=$prepareWatch.Elapsed.TotalSeconds;run_command=('./tools/run-straight.ps1 -Name '+$Name+' -Run')}
        return
    }
    if(-not(Test-Path (Join-Path $output 'transition-readiness.json'))){throw 'Prepare this identity before explicitly authorizing collection/fitting'}
    if(Test-Path (Join-Path $output 'transition-started')){throw 'Trial already attempted; no retries'}
    foreach($file in (Get-Content (Join-Path $output 'frozen-hashes.json') -Raw|ConvertFrom-Json)){if((Get-FileHash -LiteralPath $file.Path).Hash -ne $file.Hash){throw ('Frozen input changed: '+$file.Path)}}
    $started=[DateTime]::UtcNow.ToString('o');$watch=[Diagnostics.Stopwatch]::StartNew()
    $argsList=@('--run-transitions',('"'+$output+'"'),'--actor',('"'+(Join-Path $output 'initial-controller.bin')+'"'),'--examples',('"'+(Join-Path $output 'source-examples.bin')+'"'))
    $own=Start-Process -FilePath (Join-Path $output 'runner.exe') -ArgumentList $argsList -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'native.log') -RedirectStandardError (Join-Path $output 'native.err')
    $nativePid=$own.Id;Status 'running';Write-Output ('STARTED '+$started+' pid='+$nativePid)
    while(-not $own.WaitForExit(100)){
        if($watch.Elapsed.TotalSeconds -ge 25){$own.Kill();throw 'Own process wall cap reached; no retry'}
        $progress=$null;try{$progress=Get-Content (Join-Path $output 'transition-status.json') -Raw|ConvertFrom-Json}catch{}
        if($null -ne $progress){Status $progress.phase @{controls=$progress.controls;physical_substeps=$progress.physical_substeps;optimizer_steps=$progress.optimizer_steps;native_phase_unix_seconds=$progress.unix_seconds}}
    }
    $result=Get-Content (Join-Path $output 'transition-result.json') -Raw|ConvertFrom-Json
    $result|Add-Member started_utc $started;$result|Add-Member ended_utc ([DateTime]::UtcNow.ToString('o'));$result|Add-Member process_seconds $watch.Elapsed.TotalSeconds
    $result|ConvertTo-Json -Depth 6|Set-Content (Join-Path $output 'result.json')
    Status $(if($result.fault){'failed'}else{'done'}) @{controls=$result.controls;physical_substeps=$result.physical_substeps;optimizer_steps=$result.optimizer_steps;candidate_accepted=$result.candidate_accepted;fault=$result.fault}
    $result|ConvertTo-Json -Compress|Write-Output
    if($own.ExitCode){throw 'Bounded trial stopped; no retry'}
} catch {
    if($null -ne $own -and -not $own.HasExited){$own.Kill()}
    Status 'failed' @{failed_phase=$phase;error=$_.Exception.Message}
    throw
}
