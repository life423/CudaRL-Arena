param(
    [string]$SessionDir='build/flexible-tail/whole-beat-pond-v2',
    [string]$BuildDirectory='build/flexible-tail/whole-beat-pond-v2-build',
    [switch]$Resume,
    [ValidateRange(1,64)][int]$TargetUpdates=64,
    [ValidateRange(60,1200)][int]$WallSeconds=1200
)

$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$active=[Collections.Generic.List[object]]::new()
$lease=$null
$trainer=$null
$training=$null
Push-Location -LiteralPath $repository

function Write-AtomicJson([string]$Path,$Value) {
    $Value | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath ($Path+'.tmp') -Encoding utf8
    [IO.File]::Move($Path+'.tmp',$Path,$true)
}
function Bind-Files([string[]]$Paths) {
    foreach($path in ($Paths | Sort-Object -Unique)) {
        $absolute=(Resolve-Path -LiteralPath $path).Path
        [pscustomobject]@{path=[IO.Path]::GetRelativePath($repository,$absolute).Replace('\','/');
            sha256=(Get-FileHash -LiteralPath $absolute).Hash}
    }
}
function Assert-Bindings($Bindings) {
    foreach($binding in $Bindings) {
        $path=Join-Path $repository $binding.path
        if(!(Test-Path -LiteralPath $path) -or (Get-FileHash -LiteralPath $path).Hash -ne $binding.sha256) {
            throw "Bound source or input changed: $($binding.path)"
        }
    }
}
function Start-RecordedProcess([string]$Name,[string]$Executable,[string[]]$Arguments,[string]$Directory,[double]$LimitSeconds) {
    $null=New-Item -Path $Directory -ItemType Directory -Force
    $info=[Diagnostics.ProcessStartInfo]::new()
    $info.FileName=$Executable;$info.WorkingDirectory=$repository
    $info.UseShellExecute=$false;$info.CreateNoWindow=$true
    $info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    foreach($argument in $Arguments){$info.ArgumentList.Add($argument)}
    $stdout=[IO.File]::Open((Join-Path $Directory "$Name.stdout.log"),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $stderr=[IO.File]::Open((Join-Path $Directory "$Name.stderr.log"),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {
        $started=[DateTime]::UtcNow;$watch=[Diagnostics.Stopwatch]::StartNew()
        $process=[Diagnostics.Process]::Start($info)
        $outTask=$process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $errTask=$process.StandardError.BaseStream.CopyToAsync($stderr)
    } catch {$stdout.Dispose();$stderr.Dispose();throw}
    $job=[pscustomobject]@{name=$Name;process=$process;watch=$watch;stdout=$stdout;stderr=$stderr;outTask=$outTask;errTask=$errTask;
        executable=$Executable;arguments=$Arguments;directory=$Directory;limit_s=$LimitSeconds;started_utc=$started.ToString('o');finished=$false;receipt=$null}
    $active.Add($job)
    return $job
}
function Finish-RecordedProcess($Job,[bool]$TimedOut=$false,[bool]$Interrupted=$false) {
    if($Job.finished){return $Job.receipt}
    if(!$Job.process.HasExited){$Job.process.Kill($true);$null=$Job.process.WaitForExit(5000)}
    $Job.watch.Stop()
    $Job.outTask.GetAwaiter().GetResult();$Job.errTask.GetAwaiter().GetResult()
    $Job.stdout.Dispose();$Job.stderr.Dispose()
    $receipt=[pscustomobject]@{phase=$Job.name;started_utc=$Job.started_utc;wall_s=$Job.watch.Elapsed.TotalSeconds;
        limit_s=$Job.limit_s;exit=$Job.process.ExitCode;timeout=$TimedOut;interrupted=$Interrupted;
        executable=$Job.executable;arguments=$Job.arguments}
    Write-AtomicJson (Join-Path $Job.directory "$($Job.name).receipt.json") $receipt
    $Job.finished=$true;$Job.receipt=$receipt
    Write-Host ('{0}: {1:F3}s, exit {2}, timeout {3}' -f $Job.name,$receipt.wall_s,$receipt.exit,$TimedOut)
    return $receipt
}
function Wait-RecordedProcess($Job,[int[]]$AcceptedExits=@(0)) {
    while(!$Job.process.HasExited -and $Job.watch.Elapsed.TotalSeconds -le $Job.limit_s){$null=$Job.process.WaitForExit(250)}
    $receipt=Finish-RecordedProcess $Job (!$Job.process.HasExited)
    if($receipt.timeout -or $receipt.exit -notin $AcceptedExits){throw "$($Job.name) failed; all evidence is retained."}
    return $receipt
}
function Read-Assessment([string]$Directory,$Receipt,[string]$Horizon) {
    if($Receipt.timeout -or $Receipt.exit -notin @(0,1)){throw 'Pond evaluation failed before completing its declared scoring.'}
    $path=Join-Path $Directory 'assessment'
    $summary=Get-Content -LiteralPath (Join-Path $path 'summary.json') -Raw | ConvertFrom-Json
    if([bool]$summary.pass -ne ($Receipt.exit -eq 0) -or $summary.steering_mode -ne 'held-heading' -or
        $summary.motor_feedback_mode -ne 'whole-beat' -or $summary.horizon_mode -ne $Horizon){throw 'Pond evaluator contract differs from the registered modes.'}
    $cases=@(Import-Csv -LiteralPath (Join-Path $path 'cases.csv'))
    if(@($cases | Where-Object {[int]$_.fault -in @(1,3,6)}).Count){throw 'Pond evaluation reported invalid physics, numeric or runtime state.'}
    return $summary
}

try {
    $watch=[Diagnostics.Stopwatch]::StartNew()
    $session=[IO.Path]::GetFullPath($SessionDir);$build=[IO.Path]::GetFullPath($BuildDirectory)
    if(!$Resume -and (Test-Path -LiteralPath $session)){throw 'A new session requires an unused output directory.'}
    if($Resume -and !(Test-Path -LiteralPath (Join-Path $session 'manifest.json'))){throw 'Resume requires the existing full session manifest.'}
    if(!$Resume){$null=New-Item -Path $session -ItemType Directory}
    $lease=[IO.File]::Open((Join-Path $session '.workflow.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $gait=(Resolve-Path 'flexible-tail-preview/live_data/fitted-gait.bin').Path
    $offset=(Resolve-Path 'build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin').Path
    $metadata=(Resolve-Path 'build/flexible-tail/whole-beat-steering-v2/training/actor-56.metadata.json').Path
    $motorAssessment=(Resolve-Path 'build/flexible-tail/whole-beat-steering-v2/evaluations/update-056/assessment/result.json').Path
    $motorManifest=(Resolve-Path 'build/flexible-tail/whole-beat-steering-v2/manifest.json').Path
    $motorAudit=(Resolve-Path 'build/flexible-tail/whole-beat-steering-v2/saved-data-audit/report.json').Path
    $motor=Get-Content -LiteralPath $metadata -Raw | ConvertFrom-Json
    $qualifiedMotor=Get-Content -LiteralPath $motorAssessment -Raw | ConvertFrom-Json
    $audit=Get-Content -LiteralPath $motorAudit -Raw | ConvertFrom-Json
    if($motor.feedback -ne 'whole-beat' -or $motor.update -ne 56 -or !$audit.pass -or
        $qualifiedMotor.candidate_15_passes -ne 8 -or $qualifiedMotor.candidate_straight_retention -ne 12 -or
        !$qualifiedMotor.all_inactive_commands_exact -or !$qualifiedMotor.paired_retention_final_states_exact -or
        $qualifiedMotor.global_faults -ne 0){throw 'The motor lacks the declared pond-stage retention eligibility.'}
    $starts=(Resolve-Path 'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin').Path
    $heldout=(Resolve-Path 'flexible-tail-preview/live_data/straight-evaluation-starts.bin').Path
    $initial=(Resolve-Path 'assets/research-evidence/pond-flexible-tail-chooser-sustained-v3/data/training/actor-56.bin').Path
    $staticProof=(Resolve-Path 'build/flexible-tail/whole-beat-chooser-static-v2/evidence/report.json').Path
    $static=Get-Content -LiteralPath $staticProof -Raw | ConvertFrom-Json
    if(!$static.pass){throw 'Matched motor/chooser integration proof failed.'}
    $inputs=@(Bind-Files @($gait,$offset,$metadata,$motorAssessment,$motorManifest,$motorAudit,$starts,$heldout,$initial,
        $staticProof,
        'flexible-tail-preview/live_data/yaw-offset-update-8.bin','src/physics/flexible_tail.h','flexible-tail-preview/habitat.json',
        'tools/pond-flexible-tail-steering/BEAT_FEEDBACK_PROTOCOL.md','tools/pond-flexible-tail-chooser/POND_STAGE_PROTOCOL.md'))
    $protected=@{
        'flexible-tail-preview/live_data/fitted-gait.bin'='2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6'
        'flexible-tail-preview/live_data/yaw-offset-update-8.bin'='52BA55F110DD1227CBFEB1E47E6199E50DE74687E82E9DD651114CDA86FEC354'
        'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin'='75E6AA2504726082E84E286FD5B3ADF2B7B9D4F7D4797116EF8A682F72DEC025'
        'flexible-tail-preview/live_data/straight-evaluation-starts.bin'='4A081CCC3DEA8C7310191902CC094E50FA6C0B341606DC9DBC4E63DD581FFF39'
        'src/physics/flexible_tail.h'='B6878791EC1A3CC68112866DE8C105390A43AC143A8EDCBA9EA2BF2430CCB226'
        'flexible-tail-preview/habitat.json'='554E9E0976F22CD6B473EAAB84559C7B9887358C4B620CEF4592D5FFFEC0C7DF'
    }
    foreach($binding in $inputs){if($protected.ContainsKey($binding.path) -and $protected[$binding.path] -ne $binding.sha256){throw "Protected input differs: $($binding.path)"}}
    $paths=@(
        Get-ChildItem -LiteralPath 'src' -Recurse -File | Where-Object Extension -in @('.h','.cpp','.cu') | Select-Object -ExpandProperty FullName
        Get-ChildItem -LiteralPath $PSScriptRoot -File | Select-Object -ExpandProperty FullName
        (Resolve-Path 'tools/pond-flexible-tail-check/habitat_config.h.in').Path
    )
    $sources=@(Bind-Files $paths)
    $text=($sources | ForEach-Object {$_.path+' '+$_.sha256}) -join "`n"
    $fingerprint=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($text)))
    if($Resume){
        $manifest=Get-Content -LiteralPath (Join-Path $session 'manifest.json') -Raw | ConvertFrom-Json
        if($manifest.source_fingerprint -ne $fingerprint -or $manifest.max_updates -ne $TargetUpdates -or $manifest.training_wall_budget_s -ne $WallSeconds){throw 'Resume source or budget identity differs.'}
        Assert-Bindings $manifest.sources;Assert-Bindings $manifest.inputs
    }else{
        Write-AtomicJson (Join-Path $session 'manifest.json') @{schema=1;created_utc=[DateTime]::UtcNow.ToString('o');
            source_fingerprint=$fingerprint;sources=$sources;inputs=$inputs;worlds=1024;decisions_per_world=8;
            max_updates=$TargetUpdates;training_wall_budget_s=$WallSeconds;steering='held-heading';feedback='whole-beat';training_horizon='long-horizon';evaluation_interval=8}
        Copy-Item -LiteralPath 'tools/pond-flexible-tail-chooser/POND_STAGE_PROTOCOL.md' -Destination (Join-Path $session 'protocol.md')
        Copy-Item -LiteralPath 'tools/pond-flexible-tail-steering/BEAT_FEEDBACK_PROTOCOL.md' -Destination (Join-Path $session 'motor-protocol.md')
        foreach($binding in $sources){$destination=Join-Path (Join-Path $session 'source') $binding.path;$null=New-Item -Path (Split-Path -Parent $destination) -ItemType Directory -Force;Copy-Item -LiteralPath (Join-Path $repository $binding.path) -Destination $destination}
        Write-AtomicJson (Join-Path $session 'frozen-motor.json') @{path=$offset;sha256=(Get-FileHash $offset).Hash;metadata=$motor;qualification=$qualifiedMotor;godot_published=$false}
    }
    $trainExe=Join-Path $build 'Release/pond_flexible_tail_chooser_session.exe'
    $evalExe=Join-Path $build 'Release/pond_flexible_tail_chooser_eval.exe'
    $stamp=Join-Path $build 'beat-workflow-fingerprint.json'
    $rebuild=!(Test-Path $stamp) -or !(Test-Path $trainExe) -or !(Test-Path $evalExe)
    if(!$rebuild){$rebuild=(Get-Content $stamp -Raw | ConvertFrom-Json).source_fingerprint -ne $fingerprint}
    if($Resume -and $rebuild){throw 'Resume requires its existing bound executables.'}
    if($rebuild){
        $cmake=(Get-Command cmake).Source;$prep=Join-Path $session 'compilation'
        $null=Wait-RecordedProcess (Start-RecordedProcess 'configure' $cmake @('-S',$PSScriptRoot,'-B',$build,'-G','Visual Studio 17 2022','-A','x64') $prep 600)
        $null=Wait-RecordedProcess (Start-RecordedProcess 'compile' $cmake @('--build',$build,'--config','Release','--target','pond_flexible_tail_chooser_session','pond_flexible_tail_chooser_eval','--parallel','4') $prep 600)
        Write-AtomicJson $stamp @{source_fingerprint=$fingerprint}
    }
    if($Resume){Assert-Bindings (Get-Content (Join-Path $session 'executables.json') -Raw | ConvertFrom-Json)}
    else{Write-AtomicJson (Join-Path $session 'executables.json') @(Bind-Files @($trainExe,$evalExe))}
    if(!$Resume){
        $baseline=Join-Path $session 'frozen-chooser'
        $receipt=Wait-RecordedProcess (Start-RecordedProcess 'evaluation' $evalExe @($gait,$offset,$heldout,$initial,(Join-Path $baseline 'assessment'),'held-heading','whole-beat','original-horizon') $baseline 20) @(0,1)
        $summary=Read-Assessment $baseline $receipt 'original-horizon'
        Write-AtomicJson (Join-Path $baseline 'admission.json') @{qualified=[bool]$summary.pass;summary=$summary;chooser=$initial;godot_published=$false}
        if($summary.pass){
            Write-AtomicJson (Join-Path $session 'result.json') @{stage='qualified-frozen-chooser';qualified=$true;training_ran=$false;
                actor=$initial;motor=$offset;steering='held-heading';feedback='whole-beat';horizon='original-horizon';godot_published=$false;workflow_wall_s=$watch.Elapsed.TotalSeconds}
            return
        }
        $verification=Join-Path $session 'verification'
        $null=Wait-RecordedProcess (Start-RecordedProcess 'verification' $trainExe @('verify-resume',$gait,$offset,$starts,$initial,(Join-Path $verification 'data'),'held-heading','whole-beat','long-horizon') $verification 600)
        $proof=Get-Content (Join-Path $verification 'data/verification.json') -Raw | ConvertFrom-Json
        if(!$proof.pass -or $proof.physical_updates_executed -ne 3 -or $proof.steering_mode -ne 'held-heading' -or
            $proof.motor_feedback -ne 'whole-beat' -or $proof.horizon -ne 'long-horizon'){throw 'Exact full-state CUDA resume proof failed.'}
    }
    $invocations=Join-Path $session 'invocations';$null=New-Item -Path $invocations -ItemType Directory -Force
    $used=0.;foreach($file in Get-ChildItem $invocations -Filter training.receipt.json -File -Recurse){$r=Get-Content $file.FullName -Raw | ConvertFrom-Json;if($r.timeout -or $r.exit -ne 0){throw 'A failed training invocation cannot be silently resumed.'};$used+=$r.wall_s}
    $remaining=[math]::Floor($WallSeconds-$used)
    if($remaining -lt 1){throw 'Registered training wall budget spent.'}
    $index=@(Get-ChildItem $invocations -Directory).Count+1;$invocation=Join-Path $invocations ('run-{0:D4}' -f $index)
    $training=Join-Path $session 'training'
    $arguments=if($Resume){@('resume',$gait,$offset,$starts,$training,"$TargetUpdates","$remaining",'held-heading','whole-beat','long-horizon')}
        else{@('new',$gait,$offset,$starts,$initial,$training,"$TargetUpdates","$remaining",'held-heading','whole-beat','long-horizon')}
    $trainer=Start-RecordedProcess 'training' $trainExe $arguments $invocation ($remaining+600)
    $evalRoot=Join-Path $session 'evaluations';$null=New-Item -Path $evalRoot -ItemType Directory -Force
    $pending=[Collections.Generic.SortedSet[int]]::new();$checked=[Collections.Generic.HashSet[int]]::new()
    $evaluation=$null;$finalQueued=$false;$qualified=@()
    foreach($dir in Get-ChildItem $evalRoot -Directory){if($dir.Name -match '^update-(\d+)$'){
        $admission=Join-Path $dir.FullName 'admission.json';if(!(Test-Path $admission)){throw 'Incomplete prior evaluation is retained; no automatic retry.'}
        $saved=Get-Content $admission -Raw | ConvertFrom-Json;$null=$checked.Add([int]$Matches[1]);if($saved.qualified){$qualified+=$saved}
    }}
    while(!$trainer.finished -or $evaluation -or $pending.Count -gt 0 -or !$finalQueued){
        foreach($file in Get-ChildItem $training -Filter 'actor-*.bin' -File -ErrorAction SilentlyContinue){
            if($file.BaseName -match '^actor-(\d+)$'){$update=[int]$Matches[1];if($update -gt 0 -and $update%8 -eq 0 -and !$checked.Contains($update)){$null=$pending.Add($update)}}
        }
        if(!$trainer.finished -and ($trainer.process.HasExited -or $trainer.watch.Elapsed.TotalSeconds -gt $trainer.limit_s)){
            $r=Finish-RecordedProcess $trainer (!$trainer.process.HasExited)
            if($r.timeout -or $r.exit -ne 0){throw 'Training process faulted or reached its external cap; no restart.'}
            $status=Get-Content (Join-Path $training 'result.json') -Raw | ConvertFrom-Json
            $last=[int]$status.updates_completed;if($last -gt 0 -and !$checked.Contains($last)){$null=$pending.Add($last)};$finalQueued=$true
        }
        if($evaluation -and ($evaluation.process.HasExited -or $evaluation.watch.Elapsed.TotalSeconds -gt $evaluation.limit_s)){
            $r=Finish-RecordedProcess $evaluation (!$evaluation.process.HasExited)
            $summary=Read-Assessment $evaluation.directory $r 'long-horizon'
            $candidate=Get-Content (Join-Path $evaluation.directory 'candidate.json') -Raw | ConvertFrom-Json
            if((Get-FileHash $candidate.path).Hash -ne $candidate.sha256){throw 'Immutable chooser candidate changed during assessment.'}
            $admission=@{qualified=[bool]$summary.pass;summary=$summary;actor=$candidate;godot_published=$false}
            Write-AtomicJson (Join-Path $evaluation.directory 'admission.json') $admission
            if($summary.pass){$qualified+=$admission}
            Write-Host ('navigation {0}/12, endurance {1}/{2}, clearance stops {3}' -f $summary.diagnostic_passes,$summary.endurance_passes,$summary.endurance_attempts,$summary.clearance_stops)
            $evaluation=$null
        }
        if(!$evaluation -and $pending.Count -gt 0){
            $update=$pending.Min;$null=$pending.Remove($update);$null=$checked.Add($update)
            $directory=Join-Path $evalRoot ('update-{0:D4}' -f $update)
            if(Test-Path $directory){throw 'Incomplete assessment directory exists; no automatic retry.'}
            $null=New-Item -Path $directory -ItemType Directory
            $actor=Join-Path $training "actor-$update.bin"
            Write-AtomicJson (Join-Path $directory 'candidate.json') @{update=$update;path=$actor;sha256=(Get-FileHash $actor).Hash;
                steering='held-heading';feedback='whole-beat';horizon='long-horizon'}
            $evaluation=Start-RecordedProcess 'evaluation' $evalExe @($gait,$offset,$heldout,$actor,(Join-Path $directory 'assessment'),'held-heading','whole-beat','long-horizon') $directory 20
        }
        if(!$trainer.finished -or $evaluation){[Threading.Thread]::Sleep(200)}
    }
    Assert-Bindings $sources;Assert-Bindings $inputs
    Write-AtomicJson (Join-Path $invocation 'completion.json') @{completed_utc=[DateTime]::UtcNow.ToString('o');
        evaluated_updates=@($checked | Sort-Object);qualified=$qualified.Count -gt 0;godot_published=$false;workflow_wall_s=$watch.Elapsed.TotalSeconds}
    Write-AtomicJson (Join-Path $session 'result.json') @{stage='training-complete';training_ran=$true;qualified=$qualified.Count -gt 0;
        qualifying_candidates=$qualified;motor=$offset;steering='held-heading';feedback='whole-beat';horizon='long-horizon';
        godot_published=$false;workflow_wall_s=$watch.Elapsed.TotalSeconds}
}catch{
    $originalError=$_
    if($trainer -and !$trainer.finished -and !$trainer.process.HasExited -and $training){
        $null=New-Item -Path (Join-Path $training 'stop.request') -ItemType File -Force
        try{$null=Wait-RecordedProcess $trainer @(0,2)}catch{Write-Warning 'Trainer stopped with preserved failure evidence.'}
    }
    throw $originalError
}finally{
    foreach($job in $active){if(!$job.finished){$null=Finish-RecordedProcess $job $false $true};$job.process.Dispose()}
    if($lease){$lease.Dispose()}
    Pop-Location
}
