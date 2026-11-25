param([string]$Output='build/flexible-tail/seven-options-v6', [string]$ReusePrerequisites='',
    [string]$ContinueAfterMotor='', [string]$ResumeChooserFrom='',
    [string]$MotorBuild='build/flexible-tail/seven-stage-cap16-build',
    [string]$ChooserBuild='build/flexible-tail/seven-rate-cap16-build')
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location -LiteralPath $repository
$children=[Collections.Generic.List[object]]::new()
$script:trainingWallSpent=0.0
function Json($Path,$Value){$Value|ConvertTo-Json -Depth 18|Set-Content -LiteralPath ($Path+'.tmp') -Encoding utf8;[IO.File]::Move($Path+'.tmp',$Path,$true)}
function Start-Native($Name,$Exe,[string[]]$Arguments,$Directory,[double]$Limit){
    if($Name -eq 'training'){$Limit=[Math]::Min($Limit,3600.0-$script:trainingWallSpent);if($Limit -lt 1){throw 'Total training wall allowance spent.'}}
    $null=New-Item -Path $Directory -ItemType Directory -Force
    $info=[Diagnostics.ProcessStartInfo]::new();$info.FileName=$Exe;$info.WorkingDirectory=$repository
    $info.UseShellExecute=$false;$info.CreateNoWindow=$true;$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    foreach($argument in $Arguments){$info.ArgumentList.Add($argument)}
    $out=[IO.File]::Open((Join-Path $Directory "$Name.stdout.log"),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $err=[IO.File]::Open((Join-Path $Directory "$Name.stderr.log"),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $watch=[Diagnostics.Stopwatch]::StartNew();$p=[Diagnostics.Process]::Start($info)
    $job=[pscustomobject]@{name=$Name;process=$p;watch=$watch;out=$out;err=$err;ot=$p.StandardOutput.BaseStream.CopyToAsync($out);et=$p.StandardError.BaseStream.CopyToAsync($err);directory=$Directory;exe=$Exe;arguments=$Arguments;limit=$Limit;finished=$false}
    $children.Add($job);return $job
}
function Finish-Native($Job){
    if($Job.finished){return}
    $timeout=!$Job.process.HasExited
    if($timeout){$Job.process.Kill($true);$null=$Job.process.WaitForExit(5000)}
    $null=$Job.ot.GetAwaiter().GetResult();$null=$Job.et.GetAwaiter().GetResult();$Job.out.Dispose();$Job.err.Dispose();$Job.watch.Stop()
    $Job.finished=$true
    if($Job.name -eq 'training'){$script:trainingWallSpent+=$Job.watch.Elapsed.TotalSeconds}
    $receipt=[pscustomobject]@{name=$Job.name;wall_s=$Job.watch.Elapsed.TotalSeconds;exit=$Job.process.ExitCode;timeout=$timeout;executable=$Job.exe;arguments=$Job.arguments}
    Json (Join-Path $Job.directory "$($Job.name).receipt.json") $receipt
    Write-Host ('{0}: {1:F3}s exit{2}' -f $Job.name,$receipt.wall_s,$receipt.exit)
    if(!$timeout -and $Job.exe -eq $script:rateEval -and $receipt.exit -in @(0,1)){
        $path=Join-Path $Job.directory 'assessment/summary.json'
        $r=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
        foreach($field in @('pass','other_native_faults','attempted_substeps','accepted_substeps','diagnostic_cases','endurance_attempts','resets','optimizer_updates')){
            if($field -notin $r.PSObject.Properties.Name){throw "Incomplete chooser evaluation: $field"}
        }
        if($r.other_native_faults -ne 0 -or $r.resets -ne 0 -or $r.optimizer_updates -ne 0 -or $r.diagnostic_cases -ne 12 -or $r.endurance_attempts -notin @(0,4) -or $r.attempted_substeps -gt 2112000 -or $r.accepted_substeps -gt $r.attempted_substeps -or ($r.pass -ne ($receipt.exit -eq 0))){throw 'Chooser evaluation native fault or incomplete accounting.'}
        $rows=@(Import-Csv -LiteralPath (Join-Path $Job.directory 'assessment/cases.csv'))
        if($rows.Count -ne (12+$r.endurance_attempts) -or @($rows|Where-Object suite -eq 'diagnostic').Count -ne 12){throw 'Incomplete chooser case evidence.'}
        foreach($row in $rows){
            if(($row.fault -eq '0' -and ($row.outcome -notin @('pass','metrics') -or [double]$row.simulated_s -ne [double]$row.horizon_s)) -or ($row.fault -ne '0' -and ($row.fault -ne '2' -or $row.outcome -ne 'native_stop' -or [double]$row.simulated_s -le 0))){throw "Chooser evaluation runtime/native fault: $($row.case) $($row.outcome)"}
        }
        if(($rows|Measure-Object attempted_substeps -Sum).Sum -ne $r.attempted_substeps -or ($rows|Measure-Object accepted_substeps -Sum).Sum -ne $r.accepted_substeps){throw 'Chooser case/summary work differs.'}
        return # Exit1 is a valid, completed unqualified assessment, not a process fault.
    }
    if($timeout -or $receipt.exit -ne 0){throw "Native phase failed: $($Job.name), evidence retained."}
}
function Wait-Native($Job){while(!$Job.process.HasExited -and $Job.watch.Elapsed.TotalSeconds -lt $Job.limit){$null=$Job.process.WaitForExit(200)};Finish-Native $Job}
function Bind-Files([string[]]$Paths){foreach($p in $Paths|Sort-Object -Unique){$f=(Resolve-Path -LiteralPath $p).Path;[pscustomobject]@{path=[IO.Path]::GetRelativePath($repository,$f).Replace('\','/');sha256=(Get-FileHash -LiteralPath $f).Hash}}}
function Assert-Frozen {foreach($b in $script:bindings){if((Get-FileHash -LiteralPath $b.path).Hash -ne $b.sha256){throw "Bound source/input changed: $($b.path)"}}}
function Score($Result,$Update,$Actor,$Checkpoint,$Assessment,$Task,$Study,$Material){
    $r=Get-Content -LiteralPath $Result -Raw|ConvertFrom-Json
    if($r.global_faults -ne 0){throw 'Native motor assessment fault; affected work stopped.'}
    $errorMean=1000.0
    $windows=Join-Path $Assessment 'windows.csv'
    if(Test-Path -LiteralPath $windows){$values=@(Import-Csv -LiteralPath $windows|Where-Object {$_.policy -eq 'candidate' -and $_.case -like 'phase_*' -and $_.course_error_deg -ne ''}|ForEach-Object {[double]$_.course_error_deg});if($values.Count){$errorMean=($values|Measure-Object -Average).Average}}
    $ratePass=0;$rateError=1000.0
    $probe=Join-Path $Assessment 'program-probe-windows.csv'
    if(Test-Path -LiteralPath $probe){$rates=@(Import-Csv -LiteralPath $probe|Where-Object {$_.case -like 'rate_*'});$ratePass=@($rates|Where-Object motion_pass -eq '1').Count;if($rates.Count){$rateError=($rates|ForEach-Object {[double]$_.error_deg}|Measure-Object -Average).Average}}
    [pscustomobject]@{update=$Update;actor=$Actor;checkpoint=$Checkpoint;task=$Task;study=$Study;material=$Material;assessment=$Assessment;rate_passes=$ratePass;rate_error=$rateError;
        physical=$r.candidate_straight_retention -eq 12;gentle=$r.candidate_main_15_passes;main=$r.candidate_main_stroke_turn_passes;
        thirty=$r.candidate_main_30_passes;sixty=$r.candidate_main_60_passes;ninety=$r.candidate_main_90_passes;
        legacy=$r.candidate_turn_passes;error=$errorMean;result=$Result}
}
function Motor-Eval($Actor,$Study,$Material,$Directory,$Supplement=$false){
    $kind=if($Supplement){'supplement'}else{'fixed'};$body=if($Material -eq 'half-yaw'){'yaw-half-stiffness'}else{'baseline'}
    Start-Native 'evaluation' $script:eval @($script:gait,$script:baseOffset,$Actor,$script:turns,$script:straight,(Join-Path $Directory 'assessment'),'phase-aware',$Study,$kind,$body) $Directory 30
}
function Run-Motor($Number,$Study,$Material,$Parent,$FocusRate=$false){
    Assert-Frozen;$directory=Join-Path $script:root ('stage-{0:D2}-{1}-{2}' -f $Number,$Study,$Material)
    $null=New-Item -Path $directory -ItemType Directory
    Json (Join-Path $directory 'parent.json') $Parent
    $zero=Join-Path $directory 'zero-step';$body=if($Material -eq 'half-yaw'){'yaw-half-stiffness'}else{'baseline'}
    Wait-Native (Start-Native 'zero-step' $script:check @($script:gait,$script:baseOffset,$script:starts,$script:turns,$script:straight,(Join-Path $zero 'data'),'phase-aware',$Study,$Parent.actor,$body) $zero 30)
    $initial=Join-Path $zero 'data/initial-program-actor.bin'
    $baseline=Join-Path $directory baseline
    if($Number -eq 1 -and $script:reuse){
        $priorStage=Join-Path $script:reuse 'stage-01-rhythm-baseline'
        if((Get-FileHash -LiteralPath $initial).Hash -ne (Get-FileHash -LiteralPath (Join-Path $priorStage 'zero-step/data/initial-program-actor.bin')).Hash){throw 'Initial actor changed: baseline evidence cannot be reused.'}
        Copy-Item -LiteralPath (Join-Path $priorStage baseline) -Destination $directory -Recurse
        Json (Join-Path $baseline 'reuse.json') @{original=(Join-Path $priorStage baseline);reason='critic-only cached-COM correction; actor/physics/requests unchanged';new_physics_steps=0}
    }else{Wait-Native (Motor-Eval $initial $Study $Material $baseline $FocusRate)}
    $baselineResult=Get-Content -LiteralPath (Join-Path $baseline assessment/result.json) -Raw|ConvertFrom-Json
    if($baselineResult.global_faults -ne 0){throw 'Motor baseline native fault.'}
    if($baselineResult.candidate_straight_retention -ne 12){
        Json (Join-Path $directory 'closed.json') @{stage=$Number;outcome='initial physical swimming regression';training_updates=0;parent_retained=$true}
        return $Parent
    }
    $verification=Join-Path $directory verification
    Wait-Native (Start-Native 'verification' $script:trainer @('verify-program',$script:gait,$Parent.actor,$script:starts,(Join-Path $verification data),'590',$Study,$Material,$Parent.checkpoint,$Parent.task) $verification 600)
    $proof=Get-Content -LiteralPath (Join-Path $verification data/verification.json) -Raw|ConvertFrom-Json
    if(!$proof.pass -or $proof.physical_updates_executed -ne 3){throw 'Program exact resume failed.'}
    if((Get-FileHash -LiteralPath $initial).Hash -ne (Get-FileHash -LiteralPath (Join-Path $verification data/actor-0.bin)).Hash){throw 'CPU transfer differs from native initialization.'}
    $training=Join-Path $directory training
    $job=Start-Native 'training' $script:trainer @('program',$script:gait,$Parent.actor,$script:starts,$training,'128','1200',$Study,$Material,$Parent.checkpoint,$Parent.task) $directory 1800
    $pending=[Collections.Generic.Queue[int]]::new();$scheduled=[Collections.Generic.HashSet[int]]::new();$evalJob=$null;$evalUpdate=0;$candidates=@()
    while(!$job.finished -or $evalJob -or $pending.Count){
        for($u=8;$u -le 128;$u+=8){if((Test-Path -LiteralPath (Join-Path $training "actor-$u.bin")) -and $scheduled.Add($u)){$pending.Enqueue($u)}}
        if(!$evalJob -and $pending.Count){$evalUpdate=$pending.Dequeue();$where=Join-Path $directory ('evaluations/update-{0:D3}' -f $evalUpdate);$evalJob=Motor-Eval (Join-Path $training "actor-$evalUpdate.bin") $Study $Material $where $FocusRate}
        if($evalJob -and ($evalJob.process.HasExited -or $evalJob.watch.Elapsed.TotalSeconds -ge $evalJob.limit)){
            Finish-Native $evalJob
            $result=Join-Path $evalJob.directory assessment/result.json;$r=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
            $task=(Get-Content -LiteralPath (Join-Path $training result.json) -Raw|ConvertFrom-Json).task_identity
            $candidates+=Score $result $evalUpdate (Join-Path $training "actor-$evalUpdate.bin") (Join-Path $training "checkpoints/session-$evalUpdate.bin") (Join-Path $evalJob.directory assessment) $task $Study $Material
            Write-Host ("stage{0} update{1}: main15={2}/8 main30={3}/8 main60={4}/8 main90={5}/8 swim={6}/12" -f $Number,$evalUpdate,$r.candidate_main_15_passes,$r.candidate_main_30_passes,$r.candidate_main_60_passes,$r.candidate_main_90_passes,$r.candidate_straight_retention)
            if($r.global_faults -ne 0){throw 'Motor evaluation numerical/native fault.'}
            if($r.candidate_straight_retention -ne 12){$null=New-Item -Path (Join-Path $training STOP) -ItemType File -Force}
            $evalJob=$null
        }
        if(!$job.finished -and ($job.process.HasExited -or $job.watch.Elapsed.TotalSeconds -ge $job.limit)){Finish-Native $job}
        if(!$job.finished -or $evalJob -or $pending.Count){[Threading.Thread]::Sleep(100)}
    }
    $task=(Get-Content -LiteralPath (Join-Path $training result.json) -Raw|ConvertFrom-Json).task_identity
    $candidates+=Score (Join-Path $baseline assessment/result.json) 0 (Join-Path $training actor-0.bin) (Join-Path $training checkpoints/session-0.bin) (Join-Path $baseline assessment) $task $Study $Material
    $best=$candidates|Where-Object physical|Sort-Object @{Expression={if($_.gentle -eq 8){1}else{0}};Descending=$true},@{Expression='main';Descending=$true},@{Expression='error';Ascending=$true}|Select-Object -First 1
    if($FocusRate){$best=$candidates|Where-Object {$_.physical -and $_.gentle -eq 8 -and $_.thirty -eq 8}|Sort-Object @{Expression='rate_passes';Descending=$true},@{Expression='rate_error';Ascending=$true}|Select-Object -First 1}
    if(!$best){throw 'No safe retained motor candidate.'}
    $selectionRule=if($FocusRate){'physical12, all8main15, all8main30, rate12, rate-courseerror'}else{'physical12, all8main15, main32, courseerror'}
    Json (Join-Path $directory 'selection.json') @{selected=$best;all=$candidates;rule=$selectionRule}
    if(!$FocusRate){$final=Join-Path $directory final;Wait-Native (Motor-Eval $best.actor $Study $Material $final $true)}
    Assert-Frozen;return $best
}
function Run-Chooser($Motor,$TargetUpdates=64){
    $directory=Join-Path $script:root 'stage-06-bank';$null=New-Item -Path $directory -ItemType Directory
    Json (Join-Path $directory motor.json) $Motor
    $material=if($Motor.material -eq 'half-yaw'){'1'}else{'0'}
    $verification=Join-Path $directory verification
    $training=Join-Path $directory training
    $target=$TargetUpdates;$mode='new';$candidates=@()
    if($script:chooserResume){
        $prior=Join-Path $script:chooserResume 'stage-06-bank'
        $receipt=Get-Content -LiteralPath (Join-Path $prior training.receipt.json) -Raw|ConvertFrom-Json
        if((Get-FileHash -LiteralPath $script:rate).Hash -ne (Get-FileHash -LiteralPath $receipt.executable).Hash){throw 'Chooser executable changed; saved diagnostic cannot be reused.'}
        if(Test-Path -LiteralPath (Join-Path $prior training/actor-1.bin)){throw 'Expected interrupted first update with only checkpoint0.'}
        if((Get-Content -LiteralPath (Join-Path $prior motor.json) -Raw|ConvertFrom-Json).actor -ne $Motor.actor){throw 'Chooser motor parent changed.'}
        Copy-Item -LiteralPath (Join-Path $prior verification) -Destination $directory -Recurse
        if(!(Get-Content -LiteralPath (Join-Path $verification data/verification.json) -Raw|ConvertFrom-Json).pass){throw 'Saved chooser exact-resume proof failed.'}
        & (Join-Path $PSScriptRoot 'prepare-chooser-continuation.ps1') -SourceTraining (Join-Path $prior training) -DestinationTraining $training | Out-Host
        $target=63;$mode='resume'
        $where=Join-Path $directory 'evaluations/update-000';$null=New-Item -Path (Split-Path -Parent $where) -ItemType Directory -Force
        Copy-Item -LiteralPath (Join-Path $prior 'evaluations/update-000') -Destination $where -Recurse
        $summary=Join-Path $where assessment/summary.json;$r=Get-Content -LiteralPath $summary -Raw|ConvertFrom-Json
        if($r.other_native_faults -ne 0 -or $r.diagnostic_cases -ne 12){throw 'Saved baseline invalid.'}
        $candidates+=@{update=0;actor=(Join-Path $training actor-0.bin);summary=$summary;results=$r;reused=$true}
        Json (Join-Path $directory continuation.json) @{previous=$prior;reason='valid evaluator exit1 was misclassified by runner';reserved_unknown_updates=1;reserved_unknown_transitions_maximum=8192;reserved_unknown_adam_calls_per_optimizer_maximum=32;reserved_unknown_substeps_maximum=10485760;new_complete_update_cap=63;prior_wall_s=$receipt.wall_s;new_diagnostics=0;reused_baseline=$true}
    }else{
        Wait-Native (Start-Native 'verification' $script:rate @('verify',$script:gait,$Motor.actor,$script:starts,(Join-Path $verification data),'2','590',$material) $verification 600)
        if(!(Get-Content -LiteralPath (Join-Path $verification data/verification.json) -Raw|ConvertFrom-Json).pass){throw 'Rate chooser resume verification failed.'}
    }
    $job=Start-Native 'training' $script:rate @($mode,$script:gait,$Motor.actor,$script:starts,$training,"$target",'1200',$material) $directory 1800
    $pending=[Collections.Generic.Queue[int]]::new();$scheduled=[Collections.Generic.HashSet[int]]::new();$evalJob=$null
    if($script:chooserResume){$null=$scheduled.Add(0)}
    while(!$job.finished -or $evalJob -or $pending.Count){
        $checks=@(for($checkUpdate=0;$checkUpdate -le $target;$checkUpdate+=8){$checkUpdate})+@($target)
        foreach($u in $checks){if((Test-Path -LiteralPath (Join-Path $training "actor-$u.bin")) -and $scheduled.Add($u)){$pending.Enqueue($u)}}
        if(!$evalJob -and $pending.Count){$u=$pending.Dequeue();$where=Join-Path $directory ('evaluations/update-{0:D3}' -f $u);$evalJob=Start-Native 'evaluation' $script:rateEval @($script:gait,$Motor.actor,(Join-Path $training "actor-$u.bin"),$script:straight,(Join-Path $where assessment),$material) $where 30;$evalJob|Add-Member NoteProperty update $u}
        if($evalJob -and ($evalJob.process.HasExited -or $evalJob.watch.Elapsed.TotalSeconds -ge $evalJob.limit)){
            Finish-Native $evalJob;$p=Join-Path $evalJob.directory assessment/summary.json;$r=Get-Content -LiteralPath $p -Raw|ConvertFrom-Json
            $candidates+=@{update=$evalJob.update;actor=(Join-Path $training "actor-$($evalJob.update).bin");summary=$p;results=$r}
            Write-Host ("bank update{0}: {1}" -f $evalJob.update,($r|ConvertTo-Json -Compress -Depth 3));$evalJob=$null
        }
        if(!$job.finished -and ($job.process.HasExited -or $job.watch.Elapsed.TotalSeconds -ge $job.limit)){Finish-Native $job}
        if(!$job.finished -or $evalJob -or $pending.Count){[Threading.Thread]::Sleep(100)}
    }
    $best=$candidates|Sort-Object @{Expression={$_.results.diagnostic_passes};Descending=$true},@{Expression={$_.results.endurance_passes};Descending=$true},@{Expression={$_.results.diagnostic_coverage_fraction};Descending=$true}|Select-Object -First 1
    Json (Join-Path $directory selection.json) @{selected=$best;all=$candidates};return $best
}
try{
    $script:root=[IO.Path]::GetFullPath($Output);if(Test-Path -LiteralPath $root){throw 'Seven-stage output must be unused.'};$null=New-Item -Path $root -ItemType Directory
    $script:reuse=if($ReusePrerequisites){(Resolve-Path -LiteralPath $ReusePrerequisites).Path}else{''}
    $script:continuation=if($ContinueAfterMotor){(Resolve-Path -LiteralPath $ContinueAfterMotor).Path}else{''}
    $script:chooserResume=if($ResumeChooserFrom){(Resolve-Path -LiteralPath $ResumeChooserFrom).Path}else{''}
    if($script:chooserResume -and !$script:continuation){throw 'Chooser continuation requires the original five-stage motor root.'}
    if($script:chooserResume){
        $priorBindings=Get-Content -LiteralPath (Join-Path $script:chooserResume bindings.json) -Raw|ConvertFrom-Json
        foreach($b in $priorBindings|Where-Object {$_.path -ne 'tools/pond-flexible-tail-steering/run-seven-stages.ps1'}){
            if((Get-FileHash -LiteralPath $b.path).Hash -ne $b.sha256){throw "Chooser continuation identity changed: $($b.path)"}
        }
    }
    if($script:reuse -and $script:continuation){throw 'Choose one explicit prerequisite reuse or post-motor continuation.'}
    $cmake=(Get-Command cmake).Source
    $motorDirectory=[IO.Path]::GetFullPath($MotorBuild);$chooserDirectory=[IO.Path]::GetFullPath($ChooserBuild)
    $compileDirectory=Join-Path $root compilation
    Wait-Native (Start-Native 'configure-motor' $cmake @('-S',$PSScriptRoot,'-B',$motorDirectory,'-G','Visual Studio 17 2022','-A','x64') $compileDirectory 600)
    Wait-Native (Start-Native 'compile-motor' $cmake @('--build',$motorDirectory,'--config','Release','--parallel','4') $compileDirectory 600)
    $chooserSource=Join-Path (Split-Path -Parent $PSScriptRoot) 'pond-flexible-tail-chooser'
    Wait-Native (Start-Native 'configure-chooser' $cmake @('-S',$chooserSource,'-B',$chooserDirectory,'-G','Visual Studio 17 2022','-A','x64') $compileDirectory 600)
    Wait-Native (Start-Native 'compile-chooser' $cmake @('--build',$chooserDirectory,'--config','Release','--target','pond_flexible_tail_rate_chooser','pond_flexible_tail_rate_evaluate','--parallel','4') $compileDirectory 600)
    $script:trainer=(Resolve-Path (Join-Path $motorDirectory Release/pond_flexible_tail_steering_ppo.exe)).Path
    $script:eval=(Resolve-Path (Join-Path $motorDirectory Release/pond_flexible_tail_steering_evaluate.exe)).Path
    $script:check=(Resolve-Path (Join-Path $motorDirectory Release/pond_flexible_tail_beat_feedback_check.exe)).Path
    $script:rate=(Resolve-Path (Join-Path $chooserDirectory Release/pond_flexible_tail_rate_chooser.exe)).Path
    $script:rateEval=(Resolve-Path (Join-Path $chooserDirectory Release/pond_flexible_tail_rate_evaluate.exe)).Path
    $script:gait=(Resolve-Path flexible-tail-preview/live_data/fitted-gait.bin).Path
    $script:baseOffset=(Resolve-Path build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin).Path
    $script:starts=(Resolve-Path build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin).Path
    $script:turns=(Resolve-Path build/flexible-tail/yaw-beat-v1/capture/eval-turn-worlds.bin).Path
    $script:straight=(Resolve-Path flexible-tail-preview/live_data/straight-evaluation-starts.bin).Path
    $parent=[pscustomobject]@{actor=(Resolve-Path build/flexible-tail/phase-mixed-v5/training/actor-88.bin).Path;checkpoint=(Resolve-Path build/flexible-tail/phase-mixed-v5/training/checkpoints/session-88.bin).Path;task='e77d7688e076201850f5aa9eb2c1960a1b10a70986bb1b934f57d4a88e6035fa';update=88;study='none';material='baseline'}
    $priorResults=@()
    if($script:continuation){
        $priorResults=@(Get-Content -LiteralPath (Join-Path $script:continuation progress.json) -Raw|ConvertFrom-Json)
        if($priorResults.Count -ne 5 -or $priorResults[-1].stage -ne 5){throw 'Continuation requires exactly five completed motor stages.'}
        $parent=$priorResults[-1].selected
        $old=Get-Content -LiteralPath (Join-Path $script:continuation bindings.json) -Raw|ConvertFrom-Json
        $allowed=@('src/rl/ppo_math.h','src/rl/ppo_kernels.cu','tools/pond-flexible-tail-steering/run-seven-stages.ps1')
        foreach($b in $old|Where-Object {$_.path -like 'src/*' -or $_.path -like 'tools/*'}){
            if($b.path -notin $allowed -and (Get-FileHash -LiteralPath $b.path).Hash -ne $b.sha256){throw "Unexpected source change in continuation: $($b.path)"}
        }
        $failed=Get-Content -LiteralPath (Join-Path $script:continuation 'stage-06-bank/verification/data/result.json') -Raw|ConvertFrom-Json
        if($failed.samples -ne 0 -or $failed.actor_steps -ne 0){throw 'Chooser repair continuation requires the zero-action capacity failure.'}
        foreach($f in Get-ChildItem -LiteralPath $script:continuation -Filter training.receipt.json -File -Recurse){$script:trainingWallSpent+=(Get-Content -LiteralPath $f.FullName -Raw|ConvertFrom-Json).wall_s}
        if($script:chooserResume){$script:trainingWallSpent+=(Get-Content -LiteralPath (Join-Path $script:chooserResume 'stage-06-bank/training.receipt.json') -Raw|ConvertFrom-Json).wall_s}
        Json (Join-Path $root previous-attempt.json) @{root=$script:continuation;preserved_stages=@(1,2,3,4,5);failed_chooser_warmup_substeps=$failed.attempted_substeps;failed_chooser_optimizer_steps=0;training_wall_already_spent=$script:trainingWallSpent;second_reserved_chooser_diagnostic=$true}
    }
    $paths=@(Get-ChildItem src -File -Recurse|Where-Object Extension -in @('.h','.cpp','.cu')|Select-Object -ExpandProperty FullName)
    $paths+=@(Get-ChildItem tools/pond-flexible-tail-steering,tools/pond-flexible-tail-chooser -File|Select-Object -ExpandProperty FullName)
    $paths+=@($trainer,$eval,$check,$rate,$rateEval,$gait,$baseOffset,$starts,$turns,$straight,$parent.actor,$parent.checkpoint,'flexible-tail-preview/live_data/yaw-offset-update-8.bin','flexible-tail-preview/habitat.json','flexible-tail-preview/addons/flexible_tail_live/bin/flexible_tail_live.dll')
    $paths+=@(Get-ChildItem tools/pond-flexible-tail-check/material-qualification -File|Select-Object -ExpandProperty FullName)
    $script:bindings=@(Bind-Files $paths);Json (Join-Path $root bindings.json) $bindings
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot SEVEN_STAGE_PROTOCOL.md) -Destination (Join-Path $root protocol.md)
    $materialExe=(Resolve-Path build/flexible-tail/material-bank-v5-build/Release/pond_flexible_tail_material_check.exe).Path
    $materialDir=Join-Path $root material-qualification
    if($script:continuation){
        Copy-Item -LiteralPath (Join-Path $script:continuation material-qualification) -Destination $root -Recurse
    }elseif($script:reuse){
        $priorBindings=Get-Content -LiteralPath (Join-Path $script:reuse bindings.json) -Raw|ConvertFrom-Json
        foreach($b in $priorBindings|Where-Object {$_.path -in @('src/physics/flexible_tail.h','src/rl/flexible_tail_transition.h','src/rl/flexible_tail_bank_observation.h')}){if((Get-FileHash -LiteralPath $b.path).Hash -ne $b.sha256){throw 'Material prerequisite changed; reuse rejected.'}}
        if((Get-FileHash -LiteralPath $materialExe).Hash -ne '5833F3213FA08C8B97B3A24DA49B80DF708B8AE5299645D2AC7DB4D8674F84F5'){throw 'Original material qualifier binary changed.'}
        if(Test-Path -LiteralPath (Join-Path $script:reuse 'stage-01-rhythm-baseline/training')){throw 'This reuse path only permits a failed prerequisite before main training.'}
        Copy-Item -LiteralPath (Join-Path $script:reuse material-qualification) -Destination $root -Recurse
        $failed=Get-Content -LiteralPath (Join-Path $script:reuse 'stage-01-rhythm-baseline/verification/data/result.json') -Raw|ConvertFrom-Json
        if($failed.invocation_sampled_actions -gt 196608){throw 'First diagnostic exceeded reserved allowance.'}
        Json (Join-Path $root previous-attempt.json) @{root=$script:reuse;diagnostic_samples=$failed.invocation_sampled_actions;diagnostic_substeps=$failed.invocation_attempted_substeps;diagnostic_actor_steps=$failed.invocation_actor_steps;second_reserved_attempt=$true;main_updates_previously=0}
    }else{try{Wait-Native (Start-Native 'qualification' $materialExe @((Join-Path $materialDir result.json),'20') $materialDir 20)}catch{Write-Host 'Material qualifier failed; failed material evidence retained.'}}
    $materialPass=(Get-Content -LiteralPath (Join-Path $materialDir result.json) -Raw|ConvertFrom-Json).pass
    $results=@($priorResults)
    if(!$script:continuation){foreach($item in @(@(1,'rhythm'),@(2,'joint'),@(3,'nonlinear'),@(4,'speed'),@(5,'rate'))){$parent=Run-Motor $item[0] $item[1] 'baseline' $parent;$results+=@{stage=$item[0];selected=$parent};Json (Join-Path $root progress.json) $results}}
    $chooser=Run-Chooser $parent;$results+=@{stage=6;selected=$chooser};Json (Join-Path $root progress.json) $results
    if($materialPass){$body=Run-Motor 7 'rate' 'half-yaw' $parent;$results+=@{stage=7;selected=$body}
        if($body.material -eq 'half-yaw'){$where=Join-Path $root final-body-pond;Wait-Native (Start-Native 'evaluation' $rateEval @($gait,$body.actor,$chooser.actor,$straight,(Join-Path $where assessment),'1') $where 30)}
    }else{$results+=@{stage=7;outcome='material physics qualification failed; no body PPO authorized through gate'}}
    Assert-Frozen;Json (Join-Path $root result.json) @{completed=$true;stages=$results;stopped_after_seven=$true;godot_replacement=$false}
}finally{foreach($j in $children){if(!$j.finished){
    if(!$j.process.HasExited -and $j.name -eq 'training'){
        $null=New-Item -Path (Join-Path $j.arguments[4] STOP) -ItemType File -Force
        $until=$j.watch.Elapsed.TotalSeconds+15
        while(!$j.process.HasExited -and $j.watch.Elapsed.TotalSeconds -lt $until){$null=$j.process.WaitForExit(200)}
    }
    try{Finish-Native $j}catch{}
}};Pop-Location}
