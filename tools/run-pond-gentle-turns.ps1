param(
    [Parameter(Mandatory)][string]$Protocol,
    [Parameter(Mandatory)][string]$ProtocolSha256,
    [Parameter(Mandatory)][string]$GodotExe,
    [switch]$Run,
    [switch]$PublishRecorded
)
$ErrorActionPreference='Stop'
$wall=[Diagnostics.Stopwatch]::StartNew()
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$planFile=(Resolve-Path -LiteralPath $Protocol).Path
$engine=(Resolve-Path -LiteralPath $GodotExe).Path
$plan=Get-Content -Raw -LiteralPath $planFile | ConvertFrom-Json
$signedCourse=$plan.protocol_id -eq 'pond-motor-ppo-v6-signed-course-2026-11-24'
if((!$signedCourse -and $plan.protocol_id -ne 'pond-motor-ppo-v5-half-second-2026-11-24') -or $plan.max_updates -ne 8){throw 'Select a supported eight-update course protocol.'}
$root=Split-Path $plan.ledger_path_absolute
$studyId='turn35-v5-half-second-2026-11-24'
$evidence='assets/research-evidence/pond-turning-v5-half-second'
$studyLabel='Gentle turning · half-second course'
if($signedCourse){
    $studyId='turn36-v6-signed-course-2026-11-24'
    $studyLabel='Gentle turning · signed course input'
    $evidence='assets/research-evidence/pond-turning-v6-signed-course'
}
$destination=Join-Path $repo $evidence
if($Run -and $PublishRecorded){throw 'Choose either a new run or publication of the completed record.'}
if(($Run -or $PublishRecorded) -and (Test-Path (Join-Path $repo "live/research_data/studies/$studyId"))){throw 'A published run already exists for this study.'}

$arguments=@{Protocol=$planFile;ProtocolSha256=$ProtocolSha256;StopAfterUpdate=8;Run=$Run}
if(!$PublishRecorded){
    $output=& (Join-Path $PSScriptRoot 'run-pond-turning.ps1') @arguments
    if(!$Run){$output;return}
}
$originalStart=Get-Content -Raw -LiteralPath (Join-Path $root 'runner/started.json') | ConvertFrom-Json
if((Get-FileHash -LiteralPath $planFile).Hash -ne $ProtocolSha256 -or $originalStart.protocol_sha256 -ne $ProtocolSha256){throw 'The completed run differs from the supplied protocol.'}
$runResult=Get-Content -Raw -LiteralPath (Join-Path $root 'runner/result.json') | ConvertFrom-Json
$checkedSeconds=if(!$PublishRecorded){$wall.Elapsed.TotalSeconds}else{$runResult.total_host_wall_ms / 1000}
if($runResult.status -ne 'COMPLETED_ALLOCATION' -or $runResult.last_passed_training_chunk_update -ne 8){throw "The run stopped. See $root/runner/result.json"}
$candidate=Join-Path $root 'updates-1-8/candidate-8.bin'
$evalDir=Join-Path $root 'eval8'
$evalResult=Get-Content -Raw -LiteralPath (Join-Path $evalDir 'result.json') | ConvertFrom-Json
if($evalResult.update -ne 8 -or $evalResult.turn_cases_attempted -ne 16 -or $evalResult.retention_cases_attempted -ne 20){throw 'The final assessment is incomplete.'}
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$publishedCandidate=Join-Path $destination 'candidate-8.bin'
if(Test-Path -LiteralPath $publishedCandidate){
    if((Get-FileHash $candidate).Hash -ne (Get-FileHash $publishedCandidate).Hash){throw 'A different research checkpoint already exists.'}
}else{Copy-Item -LiteralPath $candidate -Destination $destination}
$publication=@{
    StudyId=$studyId;StudyLabel=$studyLabel
    TrainingResult=(Join-Path $root 'updates-1-8/result.json');EvaluationDirectory=$evalDir
    Checkpoint=$candidate;Status='completed';RepositoryCheckpoint="$evidence/candidate-8.bin"
    BaselineDirectory=(Join-Path $root 'baseline0')
}
& (Join-Path $PSScriptRoot 'publish-study-status.ps1') @publication | Out-Host
& (Join-Path $PSScriptRoot 'test-live-package.ps1') | Out-Host
$entryResult=Join-Path $root 'godot-details-result.json'
& $engine --headless --path (Join-Path $repo 'live') --script (Join-Path $repo 'tests/godot/pond_details_entry.gd') -- "--expected-study=$studyId" "--entry-result=$entryResult" | Out-Host
if($LASTEXITCODE -ne 0 -or !(Get-Content -Raw -LiteralPath $entryResult | ConvertFrom-Json).passed){throw 'Godot did not confirm the recorded study and original motor.'}
$timing=[ordered]@{
    protocol_id=$plan.protocol_id;updates=8;transitions=$runResult.native_training_accounting.attempted_controls
    through_training_and_final_skill_checks_s=$checkedSeconds
    publication_only=$PublishRecorded.IsPresent
    this_invocation_through_headless_Godot_s=$wall.Elapsed.TotalSeconds
    turns_passed=$evalResult.turn_cases_passed;turns_attempted=$evalResult.turn_cases_attempted
    long_windows_retained=$evalResult.retained_parent_long_window_passes
    short_swims_passed=$evalResult.short_swim_cases_passed;rests_passed=$evalResult.rest_cases_passed
}
$timing|ConvertTo-Json|Set-Content -Encoding utf8 -LiteralPath (Join-Path $root 'timed-workflow.json')
$timing|ConvertTo-Json
