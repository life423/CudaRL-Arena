param(
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9][a-z0-9-]*$')][string]$StudyId,
    [Parameter(Mandatory)][string]$TrainingResult,
    [Parameter(Mandatory)][string]$EvaluationDirectory,
    [Parameter(Mandatory)][string]$Checkpoint,
    [ValidateSet('stopped','completed')][string]$Status='stopped',
    [ValidateLength(0,48)][ValidatePattern('^[^\[\]\r\n]*$')][string]$StudyLabel='',
    [string]$RepositoryCheckpoint='',
    [string]$BaselineDirectory=''
)
# Publish small recorded evidence only. Never simulates, trains or adopts a policy.
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$culture=[Globalization.CultureInfo]::InvariantCulture
function FullPath([string]$path) {
    if([IO.Path]::IsPathRooted($path)){return [IO.Path]::GetFullPath($path)}
    return [IO.Path]::GetFullPath((Join-Path $repo $path))
}
function Sha([string]$path){return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
function WriteJson([string]$path,$value){$value|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $path -Encoding UTF8}
function CheckpointRecord([string]$path) {
    $file=Get-Item -LiteralPath $path
    $stream=[IO.File]::OpenRead($path)
    $reader=[IO.BinaryReader]::new($stream)
    try {
        if($reader.ReadUInt64() -ne [uint64]0x31504B434C524443){throw 'Wrong native checkpoint envelope'}
        $payloadBytes=$reader.ReadUInt64();$checksum=$reader.ReadUInt64()
        if($payloadBytes -ne $file.Length-24){throw 'Truncated checkpoint envelope'}
        $header=@();for($i=0;$i -lt 11;$i++){$header+=$reader.ReadUInt32()}
        if($header[0] -ne 0x31505044 -or $header[1] -ne 1 -or $header[2] -ne 4 -or $header[3] -ne 2 -or $header[4] -notin @(4,5) -or $header[7] -ne 41 -or $header[8] -ne 4){throw 'Unsupported recorded candidate header'}
        for($i=0;$i -lt 2;$i++){$count=$reader.ReadUInt64();if($count -ne 64){throw 'Wrong parent identity size'};$null=$reader.ReadBytes(64)}
        $null=$reader.ReadUInt64();$protocol=$reader.ReadUInt64();$source=$reader.ReadUInt64();$serial=$reader.ReadUInt64()
        $null=$reader.ReadDouble();$null=$reader.ReadUInt64()
        $null=$reader.ReadUInt32();$worldStride=$reader.ReadUInt32();$taskStride=$reader.ReadUInt32()
        $updates=$reader.ReadUInt64();$controls=$reader.ReadUInt64();$substeps=$reader.ReadUInt64();$actorAdam=$reader.ReadUInt64();$criticAdam=$reader.ReadUInt64()
        if($updates -gt 32 -or $controls -ne $updates*32768 -or $substeps -ne $controls*10 -or $actorAdam -ne $updates*32 -or $criticAdam -ne $updates*32 -or $serial -ne $updates*64){throw 'Recorded candidate counters disagree'}
        return [ordered]@{schema=1;file_sha256=(Sha $path);file_bytes=$file.Length;payload_checksum_fnv1a64=$checksum.ToString($culture);protocol_checksum_fnv1a64=$protocol.ToString($culture);source_identity_fnv1a64=$source.ToString($culture);runtime_version=$header[2];collision_envelope_version=$header[3];observation_schema=$header[4];completed_updates=$updates;training_controls=$controls;training_substeps=$substeps;actor_Adam_calls=$actorAdam;critic_Adam_calls=$criticAdam;world_stride=$worldStride;task_stride=$taskStride;raw_checkpoint_in_app=$false;repository_checkpoint=$RepositoryCheckpoint;validation='Native evaluation parsed this checkpoint; this export records its envelope, header, counters and file digest. It performs no new policy evaluation.'}
    } finally {$reader.Dispose();$stream.Dispose()}
}
$trainingPath=FullPath $TrainingResult
$evaluationRoot=FullPath $EvaluationDirectory
$checkpointPath=FullPath $Checkpoint
$assessmentPath=Join-Path $evaluationRoot 'result.json'
$training=Get-Content -LiteralPath $trainingPath -Raw|ConvertFrom-Json
$assessmentRaw=Get-Content -LiteralPath $assessmentPath -Raw
$assessment=$assessmentRaw|ConvertFrom-Json
$digest=CheckpointRecord $checkpointPath
if($RepositoryCheckpoint){
    if([IO.Path]::IsPathRooted($RepositoryCheckpoint) -or $RepositoryCheckpoint.Replace('\','/').Split('/') -contains '..'){throw 'Published checkpoint reference must stay repository-relative'}
    $published=FullPath $RepositoryCheckpoint
    if(-not (Test-Path -LiteralPath $published) -or (Sha $published) -ne $digest.file_sha256){throw 'Published terminal checkpoint is missing or differs'}
}
$identity=[regex]::Match($assessmentRaw,'"checkpoint_checksum"\s*:\s*([0-9]+)')
if(-not $identity.Success -or $identity.Groups[1].Value -ne $digest.payload_checksum_fnv1a64 -or $assessment.update -ne $digest.completed_updates -or $training.completed_updates -ne $digest.completed_updates -or $assessment.preflight_only){throw 'Native assessment/training/checkpoint identity differs'}
$output=Join-Path $repo ('live/research_data/studies/'+$StudyId)
if(Test-Path -LiteralPath $output){throw 'Immutable public study folder already exists'}
New-Item -ItemType Directory -Path $output|Out-Null
Copy-Item -LiteralPath $trainingPath -Destination (Join-Path $output 'training-result.json')
Copy-Item -LiteralPath $assessmentPath -Destination (Join-Path $output 'assessment-result.json')
WriteJson (Join-Path $output 'checkpoint-digest.json') $digest
$controlsPath=Join-Path $evaluationRoot 'controls.csv'
$columns=([IO.File]::ReadLines($controlsPath)|Select-Object -First 1).Split(',')
$index=@{};for($i=0;$i -lt $columns.Length;$i++){$index[$columns[$i]]=$i}
foreach($column in @('suite','case','lane_m','depth_m','tilt_deg')){if(-not $index.ContainsKey($column)){throw ('Missing native metric '+$column)}}
$suiteIndex=$index.suite;$caseIndex=$index.case
$laneIndex=$index.lane_m;$depthIndex=$index.depth_m;$tiltIndex=$index.tilt_deg
$splitLimit=2+[int](($suiteIndex,$caseIndex,$laneIndex,$depthIndex,$tiltIndex|Measure-Object -Maximum).Maximum)
$metrics=[ordered]@{};$rows=0;$first=$true;$lastKey=$null
foreach($line in [IO.File]::ReadLines($controlsPath)) {
    if($first){$first=$false;continue};if(-not $line){continue}
    $fields=$line.Split([char]',',$splitLimit,[StringSplitOptions]::None)
    $key=$fields[$suiteIndex]+':'+$fields[$caseIndex]
    if($key -ne $lastKey){
        if(-not $metrics.Contains($key)){
            $metrics[$key]=[ordered]@{suite=$fields[$suiteIndex];case=[int]$fields[$caseIndex];accepted_control_rows=0;maximum_lane_m=0.;maximum_depth_m=0.;maximum_tilt_deg=0.}
        }
        $lastKey=$key
        $entry=$metrics[$key]
    }
    $lane=[double]::Parse($fields[$laneIndex],$culture)
    $depth=[double]::Parse($fields[$depthIndex],$culture)
    $tilt=[double]::Parse($fields[$tiltIndex],$culture)
    if(!([double]::IsFinite($lane) -and [double]::IsFinite($depth) -and [double]::IsFinite($tilt))){throw 'Nonfinite recorded metric'}
    if($lane -gt $entry.maximum_lane_m){$entry.maximum_lane_m=$lane}
    if($depth -gt $entry.maximum_depth_m){$entry.maximum_depth_m=$depth}
    if($tilt -gt $entry.maximum_tilt_deg){$entry.maximum_tilt_deg=$tilt}
    $entry.accepted_control_rows++;$rows++
}
$summary=[ordered]@{schema=1;source_controls_sha256=(Sha $controlsPath);source_controls_bytes=(Get-Item -LiteralPath $controlsPath).Length;accepted_control_rows=$rows;raw_controls_in_app=$false;cases=@($metrics.Values)}
WriteJson (Join-Path $output 'movement-summary.json') $summary
foreach($name in @('cases.csv','long-windows.csv')){if(Test-Path -LiteralPath (Join-Path $evaluationRoot $name)){Copy-Item -LiteralPath (Join-Path $evaluationRoot $name) -Destination (Join-Path $output $name)}}
if(-not $BaselineDirectory){$BaselineDirectory=Join-Path (Split-Path -Parent $evaluationRoot) 'baseline0'}
$baseline=Join-Path (FullPath $BaselineDirectory) 'result.json'
if(Test-Path -LiteralPath $baseline){Copy-Item -LiteralPath $baseline -Destination (Join-Path $output 'baseline-assessment.json')}
$sources=[ordered]@{}
foreach($pair in @(@('checkpoint','checkpoint-digest.json'),@('training','training-result.json'),@('evaluation','assessment-result.json'),@('movement','movement-summary.json'),@('cases','cases.csv'))){$path=Join-Path $output $pair[1];$sources[$pair[0]]=[ordered]@{path=('res://research_data/studies/'+$StudyId+'/'+$pair[1]);sha256=(Sha $path)}}
if(Test-Path -LiteralPath (Join-Path $output 'baseline-assessment.json')){$sources.baseline=[ordered]@{path=('res://research_data/studies/'+$StudyId+'/baseline-assessment.json');sha256=(Sha (Join-Path $output 'baseline-assessment.json'))}}
$reason='retention_failed'
if($assessment.hard_failure){$reason='physical_check_failed'}
elseif($assessment.retention_passed){$reason=if($assessment.turning_qualified){'requested_turns_qualified'}else{'turning_unqualified'}}
elseif($assessment.reason -like '*hold*'){$reason='rest_retention_failed'}
elseif($assessment.reason -like '*long*'){$reason=if(@($metrics.Values|Where-Object{$_.suite -eq 'long' -and $_.maximum_lane_m -gt .002}).Count -gt 0){'long_swim_lane_drift'}else{'long_swim_retention_failed'}}
$receipt=[ordered]@{schema=2;study_id=$StudyId;study_label=$StudyLabel;status=$Status;completed_updates=$digest.completed_updates;reason=$reason;playing_package_sha256='4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d';lane_limit_m=.002;candidate_adopted=$false;sources=$sources;disclosure='Recorded study result. Compact native outputs and checkpoint digests are portable; omitted raw controls/checkpoints remain preserved research evidence. No study is rerun by the app.'}
WriteJson (Join-Path $repo 'live/research_data/candidate-study.json') $receipt
$bindingPath=Join-Path $repo 'live/research_data/shoreline-runtime.json'
$binding=Get-Content -LiteralPath $bindingPath -Raw|ConvertFrom-Json
$binding|Add-Member -Force -NotePropertyName candidate_source -NotePropertyValue ([ordered]@{path='res://research_data/candidate-study.json';sha256=(Sha (Join-Path $repo 'live/research_data/candidate-study.json'))})
WriteJson $bindingPath $binding
Write-Output ([ordered]@{study_id=$StudyId;status=$Status;completed_updates=$digest.completed_updates;reason=$reason;public_bytes=(Get-ChildItem -LiteralPath $output -File|Measure-Object Length -Sum).Sum;checkpoint_sha256=$digest.file_sha256;candidate_adopted=$false;physical_controls=0;optimizer_steps=0}|ConvertTo-Json)
