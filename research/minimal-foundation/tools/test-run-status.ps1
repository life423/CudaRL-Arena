$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'run-status.psm1') -Force
$probe=Join-Path ([IO.Path]::GetTempPath()) ('cudarl-status-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $probe|Out-Null
$path=Join-Path $probe 'status.json'
try {
    $count=0
    foreach($phase in @('preparing','building','checking','ready','running','training','evaluating_candidate','done','failed')){
        $value=@{phase=$phase;pid=123;heartbeat_utc=(Get-Date -AsUTC -Format o);controls=$count;result_path='result.json'}
        if($phase -eq 'failed'){$value.error='synthetic missing executable';$value.failed_phase='checking'}
        Write-RunStatus -Path $path -Value $value
        $read=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
        if($read.phase -ne $phase -or $read.pid -ne 123 -or $read.controls -ne $count){throw 'Status lost identity/progress'}
        ++$count
    }
    if($read.error -ne 'synthetic missing executable' -or $read.failed_phase -ne 'checking'){throw 'Failure evidence lost'}
    $before=[IO.File]::ReadAllText($path)
    $rejected=$false
    try{Write-RunStatus -Path (Join-Path $probe 'missing/status.json') -Value @{phase='bad'}}catch{$rejected=$true}
    if(-not $rejected -or [IO.File]::ReadAllText($path) -ne $before){throw 'Write failure damaged last known status'}
    if(@(Get-ChildItem -LiteralPath $probe -Filter '*.tmp').Count){throw 'Temporary status leaked'}
    'Status checks passed: nine phases, progress/identity, failure, failed-write preservation; zero native processes/physical advances.'
} finally {
    if([IO.File]::Exists($path)){[IO.File]::Delete($path)}
    [IO.Directory]::Delete($probe)
}
