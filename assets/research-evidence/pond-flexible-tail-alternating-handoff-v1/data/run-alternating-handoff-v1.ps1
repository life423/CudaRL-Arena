
$ErrorActionPreference='Stop'
$repo=(Get-Location).Path
$root=(Resolve-Path 'build/flexible-tail').Path
$study=Join-Path $root 'alternating-handoff-v1'
if(Test-Path -LiteralPath $study){throw 'This diagnostic allocation already ran'}
$old=Join-Path $root 'yaw-beat-v1'
$oldCue=Join-Path $old 'capture'
$offset=Join-Path $old 'training/actor-8.bin'
$gait=Join-Path $repo 'assets/research-evidence/pond-flexible-tail-gait-v1/passing-actor.bin'
$protocol=Join-Path $repo 'tools/pond-flexible-tail-motor/ALTERNATING_HANDOFF_PROTOCOL.md'
$fixture=Join-Path $root 'alternating-handoff-build/Release/pond_flexible_tail_motor_yaw_offset_fixture.exe'
$physics=Join-Path $repo 'src/physics/flexible_tail.h'
$required=[ordered]@{
    $offset='52BA55F110DD1227CBFEB1E47E6199E50DE74687E82E9DD651114CDA86FEC354'
    $gait='2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6'
    (Join-Path $root 'beat-timing-v1/assessment/segments.csv')='28858D18E17DBF8033173E2623BFA65A4CCE2EA75142DDB06F4A43C6CA12F2FB'
    (Join-Path $root 'beat-timing-v1/assessment/beat-events.csv')='A28375AEB7DD40551A12B913BA2D4C4E776FDB459C191E0C0A858A24AD288625'
    $physics='B6878791EC1A3CC68112866DE8C105390A43AC143A8EDCBA9EA2BF2430CCB226'
    (Join-Path $oldCue 'eval-turn-worlds.bin')='4902A5050A09101D8E7115EE43DDF99F3288B99EF539A525548A5D3FC727BF1E'
}
$prior=Get-Content -LiteralPath (Join-Path $old 'bindings.json') -Raw|ConvertFrom-Json
foreach($relative in @('src/rl/flexible_tail_gait.h','src/rl/flexible_tail_yaw_offset.h',
    'src/rl/flexible_tail_navigation.h','src/rl/flexible_tail_transition.h','flexible-tail-preview/habitat.json')){
    $path=(Resolve-Path -LiteralPath $relative).Path
    $entry=@($prior|Where-Object Path -eq $path)
    if($entry.Count -ne 1){throw "Missing archive binding for $relative"}
    $required[$path]=$entry[0].Hash
}
foreach($path in $required.Keys){
    if((Get-FileHash -LiteralPath $path).Hash -ne $required[$path]){throw "Previous evidence hash differs $path"}
}
$inputs=@($protocol,$fixture,$PSCommandPath,'tools/pond-flexible-tail-motor/yaw-offset-fixture.cpp',
    'tools/pond-flexible-tail-motor/bank-room-replay.h','src/environment/flexible_tail_clearance.h',
    'src/environment/pond_geometry_v1.h','build/flexible-tail/analyze-alternating-handoff-v1.ps1')+@($required.Keys)
$bindings=Get-FileHash -LiteralPath $inputs
$null=New-Item -Path $study -ItemType Directory
Copy-Item -LiteralPath $protocol -Destination (Join-Path $study 'protocol.md')
$bindings|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $study 'bindings.json')
$info=[Diagnostics.ProcessStartInfo]::new()
$info.FileName=$fixture
$info.UseShellExecute=$false
$info.CreateNoWindow=$true
$info.RedirectStandardOutput=$true
$info.RedirectStandardError=$true
$arguments=@('alternating-handoff',$gait,$offset,$oldCue,(Join-Path $study 'assessment'))
foreach($arg in $arguments){$info.ArgumentList.Add($arg)}
$clock=[Diagnostics.Stopwatch]::StartNew()
$process=[Diagnostics.Process]::Start($info)
try{
    $out=$process.StandardOutput.ReadToEndAsync()
    $err=$process.StandardError.ReadToEndAsync()
    $finished=$process.WaitForExit([int][math]::Max(0,4000-$clock.ElapsedMilliseconds))
    $timedOut=!$finished -or $clock.Elapsed.TotalMilliseconds -gt 4000
    if($timedOut -and !$process.HasExited){
        $process.Kill($true)
        if(!$process.WaitForExit(500)){throw 'Could not stop timing diagnostic at its deadline'}
    }
    $clock.Stop()
    $record=[ordered]@{wall_s=$clock.Elapsed.TotalSeconds;timeout=$timedOut;exit=$process.ExitCode;exe=$fixture;args=$arguments}
    $record|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $study 'assessment.receipt.json')
    $stdout=$out.GetAwaiter().GetResult()
    $stderr=$err.GetAwaiter().GetResult()
    Set-Content -LiteralPath (Join-Path $study 'assessment.stdout.log') -Value $stdout
    Set-Content -LiteralPath (Join-Path $study 'assessment.stderr.log') -Value $stderr
    Write-Host ("Assessment {0:F3}s exit={1} timeout={2}" -f $record.wall_s,$record.exit,$timedOut)
    Write-Host $stdout
    if($stderr){Write-Host $stderr}
    if($timedOut -or $process.ExitCode -ne 0){throw 'Timing diagnostic stopped before completion'}
}finally{
    $process.Dispose()
    foreach($entry in $bindings){
        if((Get-FileHash -LiteralPath $entry.Path).Hash -ne $entry.Hash){throw "Diagnostic binding changed $($entry.Path)"}
    }
}

