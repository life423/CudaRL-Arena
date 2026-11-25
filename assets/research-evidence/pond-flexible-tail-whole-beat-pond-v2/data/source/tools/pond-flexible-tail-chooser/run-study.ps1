
param([Parameter(Mandatory)][string]$StudyDir, [switch]$BankCurriculum)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot | Split-Path -Parent | Split-Path -Parent
Push-Location -LiteralPath $repo
try {
    $output = [IO.Path]::GetFullPath($StudyDir)
    if (Test-Path -LiteralPath $output) { throw "The chooser allocation already exists at $output" }
    $build = (Resolve-Path 'build/flexible-tail/chooser-v1-build/Release').Path
    $trainerBuild = if ($BankCurriculum) { (Resolve-Path 'build/flexible-tail/chooser-bank-v2-build/Release').Path } else { $build }
    $train = Join-Path $trainerBuild 'pond_flexible_tail_chooser_ppo.exe'
    $evaluate = Join-Path $build 'pond_flexible_tail_chooser_eval.exe'
    $gait = (Resolve-Path 'flexible-tail-preview/live_data/fitted-gait.bin').Path
    $offset = (Resolve-Path 'flexible-tail-preview/live_data/yaw-offset-update-8.bin').Path
    $learnStarts = (Resolve-Path 'build/flexible-tail/yaw-beat-v1/capture/train-worlds.bin').Path
    $heldout = (Resolve-Path 'flexible-tail-preview/live_data/straight-evaluation-starts.bin').Path

    $expected = [ordered]@{
        $gait = '2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6'
        $offset = '52BA55F110DD1227CBFEB1E47E6199E50DE74687E82E9DD651114CDA86FEC354'
        $learnStarts = '75E6AA2504726082E84E286FD5B3ADF2B7B9D4F7D4797116EF8A682F72DEC025'
        $heldout = '4A081CCC3DEA8C7310191902CC094E50FA6C0B341606DC9DBC4E63DD581FFF39'
        (Resolve-Path 'src/physics/flexible_tail.h').Path = 'B6878791EC1A3CC68112866DE8C105390A43AC143A8EDCBA9EA2BF2430CCB226'
        (Resolve-Path 'flexible-tail-preview/habitat.json').Path = '554E9E0976F22CD6B473EAAB84559C7B9887358C4B620CEF4592D5FFFEC0C7DF'
    }
    $protocolName = 'PROTOCOL.md'
    if ($BankCurriculum) {
        $previous = (Resolve-Path 'assets/research-evidence/pond-flexible-tail-chooser-v1/data/training/actor-final.bin').Path
        $expected[$previous] = '4ED0AC7FEA860C6488ED4194FBD60E8C139AD9864D798FA034BD49A39155F93E'
        $expected[$evaluate] = 'F623C8BCEAD1F121260C6C90981A669BFEA720D437FACFE69A8FCB905E92B342'
        $priorEvidence = Get-Content -LiteralPath 'assets/research-evidence/pond-flexible-tail-chooser-v1/data/bindings.json' -Raw | ConvertFrom-Json
        foreach ($relative in @('src/rl/flexible_tail_chooser.h', 'src/rl/flexible_tail_chooser_checkpoint.h',
            'src/rl/flexible_tail_gait.h', 'src/rl/flexible_tail_yaw_offset.h',
            'src/rl/flexible_tail_transition.h', 'src/rl/flexible_tail_navigation.h',
            'src/environment/pond_geometry_v1.h', 'src/environment/flexible_tail_clearance.h',
            'tools/pond-flexible-tail-chooser/evaluate.cpp')) {
            $path = (Resolve-Path -LiteralPath $relative).Path
            $binding = @($priorEvidence | Where-Object Path -eq $path)
            if ($binding.Count -ne 1) { throw "Missing original study binding for $relative" }
            $expected[$path] = $binding[0].Hash
        }
        $protocolName = 'BANK_CURRICULUM_PROTOCOL.md'
    }
    foreach ($path in $expected.Keys) {
        if ((Get-FileHash -LiteralPath $path).Hash -ne $expected[$path]) { throw "Frozen input changed at $path" }
    }
    $source = Get-ChildItem -LiteralPath $PSScriptRoot -File | Select-Object -ExpandProperty FullName
    $bindings = @(Get-FileHash -LiteralPath (@($expected.Keys) + $source + @(
        $train, $evaluate, 'src/environment/flexible_tail_clearance.h', 'src/environment/pond_geometry_v1.h',
        'src/rl/flexible_tail_chooser.h', 'src/rl/flexible_tail_chooser_checkpoint.h', 'src/rl/flexible_tail_gait.h', 'src/rl/flexible_tail_yaw_offset.h',
        'src/rl/flexible_tail_transition.h', 'src/rl/flexible_tail_navigation.h')))
    $null = New-Item -Path $output -ItemType Directory
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $protocolName) -Destination (Join-Path $output 'protocol.md')
    $bindings | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'bindings.json')

    function Invoke-Capped([string]$phase, [string]$exe, [string[]]$arguments) {
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $exe
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        foreach ($argument in $arguments) { $info.ArgumentList.Add($argument) }
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $process = [Diagnostics.Process]::Start($info)
        try {
            $readOut = $process.StandardOutput.ReadToEndAsync()
            $readErr = $process.StandardError.ReadToEndAsync()
            $finished = $process.WaitForExit([int][math]::Max(0, 20000 - $clock.ElapsedMilliseconds))
            $timedOut = !$finished -or $clock.Elapsed.TotalMilliseconds -gt 20000
            if ($timedOut -and !$process.HasExited) {
                $process.Kill($true)
                if (!$process.WaitForExit(500)) { throw "Could not stop $phase" }
            }
            $clock.Stop()
            $receipt = [ordered]@{phase=$phase; wall_s=$clock.Elapsed.TotalSeconds; timeout=$timedOut;
                exit=$process.ExitCode; executable=$exe; arguments=$arguments}
            $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output "$phase.receipt.json")
            $stdout = $readOut.GetAwaiter().GetResult()
            $stderr = $readErr.GetAwaiter().GetResult()
            $stdout | Set-Content -LiteralPath (Join-Path $output "$phase.stdout.log")
            $stderr | Set-Content -LiteralPath (Join-Path $output "$phase.stderr.log")
            Write-Host ("{0} wall={1:F3}s exit={2} timeout={3}" -f $phase, $receipt.wall_s, $receipt.exit, $timedOut)
            if ($stdout) { Write-Host $stdout }
            if ($stderr) { Write-Host $stderr }
            return [pscustomobject]$receipt
        } finally {
            $process.Dispose()
        }
    }

    try {
        $training = Join-Path $output 'training'
        $trainArgs = if ($BankCurriculum) {
            @('bank-curriculum', $gait, $offset, $learnStarts, $previous, $training)
        } else {
            @($gait, $offset, $learnStarts, $training)
        }
        $trainingReceipt = Invoke-Capped 'training' $train $trainArgs
        if ($trainingReceipt.timeout -or $trainingReceipt.exit -ne 0) {
            throw 'Training process stopped. See the preserved receipt and output.'
        }
        $candidate = Join-Path $training 'actor-final.bin'
        if (!(Test-Path -LiteralPath $candidate)) { throw 'Training produced no completed actor.' }
        $candidateHash = Get-FileHash -LiteralPath $candidate
        $candidateHash | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'candidate.json')
        $evalDir = Join-Path $output 'heldout'
        $evalReceipt = Invoke-Capped 'evaluation' $evaluate @($gait, $offset, $heldout, $candidate, $evalDir)
        if ($evalReceipt.timeout -or $evalReceipt.exit -notin @(0, 1)) {
            throw 'Held-out process stopped. See the preserved receipt and output.'
        }
        if ((Get-FileHash -LiteralPath $candidate).Hash -ne $candidateHash.Hash) { throw 'Evaluation changed its actor' }
        $summary = Get-Content -LiteralPath (Join-Path $evalDir 'summary.json') -Raw | ConvertFrom-Json
        if ($summary.pass -ne ($evalReceipt.exit -eq 0)) { throw 'Held-out outcome disagrees with exit status' }
    } finally {
        foreach ($binding in $bindings) {
            if ((Get-FileHash -LiteralPath $binding.Path).Hash -ne $binding.Hash) {
                throw "Bound source or model changed at $($binding.Path)"
            }
        }
    }
} finally {
    Pop-Location
}

