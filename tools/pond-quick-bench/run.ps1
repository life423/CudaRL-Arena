param(
    [string]$Executable = "$PSScriptRoot/../../build/quick32/native/Release/pond_quick_bench.exe",
    [string]$Module = "$PSScriptRoot/../../build/quick32/module/pond_ppo.1.sm_120.cubin",
    [string]$Out = "$PSScriptRoot/../../build/quick-layout-$([DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))"
)
$ErrorActionPreference = 'Stop'
$commandWatch = [Diagnostics.Stopwatch]::StartNew()
$repo = [IO.Path]::GetFullPath("$PSScriptRoot/../..")
$evidence = Join-Path $repo 'assets/research-evidence/pond-quick-bench-v1'
$binding = Get-Content -LiteralPath (Join-Path $evidence 'bindings.json') -Raw | ConvertFrom-Json
$fixture = Join-Path $evidence 'fixtures.bin'
if ((Get-FileHash -LiteralPath $fixture -Algorithm SHA256).Hash.ToLowerInvariant() -ne $binding.fixture_sha256) { throw 'Prepared fixture hash mismatch' }
if ((Get-FileHash -LiteralPath $Module -Algorithm SHA256).Hash.ToLowerInvariant() -ne $binding.module_sha256) { throw 'The exact frozen v2 CUBIN is required; see README' }
if (Test-Path -LiteralPath $Out) { throw 'Choose a fresh output directory; previous evidence is preserved' }
$processWatch = [Diagnostics.Stopwatch]::StartNew()
& $Executable --run --fixture $fixture --module $Module --symbol $binding.step_symbol --out $Out
$processCode = $LASTEXITCODE
$processWatch.Stop()
$commandWatch.Stop()
[ordered]@{
    exit_code = $processCode
    process_wall_ms = $processWatch.Elapsed.TotalMilliseconds
    command_wall_ms = $commandWatch.Elapsed.TotalMilliseconds
    includes_fixture_module_hashes = $true
    includes_process_startup_and_teardown = $true
    optimizer_calls = 0
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'command-wall.json') -Encoding utf8
if ($processCode -ne 0) { throw "Benchmark failed ($processCode); preserve the output and inspect result.json" }
