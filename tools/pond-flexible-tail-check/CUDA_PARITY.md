# Flexible-tail CUDA spot check

Build from the repository root in a Visual Studio Developer PowerShell with
`nvcc.exe` on PATH. The recorded run used CUDA 13.2 and a GeForce RTX 5070.

```powershell
$output = 'build/flexible-tail/cuda-parity'
New-Item -ItemType Directory -Path $output -Force | Out-Null
nvcc.exe -std=c++17 -O2 -arch=sm_120 --fmad=false '-Xcompiler=/O2,/fp:precise' -Isrc -Isrc/nn tools/pond-flexible-tail-check/cuda-parity.cu -o "$output/spot.exe"
if ($LASTEXITCODE -ne 0) { throw "CUDA parity build failed with exit code $LASTEXITCODE" }

$watch = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath (Resolve-Path "$output/spot.exe") -WindowStyle Hidden -PassThru -RedirectStandardOutput "$output/result.json" -RedirectStandardError "$output/stderr.txt"
if (!$process.WaitForExit([Math]::Max(0, 4000 - [int]$watch.ElapsedMilliseconds))) {
    $process.Kill()
    $process.WaitForExit()
    throw 'CUDA parity hit the four-second wall cap'
}
Get-Content -LiteralPath "$output/result.json", "$output/stderr.txt"
if ($process.ExitCode -ne 0) { throw "CUDA parity failed with exit code $($process.ExitCode)" }
```

One moving, tilted body receives the same 64 yaw and pitch targets on
the CPU and CUDA at 2.5 ms per substep. The check compares position,
orientation, joint angles, velocities, torque and energy at each substep
in unbounded calm water. It requires nonzero whole-body travel.

Device call boundaries on five flexible-tail functions kept the optimized
build to **5.98 s** including compiler setup. The CPU functions remain
inline. The process completed in **0.328 s** and passed **64/64** substeps
with zero measured COM position error, at most **9.32 × 10⁻¹⁰ m/s**
COM velocity error and at most **9.54 × 10⁻⁷ rad/s** angular-rate error.
[Recorded build, runtime and results](../../assets/research-evidence/pond-flexible-tail-v1/cuda-parity).

The 91.8 ms kernel measurement used one thread and one trajectory. Measure
a batched collector with representative body states before estimating
training throughput. Use the existing optimized compiler settings and
measure register use, stack size and spills with `-Xptxas=-v`. Larger
controls, shore contact, long-run parity and PPO remain outside this check.
