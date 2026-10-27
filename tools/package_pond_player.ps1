param(
    [Parameter(Mandatory=$true)][string]$GodotPath,
    [switch]$CpuOnly
)
$ErrorActionPreference='Stop'
$workspace=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$tag=if($CpuOnly) {'cpu-package'} else {'package'}
$destination="build/portable/$tag"
& (Join-Path $PSScriptRoot 'stage_pond_player.ps1') -GodotPath $GodotPath -Destination $destination -CpuOnly:$CpuOnly
$package=Join-Path $workspace $destination
$project=Join-Path $package 'player'
$log=Join-Path $workspace "build/portable/$tag-import"
# Start-Process joins its argument array, so quote the user-provided path explicitly.
$process=Start-Process -FilePath $GodotPath -ArgumentList @('--headless','--editor','--path',('"'+$project+'"'),'--import') -WindowStyle Hidden -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
$process.WaitForExit(); $process.Refresh()
if($process.ExitCode -ne 0 -or (Get-Content "$log.err" -Raw) -match 'SCRIPT ERROR|ERROR:') {throw "Godot import failed; inspect $log.err"}
$name=if($CpuOnly) {'CudaRL-Pond-CPU-only.zip'} else {'CudaRL-Pond-Windows.zip'}
$archive=Join-Path $workspace "build/portable/$name"
Add-Type -AssemblyName System.IO.Compression
$stream=[IO.File]::Open($archive,[IO.FileMode]::Create)
$zip=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create)
try {
    foreach($file in Get-ChildItem -LiteralPath $package -File -Recurse -Force) {
        $relative=$file.FullName.Substring($package.Length+1).Replace('\','/')
        if($relative -match '(^|/)\.godot/(editor|shader_cache)/' -or $relative -match '\.(log|out|err)$') {continue}
        # Early staging versions placed duplicate notices at this level.
        if($relative -in @('licenses/LICENSE.txt','licenses/COPYRIGHT.txt')) {continue}
        $entry=$zip.CreateEntry($relative,[IO.Compression.CompressionLevel]::Optimal)
        $input=[IO.File]::OpenRead($file.FullName)
        $output=$entry.Open()
        try {$input.CopyTo($output)} finally {$output.Dispose(); $input.Dispose()}
    }
} finally {$zip.Dispose(); $stream.Dispose()}
Get-Item -LiteralPath $archive | Select-Object FullName,Length
Get-FileHash -LiteralPath $archive -Algorithm SHA256
