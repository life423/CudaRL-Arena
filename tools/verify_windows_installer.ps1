param([string]$Installer='build/windows-release/TadpolePondSetup.exe')
$ErrorActionPreference='Stop'
$workspace=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$qa=Join-Path $workspace ('build/installer-qa-'+[DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))
$installed=Join-Path $qa 'Tadpole Pond'
New-Item -ItemType Directory -Path $qa | Out-Null
$desktop=Join-Path ([Environment]::GetFolderPath('Desktop')) 'Tadpole Pond.lnk'
$start=Join-Path ([Environment]::GetFolderPath('Programs')) 'Tadpole Pond.lnk'
$registration='HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\TadpolePond'
# Exercise real per-user integration without overwriting an existing installation.
if((Test-Path -LiteralPath $desktop) -or (Test-Path -LiteralPath $start) -or (Test-Path -LiteralPath $registration)) {
    throw 'An existing install/shortcut is present. Preserve it; run the isolated /TESTINSTALL flow instead.'
}
$checks=0
function Check([bool]$ok,[string]$message) {
    if(-not $ok) {throw $message}
    $script:checks++
}
$process=Start-Process -FilePath (Resolve-Path $Installer).Path -ArgumentList @('/S',('/D='+$installed)) -WindowStyle Hidden -PassThru
$process.WaitForExit();$process.Refresh()
Check ($process.ExitCode -eq 0) 'Installer failed'
$manifest=Get-Content -LiteralPath (Join-Path $installed 'release-manifest.json') -Raw | ConvertFrom-Json
foreach($file in $manifest.files) {
    Check ((Get-FileHash -LiteralPath (Join-Path $installed $file.path) -Algorithm SHA256).Hash.ToLowerInvariant() -eq $file.sha256) "Installed file mismatch: $($file.path)"
}
$shell=New-Object -ComObject WScript.Shell
foreach($shortcut in @($desktop,$start)) {
    Check (Test-Path -LiteralPath $shortcut) 'Shortcut missing'
    Check ($shell.CreateShortcut($shortcut).TargetPath -eq (Join-Path $installed 'TadpolePond.exe')) 'Shortcut target incorrect'
}
$registered=Get-ItemProperty -LiteralPath $registration
Check ($registered.InstallLocation -eq $installed) 'Per-user install registration missing'
Check ($registered.UninstallString -eq ('"'+$installed+'\Uninstall.exe"')) 'Uninstall command is not quoted correctly'
$exe=Join-Path $installed 'TadpolePond.exe'
$process=Start-Process -FilePath $exe -ArgumentList @('--script','res://qa/live_feeding_smoke.gd','--','--pond-backend=cpu') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $qa 'cpu.out') -RedirectStandardError (Join-Path $qa 'cpu.err')
$process.WaitForExit();$process.Refresh()
Check ($process.ExitCode -eq 0) 'Installed app CPU learning failed'
$appOutput=Get-Content -LiteralPath (Join-Path $qa 'cpu.out') -Raw
Check ($appOutput -match 'LIVE_FEEDING checks=\d+ failures=0') 'Installed app did not complete its learning/resume check'
Check (-not ((Get-Content -LiteralPath (Join-Path $qa 'cpu.err') -Raw) -match 'ERROR|SCRIPT ERROR')) 'Installed app reported errors'
# An unrelated file must survive uninstall; the script must never recursively delete a selected folder.
$sentinel=Join-Path $installed 'keep-user-file.txt'
'preserve this file' | Set-Content -LiteralPath $sentinel
$uninstaller=Join-Path $installed 'Uninstall.exe'
# _?= runs in place, allowing a reliable wait for the uninstall process.
$process=Start-Process -FilePath $uninstaller -ArgumentList @('/S',('_?='+$installed)) -WindowStyle Hidden -PassThru
$process.WaitForExit();$process.Refresh()
Check ($process.ExitCode -eq 0) 'Uninstaller failed'
Check (Test-Path -LiteralPath $sentinel) 'Uninstaller deleted unrelated files'
Check (-not (Test-Path -LiteralPath $exe)) 'Uninstaller left the game executable'
Check (-not (Test-Path -LiteralPath $desktop)) 'Uninstaller left its desktop shortcut'
Check (-not (Test-Path -LiteralPath $start)) 'Uninstaller left its Start Menu shortcut'
Check (-not (Test-Path -LiteralPath $registration)) 'Uninstaller left its registration'
# QA data and full learner saves are deliberately retained for inspection.
[ordered]@{checks=$checks;result='PASS';appCheck=($appOutput -split "`n" | Where-Object {$_ -match 'LIVE_FEEDING checks='});output=$qa;manifestSource=$manifest.sourceCommit;sourceDirty=$manifest.sourceDirty} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $qa 'result.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $qa 'result.json')
