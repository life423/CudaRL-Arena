# The single human-facing status is replaced atomically on its own volume.
function Write-RunStatus {
    param([Parameter(Mandatory)][string]$Path,[Parameter(Mandatory)][hashtable]$Value)
    $temporary=$Path+'.'+[guid]::NewGuid().ToString('N')+'.tmp'
    try {
        [IO.File]::WriteAllText($temporary,($Value|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
        [IO.File]::Move($temporary,$Path,$true)
    } finally {
        if([IO.File]::Exists($temporary)){[IO.File]::Delete($temporary)}
    }
}
Export-ModuleMember -Function Write-RunStatus
