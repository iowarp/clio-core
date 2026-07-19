# Close an orphaned TCP listener handle that some unrelated process
# (typically the IDE: Cursor/VS Code/...) inherited from one of our test
# binaries and is now leaking, holding the port in LISTEN state across
# subsequent test runs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\close_orphan_socket.ps1 -Port 9416
#
# Mechanism: download Sysinternals handle.exe if needed, find every PID
# that's listening on the given port, list each PID's handles, and close
# every "<Unknown type>" handle (sockets) until netstat shows the port
# free. We deliberately don't try to identify "the" socket handle: handle.exe
# can't surface which TCP port maps to which handle, so the only reliable
# move is "close until the port frees". The collateral risk is breaking
# other sockets in the holder process; in practice the holder is an IDE
# whose own sockets won't be on our test ports.
#
# Long-term: libzmq on Windows should set HANDLE_FLAG_INHERIT=0 on its
# listener sockets so child processes (and grandparents that inherited
# them from us) can't leak them. Until then, this script is the rescue.

param(
    [int]$Port = 9416
)

$ErrorActionPreference = 'Stop'

# Re-elevate if needed.
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Not elevated; relaunching as Administrator...'
    $shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell' }
    Start-Process $shell -Verb RunAs -ArgumentList @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $MyInvocation.MyCommand.Path,
        '-Port', $Port
    )
    exit
}

# Locate or fetch handle.exe.
$handleDir = "$env:TEMP\sysinternals-handle"
$handle    = Join-Path $handleDir 'handle64.exe'
if (-not (Test-Path $handle)) {
    Write-Host 'Downloading Sysinternals Handle...'
    New-Item -ItemType Directory -Force -Path $handleDir | Out-Null
    Invoke-WebRequest -Uri 'https://download.sysinternals.com/files/Handle.zip' `
        -OutFile (Join-Path $handleDir 'Handle.zip') -UseBasicParsing
    Expand-Archive -Force -Path (Join-Path $handleDir 'Handle.zip') -DestinationPath $handleDir
}

function Get-PortPids {
    param([int]$Port)
    @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
}

$pids = Get-PortPids -Port $Port
if ($pids.Count -eq 0) {
    Write-Host "Port $Port is already free."
    exit 0
}

foreach ($targetPid in $pids) {
    $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    Write-Host "Inspecting PID $targetPid ($($proc.Name)): $($proc.Path)"

    # Dump handles, find every "<Unknown type>" entry (typical for sockets).
    $output = & $handle -accepteula -p $targetPid -a 2>&1
    # Lines look like:   <hex>:  <type>  <name>
    # We want lines whose type column is `<Unknown` (the literal `<Unknown type>`).
    $candidates = @($output | Where-Object { $_ -match '^\s*([0-9A-Fa-f]+):\s*<Unknown type>' } |
                    ForEach-Object {
                        if ($_ -match '^\s*([0-9A-Fa-f]+):') { $matches[1] }
                    })
    Write-Host "  Found $($candidates.Count) candidate socket handles"

    foreach ($h in $candidates) {
        # Close the handle. handle.exe will print whether the port frees up
        # after each close so we can stop early.
        & $handle -accepteula -c $h -p $targetPid -y 2>&1 | Out-Null

        # Re-check the port; bail as soon as it's free.
        if ((Get-PortPids -Port $Port).Count -eq 0) {
            Write-Host "  Port $Port freed after closing handle 0x$h"
            exit 0
        }
    }
}

# Final state.
if ((Get-PortPids -Port $Port).Count -eq 0) {
    Write-Host "Port $Port is now free."
    exit 0
} else {
    Write-Host "Failed to free port $Port; remaining holders:"
    Get-PortPids -Port $Port | ForEach-Object {
        $p = Get-Process -Id $_ -ErrorAction SilentlyContinue
        "  PID $($_) $($p.Name)"
    }
    exit 1
}
