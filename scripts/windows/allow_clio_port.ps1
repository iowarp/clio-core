# Open the CLIO Core RPC port (ZMQ TCP) on Windows Defender Firewall so peer
# nodes in the cluster (e.g. jelly, nene) can establish inbound TCP connections
# to this node (raven). Binding a listening socket on a high port does NOT need
# Administrator, but accepting INBOUND connections from other hosts does -- the
# default firewall posture drops them. Adding the allow rule requires elevation.
#
# Usage (from a normal PowerShell prompt; this script auto-elevates):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\allow_clio_port.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows\allow_clio_port.ps1 -Port 9413
#
# Idempotent: re-running skips a rule that is already present.

param(
    [int]$Port = 9413
)

$ErrorActionPreference = 'Stop'

# Re-elevate if not already running as Administrator.
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

$ruleName = "CLIO Core RPC (TCP $Port) inbound"

# Skip if the rule already exists (idempotent re-runs).
netsh advfirewall firewall show rule name="$ruleName" *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Rule '$ruleName' already present; nothing to do."
    exit 0
}

# Inbound allow for the RPC port on all profiles (domain/private/public) so the
# cluster works regardless of which network profile the cluster NIC is on.
netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow `
    protocol=TCP localport=$Port profile=any enable=yes | Out-Null

Write-Host "Opened inbound TCP $Port for the CLIO Core cluster RPC listener."
Write-Host "Peers (jelly, nene) can now connect to this node on port $Port."
