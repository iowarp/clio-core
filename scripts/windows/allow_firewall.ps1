# Add Windows Defender Firewall allow rules for every test binary in
# build\bin so ctest doesn't pop a "Allow access?" prompt every time a
# runtime/test process binds 0.0.0.0:<port>. Must be run from an
# elevated PowerShell prompt -- netsh advfirewall firewall add rule
# requires Administrator.
#
# Usage (from a regular Windows PowerShell prompt; this script auto-elevates):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\allow_firewall.ps1
#
# Re-run after a fresh build adds new test executables; the rule names
# embed the binary stem so existing entries are skipped and only new
# binaries get added.

$ErrorActionPreference = 'Stop'

# Re-elevate if not already running as admin.
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Not elevated; relaunching as Administrator...'
    # Prefer pwsh (PowerShell Core) if available, fall back to powershell.exe
    # so this script works on both stock Windows and machines with PS7+.
    $shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell' }
    Start-Process $shell -Verb RunAs -ArgumentList @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $MyInvocation.MyCommand.Path
    )
    exit
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')

# Cover every build tree's bin dir: build\bin, build-cuda\bin, build-nocuda\bin,
# etc. ctest only runs binaries under these, so allowing them all stops the
# Defender prompts regardless of which preset you built.
$binDirs = Get-ChildItem -Path $repoRoot -Directory -Filter 'build*' |
    ForEach-Object { Join-Path $_.FullName 'bin' } |
    Where-Object { Test-Path $_ }

if (-not $binDirs) {
    Write-Host "No build*/bin directories under $repoRoot; build the project first."
    exit 1
}

# Enumerate every .exe and add an inbound + outbound allow rule per binary
# (a process binding 0.0.0.0:<port> can trigger either). The rule name embeds
# the full path so binaries with the same name in different build trees get
# distinct, idempotent rules.
$exes = foreach ($binDir in $binDirs) { Get-ChildItem $binDir -Filter *.exe -File }
$added = 0
foreach ($f in $exes) {
    $exe  = $f.FullName
    $name = "CLIO Core test - $exe"

    netsh advfirewall firewall show rule name="$name" *> $null
    if ($LASTEXITCODE -eq 0) { continue }   # already present

    netsh advfirewall firewall add rule name="$name" dir=in  action=allow `
        program="$exe" profile=any enable=yes | Out-Null
    netsh advfirewall firewall add rule name="$name" dir=out action=allow `
        program="$exe" profile=any enable=yes | Out-Null
    $added++
    Write-Host "Allowed $($f.Name)  ($($f.DirectoryName))"
}

Write-Host "Done. Added rules for $added new binaries across:"
$binDirs | ForEach-Object { Write-Host "  $_" }
Write-Host "Re-run after a fresh build adds new test executables."
