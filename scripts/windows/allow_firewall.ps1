# Add Windows Defender Firewall allow rules for every test binary under each
# build*/bin directory so ctest never pops an "Allow access?" prompt when a
# runtime/test process binds a port. Adding rules requires Administrator, so the
# script self-elevates -- but ONLY when there are new binaries to allow.
#
# How it stays prompt-free: after applying rules it records the set of allowed
# .exe paths in a marker file (.clio_firewall_allowed at the repo root). On the
# next run it compares the current binaries to the marker; if they match it
# exits immediately WITHOUT elevating, so the common "build once, run ctest many
# times" loop only triggers a single UAC prompt right after a fresh build adds
# new executables.
#
# It is best-effort: any failure (including a declined UAC prompt) exits 0 so it
# never fails the test suite -- you just get the per-binary prompts back.
#
# Usage (wired into ctest via the cr_firewall_allow setup fixture, or run by
# hand from a normal PowerShell prompt):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\allow_firewall.ps1

$ErrorActionPreference = 'SilentlyContinue'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')

# Every build tree's bin dir: build\bin, build-stackless\bin, build-cuda\bin, ...
$binDirs = Get-ChildItem -Path $repoRoot -Directory -Filter 'build*' -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName 'bin' } |
    Where-Object { Test-Path $_ }

$exePaths = @()
foreach ($d in $binDirs) {
    $exePaths += (Get-ChildItem $d -Filter *.exe -File -ErrorAction SilentlyContinue |
                  ForEach-Object { $_.FullName })
}
$exePaths = @($exePaths | Sort-Object -Unique)
if ($exePaths.Count -eq 0) { exit 0 }   # nothing built yet

# Fast path: if the marker already lists exactly these binaries, they're allowed.
$marker = Join-Path $repoRoot '.clio_firewall_allowed'
if (Test-Path $marker) {
    $allowed = @(Get-Content $marker -ErrorAction SilentlyContinue)
    if (-not (Compare-Object $exePaths $allowed -ErrorAction SilentlyContinue)) {
        Write-Host "Firewall: all $($exePaths.Count) test binaries already allowed."
        exit 0
    }
}

# Need to (re)apply rules -- self-elevate if we aren't Administrator.
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell' }
    try {
        Start-Process $shell -Verb RunAs -Wait -ErrorAction Stop -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-File', $MyInvocation.MyCommand.Path)
    } catch {
        Write-Host "Firewall: elevation declined; per-binary prompts may still appear."
    }
    exit 0   # best-effort: never fail the suite on a declined prompt
}

# Elevated: add an inbound + outbound allow rule per binary (a process binding a
# port can trip either). The rule name embeds the full path so binaries with the
# same stem in different build trees get distinct, idempotent rules.
$added = 0
foreach ($exe in $exePaths) {
    $name = "CLIO Core test - $exe"
    netsh advfirewall firewall show rule name="$name" *> $null
    if ($LASTEXITCODE -eq 0) { continue }   # already present
    netsh advfirewall firewall add rule name="$name" dir=in  action=allow `
        program="$exe" profile=any enable=yes | Out-Null
    netsh advfirewall firewall add rule name="$name" dir=out action=allow `
        program="$exe" profile=any enable=yes | Out-Null
    $added++
}
$exePaths | Set-Content $marker -ErrorAction SilentlyContinue
Write-Host "Firewall: added $added new rule(s); $($exePaths.Count) binaries allowed across:"
$binDirs | ForEach-Object { Write-Host "  $_" }
exit 0
