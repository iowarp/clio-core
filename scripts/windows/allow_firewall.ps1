# Add Windows Defender Firewall allow rules for every test binary under each
# build*/bin directory so ctest never pops an "Allow access?" prompt when a
# runtime/test process binds a port. Adding rules requires Administrator.
#
# It NEVER raises a UAC prompt on its own: a prompt nobody answers blocks until
# the caller gives up, which is how the cr_firewall_allow ctest fixture used to
# eat its full TIMEOUT and fail on the dashboard. Unelevated, it reports which
# binaries still need rules and exits 0.
#
# After applying rules it records the set of allowed .exe paths in a marker file
# (.clio_firewall_allowed at the repo root). On the next run it compares the
# current binaries to the marker; if they match it exits immediately, so the
# common "build once, run ctest many times" loop does no work at all.
#
# It is best-effort: it always exits 0 so it never fails the test suite -- if the
# rules weren't applied you just get the per-binary prompts back.
#
# Usage (wired into ctest via the cr_firewall_allow setup fixture; to actually
# apply rules, run it by hand from an ELEVATED PowerShell prompt):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\allow_firewall.ps1
#
# Set CLIO_FIREWALL_ELEVATE=1 to let an unelevated run self-elevate via UAC --
# only from an interactive shell where you can answer the prompt.

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

# Need to (re)apply rules -- which requires Administrator.
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    # Do NOT auto-elevate. `Start-Process -Verb RunAs` blocks inside
    # ShellExecuteEx until the UAC consent dialog is answered, and on an
    # unattended run (ctest -D <Dashboard>, CI, a scheduled task) nobody answers
    # it -- the fixture silently burns its whole TIMEOUT and reports a failure
    # (CDash test 446369917). No timeout on our side can rescue that: the block
    # happens before we ever get a process handle back.
    #
    # There is no reliable way to tell "a human is watching" from here -- ctest
    # does not even export CTEST_INTERACTIVE_DEBUG_MODE in dashboard mode -- so
    # elevation is opt-in only. Either run this script from an already-elevated
    # prompt (the branch below), or set CLIO_FIREWALL_ELEVATE=1 to accept the
    # prompt.
    if ($env:CLIO_FIREWALL_ELEVATE -ne '1') {
        Write-Host ("Firewall: $($exePaths.Count) test binaries still need rules; " +
                    "not elevating (a UAC prompt would hang an unattended run).")
        Write-Host ("Firewall: to apply them, run " +
                    "scripts\windows\allow_firewall.ps1 from an elevated prompt, " +
                    "or set CLIO_FIREWALL_ELEVATE=1.")
        exit 0
    }

    $shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell' }
    try {
        Start-Process $shell -Verb RunAs -Wait -ErrorAction Stop -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-File', $PSCommandPath)
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
