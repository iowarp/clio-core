# Retry a command that talks to the network (PowerShell half of CI/retry.sh).
#
# Same contract, same reasoning: chocolatey.org and the WinFsp release CDN drop
# out for tens of seconds and take a Windows job's red X with them. Wrap the
# install, never the build or the test.
#
# Usage:
#   pwsh CI/Retry.ps1 choco install winfsp -y --no-progress
#   pwsh CI/Retry.ps1 -Attempts 5 -DelaySeconds 10 <command> [args...]
#
# Do NOT put a '--' separator before the command. Unlike the bash half, this
# is an advanced function ([CmdletBinding()]), and PowerShell's parameter
# binder reads '--' as a parameter whose name is empty -- so it fails with
# "the parameter name '' is ambiguous" BEFORE a single line of this script
# runs, and the wrapped command never executes at all. That took every
# Windows wheel build red the day this helper landed: the WinFsp install
# silently never ran, and the #1004 SDK assertion downstream was what
# actually reported the failure.
#
# The command's own dashed flags need no protection -- ValueFromRemaining-
# Arguments collects '-y' and '--no-progress' into $Command unharmed.
#
# PositionalBinding is off for a second, independent reason: without it the
# first bare argument binds to -Attempts, so the two call sites that do not
# pass -Attempts (build-pip test-wheels-windows, build-packages nsis) fail with
#   Cannot convert value "choco" to type "System.Int32"
# even after the '--' above is gone. Removing the separator and turning
# positional binding off are both required; either alone still fails.

[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Attempts = 3,
    [int]$DelaySeconds = 5,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Continue'
# PowerShell 7.4 flipped $PSNativeCommandUseErrorActionPreference to $true by
# default, which turns "native command exited non-zero" into a TERMINATING
# error under the $ErrorActionPreference='Stop' that GitHub sets for pwsh
# steps. That silently killed the hand-rolled `foreach ($attempt in 1..3)`
# retry loops in this repo: they aborted on attempt 1, so a chocolatey.org
# outage read as a hard failure and the retry never happened. Turn it off here
# so this script -- and only this script -- decides what a non-zero exit means.
$PSNativeCommandUseErrorActionPreference = $false

if ($Command.Count -eq 0) {
    Write-Error 'CI/Retry.ps1: no command given'
    exit 2
}

$exe = $Command[0]
$rest = @()
if ($Command.Count -gt 1) { $rest = $Command[1..($Command.Count - 1)] }
$delay = $DelaySeconds

for ($n = 1; $n -le $Attempts; $n++) {
    & $exe @rest
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { exit 0 }

    if ($n -eq $Attempts) {
        Write-Error "CI/Retry.ps1: '$($Command -join ' ')' failed $Attempts/$Attempts attempts (last rc=$rc)"
        exit $rc
    }
    Write-Host "CI/Retry.ps1: attempt $n/$Attempts failed (rc=$rc); retrying in ${delay}s: $($Command -join ' ')"
    Start-Sleep -Seconds $delay
    $delay = $delay * 2
}
