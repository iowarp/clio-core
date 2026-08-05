# ---------------------------------------------------------------------------
# Windows (WinFsp) FUSE mount smoke test — pwsh port of CI/fuse_mount_smoke.sh.
#
# Brings up the Clio runtime, composes the CTE pool, mounts clio_cte_fuse.exe
# on a free drive letter via WinFsp, writes+reads+verifies a file through the
# mount, then tears everything down. Exercises the real WinFsp FUSE backend
# end-to-end — the unit suites (test_fuse_adapter / test_fuse_ops) never
# actually mount.
#
# Usage:  pwsh CI/fuse_mount_smoke.ps1 -BuildDir <build-dir>
#   <build-dir>  CMake binary dir whose bin/ holds clio_run.exe +
#                clio_cte_fuse.exe (this repo flattens per-config output
#                into bin/, so no Debug/ suffix).
#
# Requires the WinFsp runtime (driver + winfsp-x64.dll) to be installed —
# ci-adapters.yml installs the MSI with ADDLOCAL=ALL before building.
# ---------------------------------------------------------------------------
param([Parameter(Mandatory = $true)][string]$BuildDir)

$ErrorActionPreference = "Stop"
$bin = Join-Path (Resolve-Path $BuildDir) "bin"
$cfgDir = Resolve-Path (Join-Path $PSScriptRoot "..\context-transfer-engine\test\integration\fuse-manual")

function Info($msg) { Write-Host "[smoke] $msg" }

# --- Preflight --------------------------------------------------------------
foreach ($exe in @("clio_run.exe", "clio_cte_fuse.exe")) {
    if (-not (Test-Path (Join-Path $bin $exe))) {
        throw "[smoke] ERROR: $bin\$exe not found (was the FUSE adapter built?)"
    }
}
# winfsp-x64.dll resolution: the MSI does not put WinFsp's bin on PATH.
$winfspBin = "${env:ProgramFiles(x86)}\WinFsp\bin"
if (Test-Path $winfspBin) { $env:PATH = "$winfspBin;$env:PATH" }

$env:PATH = "$bin;$env:PATH"
$env:CLIO_SERVER_CONF = Join-Path $cfgDir "cte_config.yaml"
$env:CLIO_BIND_ADDR = "127.0.0.1"

# Free drive letter for the mount (WinFsp fuse mounts drive letters natively).
$used = (Get-PSDrive -PSProvider FileSystem).Name
$mount = $null
foreach ($l in @('X', 'Y', 'W', 'V', 'U')) {
    if ($used -notcontains $l) { $mount = "${l}:"; break }
}
if (-not $mount) { throw "[smoke] ERROR: no free drive letter found" }

$runtimeProc = $null
$fuseProc = $null
$exitCode = 1

try {
    # --- Start runtime ------------------------------------------------------
    Info "starting Clio runtime"
    $runtimeProc = Start-Process -FilePath (Join-Path $bin "clio_run.exe") `
        -ArgumentList "runtime", "start" -PassThru -NoNewWindow `
        -RedirectStandardOutput "$env:TEMP\clio_runtime.out" `
        -RedirectStandardError "$env:TEMP\clio_runtime.err"
    Start-Sleep -Seconds 5
    if ($runtimeProc.HasExited) {
        Get-Content "$env:TEMP\clio_runtime.out", "$env:TEMP\clio_runtime.err" -ErrorAction SilentlyContinue
        throw "[smoke] ERROR: runtime died on startup (exit $($runtimeProc.ExitCode))"
    }

    # --- Compose the CTE pool ----------------------------------------------
    Info "composing CTE pool"
    & (Join-Path $bin "clio_run.exe") compose (Join-Path $cfgDir "cte_compose.yaml")
    if ($LASTEXITCODE -ne 0) { throw "[smoke] ERROR: compose failed ($LASTEXITCODE)" }

    # --- Mount --------------------------------------------------------------
    Info "mounting clio_cte_fuse at $mount"
    $env:CLIO_WITH_RUNTIME = "0"
    $fuseProc = Start-Process -FilePath (Join-Path $bin "clio_cte_fuse.exe") `
        -ArgumentList $mount, "-f" -PassThru -NoNewWindow `
        -RedirectStandardOutput "$env:TEMP\clio_fuse.out" `
        -RedirectStandardError "$env:TEMP\clio_fuse.err"
    $mounted = $false
    foreach ($i in 1..20) {
        if (Test-Path "$mount\") { $mounted = $true; break }
        if ($fuseProc.HasExited) {
            Get-Content "$env:TEMP\clio_fuse.out", "$env:TEMP\clio_fuse.err" -ErrorAction SilentlyContinue
            throw "[smoke] ERROR: FUSE daemon exited before mount (exit $($fuseProc.ExitCode))"
        }
        Start-Sleep -Seconds 1
    }
    if (-not $mounted) {
        Get-Content "$env:TEMP\clio_fuse.out", "$env:TEMP\clio_fuse.err" -ErrorAction SilentlyContinue
        throw "[smoke] ERROR: mount did not appear within 20s"
    }
    Info "mount is live"

    # --- I/O round-trip -----------------------------------------------------
    $payload = "hello context-transfer-engine fuse Windows/WinFsp"
    Set-Content -Path "$mount\smoke.txt" -Value $payload -NoNewline
    $got = Get-Content -Path "$mount\smoke.txt" -Raw
    if ($got -ne $payload) {
        throw "[smoke] ERROR: readback mismatch`n  wrote: $payload`n  read:  $got"
    }
    Info "PASS: wrote and read back '$payload' through the WinFsp mount"

    # Directory listing sanity: the file we just wrote must be enumerable.
    $names = (Get-ChildItem "$mount\").Name
    if ($names -notcontains "smoke.txt") {
        throw "[smoke] ERROR: smoke.txt missing from directory listing ($names)"
    }
    Info "PASS: directory listing shows smoke.txt"
    $exitCode = 0
}
finally {
    Info "cleaning up"
    # Killing the fuse process makes WinFsp unmount the volume.
    if ($fuseProc -and -not $fuseProc.HasExited) {
        Stop-Process -Id $fuseProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($runtimeProc -and -not $runtimeProc.HasExited) {
        & (Join-Path $bin "clio_run.exe") runtime stop 2>$null | Out-Null
        Start-Sleep -Seconds 2
        if (-not $runtimeProc.HasExited) {
            Stop-Process -Id $runtimeProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
exit $exitCode
