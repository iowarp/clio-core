#!/usr/bin/env python3
"""
test_globus_assim.py - Python example for Globus data assimilation via CAE

Demonstrates how to assimilate files from a public Globus endpoint into the
local filesystem using the Content Assimilation Engine (CAE).

The script:
  1. Reads Globus tokens from the environment or /tmp/globus_tokens.sh
  2. Writes a temporary OMNI YAML describing the transfer
  3. Starts the Clio runtime (CTE + CAE compose) in the background
  4. Calls ``clio_cae <omni.yaml>`` to drive the transfer
  5. Verifies the output files exist
  6. Stops the runtime

Prerequisites:
  - Built with -DCAE_ENABLE_GLOBUS=ON
  - GLOBUS_ACCESS_TOKEN set (and GLOBUS_HTTPS_ACCESS_TOKEN for HTTPS endpoints)
  - ``clio_run``, ``clio_cae`` on PATH (or BIN_DIR env var pointing to build/bin)
  - Run ``python3 get_oauth_token.py --client-id <id> <collection_id>`` first
    to obtain tokens and save them to /tmp/globus_tokens.sh

Usage:
  export GLOBUS_ACCESS_TOKEN="<transfer_token>"
  export GLOBUS_HTTPS_ACCESS_TOKEN="<collection_token>"   # optional
  python3 test_globus_assim.py

  # Or let the script load saved tokens automatically:
  python3 test_globus_assim.py

  # Override output directory:
  OUTPUT_DIR=/tmp/my_output python3 test_globus_assim.py
"""

import os
import sys
import shutil
import signal
import subprocess
import tempfile
import time
import pathlib

# ---------------------------------------------------------------------------
# Dataset: AFRL Materials Data Facility – public GCSv5 guest collection
# ---------------------------------------------------------------------------
COLLECTION_ID = "82f1b5c6-6e9b-11e5-ba47-22000b92c6ec"
REMOTE_FILES = [
    (
        "/afrl-challenge-data/published/publication_1151/data/Input Data/HomeIn-Build B.csv",
        "HomeIn-Build_B.csv",
    ),
]

# Runtime YAML template (CTE + CAE compose pools)
_RUNTIME_CONF_TEMPLATE = """\
memory:
  main_segment_size: 256MB
  client_data_segment_size: 256MB
  runtime_data_segment_size: 256MB
networking:
  port: 9413
  neighborhood_size: 1
logging:
  level: info
  file: /tmp/clio_run_globus_test.log
runtime:
  num_threads: 4
  queue_depth: 1024
  local_sched: default
  heartbeat_interval: 1000
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: 64MB
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
    storage:
      - path: "ram::cte_ram_tier1"
        bdev_type: ram
        capacity_limit: 256MB
        score: 0.0
    dpe:
      dpe_type: max_bw
  - mod_name: clio_cae_core
    pool_name: cae_main
    pool_query: local
    pool_id: "400.0"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_saved_tokens():
    """Source /tmp/globus_tokens.sh (if it exists) into os.environ."""
    sh_path = "/tmp/globus_tokens.sh"
    if not os.path.exists(sh_path):
        return
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {sh_path} && env"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                if key.startswith("GLOBUS_"):
                    os.environ.setdefault(key, val)
    except Exception:
        pass


def _require_token():
    """Return GLOBUS_ACCESS_TOKEN or abort with instructions."""
    _load_saved_tokens()
    token = os.environ.get("GLOBUS_ACCESS_TOKEN", "")
    if not token:
        print("ERROR: GLOBUS_ACCESS_TOKEN is not set.", file=sys.stderr)
        print(
            "Run: python3 get_oauth_token.py --client-id <id> "
            f"{COLLECTION_ID}",
            file=sys.stderr,
        )
        print("Then: source /tmp/globus_tokens.sh", file=sys.stderr)
        sys.exit(1)
    return token


def _build_omni_yaml(output_dir: str, access_token: str) -> str:
    """Return an OMNI YAML string for the configured remote files."""
    lines = ['version: "1.0"', "", "transfers:"]
    for remote_path, local_name in REMOTE_FILES:
        local_path = os.path.join(output_dir, local_name)
        lines.append(f'  - src: "globus://{COLLECTION_ID}{remote_path}"')
        lines.append(f'    dst: "file::{local_path}"')
        lines.append(f'    format: "binary"')
        lines.append(f'    src_token: "{access_token}"')
        https_token = os.environ.get("GLOBUS_HTTPS_ACCESS_TOKEN", "")
        if https_token:
            lines.append(f'    dst_token: "{https_token}"')
        lines.append("")
    return "\n".join(lines)


def _find_bin_dir() -> str:
    """Return the directory that contains clio_run / clio_cae."""
    if "BIN_DIR" in os.environ:
        return os.environ["BIN_DIR"]
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    candidate = repo_root / "build" / "bin"
    if candidate.is_dir():
        return str(candidate)
    # Fall back to PATH
    return ""


def _run(cmd: list, env: dict | None = None, **kwargs):
    """Run a command, raising on non-zero exit."""
    return subprocess.run(
        cmd,
        env=env or os.environ.copy(),
        check=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_globus_assim_test(output_dir: str | None = None) -> int:
    """
    Run the Globus assimilation test.

    Returns 0 on success, non-zero on failure.
    """
    output_dir = output_dir or os.environ.get("OUTPUT_DIR", "/tmp/globus_matsci")
    os.makedirs(output_dir, exist_ok=True)

    access_token = _require_token()
    bin_dir = _find_bin_dir()

    env = os.environ.copy()
    if bin_dir:
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        )

    # Write runtime config and OMNI file to a temp directory
    tmp = tempfile.mkdtemp(prefix="globus_assim_")
    runtime_conf = os.path.join(tmp, "clio_runtime_conf.yaml")
    omni_file = os.path.join(tmp, "transfer.yaml")

    with open(runtime_conf, "w") as f:
        f.write(_RUNTIME_CONF_TEMPLATE)
    with open(omni_file, "w") as f:
        f.write(_build_omni_yaml(output_dir, access_token))

    env["CLIO_SERVER_CONF"] = runtime_conf

    print("=" * 55)
    print("Globus assimilation test")
    print("=" * 55)
    print(f"  Collection : {COLLECTION_ID}")
    print(f"  Output dir : {output_dir}")
    print(f"  OMNI file  : {omni_file}")
    print()

    runtime_proc = None
    rc = 1
    try:
        # Start runtime
        print("Starting Clio runtime...")
        runtime_proc = subprocess.Popen(
            ["clio_run", "runtime", "start"],
            env=env,
        )

        # Wait for IPC socket
        ipc_socket = f"/tmp/clio_{os.environ.get('USER', 'user')}/clio_9413.ipc"
        print(f"Waiting for runtime socket ({ipc_socket})...", end="", flush=True)
        for _ in range(60):
            if os.path.exists(ipc_socket):
                break
            time.sleep(1)
            print(".", end="", flush=True)
        print()
        time.sleep(5)  # grace period for compose pools

        # Run CAE against the OMNI file
        print("Running clio_cae...")
        result = subprocess.run(
            ["clio_cae", omni_file],
            env=env,
        )
        rc = result.returncode

        if rc == 0:
            print()
            print("=" * 55)
            print("PASSED")
            print("=" * 55)
            print(f"Output files in {output_dir}:")
            for _, local_name in REMOTE_FILES:
                path = os.path.join(output_dir, local_name)
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    print(f"  {local_name}  ({size:,} bytes)")
                else:
                    print(f"  {local_name}  MISSING")
                    rc = 1
        else:
            print()
            print("=" * 55)
            print(f"FAILED  (clio_cae exit code {rc})")
            print("=" * 55)

    finally:
        # Stop runtime
        print()
        print("Stopping runtime...")
        try:
            subprocess.run(
                ["clio_run", "runtime", "stop"],
                env=env, timeout=15, check=False,
            )
        except Exception:
            pass
        if runtime_proc and runtime_proc.poll() is None:
            runtime_proc.terminate()
            try:
                runtime_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                runtime_proc.kill()

        shutil.rmtree(tmp, ignore_errors=True)

    return rc


if __name__ == "__main__":
    sys.exit(run_globus_assim_test())
