#!/usr/bin/env python3
"""Real capacity-binding regression with private storage and owned daemons.

Run against an installed artifact, for example:
    python test_cte_capacity.py --module-dir PREFIX/iowarp_core/ext \
        --bin-dir PREFIX/iowarp_core/bin --output-dir BUILD/capacity-results

Requires psutil. Missing prerequisites fail rather than skip. The parent never
attaches a native client: each phase uses a separate subprocess, and restart
therefore proves restoration through a fresh client. Evidence is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, BinaryIO
import uuid

MIB = 1 << 20
READY_MARK = "CTE Core container created and initialized"


def arguments() -> argparse.Namespace:
    """Parse artifact paths and the internal subprocess phase arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, default=Path.cwd())
    parser.add_argument("--bin-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--phase", choices=("initial", "restart"))
    parser.add_argument("--daemon-pid", type=int)
    parser.add_argument("--daemon-created", type=float)
    return parser.parse_args()


def available_port() -> int:
    """Find an unused local cluster of five ports without contacting a service."""
    for base in range(22000 + (os.getpid() % 1000) * 5, 45000, 5):
        sockets: list[socket.socket] = []
        try:
            for offset in range(5):
                candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(candidate)
                candidate.bind(("127.0.0.1", base + offset))
            return base
        except OSError:
            continue
        finally:
            for candidate in sockets:
                candidate.close()
    raise RuntimeError("No free private test port cluster")


def config_text(root: Path, port: int, extra: bool) -> str:
    """Construct the same stable storage-index layout for launch and restart."""

    def quoted(path: Path) -> str:
        """Quote a path as a YAML-compatible JSON string."""
        return json.dumps(path.as_posix())

    storage = ""
    for filename in ["storage.bin", "extra.bin"] if extra else ["storage.bin"]:
        storage += (
            f"      - path: {quoted(root / 'store' / filename)}\n"
            "        bdev_type: file\n"
            "        capacity_limit: 8MB\n"
            "        persistence_level: temporary\n"
            "        score: 1.0\n"
        )
    return (
        f"networking:\n  port: {port}\n"
        "runtime:\n  num_threads: 2\n  queue_depth: 256\n"
        "  main_segment_size: 256MB\n  metadata_segment_size: 32MB\n"
        f"  conf_dir: {quoted(root / 'conf')}\n"
        "compose:\n"
        "  - mod_name: clio_bdev\n    pool_name: ram::capacity_working_bdev\n"
        "    pool_query: local\n    pool_id: '301.0'\n"
        "    bdev_type: ram\n    capacity: 64MB\n"
        "  - mod_name: clio_cte_core\n    pool_name: cte_capacity_test\n"
        "    pool_query: local\n    pool_id: '512.0'\n    restart: true\n"
        f"    storage:\n{storage}"
        "    dpe:\n      dpe_type: max_bw\n"
        "    performance:\n      stat_targets_period_ms: 0\n"
        f"      metadata_log_path: {quoted(root / 'store' / 'metadata.log')}\n"
        "      transaction_log_capacity: 32MB\n"
    )


def write_config(root: Path, port: int, extra: bool) -> None:
    """Atomically persist target identity before any dynamic registration."""
    temporary = root / "cte.yaml.next"
    temporary.write_text(config_text(root, port, extra), encoding="utf-8")
    os.replace(temporary, root / "cte.yaml")


def private_environment(
    root: Path, port: int, module_dir: Path, bin_dir: Path
) -> dict[str, str]:
    """Create an isolated namespace before native libraries or package setup load."""
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(root / "home"),
            "USERPROFILE": str(root / "home"),
            "USER": f"clio-capacity-{root.name}",
            "CLIO_AGENT_HOME": str(root / "agent"),
            "CLIO_RUNTIME_STATE_DIR": str(root / "runtime-state"),
            "CLIO_SERVER_CONF": str(root / "cte.yaml"),
            "CLIO_PORT": str(port),
            "CLIO_BIND_ADDR": "127.0.0.1",
            "CLIO_REPO_PATH": str(bin_dir),
            "CLIO_WITH_RUNTIME": "0",
            "CLIO_WAIT_SERVER": "10",
            "CLIO_CLIENT_RETRY_TIMEOUT": "3",
            "CLIO_X": "TCP",
            "PYTHONPATH": os.pathsep.join([str(module_dir), env.get("PYTHONPATH", "")]),
            "PATH": os.pathsep.join([str(bin_dir), env.get("PATH", "")]),
        }
    )
    return env


def daemon_executable(bin_dir: Path) -> Path:
    """Require the exact executable alongside the selected installed artifact."""
    executable = bin_dir / ("clio_run.exe" if os.name == "nt" else "clio_run")
    if not executable.is_file():
        raise RuntimeError(f"Missing installed daemon executable: {executable}")
    return executable.resolve()


def owned_process(pid: int, created: float, executable: Path) -> Any:
    """Verify PID, creation time and executable before controlling a daemon."""
    psutil = importlib.import_module("psutil")
    process = psutil.Process(pid)
    if abs(process.create_time() - created) > 0.001:
        raise RuntimeError(f"Refusing reused daemon PID {pid}")
    if Path(process.exe()).resolve() != executable.resolve():
        raise RuntimeError(f"Refusing unexpected daemon executable for PID {pid}")
    if "start" not in process.cmdline():
        raise RuntimeError(f"Refusing non-start daemon process {pid}")
    return process


def start_daemon(
    root: Path, phase: str, env: dict[str, str], executable: Path
) -> tuple[subprocess.Popen[bytes], BinaryIO, float]:
    """Launch one owned foreground daemon and wait for its composed CTE pool."""
    log_path = root / f"daemon-{phase}.log"
    log = log_path.open("wb")
    process = subprocess.Popen(
        [str(executable), "start"], env=env, stdout=log, stderr=subprocess.STDOUT
    )
    created = importlib.import_module("psutil").Process(process.pid).create_time()
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(
                f"Private daemon exited {process.returncode}: {log_path}"
            )
        if READY_MARK in log_path.read_text(encoding="utf-8", errors="replace"):
            # The marker precedes the final compose reply and restart replay.
            time.sleep(0.5)
            return process, log, created
        time.sleep(0.1)
    try:
        owned_process(process.pid, created, executable).kill()
        process.wait(timeout=10)
    finally:
        log.close()
    raise TimeoutError(f"Private daemon did not compose: {log_path}")


def stop_daemon(
    root: Path,
    phase: str,
    env: dict[str, str],
    executable: Path,
    process: subprocess.Popen[bytes],
    created: float,
    log: BinaryIO,
) -> dict[str, Any]:
    """Gracefully stop the verified private PID, with a guarded cleanup fallback."""
    receipt: dict[str, Any] = {"pid": process.pid, "created": created, "forced": False}
    try:
        if process.poll() is None:
            owned = owned_process(process.pid, created, executable)
            owned.resume()  # also recover a failed suspended-daemon test
            with (root / f"stop-{phase}.log").open("wb") as stop_log:
                result = subprocess.run(
                    [str(executable), "stop", "--grace-period", "1000"],
                    env=env,
                    stdout=stop_log,
                    stderr=subprocess.STDOUT,
                    timeout=20,
                    check=False,
                )
            receipt["stop_returncode"] = result.returncode
            process.wait(timeout=15)
    finally:
        if process.poll() is None:
            owned_process(process.pid, created, executable).kill()
            process.wait(timeout=10)
            receipt["forced"] = True
        log.close()
    receipt["daemon_returncode"] = process.returncode
    receipt["reaped"] = process.poll() is not None
    return receipt


def load_binding(module_dir: Path, bin_dir: Path) -> Any:
    """Load only the selected installed extension after private environment setup."""
    sys.path.insert(0, str(module_dir))
    dll_handles: list[Any] = []
    if os.name == "nt":
        for directory in {module_dir.resolve(), bin_dir.resolve()}:
            dll_handles.append(os.add_dll_directory(str(directory)))
    cte = importlib.import_module("clio_cte_core_ext")
    if Path(cte.__file__).resolve().parent != module_dir.resolve():
        raise RuntimeError(f"Wrong binding imported: {cte.__file__}")
    cte._capacity_test_dll_handles = dll_handles
    return cte


def expected_payload(record: dict[str, Any]) -> bytes:
    """Recreate binary payloads for exact readback without serializing megabytes."""
    payload = bytes([record["seed"]]) * record["size"]
    prefix_size = record.get("prefix_size", 0)
    if prefix_size:
        payload = bytes([record["prefix_seed"]]) * prefix_size + payload[prefix_size:]
    return payload


def readback(
    client: Any, tag_id: Any, records: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Verify native status, logical size and every byte for all committed blobs."""
    hashes: dict[str, str] = {}
    for name, record in records.items():
        size_future = client.AsyncGetBlobSize(tag_id, name)
        assert size_future.wait(max_sec=10.0) == 0, name
        assert size_future.result(max_sec=10.0) == record["size"], name
        future = client.AsyncGetBlob(tag_id, name, record["size"], 0)
        assert future.wait(max_sec=10.0) == 0, name
        actual = future.result(max_sec=10.0)
        assert actual == expected_payload(record), name
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def assert_readonly(value: Any, field: str) -> None:
    """Require an immutable native result snapshot instead of mutable task state."""
    try:
        setattr(value, field, 0)
    except AttributeError:
        return
    raise AssertionError(f"{type(value).__name__}.{field} is writable")


def diagnostic_checks(cte: Any, client: Any, root: Path, tag_id: Any) -> dict[str, Any]:
    """Check copied snapshots, native names and loud missing-target failures."""
    capacity = client.AsyncGetCapacity().result(max_sec=10.0)
    assert isinstance(capacity, cte.CapacityInfo)
    assert capacity.total_capacity == 8 * MIB
    assert capacity.remaining_capacity == 8 * MIB
    assert_readonly(capacity, "total_capacity")
    names = client.AsyncListTargets().result(max_sec=10.0)
    target_name = (root / "store" / "storage.bin").as_posix() + "_node0"
    assert names == [target_name], names
    info = client.AsyncGetTargetInfo(target_name).result(max_sec=10.0)
    assert isinstance(info, cte.TargetInfo)
    assert info.remaining_space == capacity.remaining_capacity
    assert_readonly(info, "remaining_space")
    missing = client.AsyncGetTargetInfo("capacity-test-target-does-not-exist")
    assert missing.wait(max_sec=10.0) == 1
    try:
        missing.result(max_sec=10.0)
    except RuntimeError as error:
        assert "return code 1" in str(error), str(error)
    else:
        raise AssertionError("Missing target became an empty success")
    missing_blob = client.AsyncGetBlobSize(tag_id, "not-created")
    assert missing_blob.wait(max_sec=10.0) == 1
    assert client.AsyncStatTargets().wait(max_sec=10.0) == 0
    sync_rc = client.RegisterTarget(
        target_name,
        cte.BdevType.kFile,
        8 * MIB,
        cte.PoolQuery.Local(),
        cte.PoolId(512, 1),
    )
    assert type(sync_rc) is int and sync_rc == 0, sync_rc
    legacy_rc = client.AsyncRegisterTarget(
        target_name, cte.BdevType.kFile, 8 * MIB
    ).wait(max_sec=10.0)
    assert type(legacy_rc) is int, (
        legacy_rc
    )  # accepted call; null-ID failure is allowed
    return {
        "initial_total": capacity.total_capacity,
        "initial_remaining": capacity.remaining_capacity,
        "target_names": names,
        "sync_register_rc": sync_rc,
        "legacy_async_register_rc": legacy_rc,
    }


def fill_and_overwrite(
    client: Any, tag_id: Any
) -> tuple[dict[str, dict[str, Any]], str, int, dict[str, Any]]:
    """Refuse a real write, then prove zero-growth overwrites still succeed."""
    records: dict[str, dict[str, Any]] = {}
    failed_name = ""
    for index in range(16):
        name = f"capacity_{index}"
        record = {"seed": index + 17, "size": MIB}
        rc = client.AsyncPutBlob(tag_id, name, expected_payload(record), 0).wait(
            max_sec=10.0
        )
        if rc == 12:
            failed_name = name
            failed_record = record
            break
        assert rc == 0, (name, rc)
        records[name] = record
    assert failed_name and records, "Expected a bounded real full-tier refusal"
    full = client.AsyncGetCapacity().result(max_sec=10.0)
    assert full.total_capacity == 8 * MIB
    assert full.remaining_capacity < MIB, full.remaining_capacity
    first = next(iter(records))
    records[first] = {"seed": 111, "size": MIB}
    assert (
        client.AsyncPutBlob(tag_id, first, expected_payload(records[first]), 0).wait(
            max_sec=10.0
        )
        == 0
    )
    assert (
        client.AsyncGetCapacity().result(max_sec=10.0).remaining_capacity
        == full.remaining_capacity
    )
    prefix = bytes([112]) * (MIB // 2)
    assert client.AsyncPutBlob(tag_id, first, prefix, 0).wait(max_sec=10.0) == 0
    records[first].update(prefix_size=len(prefix), prefix_seed=112)
    after = client.AsyncGetCapacity().result(max_sec=10.0)
    assert after.remaining_capacity == full.remaining_capacity
    return (
        records,
        failed_name,
        failed_record["seed"],
        {
            "refused_rc": 12,
            "committed_before_growth": len(records),
            "full_remaining": full.remaining_capacity,
            "overwrite_remaining": after.remaining_capacity,
            "short_write_is_prefix_overwrite_not_truncate": True,
        },
    )


def suspend_and_settle(process: Any, timeout: float = 5.0) -> None:
    """Suspend process and wait until every thread is stopped. SIGSTOP lands
    asynchronously: a worker polling shared memory on another core can still
    serve a request submitted right after suspend() returns."""
    process.suspend()
    task_dir = Path(f"/proc/{process.pid}/task")
    if not task_dir.is_dir():
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            states = [
                (task / "stat").read_text().rsplit(")", 1)[1].split()[0]
                for task in task_dir.iterdir()
            ]
        except OSError:  # a thread exited mid-scan
            continue
        if all(state in "tT" for state in states):
            return
        time.sleep(0.001)
    raise TimeoutError(f"Daemon {process.pid} did not stop within {timeout}s")


def bounded_gil_check(
    client: Any, args: argparse.Namespace, executable: Path
) -> dict[str, Any]:
    """Freeze only the owned daemon and verify deadlines plus Python heartbeats."""
    process = owned_process(args.daemon_pid, args.daemon_created, executable)
    suspend_and_settle(process)
    errors: list[BaseException] = []
    done = threading.Event()
    waiting = threading.Event()
    try:
        future = client.AsyncGetCapacity()
        assert future.wait(max_sec=0.1) is None

        def wait_for_result() -> None:
            """Record the expected timeout while the main thread keeps running."""
            try:
                waiting.set()
                future.result(max_sec=1.0)
                errors.append(AssertionError("Suspended daemon unexpectedly answered"))
            except TimeoutError:
                pass
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        beats = 0
        first_heartbeat_elapsed: float | None = None
        start = time.monotonic()
        worker = threading.Thread(target=wait_for_result, daemon=True)
        worker.start()
        assert waiting.wait(timeout=1.0), "Native-wait worker did not start"
        while not done.wait(0.01):
            beats += 1
            if first_heartbeat_elapsed is None:
                first_heartbeat_elapsed = time.monotonic() - start
            assert time.monotonic() - start < 3.0, "Bounded diagnostic wait hung"
        worker.join(timeout=1.0)
        elapsed = time.monotonic() - start
        assert not errors, errors
        # Scheduling granularity controls heartbeat frequency. Require Python
        # progress well before the pending native wait expires, not a rate.
        # Holding the GIL through its timeout produces no early heartbeat.
        assert first_heartbeat_elapsed is not None and first_heartbeat_elapsed < 0.75, (
            first_heartbeat_elapsed,
            beats,
            elapsed,
        )
        assert elapsed < 2.0, elapsed
    finally:
        owned_process(args.daemon_pid, args.daemon_created, executable).resume()
    resumed = future.result(max_sec=10.0)
    assert resumed.total_capacity == 8 * MIB
    return {
        "heartbeat_count": beats,
        "first_heartbeat_elapsed_seconds": first_heartbeat_elapsed,
        "native_result_timeout_seconds": 1.0,
        "timeout_elapsed_seconds": elapsed,
        "wait_returned_none": True,
        "result_raised_timeout": True,
        "resumed_query_completed": True,
    }


def assert_registration_idempotent(
    cte: Any,
    client: Any,
    tag_id: Any,
    records: dict[str, dict[str, Any]],
    root: Path,
    phase: str,
) -> dict[str, Any]:
    """Repeat an occupied target's exact registration without changing storage."""
    expected_names = sorted(
        (root / "store" / filename).as_posix() + "_node0"
        for filename in ("storage.bin", "extra.bin")
    )

    def snapshot() -> dict[str, Any]:
        """Refresh real target counters and read every committed byte."""
        assert client.AsyncStatTargets().wait(max_sec=10.0) == 0
        capacity = client.AsyncGetCapacity().result(max_sec=10.0)
        names = client.AsyncListTargets().result(max_sec=10.0)
        assert len(names) == 2 and sorted(names) == expected_names, names
        return {
            "total_capacity": capacity.total_capacity,
            "remaining_capacity": capacity.remaining_capacity,
            "target_names": sorted(names),
            "readback_hashes": readback(client, tag_id, records),
        }

    receipt_path = root / f"duplicate-registration-{phase}.json"
    receipt: dict[str, Any] = {
        "phase": phase,
        "pool_id": "513.1",
        "before": snapshot(),
        "attempts": [],
        "status": "running",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for _ in range(2):
        future = client.AsyncRegisterTarget(
            (root / "store" / "extra.bin").as_posix() + "_node0",
            cte.BdevType.kFile,
            8 * MIB,
            target_query=cte.PoolQuery.Local(),
            bdev_id=cte.PoolId(513, 1),
            pool_query=cte.PoolQuery.Local(),
            attach_existing=0,
            growth_unit=1 << 30,
        )
        attempt: dict[str, Any] = {"register_rc": future.wait(max_sec=10.0)}
        receipt["attempts"].append(attempt)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        assert attempt["register_rc"] == 0, attempt
        attempt["after"] = snapshot()
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        assert attempt["after"] == receipt["before"], attempt
    receipt["status"] = "passed"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def initial_phase(
    cte: Any, client: Any, tag_id: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Exercise diagnostics/fullness and append exactly one persisted target."""
    root = args.root
    result = diagnostic_checks(cte, client, root, tag_id)
    records, failed_name, failed_seed, fullness = fill_and_overwrite(client, tag_id)
    result.update(fullness)
    result["bounded_gil"] = bounded_gil_check(
        client, args, daemon_executable(args.bin_dir)
    )
    result["before_growth_hashes"] = readback(client, tag_id, records)
    write_config(root, int(os.environ["CLIO_PORT"]), extra=True)
    extra_name = (root / "store" / "extra.bin").as_posix() + "_node0"
    future = client.AsyncRegisterTarget(
        extra_name,
        cte.BdevType.kFile,
        8 * MIB,
        target_query=cte.PoolQuery.Local(),
        bdev_id=cte.PoolId(513, 1),
        pool_query=cte.PoolQuery.Local(),
        attach_existing=0,
        growth_unit=1 << 30,
    )
    assert future.wait(max_sec=10.0) == 0
    capacity = client.AsyncGetCapacity().result(max_sec=10.0)
    assert capacity.total_capacity == 16 * MIB
    assert capacity.remaining_capacity > result["full_remaining"]
    retry_record = {"seed": failed_seed, "size": MIB}
    assert (
        client.AsyncPutBlob(
            tag_id, failed_name, expected_payload(retry_record), 0
        ).wait(max_sec=10.0)
        == 0
    )
    records[failed_name] = retry_record
    result.update(
        {
            "registered_pool_id": "513.1",
            "total_after_growth": capacity.total_capacity,
            "retried_only": failed_name,
            "records": records,
            "readback_hashes": readback(client, tag_id, records),
        }
    )
    result["duplicate_registration"] = assert_registration_idempotent(
        cte, client, tag_id, records, root, "initial"
    )
    (root / "expected.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def restart_phase(cte: Any, client: Any, tag_id: Any, root: Path) -> dict[str, Any]:
    """Read every committed byte through a fresh client after daemon restart."""
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    # Startup registration initializes canonical free space before restored
    # allocations are reflected in target statistics. Refresh the real bdev
    # counters before using the capacity snapshot to diagnose fullness.
    refresh_rc = client.AsyncStatTargets().wait(max_sec=10.0)
    assert refresh_rc == 0, refresh_rc
    capacity = client.AsyncGetCapacity().result(max_sec=10.0)
    assert capacity.total_capacity == 16 * MIB
    minimum_live_bytes = sum(record["size"] for record in expected["records"].values())
    assert capacity.remaining_capacity < capacity.total_capacity
    assert (
        0 <= capacity.remaining_capacity <= capacity.total_capacity - minimum_live_bytes
    )
    names = client.AsyncListTargets().result(max_sec=10.0)
    assert sorted(names) == sorted(
        (root / "store" / filename).as_posix() + "_node0"
        for filename in ("storage.bin", "extra.bin")
    )
    hashes = readback(client, tag_id, expected["records"])
    assert hashes == expected["readback_hashes"]
    duplicate_registration = assert_registration_idempotent(
        cte, client, tag_id, expected["records"], root, "restart"
    )
    return {
        "total_capacity": capacity.total_capacity,
        "remaining_capacity": capacity.remaining_capacity,
        "minimum_live_bytes": minimum_live_bytes,
        "one_shot_refresh_rc": refresh_rc,
        "target_names": names,
        "readback_hashes": hashes,
        "all_hashes_preserved": True,
        "duplicate_registration": duplicate_registration,
    }


def client_phase(args: argparse.Namespace) -> int:
    """Run one client phase and preserve failures without masking native errors."""
    result: dict[str, Any] = {"phase": args.phase, "status": "failed"}
    code = 1
    try:
        cte = load_binding(args.module_dir, args.bin_dir)
        assert cte.clio_init(cte.RuntimeMode.kClient, False), "clio_init"
        assert cte.initialize_cte(str(args.root / "cte.yaml"), cte.PoolQuery.Local()), (
            "initialize_cte"
        )
        client = cte.get_cte_client()
        tag_future = client.AsyncGetOrCreateTag("capacity_binding_regression")
        assert tag_future.wait(max_sec=10.0) == 0
        tag_id = tag_future.result(max_sec=10.0)
        if args.phase == "initial":
            result.update(initial_phase(cte, client, tag_id, args))
        else:
            result.update(restart_phase(cte, client, tag_id, args.root))
        result.update(status="passed", binding_path=str(cte.__file__))
        code = 0
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        (args.root / f"result-{args.phase}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    return code


def run_suite(args: argparse.Namespace) -> int:
    """Own both private daemons and require successful shutdown after each phase."""
    importlib.import_module(
        "psutil"
    )  # required; do not silently skip GIL/ownership checks
    parent = args.output_dir or Path(tempfile.mkdtemp(prefix="clio-capacity-results-"))
    root = parent.resolve() / f"run-{uuid.uuid4().hex[:12]}"
    for directory in (root, root / "home", root / "store", root / "conf"):
        directory.mkdir(parents=True, exist_ok=False)
    module_dir = args.module_dir.resolve()
    bin_dir = args.bin_dir.resolve()
    executable = daemon_executable(bin_dir)
    port = available_port()
    env = private_environment(root, port, module_dir, bin_dir)
    write_config(root, port, extra=False)
    receipt: dict[str, Any] = {
        "root": str(root),
        "port": port,
        "status": "failed",
        "phases": [],
    }
    try:
        for phase in ("initial", "restart"):
            daemon, log, created = start_daemon(root, phase, env, executable)
            phase_receipt: dict[str, Any] = {"phase": phase}
            receipt["phases"].append(phase_receipt)
            try:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--phase",
                    phase,
                    "--root",
                    str(root),
                    "--module-dir",
                    str(module_dir),
                    "--bin-dir",
                    str(bin_dir),
                    "--daemon-pid",
                    str(daemon.pid),
                    "--daemon-created",
                    repr(created),
                ]
                with (root / f"client-{phase}.log").open("wb") as client_log:
                    result = subprocess.run(
                        command,
                        env=env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                        check=False,
                    )
                phase_receipt["client_returncode"] = result.returncode
                assert result.returncode == 0, (
                    f"Client {phase} failed; see {root / ('client-' + phase + '.log')}"
                )
            finally:
                phase_receipt["shutdown"] = stop_daemon(
                    root, phase, env, executable, daemon, created, log
                )
            assert not phase_receipt["shutdown"]["forced"], phase_receipt
            assert phase_receipt["shutdown"].get("stop_returncode") == 0, phase_receipt
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc())
    (root / "suite-result.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    parsed = arguments()
    if parsed.phase:
        # Healthy clients finish every future while their daemon is alive.
        # Exit normally so native finalizers also write real coverage counters.
        # A failed phase can retain an unfinished native task; keep its nonzero
        # forced exit and let the parent reap the verified private daemon.
        exit_code = client_phase(parsed)
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code == 0:
            raise SystemExit(exit_code)
        os._exit(exit_code)
    raise SystemExit(run_suite(parsed))
