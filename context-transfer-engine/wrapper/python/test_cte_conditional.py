#!/usr/bin/env python3
"""Real multi-process tests for the existing CTE conditional-write bindings.

The controller owns one disposable RAM daemon. Every contender is a fresh
Python process connected to that daemon over TCP; filesystem gates coordinate
submission only, and never decide a winner. No runtime, binding, or result is
mocked. These tests establish single-authority atomicity and bounded uncertainty
after daemon death, not restart durability, replication, or failover.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable
import uuid

sys.path.insert(0, os.getcwd())  # The freshly built module, not an installed wheel.

PAYLOAD_SIZE = 256
PROCESS_TIMEOUT = 45.0
READY_MARK = "CTE Core container created and initialized"


def claim(nonce: str | None) -> bytes:
    """Encode a fixed-size opaque owner record for byte-exact CAS assertions."""
    return (
        json.dumps({"nonce": nonce}, sort_keys=True).encode().ljust(PAYLOAD_SIZE, b" ")
    )


def free_port() -> int:
    """Choose the daemon port together with its runtime side ports."""
    for base in range(12013, 12900, 7):
        sockets: list[socket.socket] = []
        try:
            for port in (base, base + 1, base + 3):
                connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(connection)
                connection.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
        finally:
            for connection in sockets:
                connection.close()
    raise RuntimeError("No available test port cluster")


def write_config(root: Path, port: int) -> Path:
    """Write the same RAM CTE composition used by the async binding tests."""
    config = root / "runtime.yaml"
    config.write_text(
        f"""networking:
  port: {port}
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "ram::conditional_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "64MB"
  - mod_name: clio_cte_core
    pool_name: clio_cte_core
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "ram::conditional_ram_tier"
        bdev_type: ram
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: max_bw
""",
        encoding="utf-8",
    )
    return config


def wait_until(predicate: Callable[[], bool], message: str, timeout: float) -> None:
    """Wait with a deadline, failing rather than silently skipping a check."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(message)


def connect() -> tuple[Any, Any, Any]:
    """Initialize a real client without starting another runtime."""
    import clio_cte_core_ext as cte

    assert cte.PUT_EXISTS_RC == 60
    assert cte.PUT_VERSION_MISMATCH_RC == 61
    assert cte.clio_init(cte.RuntimeMode.kClient, False), "client initialization failed"
    assert cte.initialize_cte(os.environ["CLIO_SERVER_CONF"], cte.PoolQuery.Dynamic())
    client = cte.get_cte_client()
    future = client.AsyncGetOrCreateTag(os.environ["CTE_CONDITIONAL_TAG"])
    assert future.wait(10.0) == 0, "tag creation failed"
    tag = future.result()
    assert not tag.IsNull(), "tag creation returned a null ID"
    return cte, client, tag


def result(future: Any, expected_code: int = 0) -> dict[str, Any]:
    """Check the Future code and the typed dictionary agree after completion."""
    code = future.wait(10.0)
    value = future.result(10.0)
    assert isinstance(value, dict), value
    assert type(value["code"]) is int and value["code"] == code == expected_code, value
    assert type(value["version"]) is int and value["version"] >= 0, value
    assert future.done(), "completed future is not done"
    return value


def read(client: Any, tag: Any, name: str) -> dict[str, Any]:
    """Read exact bytes and a nonzero native version, including a cold client."""
    value = result(client.AsyncGetBlobVersioned(tag, name, PAYLOAD_SIZE))
    assert isinstance(value["data"], bytes) and len(value["data"]) == PAYLOAD_SIZE, (
        value
    )
    assert value["version"] > 0, value
    return value


def expect_error(error: type[Exception], function: Callable[[], Any]) -> None:
    """Require the documented error instead of accepting a false success."""
    try:
        function()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def basic_contract(cte: Any, client: Any, tag: Any) -> dict[str, Any]:
    """Exercise validation, normal cache traffic, staging, offsets, and errors."""
    for operation in (
        lambda: client.AsyncPutBlobIfAbsent(tag, "empty", b""),
        lambda: client.AsyncPutBlobIfVersion(tag, "empty", b"", 0),
        lambda: client.AsyncGetBlobVersioned(tag, "empty", 0),
    ):
        expect_error(ValueError, operation)
    for operation in (
        lambda: client.AsyncPutBlobIfAbsent(tag, "negative", b"x", -1),
        lambda: client.AsyncPutBlobIfVersion(tag, "negative", b"x", -1),
        lambda: client.AsyncGetBlobVersioned(tag, "negative", -1),
        lambda: client.AsyncGetBlobVersioned(tag, "negative", 1, -1),
    ):
        expect_error(TypeError, operation)

    initial = result(
        client.AsyncPutBlobIfVersion(tag, "absent-version-zero", b"first", 0)
    )
    assert initial["version"] > 0
    conflict = result(
        client.AsyncPutBlobIfVersion(tag, "absent-version-zero", b"other", 0),
        cte.PUT_VERSION_MISMATCH_RC,
    )
    assert conflict["version"] == initial["version"]
    unchanged = result(client.AsyncGetBlobVersioned(tag, "absent-version-zero", 5))
    assert unchanged["data"] == b"first" and unchanged["version"] == initial["version"]

    payload = bytes(range(256)) * 32  # Deliberately invalid UTF-8.
    pending = client.AsyncPutBlobIfAbsent(tag, "binary", payload)
    del payload
    garbage = [bytes([index]) * 8192 for index in range(64)]
    created = result(pending)
    assert created["version"] > 0
    expected = bytes(range(256)) * 32
    assert client.AsyncGetBlob(tag, "binary", len(expected)).result(10.0) == expected
    for _ in range(2):  # Ordinary reads may populate the shared-memory cache.
        got = result(client.AsyncGetBlobVersioned(tag, "binary", len(expected)))
        assert got["data"] == expected and got["version"] == created["version"], got
    assert garbage[63] == b"?" * 8192  # Keep allocation churn alive until after reads.

    patched = result(
        client.AsyncPutBlobIfVersion(tag, "binary", b"\xff\x00", created["version"], 7)
    )
    assert patched["version"] > created["version"]
    part = result(client.AsyncGetBlobVersioned(tag, "binary", 2, 7))
    assert part["data"] == b"\xff\x00" and part["version"] == patched["version"]
    result(
        client.AsyncPutBlobIfVersion(tag, "binary", b"not-written", (1 << 64) - 1),
        cte.PUT_VERSION_MISMATCH_RC,
    )
    # A failed CAS version is diagnostic; reread the authoritative version.
    unchanged = result(client.AsyncGetBlobVersioned(tag, "binary", len(expected)))
    assert unchanged["version"] == patched["version"]
    assert unchanged["data"] == expected[:7] + b"\xff\x00" + expected[9:]
    missing = client.AsyncGetBlobVersioned(tag, "does-not-exist", 1)
    code = missing.wait(10.0)
    failure = missing.result(10.0)
    assert code == failure["code"] == 1 and failure["version"] == 0, failure
    assert failure["data"] is None, failure
    return {"code": 0, "version": patched["version"]}


def worker(job_path: Path) -> bool:
    """Execute one gated operation; return True only for daemon-death futures."""
    job = json.loads(job_path.read_text(encoding="utf-8"))
    cte, client, tag = connect()
    Path(job["ready"]).touch()
    wait_until(
        lambda: Path(job["gate"]).exists(), "worker gate timed out", PROCESS_TIMEOUT
    )
    operation = job["operation"]
    if operation == "basic":
        value = basic_contract(cte, client, tag)
    elif operation == "read":
        value = read(client, tag, job["name"])
    elif operation == "absent":
        future = client.AsyncPutBlobIfAbsent(tag, job["name"], claim(job["nonce"]))
        value = result(future, job.get("expected_code", 0))
    elif operation == "cas":
        future = client.AsyncPutBlobIfVersion(
            tag, job["name"], claim(job["nonce"]), job["version"]
        )
        value = result(future, job.get("expected_code", 0))
    elif operation == "contend":
        data = claim(job["nonce"])
        future = (
            client.AsyncPutBlobIfAbsent(tag, job["name"], data)
            if "version" not in job
            else client.AsyncPutBlobIfVersion(tag, job["name"], data, job["version"])
        )
        code = future.wait(10.0)
        assert code in (0, cte.PUT_EXISTS_RC, cte.PUT_VERSION_MISMATCH_RC), code
        value = result(future, code)
    elif operation == "dead-daemon":
        operations = (
            lambda: client.AsyncPutBlobIfAbsent(tag, "dead-absent", b"x"),
            lambda: client.AsyncPutBlobIfVersion(tag, "dead-cas", b"x", 0),
            lambda: client.AsyncGetBlobVersioned(tag, "dead-read", 1),
        )
        for submit in operations:
            try:
                pending = submit()
            except RuntimeError:
                continue  # Native submission refused: an explicit failure.
            assert pending.wait(0.2) is None, "dead daemon produced a completed request"
            expect_error(TimeoutError, lambda: pending.result(0.2))
        value = {"code": "unavailable", "version": None}
    else:
        raise AssertionError(f"Unknown operation {operation}")
    if isinstance(value.get("data"), bytes):
        value["data"] = value["data"].hex()
    Path(job["output"]).write_text(json.dumps(value), encoding="utf-8")
    return operation == "dead-daemon"


class Controller:
    """Own all child handles, gates, logs, and the one ephemeral daemon."""

    def __init__(self, root: Path, environment: dict[str, str]) -> None:
        """Retain the shared endpoint and unique per-run namespace."""
        self.root = root
        self.environment = environment
        self.children: list[subprocess.Popen[bytes]] = []
        self.logs: list[Any] = []
        self.sequence = 0

    def start(self, operation: str, **arguments: Any) -> dict[str, Any]:
        """Start a new initialized client and return its private test paths."""
        self.sequence += 1
        prefix = self.root / f"client-{self.sequence}"
        job = {"operation": operation, **arguments}
        for kind in ("ready", "gate", "output"):
            job.setdefault(kind, str(prefix.with_suffix(f".{kind}")))
        job_path = prefix.with_suffix(".json")
        job_path.write_text(json.dumps(job), encoding="utf-8")
        log = prefix.with_suffix(".log").open("wb")
        self.logs.append(log)
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--worker",
                str(job_path),
            ],
            env=self.environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self.children.append(process)
        job.update(process=process, log_path=prefix.with_suffix(".log"))
        return job

    def release(self, jobs: list[dict[str, Any]]) -> None:
        """Require every real client ready before releasing the contention gate."""

        def ready() -> bool:
            for job in jobs:
                if job["process"].poll() is not None:
                    raise AssertionError(job["log_path"].read_text(errors="replace"))
            return all(Path(job["ready"]).exists() for job in jobs)

        wait_until(ready, "clients did not initialize", PROCESS_TIMEOUT)
        for gate in {job["gate"] for job in jobs}:
            Path(gate).touch()

    def collect(self, job: dict[str, Any]) -> dict[str, Any]:
        """Require process success and a concrete result from its native Future."""
        code = job["process"].wait(timeout=PROCESS_TIMEOUT)
        assert code == 0, job["log_path"].read_text(errors="replace")
        return json.loads(Path(job["output"]).read_text(encoding="utf-8"))

    def run(self, operation: str, **arguments: Any) -> dict[str, Any]:
        """Run one sequential operation in a fresh client process."""
        job = self.start(operation, **arguments)
        self.release([job])
        return self.collect(job)

    def race(self, name: str, version: int | None = None) -> tuple[int, str]:
        """Require exactly one native winner and unchanged losing contenders."""
        arguments = {} if version is None else {"version": version}
        gate = str(self.root / f"race-{self.sequence}.gate")
        jobs = [
            self.start(
                "contend", name=name, nonce=uuid.uuid4().hex, gate=gate, **arguments
            )
            for _ in range(4)
        ]
        self.release(jobs)
        outcomes = [self.collect(job) for job in jobs]
        winners = [index for index, value in enumerate(outcomes) if value["code"] == 0]
        assert len(winners) == 1, outcomes
        winner = outcomes[winners[0]]
        conflict = 60 if version is None else 61
        assert all(value["code"] in (0, conflict) for value in outcomes), outcomes
        assert all(
            value["version"] == winner["version"] > (version or 0) for value in outcomes
        ), outcomes
        nonce = jobs[winners[0]]["nonce"]
        stored = self.run("read", name=name)
        assert bytes.fromhex(stored["data"]) == claim(nonce), stored
        assert stored["version"] == winner["version"], stored
        return winner["version"], nonce

    def close(self) -> None:
        """Reap only child processes started by this fixture, even on failure."""
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10.0)
        for log in self.logs:
            log.close()


def ownership_sequence(controller: Controller) -> None:
    """Prove stale releases cannot change a successor's opaque nonce record."""
    first, _ = controller.race("ownership")
    second, nonce = controller.race("ownership", first)
    stale = controller.run(
        "cas", name="ownership", nonce=None, version=first, expected_code=61
    )
    assert stale["version"] == second
    assert bytes.fromhex(controller.run("read", name="ownership")["data"]) == claim(
        nonce
    )
    released = controller.run("cas", name="ownership", nonce=None, version=second)
    assert released["version"] > second
    successor_nonce = uuid.uuid4().hex
    successor = controller.run(
        "cas", name="ownership", nonce=successor_nonce, version=released["version"]
    )
    assert successor["version"] > released["version"]
    stale = controller.run(
        "cas", name="ownership", nonce=None, version=second, expected_code=61
    )
    assert stale["version"] == successor["version"]
    stored = controller.run("read", name="ownership")
    assert stored["version"] == successor["version"]
    assert bytes.fromhex(stored["data"]) == claim(successor_nonce)
    # The acquiring client has exited. A new process must not steal the claim.
    occupied = controller.run(
        "absent", name="ownership", nonce="intruder", expected_code=60
    )
    assert occupied["version"] == successor["version"]


def main() -> None:
    """Run only against an owned disposable runtime; preserve diagnostics on fail."""
    with tempfile.TemporaryDirectory(prefix="clio-conditional-") as temporary:
        root = Path(temporary)
        port = free_port()
        config = write_config(root, port)
        environment = dict(
            os.environ,
            CLIO_SERVER_CONF=str(config),
            CLIO_PORT=str(port),
            CLIO_BIND_ADDR="127.0.0.1",
            CLIO_SERVER_ADDR="127.0.0.1",
            CLIO_X="TCP",
            CLIO_REPO_PATH=os.getcwd(),
            CLIO_RESTART_LOG=str(root / "restart.bin"),
            CLIO_WITH_RUNTIME="0",
            CLIO_WAIT_SERVER="5",
            CLIO_CLIENT_RETRY_TIMEOUT="3",
            CTE_CONDITIONAL_TAG=f"conditional-{uuid.uuid4().hex}",
        )
        executable = Path.cwd() / ("clio_run.exe" if os.name == "nt" else "clio_run")
        assert executable.is_file(), f"freshly built daemon not found: {executable}"
        daemon_log = root / "daemon.log"
        controller = Controller(root, environment)
        with daemon_log.open("wb") as log:
            daemon = subprocess.Popen(
                [str(executable), "start"],
                env=dict(environment, CLIO_WITH_RUNTIME="1"),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:

                def daemon_ready() -> bool:
                    assert daemon.poll() is None, daemon_log.read_text(errors="replace")
                    return READY_MARK in daemon_log.read_text(errors="replace")

                wait_until(daemon_ready, "daemon did not compose CTE", 60.0)
                controller.run("basic")
                ownership_sequence(controller)
                pending = controller.start("dead-daemon")
                wait_until(
                    lambda: Path(pending["ready"]).exists(),
                    "fault client did not initialize",
                    PROCESS_TIMEOUT,
                )
                daemon.kill()
                daemon.wait(timeout=10.0)
                controller.release([pending])
                assert controller.collect(pending)["code"] == "unavailable"
                print(
                    "PASS: real TCP creators/CAS, stale release, binary/versioned read, daemon-death uncertainty",
                    flush=True,
                )
            except BaseException:
                print(daemon_log.read_text(errors="replace"), file=sys.stderr)
                raise
            finally:
                try:
                    controller.close()
                finally:
                    if daemon.poll() is None:
                        daemon.kill()
                    daemon.wait(timeout=10.0)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--worker":
            force_exit = worker(Path(sys.argv[2]))
            # Healthy workers exit normally so native/gcov exit handlers run.
            # The daemon-death worker retains unanswered futures; teardown must
            # not turn its explicit uncertainty into a hang or false success.
            sys.stdout.flush()
            sys.stderr.flush()
            if force_exit:
                os._exit(0)
        else:
            main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
