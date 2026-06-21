#!/usr/bin/env python3
"""
Regression test for context_retrieve double-free (issue #192).

Tests full data round-trip: bundle -> query -> retrieve -> verify -> destroy.
On main this test crashes with SIGABRT (double-free in ContextRetrieve).
With the fix (removing manual DelTask loop), it passes.

Self-contained: starts an in-process IOWarp runtime (clio_cte_core +
clio_cae_core modules are loaded from the build's bin directory) via
clio_cte_core_ext.clio_init(kClient, True), so it no longer needs an
externally running daemon.
"""

import os
import socket
import sys
import tempfile
import time

# Add build directory to path for module import
sys.path.insert(0, os.path.join(os.getcwd(), "bin"))


def _setup_environment_paths():
    """Point ChiMod discovery at the directory holding the python extensions."""
    import clio_cte_core_ext as cte

    bin_dir = os.path.dirname(os.path.abspath(cte.__file__))
    os.environ["CLIO_REPO_PATH"] = bin_dir
    # Linux uses LD_LIBRARY_PATH, macOS uses DYLD_LIBRARY_PATH.
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        existing = os.getenv(var, "")
        os.environ[var] = f"{bin_dir}:{existing}" if existing else bin_dir
    return bin_dir


def _generate_config():
    """Write a minimal single-node runtime config and export CLIO_SERVER_CONF."""
    import yaml

    tmp = tempfile.gettempdir()
    hostfile = os.path.join(tmp, "cee_roundtrip_hostfile")
    with open(hostfile, "w") as f:
        f.write("localhost\n")

    port = None
    for candidate in range(9229, 9300):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", candidate))
                port = candidate
                break
            except OSError:
                continue
    if port is None:
        raise RuntimeError("no available port for runtime in 9229-9300")

    storage = os.path.join(tmp, "cee_roundtrip_storage")
    os.makedirs(storage, exist_ok=True)

    config = {
        "networking": {"protocol": "zmq", "hostfile": hostfile, "port": port},
        "workers": {"num_workers": 4},
        "memory": {
            "main_segment_size": "1G",
            "client_data_segment_size": "512M",
            "runtime_data_segment_size": "512M",
        },
        "devices": [{"mount_point": storage, "capacity": "1G"}],
    }
    config_path = os.path.join(tmp, "cee_roundtrip_conf.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    os.environ["CLIO_SERVER_CONF"] = config_path
    return config_path


def start_runtime():
    """Start an in-process IOWarp runtime with a storage target. Returns True."""
    _setup_environment_paths()
    config_path = _generate_config()

    import clio_cte_core_ext as cte

    if not cte.clio_init(cte.RuntimeMode.kClient, True):
        return False
    time.sleep(0.5)  # let the runtime finish coming up
    if hasattr(cte, "initialize_cte"):
        cte.initialize_cte(config_path, cte.PoolQuery.Dynamic())

    # Register a storage target so context_bundle's PutBlob has somewhere to
    # land. The in-process runtime starts with no devices, so without this the
    # assimilator fails with "Failed to store description" (return_code 11).
    client = cte.get_cte_client()
    storage_dir = tempfile.mkdtemp(prefix="iowarp_store_")
    target_path = os.path.join(storage_dir, "cee_roundtrip_target")
    try:
        client.RegisterTarget(target_path, cte.BdevType.kFile,
                              1024 * 1024 * 1024, cte.PoolQuery.Local(),
                              cte.PoolId(700, 0))
    except TypeError:
        # RegisterTarget's binding returns a std::atomic nanobind can't convert
        # to a Python type; the registration side-effect still completes.
        pass
    return True


def test_retrieve_roundtrip():
    """Put data into IOWarp, retrieve it, and verify content matches."""
    import clio_cee as cee

    ctx_interface = cee.ContextInterface()
    tag_name = "test_retrieve_roundtrip"

    tmpdir = tempfile.mkdtemp(prefix="iowarp_test_")
    test_file = os.path.join(tmpdir, "test_data.bin")

    # Create test file with known pattern
    pattern = b"IOWarp-Roundtrip-Test-" + b"0123456789ABCDEF"
    file_size = 64 * 1024  # 64 KB
    original_data = (pattern * (file_size // len(pattern) + 1))[:file_size]
    with open(test_file, "wb") as f:
        f.write(original_data)

    try:
        # 1. Bundle (put)
        ctx = cee.AssimilationCtx(
            src=f"file::{test_file}",
            dst=f"iowarp::{tag_name}",
            format="binary",
        )
        bundle_result = ctx_interface.context_bundle([ctx])
        assert bundle_result == 0, f"context_bundle failed with code {bundle_result}"

        # 2. Query (verify blobs exist)
        blobs = ctx_interface.context_query(tag_name, ".*", 0)
        assert len(blobs) > 0, "context_query returned no blobs after bundle"

        # 3. Retrieve (get data back) - this crashes on main with double-free
        packed_data = ctx_interface.context_retrieve(
            tag_name, ".*",
            max_results=1024,
            max_context_size=4 * 1024 * 1024,
            batch_size=32,
        )
        total_size = sum(len(d) for d in packed_data)
        assert total_size > 0, "context_retrieve returned no data"

        # 4. Verify content
        retrieved_bytes = b"".join(
            d.encode("latin-1") if isinstance(d, str) else d
            for d in packed_data
        )
        assert original_data in retrieved_bytes, (
            f"Retrieved data ({len(retrieved_bytes)} bytes) does not contain "
            f"original data ({len(original_data)} bytes)"
        )

        # 5. Destroy (cleanup)
        destroy_result = ctx_interface.context_destroy([tag_name])
        assert destroy_result == 0, f"context_destroy failed with code {destroy_result}"

        return True

    finally:
        os.unlink(test_file)
        os.rmdir(tmpdir)


def main():
    print("=" * 60)
    print(" Regression test: context_retrieve round-trip (#192)")
    print("=" * 60)

    try:
        if not start_runtime():
            print("\nFAIL: could not start in-process IOWarp runtime")
            return 1
        success = test_retrieve_roundtrip()
        if success:
            print("\nPASS: Data round-trip verified successfully")
            return 0
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:
        print(f"\nFAIL: {e}")
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
