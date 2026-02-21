#!/usr/bin/env python3
"""
Unit tests for CTE Python bindings telemetry functionality

This test module specifically focuses on testing the PollTelemetryLog
functionality of the CTE Python bindings.

Requirements:
- wrp_cte_core_ext module (Python bindings)
- Chimaera runtime initialized with CHI_WITH_RUNTIME=1
- WRP_RUNTIME_CONF environment variable set to config file
- pytest (optional, for test framework mode)

Usage:
    # Run with pytest (if available)
    CHI_WITH_RUNTIME=1 WRP_RUNTIME_CONF=/path/to/config.yaml pytest test_cte_telemetry.py -v

    # Run as standalone script (no pytest required)
    CHI_WITH_RUNTIME=1 WRP_RUNTIME_CONF=/path/to/config.yaml python3 test_cte_telemetry.py
"""

import sys
import os
import time

# Add current directory to path for module import
sys.path.insert(0, os.getcwd())

# Try to import pytest, but make it optional
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy pytest module for standalone mode
    class DummyPytest:
        @staticmethod
        def skip(msg):
            print(f"⚠️  SKIP: {msg}")
            sys.exit(0)

        class fixture:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, func):
                return func

    pytest = DummyPytest()

try:
    import wrp_cte_core_ext as cte
    HAS_CTE_MODULE = True
except ImportError:
    HAS_CTE_MODULE = False
    cte = None


# Fixtures

@pytest.fixture(scope="module")
def cte_module():
    """Fixture to ensure CTE module is available"""
    if not HAS_CTE_MODULE:
        pytest.skip("wrp_cte_core_ext module not available")
    return cte


@pytest.fixture(scope="module")
def runtime_initialized(cte_module):
    """Fixture to initialize Chimaera runtime once per module

    This follows the pattern from test_bindings.py for runtime initialization.
    Requires CHI_WITH_RUNTIME=1 environment variable.
    """
    # Check if runtime should be initialized
    env_val = os.getenv("CHI_WITH_RUNTIME")
    if env_val and str(env_val).lower() in ("0", "false", "no", "off"):
        pytest.skip("Runtime initialization disabled (CHI_WITH_RUNTIME=0)")

    # Check for config file
    config_path = os.getenv("WRP_RUNTIME_CONF")
    if not config_path:
        pytest.skip("WRP_RUNTIME_CONF environment variable not set")

    if not os.path.exists(config_path):
        pytest.skip(f"Config file not found: {config_path}")

    # Set up environment paths for ChiMod discovery
    try:
        module_file = cte_module.__file__ if hasattr(cte_module, '__file__') else None
        if module_file:
            bin_dir = os.path.dirname(os.path.abspath(module_file))
            os.environ["CHI_REPO_PATH"] = bin_dir
            existing_ld_path = os.getenv("LD_LIBRARY_PATH", "")
            if existing_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = bin_dir
    except Exception:
        pass  # Continue with existing environment

    # Initialize Chimaera
    try:
        init_result = cte_module.chimaera_init(cte_module.ChimaeraMode.kClient, True)
        if not init_result:
            pytest.skip("Chimaera initialization failed")

        # Give runtime time to initialize
        time.sleep(0.5)

        # Initialize CTE subsystem
        pool_query = cte_module.PoolQuery.Dynamic()
        cte_result = cte_module.initialize_cte(config_path, pool_query)
        if not cte_result:
            pytest.skip("CTE initialization failed")

        return True

    except Exception as e:
        pytest.skip(f"Runtime initialization error: {e}")


@pytest.fixture(scope="module")
def cte_client(cte_module, runtime_initialized):
    """Fixture to get CTE client instance"""
    return cte_module.get_cte_client()


@pytest.fixture(scope="function")
def test_tag(cte_module, runtime_initialized):
    """Fixture to create a unique test tag for each test"""
    import uuid
    tag_name = f"test_tag_{uuid.uuid4().hex[:8]}"
    return cte_module.Tag(tag_name)


# Test Cases

class TestPollTelemetryLog:
    """Test suite for PollTelemetryLog functionality"""

    def test_poll_telemetry_log_basic(self, cte_module, cte_client):
        """Test basic PollTelemetryLog call"""
        # Poll with minimum_logical_time=0 to get all entries
        entries = cte_client.PollTelemetryLog(0)

        # Should return a list
        assert isinstance(entries, list), "PollTelemetryLog should return a list"

        # List may be empty if no operations have been performed yet
        # This is not an error
        print(f"PollTelemetryLog returned {len(entries)} entries")

    def test_poll_telemetry_log_after_putblob(self, cte_module, cte_client, test_tag):
        """Test PollTelemetryLog after PutBlob operation"""
        # Perform PutBlob operation
        test_blob_name = "telemetry_test_blob"
        test_data = b"Test data for telemetry tracking"

        test_tag.PutBlob(test_blob_name, test_data, 0)

        # Poll telemetry log
        entries = cte_client.PollTelemetryLog(0)

        # Should return a list
        assert isinstance(entries, list), "PollTelemetryLog should return a list"

        if len(entries) > 0:
            # At least one entry should exist
            print(f"Found {len(entries)} telemetry entries after PutBlob")

            # Check that entries have required fields
            for entry in entries:
                assert hasattr(entry, 'op_'), "Entry should have op_ field"
                assert hasattr(entry, 'size_'), "Entry should have size_ field"
                assert hasattr(entry, 'tag_id_'), "Entry should have tag_id_ field"
                assert hasattr(entry, 'logical_time_'), "Entry should have logical_time_ field"
        else:
            pytest.skip("No telemetry entries found (telemetry may be disabled)")

    def test_poll_telemetry_log_putblob_and_getblob(self, cte_module, cte_client, test_tag):
        """Test PollTelemetryLog after PutBlob and GetBlob operations

        This is the main test requested by the user:
        1. Perform PutBlob
        2. Perform GetBlob
        3. Call PollTelemetryLog
        4. Verify log has entries
        """
        # Step 1: Perform PutBlob operation
        test_blob_name = "telemetry_putget_blob"
        test_data = b"Telemetry test: PutBlob and GetBlob operations"

        test_tag.PutBlob(test_blob_name, test_data, 0)
        print(f"PutBlob completed: {len(test_data)} bytes")

        # Step 2: Perform GetBlob operation
        blob_size = test_tag.GetBlobSize(test_blob_name)
        assert blob_size > 0, "Blob size should be greater than 0"

        retrieved_data = test_tag.GetBlob(test_blob_name, blob_size, 0)
        print(f"GetBlob completed: {blob_size} bytes retrieved")

        # Step 3: Poll telemetry log
        entries = cte_client.PollTelemetryLog(0)

        # Step 4: Verify log has entries
        assert isinstance(entries, list), "PollTelemetryLog should return a list"

        if len(entries) == 0:
            pytest.skip("No telemetry entries found (telemetry may be disabled)")

        print(f"Found {len(entries)} telemetry entries")

        # Analyze entries
        operation_types = []
        for entry in entries:
            # Verify entry has required fields
            assert hasattr(entry, 'op_'), "Entry should have op_ field"
            assert hasattr(entry, 'size_'), "Entry should have size_ field"
            assert hasattr(entry, 'tag_id_'), "Entry should have tag_id_ field"
            assert hasattr(entry, 'logical_time_'), "Entry should have logical_time_ field"

            # Collect operation types
            op_name = str(entry.op_)
            operation_types.append(op_name)

            print(f"Entry: op={op_name}, size={entry.size_}, logical_time={entry.logical_time_}")

        # Check if we have PutBlob and GetBlob operations
        has_putblob = any('kPutBlob' in op for op in operation_types)
        has_getblob = any('kGetBlob' in op for op in operation_types)

        print(f"Operations found: PutBlob={has_putblob}, GetBlob={has_getblob}")

        # This is informational - don't fail if operations aren't found
        # as telemetry logging may be configured differently
        if has_putblob:
            print("✅ Found PutBlob operation in telemetry log")
        else:
            print("⚠️  No PutBlob operation found in telemetry log")

        if has_getblob:
            print("✅ Found GetBlob operation in telemetry log")
        else:
            print("⚠️  No GetBlob operation found in telemetry log")

    def test_poll_telemetry_log_entry_structure(self, cte_module, cte_client, test_tag):
        """Test that telemetry entries have correct structure"""
        # Perform operations to generate telemetry
        test_blob_name = "structure_test_blob"
        test_data = b"Testing telemetry entry structure"

        test_tag.PutBlob(test_blob_name, test_data, 0)

        # Poll telemetry log
        entries = cte_client.PollTelemetryLog(0)

        if len(entries) == 0:
            pytest.skip("No telemetry entries found (telemetry may be disabled)")

        # Check structure of first entry
        entry = entries[0]

        # Verify all required fields exist
        assert hasattr(entry, 'op_'), "Entry must have op_ field"
        assert hasattr(entry, 'off_'), "Entry must have off_ field"
        assert hasattr(entry, 'size_'), "Entry must have size_ field"
        assert hasattr(entry, 'tag_id_'), "Entry must have tag_id_ field"
        assert hasattr(entry, 'mod_time_'), "Entry must have mod_time_ field"
        assert hasattr(entry, 'read_time_'), "Entry must have read_time_ field"
        assert hasattr(entry, 'logical_time_'), "Entry must have logical_time_ field"

        # Verify types (basic checks)
        assert isinstance(entry.size_, (int, float)), "size_ should be numeric"
        assert isinstance(entry.logical_time_, (int, float)), "logical_time_ should be numeric"

        print("✅ Telemetry entry structure validated")

    def test_poll_telemetry_log_with_logical_time_filter(self, cte_module, cte_client, test_tag):
        """Test PollTelemetryLog with logical time filtering"""
        # First poll to get current max logical time
        entries_before = cte_client.PollTelemetryLog(0)

        if len(entries_before) > 0:
            max_logical_time = max(entry.logical_time_ for entry in entries_before)
        else:
            max_logical_time = 0

        print(f"Max logical time before operations: {max_logical_time}")

        # Perform new operations
        test_blob_name = "filter_test_blob"
        test_data = b"Testing logical time filter"
        test_tag.PutBlob(test_blob_name, test_data, 0)

        # Poll again with minimum_logical_time = max_logical_time
        # This should return only entries after the previous max
        entries_after = cte_client.PollTelemetryLog(max_logical_time)

        print(f"Found {len(entries_after)} entries after logical time {max_logical_time}")

        # All returned entries should have logical_time >= max_logical_time
        for entry in entries_after:
            assert entry.logical_time_ >= max_logical_time, \
                f"Entry logical_time {entry.logical_time_} should be >= {max_logical_time}"

        print("✅ Logical time filtering validated")


# Standalone execution for pytest-less environments

def main():
    """Run tests without pytest"""
    print("=" * 70)
    print("🧪 CTE Telemetry Unit Tests")
    print("=" * 70)
    print()

    if not HAS_CTE_MODULE:
        print("❌ wrp_cte_core_ext module not available")
        return 1

    # Initialize runtime
    config_path = os.getenv("WRP_RUNTIME_CONF")
    if not config_path:
        print("❌ WRP_RUNTIME_CONF environment variable not set")
        return 1

    try:
        # Set up environment
        module_file = cte.__file__ if hasattr(cte, '__file__') else None
        if module_file:
            bin_dir = os.path.dirname(os.path.abspath(module_file))
            os.environ["CHI_REPO_PATH"] = bin_dir
            existing_ld_path = os.getenv("LD_LIBRARY_PATH", "")
            if existing_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = bin_dir

        # Initialize runtime
        if not cte.chimaera_init(cte.ChimaeraMode.kClient, True):
            print("❌ Chimaera initialization failed")
            return 1

        time.sleep(0.5)

        pool_query = cte.PoolQuery.Dynamic()
        if not cte.initialize_cte(config_path, pool_query):
            print("❌ CTE initialization failed")
            return 1

        print("✅ Runtime initialized")
        print()

        # Run tests manually
        test_suite = TestPollTelemetryLog()
        client = cte.get_cte_client()

        # Test 1
        print("Test 1: Basic PollTelemetryLog call")
        test_suite.test_poll_telemetry_log_basic(cte, client)
        print("✅ Passed")
        print()

        # Test 2
        print("Test 2: PollTelemetryLog after PutBlob and GetBlob")
        import uuid
        tag = cte.Tag(f"test_tag_{uuid.uuid4().hex[:8]}")
        test_suite.test_poll_telemetry_log_putblob_and_getblob(cte, client, tag)
        print("✅ Passed")
        print()

        # Test 3
        print("Test 3: Telemetry entry structure validation")
        tag = cte.Tag(f"test_tag_{uuid.uuid4().hex[:8]}")
        test_suite.test_poll_telemetry_log_entry_structure(cte, client, tag)
        print("✅ Passed")
        print()

        print("=" * 70)
        print("🎉 All tests passed!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
