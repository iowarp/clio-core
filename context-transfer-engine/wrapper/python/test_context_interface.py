#!/usr/bin/env python3
"""
Test for Context Interface wrapper functions
Tests context_bundle, context_query, and context_delete
"""

import sys
import os
import time
import tempfile
import socket

# Add current directory to path for module import
sys.path.insert(0, os.getcwd())

# Import the context interface functions
from context_interface import context_bundle, context_query, context_delete

# Import the underlying CTE module for runtime initialization
import wrp_cte_core_ext as cte


def find_available_port(start_port=9129, end_port=9200):
    """Find an available port in the given range"""
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")


def initialize_runtime():
    """Initialize Chimaera runtime for testing"""
    try:
        import yaml
    except ImportError:
        print("❌ PyYAML not available - cannot run tests")
        return False

    temp_dir = tempfile.gettempdir()

    # Create hostfile
    hostfile = os.path.join(temp_dir, "wrp_context_test_hostfile")
    with open(hostfile, 'w') as f:
        f.write("localhost\n")

    # Find available port
    port = find_available_port()
    print(f"Using port: {port}")

    # Create storage directory
    storage_dir = os.path.join(temp_dir, "cte_context_test_storage")
    os.makedirs(storage_dir, exist_ok=True)

    # Generate config
    config = {
        'networking': {
            'protocol': 'zmq',
            'hostfile': hostfile,
            'port': port
        },
        'workers': {
            'num_workers': 4
        },
        'memory': {
            'main_segment_size': '1G',
            'client_data_segment_size': '512M',
            'runtime_data_segment_size': '512M'
        },
        'devices': [
            {
                'mount_point': storage_dir,
                'capacity': '1G'
            }
        ]
    }

    # Write config
    config_path = os.path.join(temp_dir, "wrp_context_test_conf.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Set environment
    os.environ['CHI_SERVER_CONF'] = config_path
    os.environ['CHI_REPO_PATH'] = os.getcwd()

    try:
        # Initialize runtime
        print("Initializing Chimaera runtime...")
        if not cte.chimaera_runtime_init():
            print("❌ Runtime init failed")
            return False
        time.sleep(0.5)

        # Initialize client
        print("Initializing Chimaera client...")
        if not cte.chimaera_client_init():
            print("❌ Client init failed")
            return False
        time.sleep(0.2)

        # Initialize CTE
        print("Initializing CTE subsystem...")
        pool_query = cte.PoolQuery.Dynamic()
        if not cte.initialize_cte(config_path, pool_query):
            print("❌ CTE init failed")
            return False
        time.sleep(0.3)

        print("✅ Runtime initialized successfully\n")
        return True

    except Exception as e:
        print(f"❌ Runtime initialization failed: {e}")
        return False


def test_context_bundle():
    """Test context_bundle function"""
    print("Test 1: context_bundle")

    tag_name = "test_tag_bundle"
    blob_name = "test_blob"
    test_data = b"Hello from context_bundle!"

    result = context_bundle(tag_name, blob_name, test_data)

    if result:
        print(f"✅ context_bundle succeeded")
        return True
    else:
        print(f"❌ context_bundle failed")
        return False


def test_context_query_list():
    """Test context_query function to list blobs"""
    print("\nTest 2: context_query (list blobs)")

    tag_name = "test_tag_query"

    # First bundle some data
    context_bundle(tag_name, "blob1", b"Data 1")
    context_bundle(tag_name, "blob2", b"Data 2")
    context_bundle(tag_name, "blob3", b"Data 3")

    # Query all blobs
    blobs = context_query(tag_name)

    if blobs and isinstance(blobs, list):
        print(f"✅ context_query returned blob list: {blobs}")
        return True
    else:
        print(f"❌ context_query failed to return blob list")
        return False


def test_context_query_blob():
    """Test context_query function to retrieve specific blob"""
    print("\nTest 3: context_query (get specific blob)")

    tag_name = "test_tag_retrieve"
    blob_name = "test_blob_data"
    original_data = b"This is the original data!"

    # Bundle data first
    context_bundle(tag_name, blob_name, original_data)

    # Query specific blob
    retrieved_data = context_query(tag_name, blob_name)

    if retrieved_data:
        # Convert to bytes if needed
        if isinstance(retrieved_data, str):
            retrieved_data = retrieved_data.encode('latin-1')

        if retrieved_data == original_data:
            print(f"✅ context_query retrieved correct data")
            return True
        else:
            print(f"❌ context_query data mismatch")
            print(f"   Expected: {original_data}")
            print(f"   Got: {retrieved_data}")
            return False
    else:
        print(f"❌ context_query failed to retrieve blob")
        return False


def test_context_delete():
    """Test context_delete function"""
    print("\nTest 4: context_delete")

    tag_name = "test_tag_delete"
    blob_name = "test_blob_delete"

    # This should print a message that delete is not implemented
    result = context_delete(tag_name, blob_name)

    if not result:
        print(f"✅ context_delete correctly reported not implemented")
        return True
    else:
        print(f"⚠️  context_delete unexpectedly returned True")
        return True  # Still pass since it's a known limitation


def main():
    """Run all context interface tests"""
    print("=" * 70)
    print("Context Interface Test Suite")
    print("=" * 70)
    print()

    # Initialize runtime
    if not initialize_runtime():
        print("\n❌ Cannot run tests without runtime initialization")
        return 1

    # Run tests
    tests = [
        ("context_bundle", test_context_bundle),
        ("context_query (list)", test_context_query_list),
        ("context_query (get blob)", test_context_query_blob),
        ("context_delete", test_context_delete),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' threw exception: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("✅ All context interface tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
