#!/usr/bin/env python3
"""
Test for Context Interface API
Tests ContextBundle, ContextQuery, and ContextDestroy methods
"""

import sys
import os
import time
import tempfile
import socket

# Add current directory to path for module import
sys.path.insert(0, os.getcwd())

try:
    # Import the CEE API module (built by nanobind)
    import wrp_cee as cee
except ImportError as e:
    print(f"❌ Failed to import wrp_cee module: {e}")
    print("   Make sure WRP_CORE_ENABLE_PYTHON=ON and nanobind is installed")
    sys.exit(1)

# Import CTE module for Tag and runtime initialization
try:
    import wrp_cte_core_ext as cte
except ImportError as e:
    print(f"❌ Failed to import wrp_cte_core_ext module: {e}")
    print("   CTE Python bindings are required for runtime initialization")
    sys.exit(1)


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
    """Test ContextBundle method"""
    print("Test 1: ContextBundle")

    # Create test file
    temp_dir = tempfile.gettempdir()
    test_file = os.path.join(temp_dir, "test_bundle_data.bin")
    test_data = b"Hello from ContextBundle test!"

    with open(test_file, 'wb') as f:
        f.write(test_data)

    try:
        # Create ContextInterface instance
        ctx_interface = cee.ContextInterface()

        # Create AssimilationCtx
        ctx = cee.AssimilationCtx(
            src=f"file://{test_file}",
            dst="iowarp::test_bundle_tag",
            format="binary"
        )

        # Call ContextBundle
        result = ctx_interface.context_bundle([ctx])

        if result == 0:
            print(f"✅ ContextBundle succeeded")
            return True
        else:
            print(f"❌ ContextBundle failed with code: {result}")
            return False

    except Exception as e:
        print(f"❌ ContextBundle threw exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_context_query():
    """Test ContextQuery method"""
    print("\nTest 2: ContextQuery")

    try:
        # First, put some test data using CTE Tag API
        tag = cte.Tag("test_query_tag")
        tag.put_blob("blob1", b"Test data 1")
        tag.put_blob("blob2", b"Test data 2")
        tag.put_blob("blob3", b"Test data 3")
        time.sleep(0.2)  # Allow operations to complete

        # Create ContextInterface instance
        ctx_interface = cee.ContextInterface()

        # Query for blobs
        results = ctx_interface.context_query("test_query_tag", ".*", 0)

        if results and isinstance(results, list):
            print(f"✅ ContextQuery returned {len(results)} results: {results}")
            return True
        else:
            print(f"❌ ContextQuery failed to return results")
            return False

    except Exception as e:
        print(f"❌ ContextQuery threw exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_destroy():
    """Test ContextDestroy method"""
    print("\nTest 3: ContextDestroy")

    try:
        # Create a tag to destroy
        tag = cte.Tag("test_destroy_tag")
        tag.put_blob("temp_blob", b"Temporary data")
        time.sleep(0.2)

        # Create ContextInterface instance
        ctx_interface = cee.ContextInterface()

        # Destroy the tag
        result = ctx_interface.context_destroy(["test_destroy_tag"])

        if result == 0:
            print(f"✅ ContextDestroy succeeded")
            return True
        else:
            print(f"❌ ContextDestroy failed with code: {result}")
            return False

    except Exception as e:
        print(f"❌ ContextDestroy threw exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all context interface tests"""
    print("=" * 70)
    print("Context Interface Test Suite (CEE API)")
    print("=" * 70)
    print()

    # Initialize runtime
    if not initialize_runtime():
        print("\n❌ Cannot run tests without runtime initialization")
        return 1

    # Run tests
    tests = [
        ("ContextBundle", test_context_bundle),
        ("ContextQuery", test_context_query),
        ("ContextDestroy", test_context_destroy),
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
