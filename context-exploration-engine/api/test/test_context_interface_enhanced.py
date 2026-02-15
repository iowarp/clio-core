#!/usr/bin/env python3
"""
Enhanced test for Context Interface Python Bindings
Tests actual API functionality, not just imports

Coverage Areas:
- AssimilationCtx properties and serialization
- ContextInterface API methods (bundle, query, retrieve, destroy, splice)
- Python-specific features (__repr__, properties)
- Error handling and edge cases
"""

import sys
import os
import tempfile

# Start an embedded runtime so API calls don't timeout waiting for a server
os.environ['CHI_WITH_RUNTIME'] = '1'

# Add build directory to path for module import
sys.path.insert(0, os.path.join(os.getcwd(), 'bin'))


def test_module_imports():
    """Test that required modules can be imported"""
    print("Test 1: Module Imports")

    try:
        import wrp_cee as cee
        print("  ✅ wrp_cee module imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import wrp_cee module: {e}")
        return False

    try:
        import wrp_cte_core_ext as cte
        print("  ✅ wrp_cte_core_ext module imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import wrp_cte_core_ext module: {e}")
        return False

    return True


def test_assimilation_ctx_construction():
    """Test AssimilationCtx constructor variants"""
    print("\nTest 2: AssimilationCtx Construction")

    try:
        import wrp_cee as cee

        # Test default constructor
        ctx1 = cee.AssimilationCtx()
        print("  ✅ Default constructor works")

        # Test full constructor with all parameters
        ctx2 = cee.AssimilationCtx(
            src="file::/tmp/test.bin",
            dst="iowarp::test_tag",
            format="binary",
            depends_on="dep1",
            range_off=0,
            range_size=1024
        )
        print("  ✅ Full constructor with all parameters works")

        # Test partial constructor
        ctx3 = cee.AssimilationCtx(
            src="file::/data.h5",
            dst="iowarp::hdf5_tag",
            format="hdf5"
        )
        print("  ✅ Partial constructor works")

        return True
    except Exception as e:
        print(f"  ❌ AssimilationCtx construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_assimilation_ctx_properties():
    """Test AssimilationCtx property access"""
    print("\nTest 3: AssimilationCtx Properties")

    try:
        import wrp_cee as cee

        ctx = cee.AssimilationCtx(
            src="file::/tmp/source.bin",
            dst="iowarp::dest_tag",
            format="binary",
            depends_on="dep_tag",
            range_off=100,
            range_size=200
        )

        # Test property reading
        assert ctx.src == "file::/tmp/source.bin", "src property mismatch"
        assert ctx.dst == "iowarp::dest_tag", "dst property mismatch"
        assert ctx.format == "binary", "format property mismatch"
        assert ctx.depends_on == "dep_tag", "depends_on property mismatch"
        assert ctx.range_off == 100, "range_off property mismatch"
        assert ctx.range_size == 200, "range_size property mismatch"
        print("  ✅ All property reads successful")

        # Test property writing
        ctx.src = "file::/new_source.bin"
        ctx.dst = "iowarp::new_dest"
        ctx.format = "hdf5"
        ctx.depends_on = "new_dep"
        ctx.range_off = 500
        ctx.range_size = 1000

        assert ctx.src == "file::/new_source.bin", "src property write failed"
        assert ctx.dst == "iowarp::new_dest", "dst property write failed"
        assert ctx.format == "hdf5", "format property write failed"
        assert ctx.depends_on == "new_dep", "depends_on property write failed"
        assert ctx.range_off == 500, "range_off property write failed"
        assert ctx.range_size == 1000, "range_size property write failed"
        print("  ✅ All property writes successful")

        # Note: include_patterns and exclude_patterns are not exposed in Python bindings
        # These are C++ std::vector<std::string> fields that would need explicit binding
        print("  ℹ️  Pattern vectors (include_patterns, exclude_patterns) not exposed in bindings")

        return True
    except Exception as e:
        print(f"  ❌ Property access failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_assimilation_ctx_repr():
    """Test AssimilationCtx __repr__"""
    print("\nTest 4: AssimilationCtx __repr__")

    try:
        import wrp_cee as cee

        ctx = cee.AssimilationCtx(
            src="file::/test.bin",
            dst="iowarp::tag",
            format="binary"
        )

        repr_str = repr(ctx)
        assert "AssimilationCtx" in repr_str, "__repr__ doesn't contain class name"
        assert "file::/test.bin" in repr_str, "__repr__ doesn't contain src"
        assert "iowarp::tag" in repr_str, "__repr__ doesn't contain dst"
        assert "binary" in repr_str, "__repr__ doesn't contain format"

        print(f"  ✅ __repr__ output: {repr_str}")
        return True
    except Exception as e:
        print(f"  ❌ __repr__ test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_interface_construction():
    """Test ContextInterface construction"""
    print("\nTest 5: ContextInterface Construction")

    try:
        import wrp_cee as cee

        ctx_interface = cee.ContextInterface()
        print("  ✅ ContextInterface constructed successfully")

        # Should support lazy initialization (no runtime needed yet)
        return True
    except Exception as e:
        print(f"  ❌ ContextInterface construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_bundle_empty():
    """Test ContextInterface.context_bundle with empty list"""
    print("\nTest 6: ContextInterface.context_bundle (empty)")

    try:
        import wrp_cee as cee

        ctx_interface = cee.ContextInterface()

        # Call with empty bundle (should succeed but warn)
        result = ctx_interface.context_bundle([])
        print(f"  ✅ context_bundle with empty list returned: {result}")

        return True
    except Exception as e:
        print(f"  ❌ context_bundle empty test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_bundle_with_data():
    """Test ContextInterface.context_bundle with real data"""
    print("\nTest 7: ContextInterface.context_bundle (with data)")

    try:
        import wrp_cee as cee

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
            temp_file = f.name
            f.write(b"Test data for Python bindings" * 100)

        try:
            ctx_interface = cee.ContextInterface()

            # Create bundle
            ctx = cee.AssimilationCtx(
                src=f"file::{temp_file}",
                dst="iowarp::python_test_bundle",
                format="binary"
            )
            bundle = [ctx]

            # Call context_bundle
            result = ctx_interface.context_bundle(bundle)
            print(f"  ✅ context_bundle returned: {result}")

            # Allow time for async processing
            import time
            time.sleep(1)

            return True
        finally:
            # Cleanup
            os.unlink(temp_file)

    except Exception as e:
        print(f"  ❌ context_bundle with data failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_query():
    """Test ContextInterface.context_query"""
    print("\nTest 8: ContextInterface.context_query")

    try:
        import wrp_cee as cee

        ctx_interface = cee.ContextInterface()

        # Query with wildcard (may return empty if no data bundled)
        results = ctx_interface.context_query(".*", ".*", 100)
        print(f"  ✅ context_query returned {len(results)} results")

        # Query with specific pattern (likely returns empty)
        results2 = ctx_interface.context_query("python_test_.*", ".*", 10)
        print(f"  ✅ context_query with specific pattern returned {len(results2)} results")

        return True
    except Exception as e:
        print(f"  ❌ context_query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_retrieve():
    """Test ContextInterface.context_retrieve"""
    print("\nTest 9: ContextInterface.context_retrieve")

    try:
        import wrp_cee as cee

        ctx_interface = cee.ContextInterface()

        # Retrieve with default parameters
        results = ctx_interface.context_retrieve(".*", ".*")
        print(f"  ✅ context_retrieve (defaults) returned {len(results)} contexts")

        # Retrieve with custom parameters
        results2 = ctx_interface.context_retrieve(
            "python_.*",       # tag_re
            ".*",              # blob_re
            10,                # max_results
            512 * 1024,        # max_context_size (512KB)
            16                 # batch_size
        )
        print(f"  ✅ context_retrieve (custom params) returned {len(results2)} contexts")

        return True
    except Exception as e:
        print(f"  ❌ context_retrieve failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_destroy():
    """Test ContextInterface.context_destroy"""
    print("\nTest 10: ContextInterface.context_destroy")

    try:
        import wrp_cee as cee

        ctx_interface = cee.ContextInterface()

        # Destroy with empty list (should succeed)
        result = ctx_interface.context_destroy([])
        print(f"  ✅ context_destroy with empty list returned: {result}")

        # Destroy with non-existent contexts (should handle gracefully)
        result2 = ctx_interface.context_destroy(["nonexistent_context_1", "nonexistent_context_2"])
        print(f"  ✅ context_destroy with non-existent contexts returned: {result2}")

        return True
    except Exception as e:
        print(f"  ❌ context_destroy failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_splice():
    """Test ContextInterface.context_splice (not implemented)"""
    print("\nTest 11: ContextInterface.context_splice (stub)")

    try:
        import wrp_cee as cee

        ctx_interface = cee.ContextInterface()

        # Should return error code 1 (not implemented)
        result = ctx_interface.context_splice("new_ctx", "tag_.*", "blob_.*")
        assert result == 1, "context_splice should return 1 (not implemented)"
        print(f"  ✅ context_splice correctly returns not-implemented (code={result})")

        return True
    except Exception as e:
        print(f"  ❌ context_splice failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_python_workflow():
    """Test full Python workflow: bundle -> query -> retrieve -> destroy"""
    print("\nTest 12: Full Python Workflow")

    try:
        import wrp_cee as cee
        import time

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
            temp_file = f.name
            f.write(b"Python workflow test data" * 50)

        try:
            ctx_interface = cee.ContextInterface()

            # Step 1: Bundle
            ctx = cee.AssimilationCtx(
                src=f"file::{temp_file}",
                dst="iowarp::python_workflow_test",
                format="binary"
            )
            bundle_result = ctx_interface.context_bundle([ctx])
            print(f"  Step 1 (Bundle): returned {bundle_result}")
            time.sleep(1)  # Allow async processing

            # Step 2: Query
            query_results = ctx_interface.context_query("python_workflow_test", ".*", 100)
            print(f"  Step 2 (Query): found {len(query_results)} blobs")

            # Step 3: Retrieve
            retrieve_results = ctx_interface.context_retrieve(
                "python_workflow_test", ".*", 100, 1024*1024, 32)
            print(f"  Step 3 (Retrieve): got {len(retrieve_results)} contexts")

            # Step 4: Destroy
            destroy_result = ctx_interface.context_destroy(["python_workflow_test"])
            print(f"  Step 4 (Destroy): returned {destroy_result}")

            print("  ✅ Full Python workflow completed")
            return True

        finally:
            os.unlink(temp_file)

    except Exception as e:
        print(f"  ❌ Python workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all enhanced Python binding tests"""
    print("=" * 70)
    print("Enhanced Context Interface Python Bindings Test Suite")
    print("=" * 70)
    print()

    # Run tests
    tests = [
        ("Module Imports", test_module_imports),
        ("AssimilationCtx Construction", test_assimilation_ctx_construction),
        ("AssimilationCtx Properties", test_assimilation_ctx_properties),
        ("AssimilationCtx __repr__", test_assimilation_ctx_repr),
        ("ContextInterface Construction", test_context_interface_construction),
        ("context_bundle (empty)", test_context_bundle_empty),
        ("context_bundle (with data)", test_context_bundle_with_data),
        ("context_query", test_context_query),
        ("context_retrieve", test_context_retrieve),
        ("context_destroy", test_context_destroy),
        ("context_splice (stub)", test_context_splice),
        ("Full Python Workflow", test_python_workflow),
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
        print("✅ All Python binding tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    rc = main()
    # Use os._exit to skip atexit handlers. ZMQ I/O threads allocate
    # memory via Python 3.13's mimalloc; after the threads are joined,
    # mimalloc's mi_process_done atexit handler crashes trying to
    # collect their abandoned memory segments.
    os._exit(rc)
