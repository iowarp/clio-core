#!/usr/bin/env python3
"""
Test for Context Interface API
Tests module imports and basic API construction
"""

import sys
import os

# Add build directory to path for module import
sys.path.insert(0, os.path.join(os.getcwd(), 'bin'))

def test_module_imports():
    """Test that required modules can be imported"""
    print("Test 1: Module Imports")

    try:
        # Import the CEE API module (built by nanobind)
        import clio_cee as cee
        print("  ✅ clio_cee module imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import clio_cee module: {e}")
        print("     Make sure CLIO_CORE_ENABLE_PYTHON=ON and nanobind is installed")
        return False

    try:
        # Import CTE module for Tag and runtime initialization
        import clio_cte_core_ext as cte
        print("  ✅ clio_cte_core_ext module imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import clio_cte_core_ext module: {e}")
        print("     CTE Python bindings are required for runtime initialization")
        return False

    return True


def test_api_construction():
    """Test that API types can be constructed"""
    print("\nTest 2: API Type Construction")

    try:
        import clio_cee as cee

        # Test ContextInterface construction
        ctx_interface = cee.ContextInterface()
        print("  ✅ ContextInterface constructed successfully")

        # Test AssimilationCtx construction
        ctx = cee.AssimilationCtx(
            src="file::/tmp/test.bin",
            dst="iowarp::test_tag",
            format="binary"
        )
        print("  ✅ AssimilationCtx constructed successfully")

        return True
    except Exception as e:
        print(f"  ❌ API construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cte_types():
    """Test that CTE types are available"""
    print("\nTest 3: CTE Type Availability")

    try:
        import clio_cte_core_ext as cte

        # Test that key types/enums are accessible
        _ = cte.RuntimeMode.kClient
        print("  ✅ RuntimeMode enum accessible")

        _ = cte.PoolQuery.Dynamic()
        print("  ✅ PoolQuery accessible")

        _ = cte.BdevType.kFile
        print("  ✅ BdevType enum accessible")

        return True
    except Exception as e:
        print(f"  ❌ CTE type access failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all context interface tests"""
    print("=" * 70)
    print("Context Interface Test Suite (CEE API)")
    print("=" * 70)
    print()

    # Run tests
    tests = [
        ("Module Imports", test_module_imports),
        ("API Construction", test_api_construction),
        ("CTE Types", test_cte_types),
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
