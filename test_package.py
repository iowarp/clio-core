#!/usr/bin/env python3
"""
Simple test script to verify iowarp_core package structure.
This does NOT test the actual C++ builds - just the Python package structure.
"""

import sys


def test_import():
    """Test that the package can be imported."""
    try:
        import iowarp_core
        print("✓ Package imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import package: {e}")
        return False


def test_version():
    """Test that version is accessible."""
    try:
        import iowarp_core
        version = iowarp_core.get_version()
        print(f"✓ Version: {version}")
        return True
    except Exception as e:
        print(f"✗ Failed to get version: {e}")
        return False


def test_components():
    """Test that component information is accessible."""
    try:
        import iowarp_core
        components = iowarp_core.get_component_info()
        print(f"✓ Found {len(components)} components:")
        for name, info in components.items():
            print(f"  - {name}: {info['description']}")
        return True
    except Exception as e:
        print(f"✗ Failed to get component info: {e}")
        return False


def test_module_attributes():
    """Test that expected module attributes exist."""
    try:
        import iowarp_core
        required_attrs = ["__version__", "COMPONENTS", "get_component_info", "get_version"]
        missing = [attr for attr in required_attrs if not hasattr(iowarp_core, attr)]

        if missing:
            print(f"✗ Missing attributes: {', '.join(missing)}")
            return False

        print(f"✓ All required attributes present")
        return True
    except Exception as e:
        print(f"✗ Failed to check attributes: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing iowarp_core package structure")
    print("=" * 60)
    print()

    tests = [
        ("Import test", test_import),
        ("Version test", test_version),
        ("Components test", test_components),
        ("Module attributes test", test_module_attributes),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        results.append(test_func())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
