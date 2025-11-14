#!/usr/bin/env python3
"""
Compilation-focused test for Tag Python bindings
Tests that Tag bindings compile and are accessible without full runtime initialization
"""

import sys
import os

# When running with python -I (isolated mode), we need to manually add the current directory
sys.path.insert(0, os.getcwd())

def test_tag_import():
    """Test that Tag class can be imported"""
    try:
        import wrp_cte_core_ext as cte
        tag_type = cte.Tag
        print(f"✅ Tag class accessible: {tag_type}")
        return True
    except Exception as e:
        print(f"❌ Tag import failed: {e}")
        return False

def test_tag_methods_exist():
    """Test that all Tag methods are accessible"""
    try:
        import wrp_cte_core_ext as cte
        
        methods = [
            'PutBlob',
            'GetBlob',
            'GetBlobScore',
            'GetBlobSize',
            'GetContainedBlobs',
            'GetTagId'
        ]
        
        for method in methods:
            if hasattr(cte.Tag, method):
                print(f"✅ Tag.{method} exists")
            else:
                print(f"❌ Tag.{method} not found")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Tag methods test failed: {e}")
        return False

def test_tag_constructors():
    """Test that Tag constructors are accessible"""
    try:
        import wrp_cte_core_ext as cte
        
        # Verify we can see the constructor
        if hasattr(cte.Tag, '__init__'):
            print("✅ Tag constructor accessible")
            # Note: nanobind doesn't expose signatures via inspect, but the method exists
            return True
        else:
            print("❌ Tag constructor not found")
            return False
    except Exception as e:
        print(f"❌ Tag constructor test failed: {e}")
        return False

def main():
    """Run all Tag compilation tests"""
    print("🧪 Running Tag Python bindings compilation tests...")
    
    tests = [
        ("Tag Import", test_tag_import),
        ("Tag Methods", test_tag_methods_exist),
        ("Tag Constructors", test_tag_constructors),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Tag Python bindings compilation tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

