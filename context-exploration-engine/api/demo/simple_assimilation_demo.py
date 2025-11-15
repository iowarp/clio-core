#!/usr/bin/env python3
"""
Simple Context Exploration Engine Demo

This demo shows how to use the CEE API to:
1. Create a test file
2. Assimilate it into IOWarp using context_bundle
3. Query for blobs using context_query
4. Retrieve blob data using context_retrieve
5. Clean up using context_destroy

The ContextInterface constructor handles all runtime initialization internally.
You only need to call the CEE APIs.

Usage:
    python3 simple_assimilation_demo.py
"""

import sys
import os
import tempfile

# Import the CEE API module
try:
    import wrp_cee as cee
except ImportError as e:
    print(f"ERROR: Failed to import wrp_cee module: {e}")
    print("Make sure WRP_CORE_ENABLE_PYTHON=ON and the module is built/installed")
    sys.exit(1)


def create_test_file(file_path, size_mb=1):
    """Create a test file with sample data"""
    print(f"Creating {size_mb}MB test file: {file_path}")

    pattern = b"IOWarp Context Exploration Engine Demo Data - "
    pattern += b"This is sample data for assimilation testing. " * 10
    pattern += b"\n"

    bytes_to_write = size_mb * 1024 * 1024
    bytes_written = 0

    with open(file_path, 'wb') as f:
        while bytes_written < bytes_to_write:
            chunk_size = min(len(pattern), bytes_to_write - bytes_written)
            f.write(pattern[:chunk_size])
            bytes_written += chunk_size

    actual_size = os.path.getsize(file_path)
    print(f"  Created: {actual_size:,} bytes ({actual_size / 1024 / 1024:.2f} MB)")
    return actual_size


def demo_context_bundle(ctx_interface, test_file, tag_name):
    """Demonstrate context_bundle API"""
    print("\n" + "=" * 70)
    print("1. Context Bundle - Assimilate File")
    print("=" * 70)

    # Create AssimilationCtx
    ctx = cee.AssimilationCtx(
        src=f"file::{test_file}",
        dst=f"iowarp::{tag_name}",
        format="binary"
    )

    print(f"AssimilationCtx:")
    print(f"  Source:      {ctx.src}")
    print(f"  Destination: {ctx.dst}")
    print(f"  Format:      {ctx.format}")

    # Call context_bundle
    print(f"\nCalling context_bundle()...")
    result = ctx_interface.context_bundle([ctx])

    if result == 0:
        print(f"✓ Success! File assimilated to tag '{tag_name}'")
    else:
        print(f"✗ Failed with return code: {result}")

    return result


def demo_context_query(ctx_interface, tag_name):
    """Demonstrate context_query API"""
    print("\n" + "=" * 70)
    print("2. Context Query - List Blobs")
    print("=" * 70)

    print(f"Querying for blobs in tag '{tag_name}'...")
    results = ctx_interface.context_query(tag_name, ".*", 0)

    if results:
        print(f"✓ Found {len(results)} blob(s):")
        for i, blob_name in enumerate(results, 1):
            print(f"  {i}. {blob_name}")
    else:
        print("No blobs found")

    return results


def demo_context_retrieve(ctx_interface, tag_name):
    """Demonstrate context_retrieve API"""
    print("\n" + "=" * 70)
    print("3. Context Retrieve - Get Blob Data")
    print("=" * 70)

    print(f"Retrieving blob data from tag '{tag_name}'...")
    print("  Parameters:")
    print(f"    - tag_re:      '{tag_name}'")
    print(f"    - blob_re:     '.*' (all blobs)")
    print(f"    - max_results: 0 (use default limit of 1024)")

    packed_data = ctx_interface.context_retrieve(tag_name, ".*", 0)

    if packed_data:
        print(f"✓ Retrieved {len(packed_data)} packed context(s)")
        total_bytes = sum(len(data) for data in packed_data)
        print(f"  Total data: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")

        # Show a preview of the data
        if total_bytes > 0:
            preview = packed_data[0][:100].decode('utf-8', errors='replace')
            print(f"  Preview: {preview}...")
    else:
        print("No data retrieved")

    return packed_data


def demo_context_destroy(ctx_interface, tag_name):
    """Demonstrate context_destroy API"""
    print("\n" + "=" * 70)
    print("4. Context Destroy - Clean Up")
    print("=" * 70)

    print(f"Destroying tag '{tag_name}'...")
    result = ctx_interface.context_destroy([tag_name])

    if result == 0:
        print(f"✓ Success! Tag '{tag_name}' destroyed")
    else:
        print(f"✗ Failed with return code: {result}")

    return result


def main():
    """Run the CEE API demonstration"""
    print()
    print("#" * 70)
    print("# IOWarp Context Exploration Engine - API Demo")
    print("#" * 70)
    print()

    # Setup
    temp_dir = tempfile.gettempdir()
    test_file = os.path.join(temp_dir, "cee_demo_data.bin")
    tag_name = "demo_tag"

    try:
        # Create test file
        print("=" * 70)
        print("Setup")
        print("=" * 70)
        create_test_file(test_file, size_mb=1)

        # Create ContextInterface (handles all initialization internally)
        print("\nCreating ContextInterface...")
        print("  (Runtime initialization handled internally)")
        ctx_interface = cee.ContextInterface()
        print("✓ ContextInterface created")

        # Demo 1: context_bundle
        result = demo_context_bundle(ctx_interface, test_file, tag_name)
        if result != 0:
            print("\n✗ Assimilation failed, stopping demo")
            return 1

        # Demo 2: context_query
        blobs = demo_context_query(ctx_interface, tag_name)

        # Demo 3: context_retrieve
        data = demo_context_retrieve(ctx_interface, tag_name)

        # Demo 4: context_destroy
        demo_context_destroy(ctx_interface, tag_name)

        # Summary
        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        print("\nSuccessfully demonstrated:")
        print("  ✓ context_bundle()   - Assimilated 1MB file")
        print(f"  ✓ context_query()    - Found {len(blobs)} blob(s)")
        print(f"  ✓ context_retrieve() - Retrieved {len(data)} packed context(s)")
        print("  ✓ context_destroy()  - Cleaned up tag")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ Demo failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up test file: {test_file}")


if __name__ == "__main__":
    sys.exit(main())
