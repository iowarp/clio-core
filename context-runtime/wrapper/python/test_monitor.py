import sys
import os
import threading
import time

sys.path.insert(0, os.getcwd())
os.environ.pop("CLIO_WITH_RUNTIME", None)

import clio_runtime_ext as chi


def test_init():
    """Test that clio_init(kClient) succeeds."""
    ok = chi.clio_init(0)  # 0 = kClient
    if not ok:
        print("SKIPPED: No runtime available (clio_init returned False)")
        sys.exit(0)
    print("PASSED: test_init")


def test_async_monitor():
    """Test async monitor submit + wait."""
    task = chi.async_monitor("local", "status")
    results = task.wait(5.0)
    assert isinstance(results, dict)
    print(f"  async: {len(results)} containers")
    print("PASSED: test_async_monitor")


def test_multiple_sequential():
    """Test 10 sequential monitor calls."""
    for i in range(10):
        results = chi.async_monitor("local", "status").wait(5.0)
        assert isinstance(results, dict)
    print("PASSED: test_multiple_sequential (10 calls)")


def test_threaded():
    """Test monitor from multiple Python threads."""
    num_threads = 4
    results = [None] * num_threads
    errors = [None] * num_threads

    def worker(idx):
        try:
            task = chi.async_monitor("local", "worker_stats")
            results[idx] = task.wait(5.0)
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for i in range(num_threads):
        assert errors[i] is None, f"Thread {i} raised: {errors[i]}"
        assert results[i] is not None, f"Thread {i} returned None"

    print(f"PASSED: test_threaded ({num_threads} threads)")


def test_finalize():
    """Test clean shutdown."""
    chi.clio_finalize()
    print("PASSED: test_finalize")


if __name__ == "__main__":
    test_init()
    test_async_monitor()
    test_multiple_sequential()
    test_threaded()
    test_finalize()
    print("\nAll tests PASSED")
