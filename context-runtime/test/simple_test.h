/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Simple Test Framework for CLIO Runtime
 * A lightweight testing framework that doesn't depend on external libraries
 */

#pragma once

#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <exception>
#include <sstream>

#ifdef CTP_ALLOC_TRACK_SIZE
#include <chrono>
#include <thread>
#include "clio_ctp/memory/allocator/malloc_allocator.h"
#endif

namespace SimpleTest {

#ifdef CTP_ALLOC_TRACK_SIZE
// Automatic per-test leak checking. Only compiled in leak-tracking builds
// (-DCLIO_CORE_ENABLE_LEAK_CHECK=ON, which defines CTP_ALLOC_TRACK_SIZE), so
// normal builds are completely unaffected. Each test must return the runtime
// private heap (CTP_MALLOC — where AllocateBuffer/NewObj draw from) to where it
// started; an unfreed allocation (e.g. the #560/#563 FutureShm leaks) fails the
// test. See run_all_tests() for the first-test/[noleak] exemptions.
namespace detail {
// Slack for incidental per-test allocations and measurement jitter. The
// dedicated amplifying leak test (many RPCs in one case) catches sub-threshold
// per-operation leaks that this coarse per-test gate would miss.
inline constexpr size_t kLeakToleranceBytes = 4096;

inline size_t RuntimeHeapBytes() {
  return CTP_MALLOC->GetCurrentlyAllocatedSize();
}

// Poll until the runtime private-heap usage stops changing (so async server-side
// frees — which can lag the client's Wait() — have settled) or a short bound
// elapses. Returns early once stable, so the common case costs ~20 ms.
inline size_t StabilizeRuntimeHeap() {
  size_t prev = RuntimeHeapBytes();
  for (int i = 0; i < 50; ++i) {  // up to ~0.5 s
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    size_t cur = RuntimeHeapBytes();
    if (cur == prev) return cur;
    prev = cur;
  }
  return prev;
}
}  // namespace detail
#endif  // CTP_ALLOC_TRACK_SIZE

// Test statistics
struct TestStats {
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    
    void print_summary() const {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << failed_tests << std::endl;
        std::cout << "Success rate: " << (total_tests > 0 ? (passed_tests * 100.0 / total_tests) : 0) << "%" << std::endl;
    }
};

// Global test statistics
static TestStats g_stats;

// Optional finalize callback invoked after all tests run but before main
// returns.  Set this from test fixtures that allocate framework resources
// (e.g., CLIO Runtime runtime) to ensure clean shutdown before static
// destructors fire.  Type: void(*)().
using FinalizeFunc = void(*)();
static FinalizeFunc g_test_finalize = nullptr;

// Test failure exception
class TestFailure : public std::exception {
public:
    explicit TestFailure(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

// Current test context
static std::string g_current_test_name;
static std::string g_current_section_name;

// Test function type
using TestFunc = std::function<void()>;

// Test registry
class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry registry;
        return registry;
    }
    
    void add_test(const std::string& name, TestFunc func) {
        tests_.emplace_back(name, func);
    }
    
    const std::vector<std::pair<std::string, TestFunc>>& get_tests() const {
        return tests_;
    }
    
private:
    std::vector<std::pair<std::string, TestFunc>> tests_;
};

// Helper class for automatic test registration
class TestRegistrar {
public:
    TestRegistrar(const std::string& name, TestFunc func) {
        TestRegistry::instance().add_test(name, func);
    }
};

// Simplified section handling
#define SECTION(name) \
    SimpleTest::g_current_section_name = name; \
    std::cout << "  [SECTION] " << name << std::endl;

// Assertion macros
#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << "REQUIRE failed at " << __FILE__ << ":" << __LINE__ \
                << " in test '" << SimpleTest::g_current_test_name << "'"; \
            if (!SimpleTest::g_current_section_name.empty()) { \
                oss << ", section '" << SimpleTest::g_current_section_name << "'"; \
            } \
            oss << "\n  Condition: " << #condition; \
            throw SimpleTest::TestFailure(oss.str()); \
        } \
    } while(0)

#define REQUIRE_FALSE(condition) REQUIRE(!(condition))

#define REQUIRE_NOTHROW(expression) \
    do { \
        try { \
            expression; \
        } catch (...) { \
            std::ostringstream oss; \
            oss << "REQUIRE_NOTHROW failed at " << __FILE__ << ":" << __LINE__ \
                << " in test '" << SimpleTest::g_current_test_name << "'"; \
            if (!SimpleTest::g_current_section_name.empty()) { \
                oss << ", section '" << SimpleTest::g_current_section_name << "'"; \
            } \
            oss << "\n  Expression threw an exception: " << #expression; \
            throw SimpleTest::TestFailure(oss.str()); \
        } \
    } while(0)

#define FAIL(message) \
    do { \
        std::ostringstream oss; \
        oss << "FAIL at " << __FILE__ << ":" << __LINE__ \
            << " in test '" << SimpleTest::g_current_test_name << "'"; \
        if (!SimpleTest::g_current_section_name.empty()) { \
            oss << ", section '" << SimpleTest::g_current_section_name << "'"; \
        } \
        oss << "\n  Message: " << message; \
        throw SimpleTest::TestFailure(oss.str()); \
    } while(0)

#define INFO(message) \
    std::cout << "  [INFO] " << message << std::endl

// Check if a test name matches a filter (supports [tag] matching)
inline bool matches_filter(const std::string& name, const std::string& filter) {
    if (filter.empty()) return true;
    // If filter contains '[', extract all [tag] tokens and check each
    if (filter.find('[') != std::string::npos) {
        size_t pos = 0;
        while (pos < filter.size()) {
            size_t start = filter.find('[', pos);
            if (start == std::string::npos) break;
            size_t end = filter.find(']', start);
            if (end == std::string::npos) break;
            std::string tag = filter.substr(start, end - start + 1);
            if (name.find(tag) == std::string::npos) return false;
            pos = end + 1;
        }
        return true;
    }
    return name.find(filter) != std::string::npos;
}

// Main test runner with optional filter
inline int run_all_tests(const std::string& filter = "") {
    const auto& tests = TestRegistry::instance().get_tests();

    // Count tests that match filter
    int matching_tests = 0;
    for (const auto& test : tests) {
        if (matches_filter(test.first, filter)) {
            matching_tests++;
        }
    }

    if (!filter.empty()) {
        std::cout << "Running " << matching_tests << " test(s) matching filter '" << filter << "'..." << std::endl;
    } else {
        std::cout << "Running " << tests.size() << " test(s)..." << std::endl;
    }

#ifdef CTP_ALLOC_TRACK_SIZE
    int executed_count = 0;
#endif
    for (const auto& test : tests) {
        // Skip tests that don't match filter
        if (!matches_filter(test.first, filter)) {
            continue;
        }

        g_current_test_name = test.first;
        g_current_section_name = "";
        g_stats.total_tests++;

        std::cout << "\n[TEST] " << test.first << std::endl;

#ifdef CTP_ALLOC_TRACK_SIZE
        // Leak-check every test except:
        //  - the first executed test, during which lazy one-time runtime init
        //    (CHIMAERA_INIT in a fixture) allocates process-lifetime state that
        //    is legitimate, not a leak; and
        //  - tests tagged [noleak], which intentionally retain runtime-heap
        //    memory (analogous to the LSan suppressions list).
        const bool leak_check =
            executed_count > 0 &&
            test.first.find("[noleak]") == std::string::npos;
        const size_t heap_before =
            leak_check ? detail::StabilizeRuntimeHeap() : 0;
        ++executed_count;
#endif

        try {
            test.second();
#ifdef CTP_ALLOC_TRACK_SIZE
            if (leak_check) {
                const size_t heap_after = detail::StabilizeRuntimeHeap();
                if (heap_after > heap_before + detail::kLeakToleranceBytes) {
                    std::ostringstream oss;
                    oss << "Runtime-heap leak in test '" << test.first << "': "
                        << (heap_after - heap_before)
                        << " bytes still allocated after the test (before="
                        << heap_before << ", after=" << heap_after
                        << "). If this memory is intentionally retained for the "
                           "process lifetime, tag the test [noleak].";
                    throw TestFailure(oss.str());
                }
            }
#endif
            g_stats.passed_tests++;
            std::cout << "  [PASS] " << test.first << std::endl;
        } catch (const TestFailure& e) {
            g_stats.failed_tests++;
            std::cout << "  [FAIL] " << test.first << std::endl;
            std::cout << "    " << e.what() << std::endl;
        } catch (const std::exception& e) {
            g_stats.failed_tests++;
            std::cout << "  [ERROR] " << test.first << std::endl;
            std::cout << "    Unexpected exception: " << e.what() << std::endl;
        } catch (...) {
            g_stats.failed_tests++;
            std::cout << "  [ERROR] " << test.first << std::endl;
            std::cout << "    Unknown exception caught" << std::endl;
        }
    }

    g_stats.print_summary();
    return g_stats.failed_tests > 0 ? 1 : 0;
}

} // namespace SimpleTest

// Helper macro to generate unique function names.
// Renamed from UNIQUE_NAME to SIMPLE_TEST_UNIQUE_NAME because the Windows
// SDK's <nb30.h> (NetBIOS API, transitively pulled in by <windows.h>)
// defines UNIQUE_NAME as a two-arg macro for NetBIOS-name structures;
// any inclusion of windows.h before simple_test.h then poisons every
// TEST_CASE expansion.
#define SIMPLE_TEST_CONCAT_IMPL(x, y) x##y
#define SIMPLE_TEST_CONCAT(x, y) SIMPLE_TEST_CONCAT_IMPL(x, y)
#define SIMPLE_TEST_UNIQUE_NAME(base) SIMPLE_TEST_CONCAT(base, __LINE__)

// Main TEST_CASE macro that works with string names
#define TEST_CASE(test_name, tags) \
    void SIMPLE_TEST_UNIQUE_NAME(test_func_)(); \
    static SimpleTest::TestRegistrar SIMPLE_TEST_UNIQUE_NAME(test_reg_)(std::string(test_name) + " " + std::string(tags), SIMPLE_TEST_UNIQUE_NAME(test_func_)); \
    void SIMPLE_TEST_UNIQUE_NAME(test_func_)()

// On Windows the libzmq 4.3.x signaler asserts inside zmq_close() at
// static-destructor / atexit time with "WSASTARTUP not yet performed" — its
// mailbox send() runs after some part of the Winsock state has gone away.
// All tests have already passed by the time we get there, so on Windows we
// TerminateProcess() to skip every cleanup path; POSIX uses _exit() to
// match the historical behaviour some test cases relied on for skipping
// worker-join hangs. Both go through ctp::SystemInfo so this header stays
// free of <windows.h> — pulling that in here leaks Yield/min/max/SendMessage
// macros into every test TU.
#include "clio_ctp/introspect/system_info.h"

// SIMPLE_TEST_PROCESS_EXIT routes through SystemInfo::TerminateProcessNow.
// Windows: TerminateProcess(code) — skips ZMQ destructor chain that
//          aborts at exit.
// POSIX:   no-op (returns) — callers fall through and the test's normal
//          finalize + return runs, keeping leak sanitizers / coverage
//          happy.
// Test code that needs to "exit immediately on Windows but stay on the
// normal teardown path on Linux" should call this then `return result;`.
#define SIMPLE_TEST_PROCESS_EXIT(code) (::ctp::SystemInfo::TerminateProcessNow((code)))

// Main function for test executable. The flow is identical on every
// platform; the platform difference is hidden inside SIMPLE_TEST_PROCESS_EXIT.
// On Windows the TerminateProcess call returns control to the OS before
// g_test_finalize / static dtors / atexit handlers run, sidestepping the
// libzmq signaler abort. On POSIX it's a no-op and the rest of main()
// executes normally.
#define SIMPLE_TEST_MAIN() \
int main(int argc, char* argv[]) { \
    std::string filter = ""; \
    if (argc > 1) { \
        filter = argv[1]; \
    } \
    int result = SimpleTest::run_all_tests(filter); \
    SIMPLE_TEST_PROCESS_EXIT(result); \
    if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize(); \
    return result; \
}
