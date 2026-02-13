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
 * Simple Test Framework for Chimaera
 * A lightweight testing framework that doesn't depend on external libraries
 *
 * This is the unified simple_test.h header for the IOWarp Core repository.
 * Use this instead of context-runtime/test/simple_test.h.
 */

#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <exception>
#include <sstream>

namespace SimpleTest {

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

    for (const auto& test : tests) {
        // Skip tests that don't match filter
        if (!matches_filter(test.first, filter)) {
            continue;
        }

        g_current_test_name = test.first;
        g_current_section_name = "";
        g_stats.total_tests++;

        std::cout << "\n[TEST] " << test.first << std::endl;

        try {
            test.second();
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

// Helper macro to generate unique function names
#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)
#define UNIQUE_NAME(base) CONCAT(base, __LINE__)

// Main TEST_CASE macro that works with string names
#define TEST_CASE(test_name, tags) \
    void UNIQUE_NAME(test_func_)(); \
    static SimpleTest::TestRegistrar UNIQUE_NAME(test_reg_)(std::string(test_name) + " " + std::string(tags), UNIQUE_NAME(test_func_)); \
    void UNIQUE_NAME(test_func_)()

// Main function for test executable
#define SIMPLE_TEST_MAIN() \
int main(int argc, char* argv[]) { \
    std::string filter = ""; \
    if (argc > 1) { \
        filter = argv[1]; \
    } \
    int rc = SimpleTest::run_all_tests(filter); \
    _exit(rc); \
}
