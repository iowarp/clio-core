# ChiMod Unit Testing Guide

This guide covers how to create unit tests for Chimaera modules (ChiMods). The Chimaera testing framework allows both runtime and client components to be tested in the same process, enabling comprehensive integration testing without complex multi-process coordination.

## Test Environment Setup

### Environment Variables

Unit tests require specific environment variables for module discovery and configuration:

```bash
# Set the path to compiled ChiMod libraries (build directory)
export CHI_REPO_PATH="/path/to/your/project/build/bin"

# Set library path for dynamic loading (both variables are scanned for modules)
export LD_LIBRARY_PATH="/path/to/your/project/build/bin:$LD_LIBRARY_PATH"

# Optional: Enable test mode for additional debugging
export CHIMAERA_TEST_MODE=1

# Optional: Specify custom configuration file
export CHI_SERVER_CONF="/path/to/your/project/config/chimaera_default.yaml"
```

**Module Discovery Process:**
- The Chimaera runtime scans both `CHI_REPO_PATH` and `LD_LIBRARY_PATH` for ChiMod libraries
- `CHI_REPO_PATH` should point to the directory containing compiled libraries (typically `build/bin`)
- ChiMod libraries are loaded dynamically at runtime based on module registration
- Configuration files are located relative to the runtime executable or via standard paths

### Configuration Files

Tests can use custom configuration files for runtime settings. Default location: `config/chimaera_default.yaml`

```yaml
# Example test configuration
workers:
  low_latency_threads: 2
  high_latency_threads: 1

memory:
  main_segment_size: 268435456      # 256MB for tests
  client_data_segment_size: 134217728 # 128MB for tests

shared_memory:
  main_segment_name: "chi_test_main_${USER}"
  client_data_segment_name: "chi_test_client_${USER}"
```

## Test Framework Integration

The project uses a custom simple test framework:

```cpp
#include "../simple_test.h"

// Test cases use TEST_CASE macro
TEST_CASE("test_name", "[category][tags]") {
  SECTION("test_section") {
    // Test implementation
    REQUIRE(condition);
    REQUIRE_FALSE(condition);
    REQUIRE_NOTHROW(function_call());
  }
}

// Main test runner
SIMPLE_TEST_MAIN()
```

## Test Fixture Pattern

Use test fixtures for setup/teardown and utility functions:

```cpp
class ChimaeraTestFixture {
public:
  ChimaeraTestFixture() = default;
  ~ChimaeraTestFixture() { cleanup(); }

  bool initialize() {
    if (g_initialized) return true;

    // Use unified initialization (client mode with embedded runtime)
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (success) {
      g_initialized = true;
      std::this_thread::sleep_for(500ms); // Allow initialization

      // Verify core managers
      REQUIRE(CHI_CHIMAERA_MANAGER != nullptr);
      REQUIRE(CHI_IPC != nullptr);
      REQUIRE(CHI_POOL_MANAGER != nullptr);
      REQUIRE(CHI_MODULE_MANAGER != nullptr);
    }
    return success;
  }

  // Utility method for async task completion
  template<typename TaskT>
  bool waitForTaskCompletion(chi::Future<TaskT>& task, chi::u32 timeout_ms = 5000) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::milliseconds(timeout_ms);

    // Wait for completion with timeout
    while (!task.IsComplete()) {
      auto current_time = std::chrono::steady_clock::now();
      if (current_time - start_time > timeout_duration) {
        return false; // Timeout
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }

private:
  void cleanup() {
    // Framework handles automatic cleanup
  }
  
  static bool g_initialized;
};
```

## Complete Test Example

Here's a comprehensive test that demonstrates the full ChiMod testing workflow:

```cpp
/**
 * Unit tests for YourModule ChiMod
 * Tests complete functionality: container creation, operations, error handling
 */

#include "../simple_test.h"
#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// Include headers
#include <chimaera/chimaera.h>
#include <chimaera/your_module/your_module_client.h>
#include <chimaera/admin/admin_client.h>

namespace {
  // Test constants
  constexpr chi::u32 kTestTimeoutMs = 10000;
  constexpr chi::PoolId kTestPoolId = chi::PoolId(500, 0);
  
  // Global state
  bool g_initialized = false;
}

// Test fixture class (implementation as shown above)
class YourModuleTestFixture {
  // ... (fixture implementation)
};

//==============================================================================
// INITIALIZATION TESTS  
//==============================================================================

TEST_CASE("Chimaera Initialization", "[initialization]") {
  YourModuleTestFixture fixture;

  SECTION("Unified initialization should succeed") {
    REQUIRE(fixture.initialize());
    REQUIRE(CHI_CHIMAERA_MANAGER->IsInitialized());
    REQUIRE(CHI_CHIMAERA_MANAGER->IsRuntime());
    REQUIRE(CHI_IPC->IsInitialized());
  }
}

//==============================================================================
// CHIMOD FUNCTIONALITY TESTS
//==============================================================================

TEST_CASE("ChiMod Complete Workflow", "[workflow]") {
  YourModuleTestFixture fixture;
  REQUIRE(fixture.initialize());

  SECTION("Create admin pool and ChiMod container") {
    // Step 1: Create admin pool
    chimaera::admin::Client admin_client(chi::kAdminPoolId);
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    admin_client.Create(pool_query, "admin", chi::kAdminPoolId);
    std::this_thread::sleep_for(100ms);

    // Step 2: Initialize ChiMod client and create pool
    chimaera::your_module::Client module_client(kTestPoolId);
    module_client.Create(pool_query, "test_module", kTestPoolId);
    std::this_thread::sleep_for(100ms);

    // Verify creation succeeded
    REQUIRE(module_client.GetReturnCode() == 0);
  }

  SECTION("Test synchronous operations") {
    chimaera::your_module::Client module_client(kTestPoolId);
    
    std::string input = "test_data";
    std::string output;
    chi::u32 result = module_client.ProcessData(pool_query, input, output);
    
    REQUIRE(result == 0);
    REQUIRE_FALSE(output.empty());
    INFO("Sync operation: " << input << " -> " << output);
  }

  SECTION("Test asynchronous operations") {
    chimaera::your_module::Client module_client(kTestPoolId);

    auto task = module_client.AsyncProcessData(pool_query, "async_test");

    // Wait for task completion
    task.Wait();
    REQUIRE(task->result_code_ == 0);

    std::string output = task->output_data_.str();
    REQUIRE_FALSE(output.empty());
    INFO("Async operation result: " << output);
  }

  SECTION("Error handling and edge cases") {
    // Test invalid pool ID
    constexpr chi::PoolId kInvalidPoolId = chi::PoolId(9999, 0);
    chimaera::your_module::Client invalid_client(kInvalidPoolId);

    // Should not crash, but may fail
    REQUIRE_NOTHROW(invalid_client.Create(pool_query, "invalid_pool", kInvalidPoolId));

    // Test task timeout
    chimaera::your_module::Client module_client(kTestPoolId);
    auto task = module_client.AsyncProcessData(pool_query, "timeout_test");

    // Test with very short timeout
    bool completed = fixture.waitForTaskCompletion(task, 50); // 50ms timeout
    INFO("Task completed within short timeout: " << completed);
  }
}

// Test runner
SIMPLE_TEST_MAIN()
```

## CMake Integration

Add unit tests to your ChiMod's CMakeLists.txt:

```cmake
# Create unit test executable
add_executable(chimaera_your_module_tests
  test/unit/test_your_module.cc
)

# Link against ChiMod libraries and test framework
target_link_libraries(chimaera_your_module_tests
  chimaera_your_module_runtime
  chimaera_your_module_client
  chimaera_admin_runtime
  chimaera_admin_client
  chimaera
  hshm::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

# Set runtime definition for proper initialization
target_compile_definitions(chimaera_your_module_tests PRIVATE 
  CHIMAERA_RUNTIME=1
)

# Install test executable
install(TARGETS chimaera_your_module_tests
  DESTINATION bin
  COMPONENT tests
)
```

## Running Tests

### Environment Setup and Execution

```bash
# Set required environment variables
export CHI_REPO_PATH="${PWD}/build/bin"
export LD_LIBRARY_PATH="${PWD}/build/bin:${LD_LIBRARY_PATH}"
export CHI_SERVER_CONF="${PWD}/config/chimaera_default.yaml"
export CHIMAERA_TEST_MODE=1

# Build and run tests
cmake --preset debug
cmake --build build
./build/bin/chimaera_your_module_tests

# Run specific test categories
./build/bin/chimaera_your_module_tests "[initialization]"
./build/bin/chimaera_your_module_tests "[workflow]"
```

## Best Practices

1. **Initialize Once**: Use static flags to avoid redundant runtime/client initialization
2. **Use Fixtures**: Encapsulate common setup/teardown in test fixture classes  
3. **Test Both Modes**: Test runtime and client components in the same process when possible
4. **Handle Timeouts**: Always use timeouts for async operations to prevent test hangs
5. **Clean Up Resources**: Use RAII patterns and explicit cleanup for tasks and resources
6. **Test Edge Cases**: Include error conditions and boundary values in your tests

### Common Patterns

**Async Task Pattern:**
```cpp
// Submit async task and wait for completion
auto task = client.AsyncOperation(pool_query, params);
task.Wait();

// Check result
if (task->result_code_ == 0) {
  // Success - access output
  std::string output = task->output_data_.str();
}
```

**Multiple Parallel Tasks:**
```cpp
std::vector<chi::Future<TaskType>> tasks;

// Submit multiple tasks
for (int i = 0; i < num_tasks; ++i) {
  tasks.push_back(client.AsyncOperation(pool_query, params));
}

// Wait for all tasks
for (auto& task : tasks) {
  task.Wait();
  REQUIRE(task->result_code_ == 0);
}
```

This testing approach ensures your ChiMod is validated across key operational scenarios while maintaining focus on essential setup and workflow patterns.