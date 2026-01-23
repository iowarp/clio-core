# Chimaera Unit Tests

This directory contains comprehensive unit tests for the Chimaera runtime system using the Catch2 testing framework.

## Overview

The unit tests demonstrate and verify the complete Chimaera workflow:

1. **Runtime Startup** - Initialize Chimaera runtime components
2. **Client Initialization** - Connect client to the runtime
3. **Task Submission** - Create and submit MOD_NAME custom tasks
4. **Task Completion** - Wait for task completion and verify results
5. **Resource Cleanup** - Proper cleanup of all resources

## Test Categories

### Core Functionality Tests
- **Chimaera Initialization** `[initialization]` - Tests for `CHIMAERA_INIT()`
- **MOD_NAME Task Execution** `[task][mod_name]` - Custom task creation and execution
- **Async Task Handling** `[task][mod_name][async]` - Asynchronous task submission

### Reliability Tests
- **Error Handling** `[error][edge_cases]` - Graceful handling of invalid inputs
- **Memory Management** `[memory][cleanup]` - Task allocation and deallocation
- **Concurrent Execution** `[concurrent][stress]` - Multiple simultaneous tasks

### Performance Tests
- **Execution Latency** `[performance][timing]` - Task execution time measurements

## Quick Start

### Using the Test Runner Script (Recommended)

The easiest way to run tests is using the provided test runner script:

```bash
# Run all tests
./test/run_tests.sh

# Run specific test categories
./test/run_tests.sh -t runtime      # Runtime tests only
./test/run_tests.sh -t client       # Client tests only  
./test/run_tests.sh -t tasks        # Task execution tests
./test/run_tests.sh -t errors       # Error handling tests
./test/run_tests.sh -t performance  # Performance tests

# Run with verbose output
./test/run_tests.sh -v

# Run with custom Catch2 filters
./test/run_tests.sh -f "[runtime][initialization]"
./test/run_tests.sh -f "MOD_NAME*"
```

### Manual Build and Run

If you prefer to build and run manually:

```bash
# 1. Configure with tests enabled
mkdir -p build && cd build
cmake .. -DCHIMAERA_ENABLE_TESTS=ON

# 2. Build the project
make -j$(nproc)

# 3. Run tests
./bin/chimaera_unit_tests

# Run specific test categories
./bin/chimaera_unit_tests "[runtime]"
./bin/chimaera_unit_tests "[task][mod_name]"
```

### Using CTest

If you want to use CMake's CTest framework:

```bash
cd build
cmake .. -DCHIMAERA_ENABLE_TESTS=ON
make -j$(nproc)

# Run all tests via CTest
ctest

# Run specific test suites
ctest -R runtime_initialization_tests
ctest -R mod_name_task_tests

# Run with verbose output
ctest -V
```

## Test Structure

### Test Fixture: `ChimaeraRuntimeFixture`

The test fixture provides:
- **`initialize()`** - Complete Chimaera initialization (client with embedded runtime)
- **`waitForTaskCompletion()`** - Wait for async task completion with timeout
- **`createModNamePool()`** - Create MOD_NAME pool using admin client
- **`cleanup()`** - Automatic resource cleanup

### Test Organization

Tests are organized using Catch2 sections and tags:

```cpp
TEST_CASE("Test Description", "[tag1][tag2]") {
  ChimaeraRuntimeFixture fixture;
  
  SECTION("Specific scenario") {
    // Test implementation
  }
}
```

## Writing New Tests

### Basic Test Template

```cpp
TEST_CASE("My New Test", "[category][subcategory]") {
  ChimaeraRuntimeFixture fixture;
  
  SECTION("Test scenario description") {
    // 1. Setup
    REQUIRE(fixture.initialize());
    
    // 2. Execute
    // ... your test code ...
    
    // 3. Verify
    REQUIRE(/* your assertions */);
    
    // 4. Cleanup (automatic via fixture destructor)
  }
}
```

### Task Testing Pattern

```cpp
TEST_CASE("Custom Task Test", "[task][custom]") {
  ChimaeraRuntimeFixture fixture;
  
  SECTION("Task execution") {
    // Initialize everything
    REQUIRE(fixture.initialize());
    REQUIRE(fixture.createModNamePool());
    
    // Create client
    chimaera::MOD_NAME::Client client(kTestModNamePoolId);
    chi::DomainQuery pool_query;
    client.Create(pool_query);
    
    // Submit task
    auto task = client.AsyncCustom(pool_query, "test_data", 123);
    REQUIRE_FALSE(task.IsNull());
    
    // Wait for completion
    REQUIRE(fixture.waitForTaskCompletion(task));
    
    // Verify results
    REQUIRE(task->result_code_ == 0);
    
    // Cleanup
  }
}
```

## Test Configuration

### CMake Options

- **`CHIMAERA_ENABLE_TESTS`** - Enable/disable test building (default: OFF)
- **`CMAKE_BUILD_TYPE`** - Debug/Release build (Debug recommended for tests)

### Test Constants

Key test configuration in `test_chimaera_runtime.cc`:

```cpp
constexpr chi::u32 kTestTimeoutMs = 5000;     // Task completion timeout
constexpr chi::u32 kMaxRetries = 50;          // Retry attempts
constexpr chi::u32 kRetryDelayMs = 100;       // Delay between retries
constexpr chi::PoolId kTestModNamePoolId = 100; // Test pool ID
```

## Troubleshooting

### Common Issues

1. **Tests timeout** - Increase `kTestTimeoutMs` or check Chimaera initialization
2. **Pool creation fails** - Verify admin ChiMod is available and working
3. **Task submission fails** - Check IPC manager initialization and pool existence

### Debug Mode

Run tests with verbose output for debugging:

```bash
./test/run_tests.sh -v
# or
./bin/chimaera_unit_tests --success --out --durations yes
```

### Test Isolation

Each test case runs independently. The fixture ensures:
- Clean initialization state
- Proper resource cleanup
- No interference between tests

## Dependencies

The tests require:
- **Catch2** - Testing framework (included with HSHM)
- **Chimaera Core** - Main runtime library
- **Admin ChiMod** - For pool management
- **MOD_NAME ChiMod** - For custom task testing
- **HSHM** - Shared memory and threading support

## Contributing

When adding new tests:
1. Use descriptive test names and clear section descriptions
2. Follow the existing test organization patterns
3. Include both positive and negative test cases
4. Add appropriate tags for categorization
5. Update this README if adding new test categories
6. Ensure tests are deterministic and don't rely on timing

## Performance Considerations

- Tests run in Debug mode by default for better error detection
- Use Release mode (`-b Release`) for performance benchmarking
- Concurrent tests may affect timing-sensitive measurements
- Performance tests provide baseline measurements, not absolute benchmarks