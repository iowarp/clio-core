# CAE Module Unit Tests

This directory contains unit tests for the Content Assimilation Engine (CAE) Module.

## Test Organization

### Directory Structure

```
chimods/test/unit/
├── README.md                      # This file
├── clio_config.yaml                # Shared CTE configuration for all tests
└── binary_assim/                  # Binary assimilation test suite
    ├── test_binary_assim.cc       # C++ test executable
    ├── binary_assim_omni.yaml     # OMNI config file
    └── run_test.sh                # Bash test runner
```

## Prerequisites

Before running tests, ensure:

1. **Build the project with tests enabled:**
   ```bash
   cd /workspace/build
   cmake -DBUILD_TESTING=ON ..
   make
   ```

2. **Ensure runtime components are available:**
   - `chi_runtime` (Clio runtime)
   - `clio_cte_daemon` (Content Transfer Engine daemon)
   - Test executables in `build/bin/`

## Running Tests

### Binary Assimilation Test

Tests the ParseOmni API with binary file assimilation.

**Quick run:**
```bash
cd /workspace/chimods/test/unit/binary_assim
./run_test.sh
```

**What it tests:**
- AssimilationCtx serialization/deserialization using cereal
- ParseOmni API execution
- Binary file transfer to CTE
- Data chunking for files > 1MB
- CTE tag creation and blob storage
- Integration between CAE and CTE

**Test parameters (environment variables):**
- `TEST_FILE_SIZE`: Size of test file in MB (default: 256)
- `INIT_CLIO`: Set to "1" to manually initialize Clio (default: handled by runner)

**Expected output:**
```
========================================
Binary Assimilation ParseOmni Test
========================================

[STEP 1] Checking prerequisites...
[STEP 2] Starting Clio runtime...
[STEP 3] Starting CTE daemon...
[STEP 4] Running test executable...
[STEP 5] Analyzing results...

✓ CAE pool created successfully
✓ ParseOmni executed
✓ Data verified in CTE

========================================
TEST SUITE PASSED
========================================
```

## Test Coverage

### ParseOmni API Coverage

#### ✅ Covered Test Cases

1. **Happy Path**
   - Valid file source with `file://` protocol
   - Valid CTE destination with `iowarp://` protocol
   - Binary format processing
   - Full file transfer (range_size = 0)
   - Successful serialization/deserialization

2. **Integration**
   - CAE pool creation
   - CTE client connection
   - Tag creation in CTE
   - Blob storage verification
   - Multi-chunk file handling (>1MB)

3. **Data Integrity**
   - Patterned data generation
   - File size validation
   - Chunk count verification

#### 🔄 Future Test Cases

1. **Range Tests**
   - Partial file transfer with `range_off` and `range_size`
   - Multiple non-overlapping ranges
   - Edge cases (first byte, last byte, single byte)

2. **Error Handling**
   - Non-existent source file
   - Invalid source protocol
   - Invalid destination protocol
   - Out-of-range offsets
   - Corrupted serialized context

3. **Edge Cases**
   - Empty file (0 bytes)
   - Very small files (< 1KB)
   - Very large files (> 1GB)
   - Files exactly at chunk boundaries

4. **Dependency Handling**
   - Tests with `depends_on` set
   - Dependency chain validation

5. **Concurrent Operations**
   - Multiple simultaneous ParseOmni calls
   - Thread safety validation

## Test Implementation Details

### Test Architecture

```
┌─────────────────┐
│  run_test.sh    │  Bash orchestration script
└────────┬────────┘
         │
         ├─► Start chi_runtime
         ├─► Start clio_cte_daemon (with clio_config.yaml)
         │
         ▼
┌─────────────────────────┐
│  test_binary_assim.cc   │  C++ test executable
└────────┬────────────────┘
         │
         ├─► Generate test file (256MB)
         ├─► Create CAE pool (kCaePoolId = 400)
         ├─► Serialize AssimilationCtx
         ├─► Call ParseOmni API
         ├─► Verify result_code and num_tasks_scheduled
         └─► Verify data in CTE (tag + blobs)
```

### Key Components

1. **constants.h**
   - Defines `kCaePoolId` constant for consistent pool identification

2. **clio_config.yaml**
   - RAM-only CTE configuration
   - Single-node setup
   - 16GB capacity limit
   - Max bandwidth placement strategy

3. **test_binary_assim.cc**
   - Comprehensive test with 9 steps
   - Patterned data generation for validation
   - Full error checking and reporting
   - Integration with CTE for verification

4. **run_test.sh**
   - Environment setup and teardown
   - Process management (chi_runtime, clio_cte_daemon)
   - Output analysis and validation
   - Color-coded results

## Debugging

### Enable verbose output:

```bash
export CTP_LOG_LEVEL=debug
./run_test.sh
```

### Check logs:

```bash
# Clio logs
cat /tmp/clio_runtime.log

# CTE logs
cat /tmp/clio_cte_daemon.log
```

### Manual test run (without script):

```bash
# Start services manually
chi_runtime &
clio_cte_daemon /workspace/chimods/test/unit/clio_config.yaml &

# Run test
/workspace/build/bin/test_binary_assim

# Cleanup
killall chi_runtime clio_cte_daemon
rm -f /dev/shm/clio_*
```

## Adding New Tests

To add a new test:

1. **Create test directory:**
   ```bash
   mkdir -p chimods/test/unit/my_new_test
   ```

2. **Create test files:**
   - `test_my_feature.cc` - C++ test executable
   - `run_test.sh` - Bash runner script
   - `*.yaml` - Any config files needed

3. **Update CMakeLists.txt:**
   ```cmake
   add_executable(test_my_feature
     ${CMAKE_CURRENT_SOURCE_DIR}/../test/unit/my_new_test/test_my_feature.cc
   )
   target_link_libraries(test_my_feature
     PRIVATE
       clio_cae_core_client
       clio::run::cxx
   )
   install(TARGETS test_my_feature DESTINATION bin)
   ```

4. **Follow test patterns:**
   - Use AAA pattern (Arrange-Act-Assert)
   - Clear step-by-step output
   - Comprehensive error handling
   - Proper cleanup

## Contributing

When adding tests:
- Follow Google C++ style guide
- Add comprehensive comments
- Include test strategy in file header
- Update this README with coverage information
- Ensure tests are deterministic
- Clean up all resources (files, processes, memory)

## License

Part of the IOWarp Content Assimilation Engine project.
