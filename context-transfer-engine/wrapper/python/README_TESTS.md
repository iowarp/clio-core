# CTE Python Bindings Tests

This directory contains tests for the CTE (Context Transfer Engine) Python bindings.

## Test Files

### 1. `test_bindings.py`
Example-driven test suite demonstrating MCP (Model Context Protocol) integration patterns.

**Purpose**: Educational and integration testing
**Features**:
- Context bundle operations (PutBlob)
- Context query operations (GetContainedBlobs, GetBlob, GetBlobSize)
- Context delete operations (DelBlob)
- **NEW**: PollTelemetryLog operation testing
- Runtime initialization patterns
- Error handling examples

**Usage**:
```bash
# Run with runtime initialization
python3 test_bindings.py

# Run without runtime initialization (external runtime)
CHI_WITH_RUNTIME=0 python3 test_bindings.py
```

### 2. `test_cte_telemetry.py`
Focused unit test suite for telemetry functionality.

**Purpose**: Unit testing with pytest framework
**Features**:
- Comprehensive PollTelemetryLog tests
- Telemetry entry structure validation
- Logical time filtering tests
- Operation tracking (PutBlob, GetBlob) validation

**Usage with pytest**:
```bash
# Set environment variables
export CHI_WITH_RUNTIME=1
export WRP_RUNTIME_CONF=/path/to/cte_config.yaml

# Run all tests
pytest test_cte_telemetry.py -v

# Run specific test
pytest test_cte_telemetry.py::TestPollTelemetryLog::test_poll_telemetry_log_putblob_and_getblob -v

# Run with output
pytest test_cte_telemetry.py -v -s
```

**Usage as standalone script**:
```bash
CHI_WITH_RUNTIME=1 WRP_RUNTIME_CONF=/path/to/cte_config.yaml python3 test_cte_telemetry.py
```

## PollTelemetryLog Test Details

The main test requested performs these steps:

1. **PutBlob Operation**: Store test data in a blob
2. **GetBlob Operation**: Retrieve the blob data
3. **PollTelemetryLog**: Query the telemetry log with `minimum_logical_time=0`
4. **Verification**: Check that log has entries for both PutBlob and GetBlob operations

### Telemetry Entry Structure

Each `CteTelemetry` entry contains:

- `op_`: Operation type (`CteOp.kPutBlob`, `CteOp.kGetBlob`, `CteOp.kDelBlob`, etc.)
- `off_`: Byte offset in blob
- `size_`: Size of operation in bytes
- `tag_id_`: TagId where operation occurred
- `mod_time_`: Modification timestamp
- `read_time_`: Read timestamp
- `logical_time_`: Logical timestamp for ordering

### Example Usage

```python
import wrp_cte_core_ext as cte

# Initialize runtime
cte.chimaera_init(cte.ChimaeraMode.kClient, True)
cte.initialize_cte(config_path, cte.PoolQuery.Dynamic())

# Perform operations
tag = cte.Tag("my_tag")
tag.PutBlob("my_blob", b"Hello, World!", 0)

blob_size = tag.GetBlobSize("my_blob")
data = tag.GetBlob("my_blob", blob_size, 0)

# Poll telemetry log
client = cte.get_cte_client()
entries = client.PollTelemetryLog(minimum_logical_time=0)

# Analyze entries
for entry in entries:
    print(f"Operation: {entry.op_}, Size: {entry.size_}, Time: {entry.logical_time_}")
```

## Environment Variables

### Required for Runtime Initialization

- `CHI_WITH_RUNTIME`: Set to `1` to enable runtime initialization (default: `1`)
- `WRP_RUNTIME_CONF`: Path to CTE configuration YAML file

### Optional

- `CHI_REPO_PATH`: Path to ChiMod repository (auto-detected from module location)
- `LD_LIBRARY_PATH`: Library search path (auto-configured to include ChiMod directory)

## Configuration File

Both tests require a valid CTE configuration file (YAML format). Example minimal config:

```yaml
compose:
  - mod_name: wrp_cte_core
    pool_name: wrp_cte
    pool_query: local
    pool_id: 512.0

    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000

    storage:
      - path: "ram::cte_ram_tier1"
        bdev_type: "ram"
        capacity_limit: "16GB"
        score: 0.0

    dpe:
      dpe_type: "max_bw"
```

## Running Tests in Development

### From Build Directory

```bash
cd /workspace/build

# Set config path
export WRP_RUNTIME_CONF=/workspace/context-transfer-engine/test/unit/adapters/adios2/cte_config.yaml

# Run example tests
python3 /workspace/context-transfer-engine/wrapper/python/test_bindings.py

# Run unit tests with pytest
pytest /workspace/context-transfer-engine/wrapper/python/test_cte_telemetry.py -v
```

### From Source Directory

```bash
cd /workspace/context-transfer-engine/wrapper/python

# Set config and library paths
export WRP_RUNTIME_CONF=/workspace/context-transfer-engine/test/unit/adapters/adios2/cte_config.yaml
export LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/build/bin:$PYTHONPATH

# Run tests
python3 test_bindings.py
pytest test_cte_telemetry.py -v
```

## Troubleshooting

### Module Import Errors

If you see `ImportError: No module named 'wrp_cte_core_ext'`:

1. Ensure Python bindings are built: `cmake --build /workspace/build --target wrp_cte_core_ext`
2. Add build directory to PYTHONPATH: `export PYTHONPATH=/workspace/build/bin:$PYTHONPATH`
3. Check library dependencies: `export LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH`

### Runtime Initialization Failures

If runtime initialization fails:

1. Check config file exists: `ls -la $WRP_RUNTIME_CONF`
2. Verify config syntax is valid YAML
3. Check storage paths are writable
4. Ensure no other runtime instance is running
5. Check logs for detailed error messages

### Empty Telemetry Log

If `PollTelemetryLog` returns empty list:

1. Telemetry logging may be disabled in configuration
2. Operations may not have completed yet (add small delay)
3. Check `minimum_logical_time` filter (try `0` to get all entries)
4. Verify operations actually succeeded (check return codes)

## Adding New Tests

To add new tests to `test_cte_telemetry.py`:

1. Add test method to `TestPollTelemetryLog` class
2. Use fixtures: `cte_module`, `cte_client`, `test_tag`
3. Follow naming convention: `test_poll_telemetry_log_<description>`
4. Add descriptive docstring explaining test purpose
5. Use assertions to validate behavior
6. Print informative messages for debugging

Example:

```python
def test_poll_telemetry_log_custom(self, cte_module, cte_client, test_tag):
    """Test custom telemetry scenario"""
    # Your test code here
    test_tag.PutBlob("test", b"data", 0)
    entries = cte_client.PollTelemetryLog(0)
    assert len(entries) > 0, "Should have telemetry entries"
```

## Test Coverage

Current test coverage for PollTelemetryLog:

- ✅ Basic PollTelemetryLog call
- ✅ PollTelemetryLog after PutBlob
- ✅ **PollTelemetryLog after PutBlob + GetBlob (main requested test)**
- ✅ Telemetry entry structure validation
- ✅ Logical time filtering

## CI/CD Integration

To integrate these tests in CI/CD:

```yaml
# Example GitHub Actions workflow step
- name: Run CTE Python Tests
  env:
    CHI_WITH_RUNTIME: 1
    WRP_RUNTIME_CONF: /path/to/test_config.yaml
    LD_LIBRARY_PATH: ${{ github.workspace }}/build/bin
    PYTHONPATH: ${{ github.workspace }}/build/bin
  run: |
    cd context-transfer-engine/wrapper/python
    python3 test_bindings.py
    pytest test_cte_telemetry.py -v --junitxml=test-results.xml
```

## Documentation References

- [CTE Core API Documentation](../../docs/cte/cte.md)
- [Python Bindings Implementation](core_bindings.cc)
- [Module Development Guide](../../../context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md)
