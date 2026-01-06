# WrpCte - Content Transfer Engine Jarvis Package

The WrpCte package provides a Jarvis-CD service for configuring the Content Transfer Engine (CTE) within the IoWarp framework. This package generates CTE configuration files, sets up storage devices, and manages the environment for CTE operations.

## Overview

WrpCte is a **Service** type package that:
- Configures CTE storage devices and performance parameters
- Generates YAML configuration files for CTE runtime
- Sets up environment variables for CTE integration
- Supports automatic storage device discovery via resource graph
- Manages distributed cleanup of storage devices

## Package Type

**Service**: This is a configuration-only service that prepares the environment for CTE usage but does not run as a persistent process.

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devices` | `list` | `[]` | List of storage devices as tuples `(path, capacity, score)` |
| `worker_count` | `int` | `4` | Number of worker threads for CTE |
| `dpe_type` | `str` | `max_bw` | Data Placement Engine type (`random`, `round_robin`, `max_bw`) |
| `blob_cache_size_mb` | `int` | `512` | Blob cache size in megabytes |
| `max_concurrent_operations` | `int` | `64` | Maximum concurrent operations |
| `target_stat_interval_ms` | `int` | `5000` | Target statistics collection interval in milliseconds |
| `score_threshold` | `float` | `0.7` | Minimum score threshold for device selection |
| `max_targets` | `int` | `100` | Maximum number of targets |
| `default_target_timeout_ms` | `int` | `30000` | Default target timeout in milliseconds |

## Storage Device Configuration

### Device Format

Devices are specified as tuples: `(path, capacity, score)`

- **`path`**: Storage device path (string)
- **`capacity`**: Storage capacity using SizeType format (string)
- **`score`**: Performance score from 0.0 to 1.0 (float)

### Examples

```python
devices = [
    ("/mnt/nvme/cte_primary", "1TB", 0.9),      # High-performance NVMe
    ("/tmp/ram_cache", "8GB", 1.0),             # RAM storage (highest score)
    ("/mnt/ssd/cte_secondary", "500GB", 0.6)    # Secondary SSD storage
]
```

### Storage Types

The package automatically determines `bdev_type` based on the path:

- **RAM storage** (`bdev_type: ram`): Paths containing `/tmp/`, `ramdisk`, `tmpfs`, or `ram`
- **File storage** (`bdev_type: file`): All other paths

### Size Format

Capacities use Jarvis SizeType format with binary multipliers:
- `k/K`: 1024 bytes (Kilobytes)
- `m/M`: 1048576 bytes (Megabytes)  
- `g/G`: 1073741824 bytes (Gigabytes)
- `t/T`: 1099511627776 bytes (Terabytes)

Examples: `"1M"`, `"512K"`, `"2G"`, `"1.5T"`

## Resource Graph Integration

If no devices are specified, the package attempts to auto-detect storage from the resource graph:

1. Initializes `ResourceGraphManager()` (automatically loads existing resource graph)
2. Queries resource graph for common storage across cluster nodes
3. Extracts device properties: mount point, available space, device type
4. Applies 0.5x safety factor to available space (to prevent filesystem full errors)
5. Appends `hermes_data.bin` to mount points for bdev file creation
6. Calculates performance scores based on device type and benchmark data
7. Falls back to default devices if resource graph is unavailable

### Auto-detected Storage Scoring

| Storage Type | Base Score | Performance Boost |
|--------------|------------|-------------------|
| RAM/RAMdisk/tmpfs | 1.0 | - |
| NVMe | 0.9 | +0.1-0.2 if benchmarked |
| SSD | 0.7 | +0.1-0.2 if benchmarked |
| HDD | 0.4 | +0.1 if benchmarked |
| Network | 0.3 | - |
| Unknown | 0.5 | - |

**Performance Benchmarks**: If the resource graph was built with benchmarking enabled (`jarvis rg build`), scores are adjusted based on actual measured performance:
- **Sequential bandwidth** (`1m_seqwrite_bw`): +0.1 for >500MB/s, +0.2 for >1GB/s
- **Random write bandwidth** (`4k_randwrite_bw`): +0.1 for >50MB/s

## Generated Configuration

The package generates a complete CTE YAML configuration file with:

- **Worker Configuration**: Thread pool settings
- **Storage Devices**: Block device configurations with paths, types, and scores
- **Queue Configuration**: Lane counts and priorities for different operation types
- **Performance Settings**: Cache sizes, timeouts, and thresholds
- **Data Placement Engine**: Algorithm selection for data distribution
- **Target Management**: Limits and timeout configurations

## Environment Variables

The package sets the following environment variable:

- **`WRP_RUNTIME_CONF`**: Path to the generated CTE configuration file

## Usage Example

```yaml
# Pipeline configuration
wrp_cte:
  devices:
    - ["/mnt/nvme1/cte", "2TB", 0.9]
    - ["/tmp/cte_cache", "16GB", 1.0]
    - ["/mnt/storage/cte", "10TB", 0.5]
  worker_count: 8
  dpe_type: "max_bw"
  blob_cache_size_mb: 1024
  max_concurrent_operations: 128
```

## File Locations

- **Package Source**: `test/jarvis_iowarp/wrp_cte/pkg.py`
- **Configuration Output**: `{shared_dir}/cte_config.yaml`
- **Template Reference**: `config/cte_example.yaml`

## Lifecycle Operations

### Configure
1. Validates device specifications using SizeType
2. Auto-detects storage via resource graph if no devices specified
3. Generates CTE YAML configuration file
4. Sets `WRP_RUNTIME_CONF` environment variable

### Start
No persistent process - returns success status and reports configuration location

### Stop
No persistent process - returns success status

### Clean
Uses `Rm` with `PsshExecInfo` to remove storage device files across all cluster nodes:
1. Removes local configuration file
2. Cleans storage devices on all nodes via parallel SSH execution
3. Handles both manually configured and auto-detected devices

## Integration

This package is designed to work with:

- **CTE Core Module**: Uses generated configuration via `WRP_RUNTIME_CONF`
- **Jarvis Resource Graph**: For automatic storage discovery
- **IoWarp Framework**: As part of the content transfer pipeline
- **Cluster Environments**: Supports distributed storage cleanup

## Dependencies

- `jarvis_cd.core.pkg.Service`
- `jarvis_cd.util.SizeType`
- `jarvis_cd.shell.process.Rm`
- `jarvis_cd.shell.PsshExecInfo`
- `jarvis_cd.core.resource_graph.ResourceGraphManager` (optional, for automatic storage discovery)

## Best Practices

1. **Use descriptive paths** for storage devices
2. **Include RAM storage** for high-performance caching with `/tmp/` paths
3. **Set appropriate scores** based on storage performance characteristics
4. **Validate capacity formats** using SizeType notation
5. **Test resource graph integration** in cluster environments
6. **Monitor cleanup operations** across distributed nodes

## Error Handling

The package provides comprehensive error handling for:
- Invalid device specifications
- Missing storage resources
- SizeType parsing errors
- Resource graph connection issues
- File generation and cleanup failures

All errors are logged with descriptive messages and the package gracefully falls back to default configurations when possible.