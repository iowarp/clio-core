# Configuration Guide

This document describes how to configure IOWarp deployments.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration File Format](#configuration-file-format)
- [Runtime Configuration Parameters](#runtime-configuration-parameters)
- [CTE Configuration Parameters](#cte-configuration-parameters)
- [Complete Examples](#complete-examples)
- [Environment Variables](#environment-variables)
- [Docker Deployment](#docker-deployment)

---

## Overview

CTE runs on the IoWarp Runtime (Chimaera distributed task execution framework). A single YAML configuration file configures both:

1. **Runtime Infrastructure**: Workers, memory, networking, logging
2. **CTE ChiMod**: Storage devices, data placement, performance tuning

Configuration is specified via the `WRP_RUNTIME_CONF` environment variable pointing to a YAML file.

---

## Quick Start

**Basic Deployment** (compose section creates CTE automatically):

```bash
# Set configuration file
export WRP_RUNTIME_CONF=/etc/iowarp/config.yaml

# Start runtime (automatically creates CTE from compose section)
chimaera_start_runtime
```

**Alternative: Manual Pool Creation** (using chimaera_compose utility):

```bash
# Start runtime first
export WRP_RUNTIME_CONF=/etc/iowarp/config.yaml
chimaera_start_runtime &

# Wait for runtime to initialize
sleep 2

# Create CTE pool from compose configuration
chimaera_compose /etc/iowarp/config.yaml
```

The `chimaera_compose` utility is useful for:
- Setting up pools after runtime initialization
- Scripted deployment workflows
- Testing pool configurations
- Separating runtime startup from pool creation

**Note**: When using `chimaera_compose`, set `pool_query: dynamic` in the compose section. For automatic pool creation during `chimaera_start_runtime`, use `pool_query: local` for better performance.

**Minimal Configuration** (`config.yaml`):

```yaml
workers:
  sched_threads: 4
  slow_threads: 4

memory:
  main_segment_size: 1GB
  client_data_segment_size: 512MB
  runtime_data_segment_size: 512MB

networking:
  port: 5555

compose:
  - mod_name: wrp_cte_core
    pool_name: wrp_cte_core    # REQUIRED: Do not change
    pool_query: local          # Use 'local' for chimaera_start_runtime
    pool_id: "512.0"            # REQUIRED: Do not change
    storage:
      - path: /tmp/cte_storage
        bdev_type: file
        capacity_limit: 10GB
    dpe:
      dpe_type: max_bw
```

---

## Configuration File Format

The configuration file has two main parts:

### 1. Runtime Configuration (Top-Level)

```yaml
# Worker threads
workers:
  sched_threads: 8           # Fast task workers (< 50us)
  slow_threads: 8            # Slow task workers (>= 50us)

# Shared memory segments
memory:
  main_segment_size: 4GB
  client_data_segment_size: 2GB
  runtime_data_segment_size: 2GB

# Networking for distributed mode
networking:
  port: 5555
  neighborhood_size: 32
  hostfile: /etc/iowarp/hostfile  # Optional: for multi-node

# Logging (optional, can omit for defaults)
logging:
  level: info
  file: /tmp/chimaera.log

# Runtime settings (optional, can omit for defaults)
runtime:
  stack_size: 65536
  queue_depth: 10000
  lane_map_policy: round_robin
  heartbeat_interval: 1000
```

### 2. CTE Compose Configuration

The CTE ChiMod is created via the `compose` section.

**Important**: The `pool_name` and `pool_id` are hardcoded constants and must not be changed:
- `pool_name`: Must be `"wrp_cte_core"` (defined as `kCtePoolName`)
- `pool_id`: Must be `"512.0"` (defined as `kCtePoolId` with major: 512, minor: 0)

**Pool Query Selection**:
- `local`: Faster for `chimaera_start_runtime` compose (recommended for automatic startup)
- `dynamic`: Required for `chimaera_compose` utility (manual pool creation)

```yaml
compose:
  - mod_name: wrp_cte_core
    pool_name: wrp_cte_core    # REQUIRED: Do not change
    pool_query: local          # Use 'local' for chimaera_start_runtime, 'dynamic' for chimaera_compose
    pool_id: "512.0"            # REQUIRED: Do not change

    # CTE-specific configuration
    storage:
      - path: /mnt/storage1
        bdev_type: file
        capacity_limit: 100GB
        score: -1.0              # -1.0 = auto, 0.0-1.0 = manual
      - path: /mnt/storage2
        bdev_type: file
        capacity_limit: 100GB
        score: -1.0

    dpe:
      dpe_type: max_bw           # Options: random, round_robin, max_bw

    targets:
      neighborhood: 4
      default_target_timeout_ms: 30000
      poll_period_ms: 5000

    performance:
      target_stat_interval_ms: 5000
      max_concurrent_operations: 64
      score_threshold: 0.7
      score_difference_threshold: 0.05
```

---

## Runtime Configuration Parameters

### Workers (`workers`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sched_threads` | 4 | Scheduler workers for fast tasks (< 50us) |
| `slow_threads` | 4 | Workers for slow tasks (>= 50us) |

**Recommendation**: Set total threads = CPU cores (e.g., 8+8 for 16-core system)

### Memory (`memory`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `main_segment_size` | 1GB | Task metadata and control structures |
| `client_data_segment_size` | 512MB | Application data |
| `runtime_data_segment_size` | 512MB | Runtime internal state |

**Size format**: `1GB`, `512MB`, `64K`, or bytes (`1073741824`)

**Docker**: Set `shm_size` >= sum of segments + 20% overhead

### Networking (`networking`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `port` | 5555 | ZeroMQ port (must match across cluster) |
| `neighborhood_size` | 32 | Max nodes queried for range queries |
| `hostfile` | - | Path to file with cluster IPs (one per line) |

**Hostfile format** (`/etc/iowarp/hostfile`):
```
172.20.0.10
172.20.0.11
172.20.0.12
```

### Logging and Runtime (Optional)

These sections can be omitted to use defaults. See `docs/chimaera/deployment.md` for details.

---

## CTE Configuration Parameters

All CTE parameters are specified within the compose entry for `wrp_cte_core`.

### Storage Devices (`storage`)

Array of storage targets for blob storage.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `path` | Yes | Directory path for block device |
| `bdev_type` | Yes | Device type: `file` or `ram` |
| `capacity_limit` | Yes | Capacity (e.g., `10GB`, `1TB`) |
| `score` | No | Manual score 0.0-1.0, or -1.0 for auto (default: -1.0) |

**Example**:
```yaml
storage:
  # RAM-based cache storage (use ram:: prefix)
  - path: "ram::cte_cache"
    bdev_type: ram
    capacity_limit: 512MB
    score: 1.0              # Manual score - fastest tier
  # File-based storage
  - path: /mnt/nvme/cte
    bdev_type: file
    capacity_limit: 500GB
    score: 0.9              # Fast storage
  - path: /mnt/hdd/cte
    bdev_type: file
    capacity_limit: 2TB
    score: 0.3              # Slow storage
```

**Note**: RAM-based storage requires the `ram::` prefix in the path.

### Data Placement Engine (`dpe`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dpe_type` | `max_bw` | Placement algorithm: `random`, `round_robin`, `max_bw` |

### Targets (`targets`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neighborhood` | 4 | Number of storage targets CTE can buffer to |
| `default_target_timeout_ms` | 30000 | Timeout for target operations (ms) |
| `poll_period_ms` | 5000 | Period to rescan targets for stats (ms) |

### Performance (`performance`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_stat_interval_ms` | 5000 | Interval for updating target stats (ms) |
| `max_concurrent_operations` | 64 | Max concurrent I/O operations |
| `score_threshold` | 0.7 | Threshold for blob reorganization (0.0-1.0) |
| `score_difference_threshold` | 0.05 | Min score difference to trigger reorganization |

**Note**: Most users can omit the `performance` section to use optimized defaults.

---

## Complete Examples

### Single-Node Development

```yaml
workers:
  sched_threads: 4
  slow_threads: 4

memory:
  main_segment_size: 1GB
  client_data_segment_size: 512MB
  runtime_data_segment_size: 512MB

networking:
  port: 5555

compose:
  - mod_name: wrp_cte_core
    pool_name: wrp_cte_core
    pool_query: local
    pool_id: "512.0"            # REQUIRED: Do not change
    storage:
      - path: /tmp/cte_storage
        bdev_type: file
        capacity_limit: 10GB
    dpe:
      dpe_type: max_bw
```

### Multi-Node Production (4 nodes)

```yaml
# Combined Chimaera + CTE Configuration
# For use with 4-node distributed cluster

workers:
  sched_threads: 8
  slow_threads: 8

memory:
  main_segment_size: 4GB
  client_data_segment_size: 2GB
  runtime_data_segment_size: 2GB

networking:
  port: 8080
  neighborhood_size: 32
  hostfile: /etc/iowarp/hostfile

logging:
  level: info
  file: /var/log/chimaera/chimaera.log

runtime:
  stack_size: 65536
  queue_depth: 10000
  lane_map_policy: round_robin
  heartbeat_interval: 1000

compose:
  - mod_name: wrp_cte_core
    pool_name: wrp_cte_core
    pool_query: local
    pool_id: "512.0"            # REQUIRED: Do not change

    # 4 storage targets across 4 nodes
    storage:
      - path: /mnt/hdd1
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25
      - path: /mnt/hdd2
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25
      - path: /mnt/hdd3
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25
      - path: /mnt/hdd4
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25

    dpe:
      dpe_type: max_bw

    targets:
      neighborhood: 4
      default_target_timeout_ms: 30000
      poll_period_ms: 5000

    performance:
      target_stat_interval_ms: 5000
      max_concurrent_operations: 64
      score_threshold: 0.7
      score_difference_threshold: 0.05
```

### Multi-Tier Storage with RAM Cache

```yaml
workers:
  sched_threads: 8
  slow_threads: 8

memory:
  main_segment_size: 4GB
  client_data_segment_size: 2GB
  runtime_data_segment_size: 2GB

networking:
  port: 5555

compose:
  - mod_name: wrp_cte_core
    pool_name: wrp_cte_core
    pool_query: local
    pool_id: "512.0"            # REQUIRED: Do not change
    storage:
      # RAM cache tier (use ram:: prefix)
      - path: "ram::cte_cache"
        bdev_type: ram
        capacity_limit: 512MB
        score: 1.0
      # Fast tier - NVMe
      - path: /mnt/nvme/cte
        bdev_type: file
        capacity_limit: 200GB
        score: 0.9
      # Medium tier - SSD
      - path: /mnt/ssd/cte
        bdev_type: file
        capacity_limit: 500GB
        score: 0.7
      # Slow tier - HDD
      - path: /mnt/hdd/cte
        bdev_type: file
        capacity_limit: 2TB
        score: 0.3
    dpe:
      dpe_type: max_bw
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `WRP_RUNTIME_CONF` | Path to configuration YAML | `export WRP_RUNTIME_CONF=/etc/iowarp/config.yaml` |

**Note**: The runtime does NOT read individual `CHI_*` environment variables. All configuration must be in the YAML file.

---

## Docker Deployment

### Docker Compose Example

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  iowarp-cte:
    image: iowarp/chimaera-cte:latest
    container_name: iowarp-cte
    hostname: cte-node1

    # Shared memory: sum of segments + 20% overhead
    # 4GB + 2GB + 2GB = 8GB -> use 10GB
    shm_size: 10gb

    volumes:
      - ./config.yaml:/etc/iowarp/config.yaml:ro
      - ./data:/data
      - ./logs:/var/log/chimaera

    environment:
      - WRP_RUNTIME_CONF=/etc/iowarp/config.yaml

    ports:
      - "5555:5555"

    networks:
      cte_net:
        ipv4_address: 172.20.0.10

networks:
  cte_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
```

### Multi-Node Deployment

For a 3-node cluster, create 3 services with different IPs (172.20.0.10, 172.20.0.11, 172.20.0.12) and mount the same config with a hostfile:

**Hostfile** (`config/hostfile`):
```
172.20.0.10
172.20.0.11
172.20.0.12
```

**Configuration** (`config/config.yaml`):
```yaml
networking:
  port: 5555
  hostfile: /etc/iowarp/hostfile
# ... rest of configuration
```

Mount the hostfile in each container:
```yaml
volumes:
  - ./config/hostfile:/etc/iowarp/hostfile:ro
```

### Deployment Steps

1. **Create directories**:
   ```bash
   mkdir -p config data logs
   ```

2. **Create configuration** (`config/config.yaml`)

3. **Create storage paths**:
   ```bash
   mkdir -p data/cte_storage1 data/cte_storage2
   ```

4. **Start**:
   ```bash
   docker-compose up -d
   ```

5. **Verify**:
   ```bash
   docker logs iowarp-cte
   docker exec iowarp-cte chimaera_pool_list
   ```

---

## Related Documentation

- [CTE API Reference](cte.md) - Complete API documentation
- [Chimaera Deployment Guide](../chimaera/deployment.md) - Detailed runtime configuration
- [Module Development Guide](../chimaera/MODULE_DEVELOPMENT_GUIDE.md) - ChiMod development

---

**Last Updated**: 2025-11-09
**Version**: 2.0.0
