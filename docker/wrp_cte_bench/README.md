# WRP CTE Benchmark Docker Compose

This directory contains a Docker Compose configuration for running WRP CTE (Context Transfer Engine) benchmarks with a separate IOWarp runtime service.

## Overview

This setup uses a multi-service architecture:

1. **iowarp-runtime**: Provides the CTE runtime daemon
2. **wrp-cte-bench**: Runs benchmark tests that connect to the runtime

The runtime service must be running before benchmarks can execute.

## Quick Start

```bash
# Start runtime and run benchmark with default settings
docker-compose up

# Start runtime in background, then run benchmark
docker-compose up -d iowarp-runtime
docker-compose up wrp-cte-bench

# Stop all services
docker-compose down
```

## Architecture

```
┌─────────────────────┐
│  wrp-cte-bench      │  Benchmark client
│  (connects to       │  - Runs test workloads
│   runtime via       │  - Reports performance
│   ZeroMQ port 9413) │
└──────────┬──────────┘
           │
           │ ZeroMQ
           │
┌──────────▼──────────┐
│  iowarp-runtime     │  CTE Runtime daemon
│  - CTE initialization│  - Manages storage
│  - ZeroMQ service   │  - Handles I/O requests
│  - Shared memory    │
└─────────────────────┘
```

## Configuration

### CTE Configuration File

The `cte_config.yaml` file configures the CTE runtime:

```yaml
targets:
  neighborhood: 1                    # Single-node configuration
  default_target_timeout_ms: 30000   # Target timeout
  poll_period_ms: 5000               # Statistics polling period

storage:
  - path: "ram::cte_ram_tier1"       # RAM-based storage
    bdev_type: "ram"                 # Storage type
    capacity_limit: "16GB"           # Maximum capacity
    score: 0.0                       # Tier score (highest)

dpe:
  dpe_type: "max_bw"                 # Data placement strategy
```

### Environment Variables

All benchmark parameters can be configured via environment variables:

| Variable | Description | Default | Examples |
|----------|-------------|---------|----------|
| `TEST_CASE` | Test to run: Put, Get, PutGet | `Put` | Put, Get, PutGet |
| `NUM_PROCS` | Number of parallel processes | `1` | 1, 4, 8, 16 |
| `DEPTH` | Queue depth (concurrent operations) | `4` | 4, 8, 16, 32 |
| `IO_SIZE` | Size of each I/O operation | `1m` | 4k, 1m, 16m |
| `IO_COUNT` | Number of operations to perform | `100` | 100, 1000, 100000 |
| `CHI_WITH_RUNTIME` | Initialize CTE runtime in benchmark | `0` | 0 (runtime service), 1 (self-init) |
| `CHI_SERVER_CONF` | Path to CTE configuration file | `/etc/iowarp/cte_config.yaml` | Custom path |

### I/O Size Format

The `IO_SIZE` parameter supports these suffixes:
- `b` - bytes (e.g., `1024b`)
- `k` - kilobytes (e.g., `4k`)
- `m` - megabytes (e.g., `1m`)
- `g` - gigabytes (e.g., `2g`)

## Usage Examples

### Basic Usage

```bash
# Run Put benchmark with default settings
docker-compose up

# Run Get benchmark with 4 processes
TEST_CASE=Get NUM_PROCS=4 docker-compose up wrp-cte-bench

# Run PutGet benchmark with high concurrency
TEST_CASE=PutGet NUM_PROCS=8 DEPTH=16 docker-compose up wrp-cte-bench
```

### Performance Testing Scenarios

```bash
# High IOPS test - many small operations
TEST_CASE=Put NUM_PROCS=16 DEPTH=16 IO_SIZE=4k IO_COUNT=1000000 docker-compose up wrp-cte-bench

# High throughput test - large operations
TEST_CASE=Put NUM_PROCS=1 DEPTH=2 IO_SIZE=16m IO_COUNT=1000 docker-compose up wrp-cte-bench

# Write-heavy workload
TEST_CASE=Put NUM_PROCS=8 DEPTH=8 IO_SIZE=1m IO_COUNT=50000 docker-compose up wrp-cte-bench

# Read-heavy workload
TEST_CASE=Get NUM_PROCS=8 DEPTH=8 IO_SIZE=1m IO_COUNT=50000 docker-compose up wrp-cte-bench

# Mixed read/write workload
TEST_CASE=PutGet NUM_PROCS=4 DEPTH=4 IO_SIZE=1m IO_COUNT=10000 docker-compose up wrp-cte-bench
```

### Multi-Stage Workflow

```bash
# 1. Start runtime service
docker-compose up -d iowarp-runtime

# 2. Wait for runtime to be ready
docker-compose logs -f iowarp-runtime

# 3. Run multiple benchmarks sequentially
TEST_CASE=Put docker-compose up wrp-cte-bench
TEST_CASE=Get docker-compose up wrp-cte-bench
TEST_CASE=PutGet docker-compose up wrp-cte-bench

# 4. Stop runtime
docker-compose down
```

### Using Alternative Service Configurations

The docker-compose.yml file includes commented-out service definitions for common test scenarios. Uncomment the desired service and run:

```bash
# Edit docker-compose.yml to uncomment the desired service, then:
docker-compose up wrp-cte-bench-put
docker-compose up wrp-cte-bench-get
docker-compose up wrp-cte-bench-large
docker-compose up wrp-cte-bench-concurrent
```

## Test Cases

### Put Test
Write-only benchmark - measures CTE PUT operation performance.

```bash
TEST_CASE=Put NUM_PROCS=4 DEPTH=8 docker-compose up wrp-cte-bench
```

### Get Test
Read-only benchmark - measures CTE GET operation performance.

```bash
TEST_CASE=Get NUM_PROCS=4 DEPTH=8 docker-compose up wrp-cte-bench
```

### PutGet Test
Mixed read/write benchmark - alternates between PUT and GET operations.

```bash
TEST_CASE=PutGet NUM_PROCS=4 DEPTH=4 docker-compose up wrp-cte-bench
```

## Resource Configuration

The default configuration allocates:
- **Memory limit**: 8GB
- **ZeroMQ port**: 9413 (for runtime-client communication)

### Adjusting Resources

For larger workloads, increase resource limits in docker-compose.yml:

```yaml
services:
  iowarp-runtime:
    mem_limit: 16g

  wrp-cte-bench:
    mem_limit: 16g
```

Also update `cte_config.yaml`:

```yaml
storage:
  - path: "ram::cte_ram_tier1"
    capacity_limit: "32GB"  # Increase capacity
```

## Monitoring and Debugging

### View Logs

```bash
# View runtime logs
docker-compose logs -f iowarp-runtime

# View benchmark logs
docker-compose logs -f wrp-cte-bench

# View all logs
docker-compose logs -f
```

### Check Service Status

```bash
# Check which services are running
docker-compose ps

# Check resource usage
docker stats iowarp-runtime wrp-cte-bench
```

### Health Checks

The runtime service includes a health check. Wait for it to be healthy:

```bash
docker-compose ps iowarp-runtime
# Should show "healthy" in STATUS column
```

## Troubleshooting

### Runtime Service Won't Start

1. Check if port 9413 is available:
   ```bash
   netstat -an | grep 9413
   ```

2. Check runtime logs:
   ```bash
   docker-compose logs iowarp-runtime
   ```

3. Verify configuration file:
   ```bash
   cat cte_config.yaml
   ```

### Benchmark Can't Connect to Runtime

1. Ensure runtime is healthy:
   ```bash
   docker-compose ps iowarp-runtime
   ```

2. Check network configuration:
   ```bash
   docker network inspect wrp_cte_bench_default
   ```

3. Verify runtime is listening:
   ```bash
   docker-compose exec iowarp-runtime netstat -an | grep 9413
   ```

### Out of Memory Errors

1. Increase memory limits in docker-compose.yml
2. Reduce IO_SIZE or IO_COUNT
3. Reduce NUM_PROCS or DEPTH
4. Monitor with `docker stats`

### Performance Issues

- Reduce NUM_PROCS for large I/O operations
- Adjust DEPTH based on workload characteristics
- Monitor system resources with `docker stats`
- Check CTE configuration for optimal settings

## Cleanup

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Remove all stopped containers
docker-compose down --remove-orphans
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run WRP CTE Benchmark
  run: |
    cd docker/wrp_cte_bench

    # Start runtime
    docker-compose up -d iowarp-runtime

    # Wait for runtime to be healthy
    timeout 30 bash -c 'until docker-compose ps iowarp-runtime | grep healthy; do sleep 1; done'

    # Run benchmarks
    TEST_CASE=Put docker-compose up wrp-cte-bench
    TEST_CASE=Get docker-compose up wrp-cte-bench

    # Cleanup
    docker-compose down
```

## Advanced Configuration

### Custom CTE Configuration

Edit `cte_config.yaml` to customize storage tiers, capacity, and data placement:

```yaml
storage:
  # Multiple storage tiers
  - path: "ram::cte_ram_tier1"
    bdev_type: "ram"
    capacity_limit: "16GB"
    score: 0.0

  - path: "/mnt/nvme/cte_tier2"
    bdev_type: "posix"
    capacity_limit: "100GB"
    score: 0.5

dpe:
  dpe_type: "round_robin"  # Alternative placement strategies
```

### Network Configuration

To expose runtime on host network:

```yaml
services:
  iowarp-runtime:
    network_mode: host
```

### Persistent Storage

To use persistent storage instead of RAM-only:

```yaml
storage:
  - path: "/data/cte_storage"
    bdev_type: "posix"
    capacity_limit: "100GB"
```

## Notes

- Runtime service must be healthy before benchmarks can run
- IOWarp uses memfd_create() for shared memory (no /dev/shm dependency)
- All data is RAM-based by default (no persistence)
- ZeroMQ port 9413 used for client-runtime communication
- Container network sharing allows benchmark to access runtime
- Benchmark containers auto-remove after completion (restart: "no")
