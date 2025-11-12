# CAE Launch Guide

## Overview

The Content Assimilation Engine (CAE) is launched using the `chimaera_compose` utility, which creates ChiMod containers from YAML configuration files. This replaces the deprecated `wrp_cae_launch` utility.

## Prerequisites

1. **Runtime must be running**: Start the Chimaera runtime before creating CAE containers
2. **Configuration file**: Create a YAML configuration file with CAE parameters
3. **CTE dependencies**: CAE requires the CTE (Content Transformation Engine) ChiMod

## Quick Start

### 1. Start the Chimaera Runtime

```bash
export WRP_RUNTIME_CONF=/etc/iowarp/config.yaml
chimaera_start_runtime &
```

Wait for the runtime to initialize (2-3 seconds).

### 2. Create CAE Container with chimaera_compose

```bash
chimaera_compose /path/to/cae_config.yaml
```

The `chimaera_compose` utility reads the YAML configuration file and creates the CAE container based on the `compose` section.

## Configuration File Format

The CAE configuration uses the Chimaera compose format. Create a YAML file with the following structure:

```yaml
compose:
  - mod_name: wrp_cae_core
    pool_name: wrp_cae_core_pool
    pool_query: dynamic
    pool_id: "400.0"

    # CAE-specific parameters
    worker_count: 4
```

### Required Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `mod_name` | `wrp_cae_core` | ChiMod name for CAE |
| `pool_name` | `wrp_cae_core_pool` | Pool identifier for the CAE container |
| `pool_query` | `dynamic` | Use dynamic query for chimaera_compose |
| `pool_id` | `"400.0"` | Pool ID (major: 400, minor: 0) defined as `kCaePoolId` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `worker_count` | 4 | Number of worker threads for CAE operations |

## Complete Configuration Example

```yaml
# CAE Container Configuration
# For use with chimaera_compose utility

compose:
  - mod_name: wrp_cae_core
    pool_name: wrp_cae_core_pool
    pool_query: dynamic
    pool_id: "400.0"
    worker_count: 4
```

## Usage Workflow

### Full Deployment Sequence

```bash
# 1. Set runtime configuration
export WRP_RUNTIME_CONF=/etc/iowarp/runtime_config.yaml

# 2. Start Chimaera runtime
chimaera_start_runtime &

# 3. Wait for runtime initialization
sleep 2

# 4. Create CAE container from compose configuration
chimaera_compose /path/to/cae_config.yaml

# 5. Verify container creation
chimaera_pool_list
```

### Expected Output

After running `chimaera_compose`, you should see output similar to:

```
Composing pools from: /path/to/cae_config.yaml
Creating pool: wrp_cae_core_pool (ID: 400.0)
Successfully created 1 pool(s)
```

Verify with `chimaera_pool_list`:

```
Pool ID    Pool Name              ChiMod
400.0      wrp_cae_core_pool      wrp_cae_core
```

## Pool Query Selection

The `pool_query` parameter determines how the container creation task is routed:

- **`dynamic`** (recommended for chimaera_compose): Checks local cache first, then broadcasts if needed
- **`local`**: Creates container only on the local node
- **`broadcast`**: Creates container across all nodes in the cluster

For `chimaera_compose`, always use `pool_query: dynamic`.

## Integration with CTE

CAE depends on CTE (Content Transformation Engine) for blob storage. Ensure CTE is configured before using CAE:

### CTE Configuration Example

```yaml
compose:
  # CTE must be configured first
  - mod_name: wrp_cte_core
    pool_name: wrp_cte_core
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: /tmp/cte_storage
        bdev_type: file
        capacity_limit: 10GB
    dpe:
      dpe_type: max_bw

  # CAE configuration
  - mod_name: wrp_cae_core
    pool_name: wrp_cae_core_pool
    pool_query: dynamic
    pool_id: "400.0"
    worker_count: 4
```

See [CTE Configuration Guide](../cte/config.md) for detailed CTE parameters.

## Troubleshooting

### Error: "Chimaera IPC not initialized"

**Cause**: Runtime is not running

**Solution**: Start the runtime first:
```bash
export WRP_RUNTIME_CONF=/etc/iowarp/config.yaml
chimaera_start_runtime &
sleep 2
```

### Error: "Pool already exists"

**Cause**: CAE container was already created with this pool ID

**Solution**: Either:
1. Use the existing container
2. Destroy the existing pool: `chimaera_pool_destroy 400.0`
3. Use a different pool ID in the configuration

### Error: "Module not found: wrp_cae_core"

**Cause**: CAE ChiMod library not installed or not in library path

**Solution**: Ensure CAE is installed and `LD_LIBRARY_PATH` includes the installation directory

## Next Steps

After launching CAE, you can:

1. **Process OMNI files**: Use `wrp_cae_omni` to ingest data (see [OMNI Documentation](omni.md))
2. **Submit assimilation tasks**: Use the CAE client API to schedule data transfers
3. **Monitor operations**: Check runtime logs and pool statistics

## Related Documentation

- [OMNI File Format](omni.md) - OMNI configuration and wrp_cae_omni utility usage
- [CTE Configuration](../cte/config.md) - CTE storage configuration
- [Chimaera Compose](../runtime/module_dev_guide.md) - Compose configuration format
- [Module Development Guide](../runtime/module_dev_guide.md) - ChiMod development

---

**Last Updated**: 2025-11-09
**Version**: 2.0.0
