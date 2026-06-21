# Context Assimilation Engine (CAE) Configuration

This directory contains example configuration files for deploying the Context Assimilation Engine (CAE) using Clio compose.

## Configuration Files

### clio_config_example.yaml

Complete example showing how to configure both CTE and CAE ChiMods together in a unified runtime configuration.

## CAE Module Configuration

### Pool Constants

The CAE pool uses these constants (defined in `clio_cae/core/constants.h`):
- **Pool ID**: `400.0` (kCaePoolId)
- **Pool Name**: User-defined (e.g., "cae_main")
- **Module Name**: `clio_cae_core`

### Configuration Parameters

```yaml
compose:
  - mod_name: clio_cae_core      # CAE Core Module library name
    pool_name: cae_main          # User-defined pool name
    pool_query: local            # Pool query type (local, broadcast, dynamic)
    pool_id: "400.0"             # CAE pool ID (must match kCaePoolId)
```

### Parameters Explanation

- **mod_name**: Must be `clio_cae_core` (the CAE Core Module library)
- **pool_name**: User-defined name for the CAE pool
- **pool_query**:
  - `local`: Create pool only on local node
  - `broadcast`: Create pool on all nodes
  - `dynamic`: Let the runtime decide based on existing pools
- **pool_id**: Must be `"400.0"` to match the constant defined in code

## Usage with clio_run compose

```bash
# Start the Clio runtime
clio_run runtime start

# Deploy CAE using the configuration file
clio_run compose /path/to/clio_config_example.yaml

# Now CAE is available for use
clio_cae /path/to/omni_file.yaml
```

## Integration with CTE

CAE typically works alongside CTE (Context Transfer Engine) to enable data assimilation workflows:

1. **CTE** provides distributed storage and data management
2. **CAE** assimilates data from external sources into CTE

See `clio_config_example.yaml` for a complete configuration showing both ChiMods working together.

## Pool ID Reference

| Component | Pool ID | Constant Name | Module Name |
|-----------|---------|---------------|-------------|
| Admin     | 1.0     | kAdminPoolId  | clio_run_admin |
| CTE Core  | 512.0   | kCtePoolId    | clio_cte_core |
| CAE Core  | 400.0   | kCaePoolId    | clio_cae_core |

## Example Workflow

```bash
# 1. Start runtime
export CLIO_X=/path/to/clio_config_example.yaml
clio_run runtime start &

# 2. Deploy CTE and CAE
clio_run compose $CLIO_X

# 3. Use CAE to ingest data
clio_cae /path/to/my_data_transfer.yaml

# 4. Access data through CTE
# (Your application uses CTE client to access assimilated data)
```
