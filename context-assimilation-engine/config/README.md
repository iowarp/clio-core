# Context Assimilation Engine (CAE) Configuration

This directory contains example configuration files for deploying the Context Assimilation Engine (CAE) using Chimaera compose.

## Configuration Files

### wrp_config_example.yaml

Complete example showing how to configure both CTE and CAE ChiMods together in a unified runtime configuration.

## CAE ChiMod Configuration

### Pool Constants

The CAE pool uses these constants (defined in `wrp_cae/core/constants.h`):
- **Pool ID**: `400.0` (kCaePoolId)
- **Pool Name**: User-defined (e.g., "cae_main")
- **Module Name**: `wrp_cae_core`

### Configuration Parameters

```yaml
compose:
  - mod_name: wrp_cae_core      # CAE Core ChiMod library name
    pool_name: cae_main          # User-defined pool name
    pool_query: local            # Pool query type (local, broadcast, dynamic)
    pool_id: "400.0"             # CAE pool ID (must match kCaePoolId)
```

### Parameters Explanation

- **mod_name**: Must be `wrp_cae_core` (the CAE Core ChiMod library)
- **pool_name**: User-defined name for the CAE pool
- **pool_query**:
  - `local`: Create pool only on local node
  - `broadcast`: Create pool on all nodes
  - `dynamic`: Let the runtime decide based on existing pools
- **pool_id**: Must be `"400.0"` to match the constant defined in code

## Usage with chimaera compose

```bash
# Start the Chimaera runtime
chimaera runtime start

# Deploy CAE using the configuration file
chimaera compose /path/to/wrp_config_example.yaml

# Now CAE is available for use
wrp_cae_omni /path/to/omni_file.yaml
```

## Integration with CTE

CAE typically works alongside CTE (Context Transfer Engine) to enable data assimilation workflows:

1. **CTE** provides distributed storage and data management
2. **CAE** assimilates data from external sources into CTE

See `wrp_config_example.yaml` for a complete configuration showing both ChiMods working together.

## Pool ID Reference

| Component | Pool ID | Constant Name | Module Name |
|-----------|---------|---------------|-------------|
| Admin     | 1.0     | kAdminPoolId  | chimaera_admin |
| CTE Core  | 512.0   | kCtePoolId    | wrp_cte_core |
| CAE Core  | 400.0   | kCaePoolId    | wrp_cae_core |

## Example Workflow

```bash
# 1. Start runtime
export CHI_SERVER_CONF=/path/to/wrp_config_example.yaml
chimaera runtime start &

# 2. Deploy CTE and CAE
chimaera compose $CHI_SERVER_CONF

# 3. Use CAE to ingest data
wrp_cae_omni /path/to/my_data_transfer.yaml

# 4. Access data through CTE
# (Your application uses CTE client to access assimilated data)
```
