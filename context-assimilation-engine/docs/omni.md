# OMNI File Format Documentation

## Overview

OMNI (Object Migration and Negotiation Interface) is a YAML-based configuration format used by the Content Assimilation Engine (CAE) to describe data transfer operations. An OMNI file specifies one or more data transfers from source locations to destinations, with support for various formats, dependencies, and partial transfers.

## File Structure

An OMNI file is a YAML document with the following structure:

```yaml
name: <job_name>

transfers:
  - src: <source_uri>
    dst: <destination_uri>
    format: <format_type>
    depends_on: <dependency>    # optional
    range_off: <offset>          # optional
    range_size: <size>           # optional
  - src: <source_uri_2>
    dst: <destination_uri_2>
    format: <format_type_2>
    # ... more transfers
```

## Top-Level Fields

### `name` (string, required)
The name or identifier for this OMNI job. Used for logging and tracking purposes.

**Example:**
```yaml
name: terra_satellite_ingest
```

### `transfers` (array, required)
An array of transfer specifications. Each transfer describes a single data movement operation from a source to a destination.

## Transfer Fields

Each entry in the `transfers` array has the following fields:

### `src` (string, required)
The source URI specifying where to read data from. The URI format determines the source type and parser.

**Supported URI schemes:**
- `file::<path>` - Local filesystem
- `hdf5://<path>:<dataset>` - HDF5 file and dataset
- `globus://<endpoint_id>/<path>` - Globus endpoint
- `s3://<bucket>/<key>` - S3 object storage

**Examples:**
```yaml
src: file::/data/satellite/TERRA_2024.bin
src: hdf5::/data/climate.h5:/temperature/surface
src: globus://82f1b5c6-6e9b-11e5-ba47-22000b92c6ec/dataset.nc
```

### `dst` (string, required)
The destination URI specifying where to write data. Currently supports the `iowarp` scheme for CTE tag-based storage.

**Format:**
- `iowarp::<tag_name>` - Store in CTE with specified tag

**Examples:**
```yaml
dst: iowarp::terra_2024_satellite_data
dst: iowarp::climate_temperature_2024
```

### `format` (string, required)
The data format type. This determines which assimilator (parser/processor) will handle the data.

**Supported formats:**
- `binary` - Raw binary data
- `hdf5` - HDF5 hierarchical data format
- Additional formats may be registered via the AssimilatorFactory

**Examples:**
```yaml
format: binary
format: hdf5
```

### `depends_on` (string, optional)
Specifies a dependency on another transfer. The current transfer will only execute after the specified dependency completes successfully.

**Default:** `""` (empty string, no dependency)

**Examples:**
```yaml
depends_on: ""                           # No dependency
depends_on: previous_transfer_tag        # Wait for this transfer
```

### `range_off` (integer, optional)
Byte offset from the start of the source file. Used for partial file transfers.

**Default:** `0` (start from beginning)

**Examples:**
```yaml
range_off: 0           # Start from beginning
range_off: 1048576     # Start from 1MB offset
range_off: 2097152     # Start from 2MB offset
```

### `range_size` (integer, optional)
Number of bytes to read from the source. When set to `0`, reads the entire file (or remaining portion after `range_off`).

**Default:** `0` (read entire file)

**Examples:**
```yaml
range_size: 0           # Read entire file
range_size: 1048576     # Read 1MB
range_size: 10485760    # Read 10MB
```

## Complete Examples

### Example 1: Simple Binary File Transfer

```yaml
name: simple_binary_transfer

transfers:
  - src: file::/tmp/data.bin
    dst: iowarp::my_binary_data
    format: binary
```

### Example 2: HDF5 Dataset Transfer

```yaml
name: climate_data_ingest

transfers:
  - src: hdf5::/data/climate/NOAA_2024.h5:/temperature/surface
    dst: iowarp::noaa_surface_temp_2024
    format: hdf5
```

### Example 3: Partial File Transfer with Range

```yaml
name: partial_transfer_test

transfers:
  - src: file::/large_dataset/file.bin
    dst: iowarp::chunk_1
    format: binary
    range_off: 0
    range_size: 1048576    # First 1MB
  - src: file::/large_dataset/file.bin
    dst: iowarp::chunk_2
    format: binary
    range_off: 1048576
    range_size: 1048576    # Second 1MB
```

### Example 4: Multiple Transfers with Dependencies

```yaml
name: multi_stage_pipeline

transfers:
  - src: file::/raw/input.bin
    dst: iowarp::stage1_output
    format: binary
    depends_on: ""
  - src: file::/processed/data.bin
    dst: iowarp::stage2_output
    format: binary
    depends_on: stage1_output
```

### Example 5: Globus Transfer

```yaml
name: globus_satellite_data

transfers:
  - src: globus://82f1b5c6-6e9b-11e5-ba47-22000b92c6ec/satellite/TERRA_2024.h5
    dst: iowarp::terra_2024_archive
    format: hdf5
```

## Using OMNI Files

### Using the wrp_cae_omni Utility

The `wrp_cae_omni` utility is the primary tool for processing OMNI files. It loads the OMNI configuration and schedules assimilation tasks with the CAE runtime.

#### Prerequisites

1. **Chimaera runtime must be running**
2. **CAE container must be created** using `chimaera_compose` (see [Launch Guide](launch.md))
3. **CTE container must be configured** for blob storage

#### Basic Usage

```bash
wrp_cae_omni <omni_file_path>
```

**Example:**
```bash
wrp_cae_omni /path/to/transfer_config.yaml
```

#### Complete Workflow

```bash
# 1. Start runtime
export WRP_RUNTIME_CONF=/etc/iowarp/config.yaml
chimaera_start_runtime &
sleep 2

# 2. Create CAE container (if not already created)
chimaera_compose /path/to/cae_config.yaml

# 3. Process OMNI file
wrp_cae_omni /path/to/omni_file.yaml
```

#### Expected Output

```
Loading OMNI file: /path/to/omni_file.yaml
  Loaded transfer 1/2:
    src: file::/data/input.bin
    dst: iowarp::my_data_tag
    format: binary
  Loaded transfer 2/2:
    src: hdf5::/data/climate.h5:/temperature
    dst: iowarp::climate_temp
    format: hdf5
Successfully loaded 2 transfer(s) from OMNI file
Calling ParseOmni...
ParseOmni completed successfully!
  Tasks scheduled: 2
```

#### Command-Line Options

| Argument | Description |
|----------|-------------|
| `omni_file_path` | Path to the OMNI YAML configuration file |

#### Return Codes

| Code | Description |
|------|-------------|
| `0` | Success - all transfers scheduled |
| `1` | Error - see stderr for details |

#### Common Errors

**Error: "Chimaera IPC not initialized. Is the runtime running?"**
- **Cause**: Runtime not started
- **Solution**: Start runtime with `chimaera_start_runtime`

**Error: "Failed to load OMNI file"**
- **Cause**: Invalid YAML syntax or missing file
- **Solution**: Verify file exists and has valid YAML syntax

**Error: "ParseOmni failed with result code N"**
- **Cause**: CAE runtime encountered an error processing transfers
- **Solution**: Check runtime logs for details

### Loading OMNI Files in C++ Code

For programmatic access, use the `LoadOmni` function to parse an OMNI file:

```cpp
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <yaml-cpp/yaml.h>
#include <vector>

std::vector<wrp_cae::core::AssimilationCtx> LoadOmni(const std::string& omni_path);

// Usage
try {
  auto contexts = LoadOmni("/path/to/config.yaml");
  // Pass to ParseOmni
  cae_client.ParseOmni(contexts, num_tasks_scheduled);
} catch (const std::exception& e) {
  std::cerr << "Failed to load OMNI: " << e.what() << std::endl;
}
```

### Processing OMNI Files

The CAE runtime processes OMNI files through the following pipeline:

1. **Parse YAML** - Load and validate the OMNI file structure
2. **Create Contexts** - Convert each transfer into an `AssimilationCtx` object
3. **Serialize** - Transparently serialize contexts using cereal
4. **Submit Task** - Send `ParseOmniTask` to the CAE runtime
5. **Deserialize** - Runtime deserializes the contexts vector
6. **Schedule** - For each context:
   - Determine assimilator based on source URI scheme
   - Schedule the transfer operation
   - Handle dependencies and ordering

## URI Scheme Reference

### File URIs: `file::<path>`
- **Description**: Local filesystem access
- **Path**: Absolute or relative file path
- **Example**: `file::/tmp/data.bin`

### HDF5 URIs: `hdf5://<file_path>:<dataset_path>`
- **Description**: HDF5 hierarchical data format
- **File Path**: Path to the .h5 file
- **Dataset Path**: Internal HDF5 dataset path
- **Example**: `hdf5::/data/climate.h5:/temperature/surface`

### Globus URIs: `globus://<endpoint_id>/<path>`
- **Description**: Globus data transfer service
- **Endpoint ID**: UUID of the Globus endpoint
- **Path**: File path on the endpoint
- **Example**: `globus://82f1b5c6-6e9b-11e5-ba47-22000b92c6ec/data/file.nc`

### IOWarp URIs: `iowarp::<tag_name>`
- **Description**: CTE tag-based storage
- **Tag Name**: Unique identifier for the data in CTE
- **Example**: `iowarp::my_data_tag_2024`

## Validation Rules

The `LoadOmni` function validates:

1. **File exists and is valid YAML**
2. **Required field `transfers` is present and is an array**
3. **Each transfer has required fields**: `src`, `dst`, `format`
4. **Field types are correct**: strings for URIs, integers for offsets/sizes
5. **Optional fields have valid defaults**

## Error Handling

Common errors and their meanings:

| Error | Description | Solution |
|-------|-------------|----------|
| `OMNI file missing required 'transfers' key` | No `transfers` array in YAML | Add `transfers:` section |
| `Transfer N missing required 'src' field` | Missing source URI | Add `src:` to transfer N |
| `Transfer N missing required 'dst' field` | Missing destination URI | Add `dst:` to transfer N |
| `Transfer N missing required 'format' field` | Missing format specification | Add `format:` to transfer N |
| `Failed to load OMNI file: <error>` | YAML parsing error | Check YAML syntax |

## Best Practices

1. **Use descriptive names**: Choose meaningful job and tag names
2. **Specify dependencies**: Use `depends_on` for ordered transfers
3. **Handle ranges carefully**: Ensure `range_off + range_size` doesn't exceed file size
4. **Validate URIs**: Ensure source files exist before submitting
5. **Use consistent formats**: Match the `format` field to the actual data type
6. **Document transfers**: Add YAML comments to explain complex configurations

## Future Extensions

Planned enhancements to the OMNI format:

- **Metadata fields**: Custom key-value pairs for transfer metadata
- **Compression options**: Specify compression algorithms
- **Retry policies**: Configure retry behavior for failed transfers
- **Notifications**: Callbacks or webhooks on completion
- **Transforms**: Data transformation pipelines

## Related Documentation

- [CAE Launch Guide](launch.md) - How to launch CAE using chimaera_compose
- [CTE Configuration](../cte/config.md) - CTE storage configuration
- [Chimaera Compose](../runtime/module_dev_guide.md) - Compose configuration format
- [Module Development Guide](../runtime/module_dev_guide.md) - ChiMod development

---

**Last Updated**: 2025-11-09
**Version**: 2.0.0
