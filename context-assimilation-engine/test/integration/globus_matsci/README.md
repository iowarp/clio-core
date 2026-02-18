# Globus Materials Science Integration Test

This integration test demonstrates using CAE (Content Assimilation Engine) to transfer data from a Globus endpoint to the local filesystem.

## Overview

The test uses the SEM_103 materials science dataset hosted on Globus:
- **Endpoint ID**: `e8cf0e9a-f96a-11ed-9a83-83ef71fbf0ae`
- **Dataset Path**: `/SEM_103/`
- **Web Interface**: https://app.globus.org/file-manager?origin_id=e8cf0e9a-f96a-11ed-9a83-83ef71fbf0ae&origin_path=%2FSEM_103%2F

## Prerequisites

1. **Globus Access Token**: You must have a valid Globus access token
2. **POCO Libraries**: Required for HTTP/HTTPS requests (should already be installed)
3. **Build with Globus Support**: CAE must be built with `CAE_ENABLE_GLOBUS=ON`

## Obtaining a Globus Access Token

1. Visit: https://app.globus.org/settings/developers
2. Create a new app or use an existing one
3. Generate a new access token with appropriate scopes:
   - `urn:globus:auth:scope:transfer.api.globus.org:all` (for transfer operations)
4. Copy the token and export it as an environment variable

## Running the Test

```bash
# Set your Globus access token
export GLOBUS_ACCESS_TOKEN="your_token_here"

# Navigate to the test directory
cd test/integration/globus_matsci

# Run the test script
./run_test.sh
```

## What the Test Does

The `run_test.sh` script performs the following steps:

1. **Start Chimaera Runtime**: Launches the Chimaera runtime in the background
2. **Launch CTE**: Starts the Content Transfer Engine
3. **Launch CAE**: Starts the Content Assimilation Engine with Globus support
4. **Process OMNI File**: Uses `wrp_cae_omni` to process the transfer requests
5. **Cleanup**: Stops the Chimaera runtime

## Test Files

- **`matsci_transfer.omni`**: OMNI configuration file defining the Globus-to-local transfers
- **`wrp_conf.yaml`**: CTE configuration (RAM-only storage for testing)
- **`run_test.sh`**: Bash script to orchestrate the test
- **`README.md`**: This file

## OMNI File Format

The OMNI file uses the following format for Globus transfers:

```yaml
name: globus_matsci_transfer

transfers:
  - src: "globus://<endpoint_id>/<path>"
    dst: "file::/local/path/to/destination"
    format: "binary"
    src_token: "${GLOBUS_ACCESS_TOKEN}"
```

The `src_token` parameter can contain environment variables using `${VAR_NAME}` syntax. These will be expanded when the OMNI file is loaded. Alternatively, you can omit `src_token` and the system will fall back to the `GLOBUS_ACCESS_TOKEN` environment variable.

### URI Formats

- **Globus Source**: `globus://<endpoint-id>/<path>`
  - Example: `globus://e8cf0e9a-f96a-11ed-9a83-83ef71fbf0ae/SEM_103/sample_data.tif`

- **Local Destination**: `file::<absolute-path>`
  - Example: `file::/tmp/globus_matsci/sample_data.tif`

## Output

Transferred files will be stored in `/tmp/globus_matsci/` by default. You can modify the destination paths in the OMNI file.

## Customizing the Test

To transfer different files from the Globus dataset:

1. Browse the dataset at the web interface URL above
2. Note the file paths you want to transfer
3. Edit `matsci_transfer.omni` and add/modify transfer entries
4. Ensure destination directories exist or will be created

## Troubleshooting

### "GLOBUS_ACCESS_TOKEN environment variable not set"
- Make sure you've exported the token: `export GLOBUS_ACCESS_TOKEN="your_token"`

### "Globus to local filesystem not yet fully implemented"
- The current implementation supports Globus-to-Globus transfers
- Local filesystem support is planned (see `globus_file_assimilator.cc:78`)

### "Failed to launch CTE/CAE"
- Ensure the executables are installed and available in your PATH
- Check that `WRP_CTE_CONF` points to a valid configuration file
- Verify installation with: `which chimaera runtime start wrp_launch_cte wrp_launch_cae wrp_cae_omni`

### Transfer Timeout
- Large files may take longer to transfer
- The default timeout is 5 minutes (30 polls × 10 seconds)
- Adjust in `globus_file_assimilator.cc:391-392` if needed

## Implementation Notes

The Globus file assimilator uses:
- **POCO Net/JSON**: For HTTPS requests to Globus Transfer API
- **nlohmann_json**: For parsing API responses
- **Globus Transfer API v0.10**: For transfer operations

See `core/src/factory/globus_file_assimilator.cc` for implementation details.
