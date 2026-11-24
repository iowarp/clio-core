#!/bin/bash
#
# Wrapper script to run IOWarp Gray-Scott with conda libraries
#
# WHY THIS IS NEEDED:
# ADIOS2 (used by gray-scott) depends on conda's libcurl, which requires
# OpenSSL 3.2.0. System OpenSSL is 3.0.x. Even though gray-scott has RPATH
# configured, RPATH only affects direct dependencies, not transitive ones.
# When libcurl (loaded by ADIOS2) needs libssl, the dynamic linker searches:
#   1. LD_LIBRARY_PATH (may contain system paths)
#   2. libcurl's RPATH ($ORIGIN/.)
#   3. System default paths
# If LD_LIBRARY_PATH has system paths, system libssl is found first, causing
# version mismatch errors. This script prepends conda's lib to fix that.
#

# Detect conda installation
CONDA_PREFIX="${CONDA_PREFIX:-/home/iowarp/miniconda3}"

if [ ! -d "$CONDA_PREFIX/lib" ]; then
    echo "Error: Conda installation not found at $CONDA_PREFIX"
    echo "Set CONDA_PREFIX environment variable to your conda installation path"
    exit 1
fi

# Prepend conda lib to LD_LIBRARY_PATH (must come before system paths)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Execute the command with all arguments
exec "$@"
