#!/bin/bash
#==============================================================================
# install_docker.sh - IOWarp Core Docker-based pip installation
#==============================================================================
# This script builds and starts the minimal Docker container, then performs
# a pip-based installation of IOWarp Core within a Python virtual environment.
#
# Usage:
#   ./install_docker.sh
#
# What this does:
#   1. Builds the minimal Docker image with source code
#   2. Starts a container and runs pip install inside it
#   3. Installs IOWarp Core to the container's virtual environment (/opt/venv)
#   4. Verifies the installation by checking for libraries and CMake configs
#   Note: Dependencies are automatically downloaded/built by install.sh if not found
#==============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

echo "======================================================================"
echo "IOWarp Core - Docker-based pip Installation"
echo "======================================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Verifying project root contents..."
ls -la "$PROJECT_ROOT" | head -20
echo ""

# Verify that critical files exist
if [ ! -f "$PROJECT_ROOT/setup.py" ]; then
    echo "ERROR: setup.py not found in $PROJECT_ROOT"
    echo "Please run this script from the docker/ directory"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/install.sh" ]; then
    echo "ERROR: install.sh not found in $PROJECT_ROOT"
    echo "Please run this script from the docker/ directory"
    exit 1
fi

echo "✓ Found setup.py and install.sh in project root"
echo ""

#------------------------------------------------------------------------------
# Build the minimal Docker image
#------------------------------------------------------------------------------
echo ">>> Building minimal Docker image..."
docker build -t iowarp/minimal:latest -f "${SCRIPT_DIR}/minimal.Dockerfile" "${PROJECT_ROOT}"
echo ""

#------------------------------------------------------------------------------
# Run container and perform pip installation
#------------------------------------------------------------------------------
echo "======================================================================"
echo ">>> Starting container and installing with pip..."
echo "======================================================================"
echo "Source is baked into the Docker image during build"
echo ""

docker run --rm \
    iowarp/minimal:latest /bin/bash -c "
set -e

echo '==================================================================='
echo 'Container Environment Information'
echo '==================================================================='
echo 'Working directory:' \$(pwd)
echo 'Source directory contents:'
ls -la /iowarp-core | head -15
echo ''

echo '==================================================================='
echo 'Step 0: Checking git repository'
echo '==================================================================='
# Configure git to trust the directory
git config --global --add safe.directory /iowarp-core 2>/dev/null || true

# Check if this is a git repository
if [ -d .git ]; then
    echo 'Git repository detected'
else
    echo 'Not a git repository'
fi

echo 'Note: Dependencies will be downloaded/built by install.sh if not found on system'

echo ''
echo '==================================================================='
echo 'Step 1: Verifying Python virtual environment'
echo '==================================================================='
echo 'Using pre-created virtual environment at /opt/venv'
echo ''
echo 'Python environment:'
which python3
python3 --version
which pip
pip --version
echo ''
echo 'Virtual environment is already activated via PATH'

echo ''
echo '==================================================================='
echo 'Step 2: Installing IOWarp Core with pip'
echo '==================================================================='
echo 'This will:'
echo '  - Run setup.py which calls install.sh'
echo '  - Download and build missing dependencies from GitHub releases'
echo '  - Build and install IOWarp Core libraries'
echo '  - Install to virtual environment at /opt/venv'
echo ''
echo 'Current directory:' \$(pwd)
echo 'Files in current directory:'
ls -la | head -20
echo ''

# pip install -v .
bash install.sh

echo ''
echo '==================================================================='
echo 'Step 3: Verifying installation'
echo '==================================================================='
echo 'Checking installed libraries in /opt/venv/lib:'
ls -lh /opt/venv/lib/libchimaera* 2>/dev/null || echo 'No chimaera libraries found'
ls -lh /opt/venv/lib/libhshm* 2>/dev/null || echo 'No hshm libraries found'
ls -lh /opt/venv/lib/libwrp* 2>/dev/null || echo 'No wrp libraries found'

echo ''
echo 'Checking CMake config files:'
ls -d /opt/venv/lib/cmake/*/ 2>/dev/null || echo 'No CMake config files found'

echo ''
echo '==================================================================='
echo '✓ Installation test complete!'
echo '==================================================================='
"

echo ""
echo "======================================================================"
echo "✓ pip installation test finished successfully!"
echo "======================================================================"
