#!/bin/bash

# Script to build and install the iowarp-core conda package locally
# This script builds IOWarp Core from your current local source code

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RECIPE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IOWarp Core - Local Conda Package Build${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_PREFIX" ]; then
    echo -e "${RED}Error: No conda environment detected${NC}"
    echo "Please activate a conda environment first:"
    echo "  conda create -n iowarp-build"
    echo "  conda activate iowarp-build"
    exit 1
fi

echo -e "${GREEN}Conda environment detected: $CONDA_PREFIX${NC}"
echo ""

# Check if conda-build is installed
if ! command -v conda-build &> /dev/null; then
    echo -e "${YELLOW}conda-build is not installed. Installing...${NC}"
    conda install -y conda-build
    echo ""
fi

# Navigate to repository root
cd "$REPO_ROOT"

# Check if submodules are initialized
echo -e "${BLUE}Checking git submodules...${NC}"
if [ ! -f "context-transport-primitives/CMakeLists.txt" ]; then
    echo -e "${YELLOW}Submodules not initialized. Initializing now...${NC}"
    git submodule update --init --recursive
    echo -e "${GREEN}Submodules initialized successfully${NC}"
else
    echo -e "${GREEN}Submodules are initialized${NC}"
fi
echo ""

# Show build configuration
echo -e "${BLUE}Build Configuration:${NC}"
echo "  Recipe directory:  $RECIPE_DIR"
echo "  Source directory:  $REPO_ROOT"
echo "  Conda environment: $CONDA_PREFIX"
echo ""

# Build the package with conda-forge channel
echo -e "${BLUE}Building conda package...${NC}"
echo -e "${YELLOW}This may take 10-30 minutes depending on your system${NC}"
echo ""

if conda build "$RECIPE_DIR" -c conda-forge; then
    BUILD_SUCCESS=true
else
    BUILD_SUCCESS=false
fi

echo ""

if [ "$BUILD_SUCCESS" = true ]; then
    # Get the output package path
    PACKAGE_PATH=$(conda build "$RECIPE_DIR" --output 2>/dev/null)

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Package built successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Package location:${NC}"
    echo "  $PACKAGE_PATH"
    echo ""

    # Offer installation options
    echo -e "${BLUE}Installation Options:${NC}"
    echo ""
    echo -e "${YELLOW}Option 1: Install using --use-local${NC}"
    echo "  conda install --use-local iowarp-core"
    echo ""
    echo -e "${YELLOW}Option 2: Install directly from package file${NC}"
    echo "  conda install \"$PACKAGE_PATH\""
    echo ""
    echo -e "${YELLOW}Option 3: Create a new environment and install${NC}"
    echo "  conda create -n iowarp-env iowarp-core --use-local -c conda-forge"
    echo "  conda activate iowarp-env"
    echo ""

    # Ask if user wants to install now
    read -p "Would you like to install the package now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${BLUE}Installing iowarp-core...${NC}"
        if conda install --use-local iowarp-core -y; then
            echo ""
            echo -e "${GREEN}Installation successful!${NC}"
            echo ""
            echo -e "${BLUE}Verify installation:${NC}"
            echo "  conda list iowarp-core"
            echo ""
            echo -e "${BLUE}Test the installation:${NC}"
            echo "  chimaera_start_runtime --help"
            echo "  wrp_cte --help"
            echo ""
        else
            echo ""
            echo -e "${RED}Installation failed. You can try installing manually:${NC}"
            echo "  conda install --use-local iowarp-core"
        fi
    else
        echo ""
        echo -e "${YELLOW}Skipping installation. You can install later using:${NC}"
        echo "  conda install --use-local iowarp-core"
    fi

    echo ""
    echo -e "${BLUE}Additional Information:${NC}"
    echo "  Documentation: $REPO_ROOT/conda/README.md"
    echo "  List packages: conda list iowarp-core"
    echo "  Remove package: conda remove iowarp-core"
    echo ""

else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo ""
    echo "1. Check that submodules are initialized:"
    echo "   git submodule update --init --recursive"
    echo ""
    echo "2. Verify conda-forge channel is configured:"
    echo "   conda config --show channels"
    echo ""
    echo "3. Try building with verbose output:"
    echo "   conda build conda/ -c conda-forge --debug"
    echo ""
    echo "4. Check build logs in:"
    echo "   $CONDA_PREFIX/conda-bld/"
    echo ""
    echo "For more help, see: $REPO_ROOT/conda/README.md"
    echo ""
    exit 1
fi
