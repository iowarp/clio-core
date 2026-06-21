#!/bin/bash

# Script to compile the external-chimod test with proper environment setup
# This script loads the required modules and compiles the external ChiMod development test

set -e  # Exit on any error

echo "Setting up environment for external-chimod test compilation..."

# Load required modules
echo "Loading iowarp-runtime module..."
module load iowarp-runtime || {
    echo "Warning: Failed to load iowarp-runtime module - continuing anyway"
}

echo "Loading iowarp spack module..."
spack load iowarp || {
    echo "Warning: Failed to load iowarp spack module - continuing anyway"
}

echo "Environment setup complete."
echo ""

# Show current directory
echo "Current directory: $(pwd)"
echo ""

# Create build directory and compile
echo "Creating build directory..."
rm -rf build
mkdir -p build
cd build

echo "Configuring CMake for external ChiMod development test..."
cmake .. || {
    echo "CMake configuration failed!"
    exit 1
}

echo "Building external ChiMod test (includes custom simple_mod ChiMod)..."
make || {
    echo "Build failed!"
    exit 1
}

echo ""
echo "Build completed successfully!"
echo "Test executable created: $(pwd)/test_simple_mod"
echo ""

# Optionally run the test
echo "Running external ChiMod development test..."
./test_simple_mod || {
    echo "Test execution failed!"
    exit 1
}

echo ""
echo "External ChiMod development test compilation and execution completed successfully!"
echo ""
echo "This test demonstrates:"
echo "✓ Custom namespace (external_test vs clio_run)"
echo "✓ External clio_repo.yaml configuration" 
echo "✓ add_clio_module_client() and add_clio_module_runtime() CMake functions"
echo "✓ External module directory structure"
echo "✓ Proper find_package(clio_run) linking"