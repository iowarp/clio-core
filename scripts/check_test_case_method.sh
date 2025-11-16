#!/bin/bash
# Script to find all instances of TEST_CASE_METHOD in test files
# This helps ensure all TEST_CASE_METHOD instances are replaced with TEST_CASE + singleton pattern

echo "Searching for TEST_CASE_METHOD instances in test files..."
echo "============================================================"

# Find all instances of TEST_CASE_METHOD in .cc and .cpp files
grep -rn "TEST_CASE_METHOD" --include="*.cc" --include="*.cpp" /workspace

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "WARNING: Found TEST_CASE_METHOD instances. These should be replaced with TEST_CASE + singleton pattern."
    exit 1
else
    echo ""
    echo "SUCCESS: No TEST_CASE_METHOD instances found."
    exit 0
fi
