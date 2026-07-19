# CLIO Runtime Modules (ChiMods) Directory
# This directory contains standard CLIO Runtime modules

cmake_minimum_required(VERSION 3.10)

message(STATUS "Building Clio modules (ChiMods)...")

# Add individual module subdirectories
add_subdirectory(MOD_NAME)  # Test module for Wait functionality

# Add admin module (critical system module)
add_subdirectory(admin)

# Add bdev module (block device operations)
add_subdirectory(bdev)

# Add safe-bdev module (declustered, erasure-coded block device)
add_subdirectory(safe-bdev)

# Additional modules can be added here as they are developed
# add_subdirectory(other_module_name)