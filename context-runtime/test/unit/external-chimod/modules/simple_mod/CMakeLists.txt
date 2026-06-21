cmake_minimum_required(VERSION 3.10)

# Create client library for simple_mod
# Automatically reads namespace from clio_repo.yaml (searches up directory tree)
# This creates the following target:
#   - external_test_simple_mod_client:  Client-side API for user processes (shared library)
add_clio_module_client(
  SOURCES
    src/simple_mod_client.cc
)

# Create runtime library for simple_mod  
# This creates the following target:
#   - external_test_simple_mod_runtime: Server-side execution logic (shared library)
add_clio_module_runtime(
  SOURCES
    src/simple_mod_runtime.cc
    src/autogen/simple_mod_lib_exec.cc
)

# Note: For this test, we don't install the ChiMod libraries
# In a real external ChiMod project, you would add:
# install_chimod(CHIMOD_NAME simple_mod)