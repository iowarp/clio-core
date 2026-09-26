cmake_minimum_required(VERSION 3.20)

#------------------------------------------------------------------------------
# llm-hooks: IOWarp integration for LLM inference with llama.cpp
#
# Two integration points:
#   kvcache/  - KV cache tiered offloading via CTE (MOONCAKE-style prefix caching)
#               Standalone: only needs CLIO_CORE_ENABLE_CTE=ON.
#   weights/  - Model weight offloading via GpuVMM (FlexGen-style demand paging)
#               Requires CUDA + llama.cpp (CLIO_CORE_ENABLE_LLAMA_SERVER=ON) because
#               it links against the 'ggml' target from external/llama.cpp.
#               CUDA separable-compilation device-link fails during CMake generate
#               when 'ggml' is absent, so we guard this carefully.
#------------------------------------------------------------------------------

add_subdirectory(kvcache)
add_subdirectory(lmcache)

# weights/ links against 'ggml' (from external/llama.cpp) and uses CUDA separable
# compilation.  Only add it when llama.cpp is also being built to keep ggml target
# available for the device-link step; otherwise the CMake generate step aborts with
# "_CMAKE_CUDA_RDC_FLAG not set" because the unresolved 'ggml' target breaks CUDA
# property propagation.
# weights/ registers a custom ggml BACKEND, so it is pinned to ggml's
# ggml_backend_i / ggml_backend_buffer_i layout. Upstream reshaped those
# structs, so its positional initializers now bind the wrong slots and it does
# not compile (invalid conversions at ggml_iowarp_backend.cc:92-105, 757-765).
# The switch exists so an otherwise-good tree can still be built while that is
# repaired; it defaults ON so nobody silently loses the backend.
option(CLIO_CORE_ENABLE_LLAMA_WEIGHTS_HOOK
       "Build the llm-hooks custom ggml weight backend (stale vs current ggml)" ON)
if(CLIO_CORE_ENABLE_CUDA AND CLIO_CORE_ENABLE_LLAMA_SERVER
   AND CLIO_CORE_ENABLE_LLAMA_WEIGHTS_HOOK)
    add_subdirectory(weights)
else()
    message(STATUS "llm-hooks/weights: skipped (requires CUDA + CLIO_CORE_ENABLE_LLAMA_SERVER=ON)")
endif()

# NOTE: Patching of server-context and llama targets with IOWARP_LLM_KVCACHE
# is done in the root CMakeLists.txt AFTER add_subdirectory(external/llama.cpp),
# because those targets do not exist yet when this file is processed.

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
