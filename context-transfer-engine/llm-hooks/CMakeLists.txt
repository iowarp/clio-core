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

# weights/ links against 'ggml' (from external/llama.cpp) and uses CUDA separable
# compilation.  Only add it when llama.cpp is also being built to keep ggml target
# available for the device-link step; otherwise the CMake generate step aborts with
# "_CMAKE_CUDA_RDC_FLAG not set" because the unresolved 'ggml' target breaks CUDA
# property propagation.
# The GpuVMM weight-offload backend (ggml_iowarp_backend.cc) is stale against
# current llama.cpp's ggml-backend-impl.h vtables and is superseded by the
# gpu_vector read_range integration (GGML_CLIO_VECTOR in ggml-cuda). Off by
# default so it doesn't break the llama build; opt in with
# -DCLIO_CORE_ENABLE_LLM_WEIGHTS_GPUVMM=ON once its vtables are updated.
option(CLIO_CORE_ENABLE_LLM_WEIGHTS_GPUVMM "Build the legacy GpuVMM weight-offload backend" OFF)
if(CLIO_CORE_ENABLE_CUDA AND CLIO_CORE_ENABLE_LLAMA_SERVER AND CLIO_CORE_ENABLE_LLM_WEIGHTS_GPUVMM)
    add_subdirectory(weights)
else()
    message(STATUS "llm-hooks/weights: skipped (legacy GpuVMM backend disabled)")
endif()

# NOTE: Patching of server-context and llama targets with IOWARP_LLM_KVCACHE
# is done in the root CMakeLists.txt AFTER add_subdirectory(external/llama.cpp),
# because those targets do not exist yet when this file is processed.

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
