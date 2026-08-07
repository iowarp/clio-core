cmake_minimum_required(VERSION 3.20)

# clio_llm_kvcache — KV cache tiered offloading library.
# Links against the CTE core client to use PutBlob/GetBlob for tiered storage.

set(KVCACHE_SOURCES
    src/kvcache_manager.cc
)

add_library(clio_llm_kvcache SHARED ${KVCACHE_SOURCES})

# On Windows a SHARED lib with no dllexport'd symbols produces no import .lib,
# so anything linking it (llama, patched with IOWARP_LLM_KVCACHE) fails with
# LNK1181. Auto-export all symbols so the import lib is generated.
set_target_properties(clio_llm_kvcache PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(clio_llm_kvcache
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# CTE core client provides PutBlob/GetBlob/GetOrCreateTag etc.
# It also brings in the CUDA runtime transitively when CUDA is enabled,
# so we do NOT add an explicit 'cuda' link here (avoids CUDA RDC linker
# errors from mixing CUDA-separable and non-CUDA targets).
target_link_libraries(clio_llm_kvcache
    PUBLIC
        clio_cte_core_client
)

# Enable D2H/H2D staging-copy code paths inside kvcache_manager.cc.
# Also provide CUDA toolkit include dirs (cuda_runtime.h) for that code.
if(CLIO_CORE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(clio_llm_kvcache PRIVATE IOWARP_LLM_ENABLE_CUDA)
    target_link_libraries(clio_llm_kvcache PRIVATE CUDA::cudart)
endif()

install(TARGETS clio_llm_kvcache
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
