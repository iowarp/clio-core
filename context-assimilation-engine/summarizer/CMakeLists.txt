cmake_minimum_required(VERSION 3.10)
project(clio_cae_summarizer)

#------------------------------------------------------------------------------
# Summarizer chimod — the transparent LLM summarization layer split out of
# clio_cae_core. An interposer on the CTE core's task interface; built via the
# standard ChiMod build functions.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# CAE_SUMMARIZATION_ENABLED (libcurl + nlohmann/json → OllamaGenerate) is
# detected in ../CMakeLists.txt so both this module and the unit-test tree,
# which is added from the sibling core/ scope, can see it. When it is FALSE,
# label_client compiles as a stub that returns false: the module still builds
# and forwards, just without summarization.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Client Library (declared first so the runtime auto-links against it)
#------------------------------------------------------------------------------
add_clio_module_client(
  SOURCES
    src/summarizer_client.cc
  LINK_LIBRARIES
    clio_cte_core_client
)

#------------------------------------------------------------------------------
# Runtime Library
#------------------------------------------------------------------------------
set(SUMMARIZER_RUNTIME_LINK_LIBRARIES
  clio_cte_core_client
)

if(CAE_SUMMARIZATION_ENABLED)
  add_compile_definitions(CLIO_CAE_ENABLE_SUMMARIZATION)
  list(APPEND SUMMARIZER_RUNTIME_LINK_LIBRARIES CURL::libcurl)
endif()

add_clio_module_runtime(
  SOURCES
    src/summarizer_runtime.cc
    src/label_client.cc
    src/autogen/summarizer_lib_exec.cc
  LINK_LIBRARIES
    ${SUMMARIZER_RUNTIME_LINK_LIBRARIES}
)

# nlohmann_json is PRIVATE: it may be a FetchContent target (non-imported),
# and PUBLIC linking would require it to be in the install export set.
if(CAE_SUMMARIZATION_ENABLED)
  target_link_libraries(clio_cae_summarizer_runtime PRIVATE nlohmann_json::nlohmann_json)
endif()

# The summarizer drives the pool below it through the CTE core's public client
# and interposes with clio_cte/core/core_interposer.h.
target_include_directories(clio_cae_summarizer_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cae_summarizer_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
