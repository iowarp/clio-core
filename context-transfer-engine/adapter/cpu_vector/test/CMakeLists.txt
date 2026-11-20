#------------------------------------------------------------------------------
# cpu_vector tests: host-only, against an in-process runtime with a RAM tier.
#------------------------------------------------------------------------------

add_executable(test_cpu_vector test_cpu_vector.cc)
target_link_libraries(test_cpu_vector
    clio::cte::cpu_vector
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(test_cpu_vector PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test   # simple_test.h
)

add_test(NAME cte_cpu_vector COMMAND test_cpu_vector)
set_tests_properties(cte_cpu_vector PROPERTIES
    TIMEOUT 300
    LABELS "cpu_vector;cte"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

#------------------------------------------------------------------------------
# cpu_vector matrix: the storage path gpu_vector uses, exercised from the host.
# One ctest entry per (codec, tier layout); see test_cpu_vector_matrix.cc.
# Codecs are added only when the build found their library, so every entry
# that exists is expected to pass.
#------------------------------------------------------------------------------
add_executable(test_cpu_vector_matrix test_cpu_vector_matrix.cc)
target_link_libraries(test_cpu_vector_matrix
    clio::cte::cpu_vector
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(test_cpu_vector_matrix PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test   # simple_test.h
)

# Raw pages always; through the compressor module, raw plus every CPU codec
# the build found (ids from ctp's compress_factory.h).
set(_cvm_cells "raw:0:0")
if(TARGET clio_cte_compressor_runtime)
  add_dependencies(test_cpu_vector_matrix
      clio_cte_compressor_runtime clio_cte_compressor_client)
  list(APPEND _cvm_cells "raw_cmp:0:1")
  if(CLIO_CTP_ENABLE_BZIP2)
    list(APPEND _cvm_cells "bzip2:1:1")
  endif()
  if(CLIO_CTP_ENABLE_LZ4)
    list(APPEND _cvm_cells "lz4:4:1")
  endif()
  if(CLIO_CTP_ENABLE_ZSTD)
    list(APPEND _cvm_cells "zstd:10:1")
  endif()
endif()

set(_cvm_port 9471)
foreach(_cell IN LISTS _cvm_cells)
  string(REPLACE ":" ";" _parts "${_cell}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _codec)
  list(GET _parts 2 _cmp)
  foreach(_tiers single spill)
    set(_test cte_cpu_vector_matrix_${_name}_${_tiers})
    add_test(NAME ${_test} COMMAND test_cpu_vector_matrix)
    set_tests_properties(${_test} PROPERTIES
        TIMEOUT 300
        RUN_SERIAL TRUE
        LABELS "cpu_vector;cte;cpu_vector_matrix"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;CLIO_SERVER_ADDR=127.0.0.1;CLIO_CVM_CODEC=${_codec};CLIO_CVM_TIERS=${_tiers};CLIO_CVM_COMPRESSOR=${_cmp};CLIO_CVM_PORT=${_cvm_port}")
    math(EXPR _cvm_port "${_cvm_port} + 4")
  endforeach()
endforeach()
