# ------------------------------------------------------------------------------
# Replay simulation field dumps through Clio's compressor (issue #693).
#
# The sibling LAMMPS example embeds a live simulation; this one reads flat
# binary field files a simulation already wrote. That is the shape upstream
# NeuroPress uses for its Nyx benchmark -- patch the application to dump raw
# float32 fields, then sweep them offline -- and it is the only practical shape
# for codes with no library interface to embed.
#
# Plain C++ against the Clio client API: no simulation dependency at all, so it
# builds anywhere the compressor does.
# ------------------------------------------------------------------------------

if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "NeuroPress field replay example: skipped (CUDA off)")
  return()
endif()

message(STATUS "NeuroPress field replay example: enabled")

add_executable(neuropress_field_replay neuropress_field_replay.cc)

target_link_libraries(neuropress_field_replay
  clio_cte_core_client
  clio_cte_core_runtime
  clio_cte_compressor_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS neuropress_field_replay RUNTIME DESTINATION bin)
