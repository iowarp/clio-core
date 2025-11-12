# FindHSHMTarget.cmake
# Helper to determine which HermesShm target to use (handles both installed and subdirectory modes)

if(NOT DEFINED HSHM_TARGET)
  if(TARGET hshm::cxx)
    set(HSHM_TARGET hshm::cxx CACHE STRING "HermesShm target name")
  elseif(TARGET cxx)
    set(HSHM_TARGET cxx CACHE STRING "HermesShm target name")
  else()
    message(FATAL_ERROR "Neither hshm::cxx nor cxx target found")
  endif()
  message(STATUS "Using HermesShm target: ${HSHM_TARGET}")
endif()
