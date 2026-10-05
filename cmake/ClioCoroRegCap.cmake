# ClioCoroRegCap -- automatic, measured register ceilings for GPU coroutine
# kernels. Nothing in this tree writes a register count or a launch bound by
# hand.
#
# WHAT THIS REPLACES. `-Xcuda-ptxas --maxrregcount=N` is translation-unit wide:
# it clamps every kernel in the file to one hand-picked number, including plain
# kernels whose allocation was already honest. This module instead measures.
#
# HOW. For a target T it builds T_regprobe -- the same sources, the same link
# line, the pass disabled -- reads the real per-kernel register counts out of
# its cubin with cuobjdump, and derives a cap file. T is then compiled with the
# NVPTXCoroCap plugin pointed at that file. The plugin stamps nvvm.maxnreg on
# exactly the kernels that (transitively) execute a device coroutine AND were
# measured over budget. Plain kernels are never touched; coroutine kernels that
# already fit keep their natural allocation.
#
# COST: T is compiled and linked twice. That is the price of a measured number
# instead of a guessed one, and it is why this is opt-in per target.
#
# WHAT THE MEASUREMENT CURRENTLY SHOWS. Because NVPTX has no tail calls,
# CoroSplit merges every resume segment into one function and the register
# allocator takes the liveness union across all suspend points -- so today
# every coroutine kernel measures the SAME count regardless of body (12 kernels
# in the MD bench: eleven coroutine kernels all at 138, one plain kernel at 8).
# There is currently no per-kernel variation among coroutine kernels to
# exploit. This wiring exists so that the number is derived rather than typed,
# plain kernels are provably spared, and the build adapts by itself the day the
# backend stops erasing the difference. It is not, today, buying per-kernel
# spread that does not exist.

include_guard(GLOBAL)

# The occupancy target -- the one genuine policy input. Coroutine kernels are
# latency-bound by construction (their duty cycle is waiting on pages, not
# arithmetic), so occupancy is the lever. 4 blocks/SM at 256 threads on a 64K
# register file = 64 registers, the count these kernels were validated at.
set(CLIO_CORO_REGS_PER_SM 65536 CACHE STRING
    "Register file per SM, for the coroutine occupancy budget")
set(CLIO_CORO_REF_THREADS 256 CACHE STRING
    "Reference block size for the coroutine occupancy budget")
set(CLIO_CORO_TARGET_BLOCKS 4 CACHE STRING
    "Target blocks/SM for coroutine kernels (the occupancy purchase)")
mark_as_advanced(CLIO_CORO_REGS_PER_SM CLIO_CORO_REF_THREADS
                 CLIO_CORO_TARGET_BLOCKS)

# --------------------------------------------------------------------------
# The pass plugin, built once for the whole project.
# --------------------------------------------------------------------------
function(_clio_coro_regcap_plugin out_var)
  if(TARGET clio_coro_cap_plugin)
    get_property(_so GLOBAL PROPERTY CLIO_CORO_CAP_SO)
    set(${out_var} "${_so}" PARENT_SCOPE)
    return()
  endif()

  find_program(CLIO_LLVM_CONFIG NAMES llvm-config-22 llvm-config)
  if(NOT CLIO_LLVM_CONFIG)
    message(STATUS "coro_regcap: llvm-config not found -- coroutine kernels "
                   "keep their natural register allocation")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()
  execute_process(COMMAND ${CLIO_LLVM_CONFIG} --cxxflags
                  OUTPUT_VARIABLE _llvm_cxxflags
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  separate_arguments(_llvm_cxxflags_list NATIVE_COMMAND "${_llvm_cxxflags}")

  set(_src ${CMAKE_SOURCE_DIR}/tools/coro_regcap/NVPTXCoroCap.cpp)
  set(_so ${CMAKE_BINARY_DIR}/lib/libNVPTXCoroCap.so)
  add_custom_command(
    OUTPUT ${_so}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_CXX_COMPILER} -shared -fPIC ${_llvm_cxxflags_list}
            ${_src} -o ${_so}
    DEPENDS ${_src}
    COMMENT "coro_regcap: building NVPTXCoroCap pass plugin")
  add_custom_target(clio_coro_cap_plugin DEPENDS ${_so})
  set_property(GLOBAL PROPERTY CLIO_CORO_CAP_SO "${_so}")
  set(${out_var} "${_so}" PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------
# clio_coro_regcap(<target>)
#
# Give <target> a measured, per-kernel coroutine register ceiling. Must be
# called AFTER the target is fully defined (sources, includes, link libraries),
# because it mirrors those onto the probe.
# --------------------------------------------------------------------------
function(clio_coro_regcap tgt)
  if(NOT CLIO_GPU_CLANG)
    return()   # no device coroutines in this configuration; nothing to cap
  endif()
  if(NOT CMAKE_CUDA_COMPILER_ID MATCHES "Clang")
    message(STATUS "coro_regcap: ${tgt} skipped -- needs clang -x cuda "
                   "(-fpass-plugin), have ${CMAKE_CUDA_COMPILER_ID}")
    return()
  endif()
  _clio_coro_regcap_plugin(_so)
  if(NOT _so)
    return()
  endif()
  find_program(CLIO_CUOBJDUMP NAMES cuobjdump
               HINTS ${CUDAToolkit_BIN_DIR} /usr/local/cuda/bin)
  if(NOT CLIO_CUOBJDUMP)
    message(STATUS "coro_regcap: cuobjdump not found -- ${tgt} keeps natural "
                   "register allocation")
    return()
  endif()

  get_target_property(_srcs ${tgt} SOURCES)
  get_target_property(_incs ${tgt} INCLUDE_DIRECTORIES)
  get_target_property(_libs ${tgt} LINK_LIBRARIES)
  get_target_property(_defs ${tgt} COMPILE_DEFINITIONS)
  get_target_property(_opts ${tgt} COMPILE_OPTIONS)

  # THE PROBE COMPILES ITS OWN COPY OF THE SOURCES. It must not share source
  # files with ${tgt}: the real target's sources carry an OBJECT_DEPENDS on the
  # cap file, and the cap file is produced BY the probe -- sharing the sources
  # would make that a dependency cycle.
  set(_probe_dir ${CMAKE_CURRENT_BINARY_DIR}/${tgt}_regprobe)
  file(MAKE_DIRECTORY ${_probe_dir})
  set(_probe_srcs "")
  foreach(_s IN LISTS _srcs)
    if(NOT IS_ABSOLUTE "${_s}")
      set(_s "${CMAKE_CURRENT_SOURCE_DIR}/${_s}")
    endif()
    get_filename_component(_base "${_s}" NAME)
    set(_copy "${_probe_dir}/${_base}")
    add_custom_command(OUTPUT "${_copy}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_s}" "${_copy}"
      DEPENDS "${_s}"
      COMMENT "coro_regcap: staging ${_base} for the ${tgt} probe build")
    get_source_file_property(_lang "${_s}" LANGUAGE)
    if(_lang)
      set_source_files_properties("${_copy}" PROPERTIES LANGUAGE ${_lang})
    endif()
    list(APPEND _probe_srcs "${_copy}")
  endforeach()

  add_executable(${tgt}_regprobe EXCLUDE_FROM_ALL ${_probe_srcs})
  add_dependencies(${tgt}_regprobe clio_coro_cap_plugin)
  # The probe is the MEASUREMENT: the plugin is loaded (so the two builds differ
  # in nothing but the cap file) but names no caps, so every kernel reports its
  # natural allocation.
  target_compile_options(${tgt}_regprobe PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-fpass-plugin=${_so}>)
  if(_opts)
    target_compile_options(${tgt}_regprobe PRIVATE ${_opts})
  endif()
  # The original source dir, so quoted includes next to the source still resolve
  # from the staged copy.
  target_include_directories(${tgt}_regprobe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(_incs)
    target_include_directories(${tgt}_regprobe PRIVATE ${_incs})
  endif()
  if(_libs)
    target_link_libraries(${tgt}_regprobe PRIVATE ${_libs})
  endif()
  if(_defs)
    target_compile_definitions(${tgt}_regprobe PRIVATE ${_defs})
  endif()
  foreach(_prop CUDA_SEPARABLE_COMPILATION CUDA_ARCHITECTURES
                CUDA_RESOLVE_DEVICE_SYMBOLS CXX_STANDARD CUDA_STANDARD
                POSITION_INDEPENDENT_CODE)
    get_target_property(_v ${tgt} ${_prop})
    if(NOT _v STREQUAL "_v-NOTFOUND")
      set_target_properties(${tgt}_regprobe PROPERTIES ${_prop} "${_v}")
    endif()
  endforeach()
  set_target_properties(${tgt}_regprobe PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${_probe_dir})

  # Measure -> derive.
  set(_caps ${_probe_dir}/coro_caps.txt)
  add_custom_command(
    OUTPUT ${_caps}
    COMMAND ${CMAKE_COMMAND} -E env python3
            ${CMAKE_SOURCE_DIR}/tools/coro_regcap/derive_caps.py
            $<TARGET_FILE:${tgt}_regprobe> ${_caps}
            --cuobjdump ${CLIO_CUOBJDUMP}
            --regs-per-sm ${CLIO_CORO_REGS_PER_SM}
            --ref-threads ${CLIO_CORO_REF_THREADS}
            --target-blocks ${CLIO_CORO_TARGET_BLOCKS}
    DEPENDS ${tgt}_regprobe
            ${CMAKE_SOURCE_DIR}/tools/coro_regcap/derive_caps.py
    COMMENT "coro_regcap: measuring ${tgt} kernel register usage")
  add_custom_target(${tgt}_regcaps DEPENDS ${_caps})

  # Apply. The OBJECT_DEPENDS is what makes a changed cap file actually
  # recompile the real target rather than silently keeping the old ceiling.
  add_dependencies(${tgt} ${tgt}_regcaps)
  target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-fpass-plugin=${_so}>)
  # THE CAP FILE TRAVELS IN THE ENVIRONMENT, NOT IN -mllvm. clang loads an
  # -fpass-plugin library in the backend but parses -mllvm into LLVM's cl
  # registry before that, so on clang <= 19 the plugin's own option does not
  # exist yet when the driver reads it and every TU dies with "Unknown command
  # line argument '-clio-coro-cap-file=...'". A compiler launcher sets the
  # variable for exactly this target's CUDA compiles instead; the plugin reads
  # the option first and the variable second, so a newer clang is unaffected.
  get_target_property(_launcher ${tgt} CUDA_COMPILER_LAUNCHER)
  if(_launcher STREQUAL "_launcher-NOTFOUND" OR NOT _launcher)
    set(_launcher "")   # an EMPTY leading element becomes `sh -c ""`: exit 127
  else()
    set(_launcher "${_launcher};")
  endif()
  set_target_properties(${tgt} PROPERTIES CUDA_COMPILER_LAUNCHER
      "${_launcher}${CMAKE_COMMAND};-E;env;CLIO_CORO_CAP_FILE=${_caps}")
  foreach(_s IN LISTS _srcs)
    set_source_files_properties("${_s}" TARGET_DIRECTORY ${tgt}
                                PROPERTIES OBJECT_DEPENDS "${_caps}")
  endforeach()
endfunction()
