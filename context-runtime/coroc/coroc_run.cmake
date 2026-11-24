# Run clio-coroc on one translation unit, at build time (cmake -P script).
#
# Inputs (-D):
#   COROC   the transpiler executable
#   TU      absolute path of the translation unit to transpile
#   ROOT    rewrite root (the source tree); the tool rewrites files under it
#   GEN     mirror directory the rewritten tree is written to
#   EXPECT  '@@'-joined paths, relative to ROOT, that must exist under GEN
#           afterwards (the TU itself is always checked)
#   LOG     file that receives the tool's output
#   ARGS    '@@'-joined compiler arguments the tool parses with
#
# Lists cross -D as '@@'-joined strings: ';' would split the argument and '|'
# is a pipe to cmd.exe, which runs Windows custom commands.
#
# The TU is copied into GEN first, so the target always has a file to compile
# even if the tool rewrote nothing in it; the expectations then check that the
# rewrite actually happened.

foreach(_v COROC TU ROOT GEN LOG)
  if(NOT DEFINED ${_v})
    message(FATAL_ERROR "coroc_run.cmake: ${_v} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${GEN}")
cmake_path(RELATIVE_PATH TU BASE_DIRECTORY "${ROOT}" OUTPUT_VARIABLE _rel)
cmake_path(GET _rel PARENT_PATH _reldir)
file(MAKE_DIRECTORY "${GEN}/${_reldir}")
file(COPY_FILE "${TU}" "${GEN}/${_rel}")

string(REPLACE "@@" ";" _args "${ARGS}")
execute_process(
    COMMAND "${COROC}" "${TU}" --verbose "--rewrite-root=${ROOT}"
            "--mirror-to=${GEN}" -- ${_args}
    OUTPUT_FILE "${LOG}" ERROR_FILE "${LOG}"
    RESULT_VARIABLE _rc)
file(READ "${LOG}" _out)
if(_out MATCHES "clio-coroc: FATAL")
  message(FATAL_ERROR "clio-coroc: fatal parse error transpiling ${TU}; see ${LOG}")
endif()

string(REPLACE "@@" ";" _expect "${EXPECT}")
foreach(_e IN LISTS _expect)
  if(NOT EXISTS "${GEN}/${_e}")
    message(FATAL_ERROR "clio-coroc did not rewrite ${_e} (rc=${_rc}); see ${LOG}")
  endif()
endforeach()
# The TU must have been rewritten, not merely copied: if the generated file is
# byte-identical to the source, the tool changed nothing in it.
file(SHA256 "${TU}" _src_hash)
file(SHA256 "${GEN}/${_rel}" _gen_hash)
if(_src_hash STREQUAL _gen_hash)
  message(FATAL_ERROR "clio-coroc did not rewrite ${_rel} (rc=${_rc}); see ${LOG}")
endif()
file(TOUCH "${GEN}/${_rel}")
