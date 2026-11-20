# ClioCliAlias -- version stamping and legacy-name aliases for the user-facing
# CLI binaries (clio-run, clio-cae).
#
# VERSION. `clio_core_version` is an INTERFACE target carrying
# CLIO_CORE_VERSION_STRING="<project version>" taken from the top-level
# project(iowarp-core VERSION ...), so bumping that one line is all a release
# needs; `clio-run --version` / `clio-cae --version` print it. Link it
# PRIVATE: it is a build-time detail, never part of an exported interface.
#
# ALIASES. The canonical binaries are hyphenated (clio-run, clio-cae). The old
# underscore names keep working as aliases rather than second copies:
#   * POSIX: a RELATIVE symlink next to the binary, both in the build tree
#     (so tests and scripts using build/bin/clio_run keep working) and in the
#     install tree. Relative links survive DESTDIR staging and relocation, and
#     every native package format (deb, rpm, conda, spack) carries symlinks
#     as-is, so the alias costs no disk.
#   * Windows: symlinks need privileges there, so the alias is a copy.
#   * pip wheels: a wheel is a zip and cannot hold a symlink. The wheel does
#     not install the alias at all; both names are [project.scripts] entry
#     points that exec the one bundled binary (see pyproject.toml).

if(NOT TARGET clio_core_version)
  add_library(clio_core_version INTERFACE)
  target_compile_definitions(clio_core_version INTERFACE
    CLIO_CORE_VERSION_STRING="${PROJECT_VERSION}")
endif()

# clio_add_cli_alias(<target> <alias> [COMPONENTS <c>...])
#
# Create <alias> next to <target>'s output (build tree) and next to its
# installed copy in <prefix>/bin for each listed install component (default:
# the default "Unspecified" component, which deb/rpm/spack/conda consume).
function(clio_add_cli_alias target alias)
  cmake_parse_arguments(ARG "" "" "COMPONENTS" ${ARGN})

  # Build tree. create_symlink replaces whatever is there, including a real
  # binary left behind by a build from before the rename.
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:${target}>"
              "$<TARGET_FILE_DIR:${target}>/${alias}${CMAKE_EXECUTABLE_SUFFIX}"
      VERBATIM)
  else()
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "$<TARGET_FILE_NAME:${target}>"
              "$<TARGET_FILE_DIR:${target}>/${alias}"
      VERBATIM)
  endif()

  # Install tree. The installed file name is computed here, at configure
  # time, rather than with $<TARGET_FILE_NAME:...>: generator expressions in
  # install(CODE) are only expanded under policy CMP0087, and a caller whose
  # directory declares cmake_minimum_required(VERSION < 3.14) (as
  # context-runtime/util does) gets it OLD -- the literal "$<...>" string then
  # never matched a file and the alias was silently not installed.
  # DESTDIR is honoured so staged installs (cpack, rpmbuild, conda) work.
  get_target_property(_out_name ${target} OUTPUT_NAME)
  if(NOT _out_name)
    set(_out_name ${target})
  endif()
  set(_src_name "${_out_name}${CMAKE_EXECUTABLE_SUFFIX}")

  set(_components ${ARG_COMPONENTS})
  if(NOT _components)
    set(_components Unspecified)
  endif()
  foreach(_comp IN LISTS _components)
    install(CODE "
      set(_bindir \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin\")
      set(_src \"${_src_name}\")
      set(_dst \"\${_bindir}/${alias}${CMAKE_EXECUTABLE_SUFFIX}\")
      if(NOT EXISTS \"\${_bindir}/\${_src}\")
        message(FATAL_ERROR
          \"CLI alias ${alias}: \${_bindir}/\${_src} was not installed\")
      endif()
      file(REMOVE \"\${_dst}\")
      if(WIN32)
        file(COPY_FILE \"\${_bindir}/\${_src}\" \"\${_dst}\")
      else()
        file(CREATE_LINK \"\${_src}\" \"\${_dst}\" SYMBOLIC)
      endif()
      message(STATUS \"Installing: \${_dst} -> \${_src}\")
    " COMPONENT ${_comp})
  endforeach()
endfunction()
