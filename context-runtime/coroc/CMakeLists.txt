# clio-coroc: the CO_AWAIT transpiler.
#
# Built standalone against a stock LLVM/Clang. The tool only READS source, so
# its clang need not be -- and generally is not -- the compiler that builds the
# project: on Aurora the project is built by icpx and this is built by whatever
# LLVM happens to be installed. They never read the same headers.
cmake_minimum_required(VERSION 3.20)
project(clio_coroc CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

add_executable(clio-coroc main.cc)
target_compile_features(clio-coroc PRIVATE cxx_std_20)
target_include_directories(clio-coroc SYSTEM PRIVATE
                           ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_compile_definitions(clio-coroc PRIVATE ${LLVM_DEFINITIONS})

# A LibTooling binary is not clang: bake in where clang's builtin headers and
# the host C++ library live (as build.sh does), so callers pass only -I/-D.
set(_coroc_resource_dir
    "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
get_filename_component(_coroc_gcc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_coroc_gcc_toolchain "${_coroc_gcc_bin}" DIRECTORY)
target_compile_definitions(clio-coroc PRIVATE
    CLIO_COROC_RESOURCE_DIR="${_coroc_resource_dir}"
    CLIO_COROC_GCC_TOOLCHAIN="${_coroc_gcc_toolchain}")

if(TARGET clang-cpp)
  target_link_libraries(clio-coroc PRIVATE clang-cpp LLVM)
else()
  # clangAnalysis is the easy one to forget -- CFG and LiveVariables live there.
  target_link_libraries(clio-coroc PRIVATE
    clangTooling clangFrontend clangSerialization clangDriver clangParse
    clangSema clangAnalysis clangASTMatchers clangAST clangRewrite
    clangEdit clangLex clangBasic)
  llvm_map_components_to_libnames(_llvm_libs support core option)
  target_link_libraries(clio-coroc PRIVATE ${_llvm_libs})
endif()

# LLVM is built without RTTI by default; a tool inheriting its classes must
# match or the vtables will not link.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(clio-coroc PRIVATE -fno-rtti)
endif()
