# SYCL LZ4 kernels library -- the device half of the SYCL nvcomp-equivalent.
#
# Built with the project's own SYCL machinery: when CLIO_CORE_ENABLE_SYCL is on
# the C++ compiler IS a SYCL compiler (DPC++/AdaptiveCpp), so this TU is just a
# normal target with wrp_core_apply_sycl_flags() -- the same pattern the CTP
# SYCL tests (test_gpu_sycl) use. Guarded by CLIO_CTP_ENABLE_SYCL_LZ4 (OFF by
# default) so ordinary builds are untouched. The header-only ctp::SyclLz4
# wrapper (compiled by the same toolchain) sees only the extern "C" ABI here.
#
# build_and_test.sh remains the minimal standalone recipe (DPC++ directly).

if(NOT CLIO_CTP_ENABLE_SYCL_LZ4)
  return()
endif()

if(NOT DEFINED CLIO_SYCL_COMPILER)
  message(FATAL_ERROR
    "CLIO_CTP_ENABLE_SYCL_LZ4=ON requires CLIO_CORE_ENABLE_SYCL=ON "
    "(the C++ compiler must be a SYCL compiler). ")
endif()

# OBJECT (not STATIC): DPC++ can drop a static archive's SYCL device image
# during the final link, which aborts at kernel launch (the host symbols link,
# but no device image is present for the CUDA backend). An object library embeds
# the objects -- and their device code -- directly into every consumer, so the
# kernel image is always present.
add_library(clio_ctp_sycl_lz4 OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/../sycl_lz4_kernels.cc)
set_target_properties(clio_ctp_sycl_lz4 PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -fsycl on both compile and link; wrp_core_apply_sycl_flags applies it PRIVATE,
# so also expose it to consumers (INTERFACE) -- a target that links this static
# library needs -fsycl at its own link step to pull in the SYCL runtime.
wrp_core_apply_sycl_flags(clio_ctp_sycl_lz4)
target_compile_options(clio_ctp_sycl_lz4 INTERFACE -fsycl)
target_link_options(clio_ctp_sycl_lz4 INTERFACE -fsycl)
if(SYCL_TARGET)
  target_link_options(clio_ctp_sycl_lz4 INTERFACE -fsycl-targets=${SYCL_TARGET})
endif()
