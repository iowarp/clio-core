/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_RUN_API_H_
#define CLIO_RUN_API_H_

/**
 * Per-DLL export macro for clio_run_cxx.
 *
 * On Windows, data globals defined inside a DLL and referenced from
 * another module (e.g. g_runtime_manager defined in clio_run_cxx.dll
 * and read from clio_admin_runtime.dll) need explicit dllexport on the
 * defining side and dllimport on the consuming side. CMake's
 * WINDOWS_EXPORT_ALL_SYMBOLS only handles function symbols, not data.
 *
 * The build system sets CLIO_RUN_BUILDING_DLL=1 PRIVATE on the
 * clio_run_cxx target only; every other target sees the import form.
 * On non-Windows platforms this macro expands to nothing — visibility
 * defaults handle the same job.
 */
#if defined(_WIN32)
#  ifdef CLIO_RUN_BUILDING_DLL
#    define CLIO_RUN_API __declspec(dllexport)
#  else
#    define CLIO_RUN_API __declspec(dllimport)
#  endif
#else
#  define CLIO_RUN_API
#endif

/**
 * Export macro for the GPU companion lib (clio_run_cxx_gpu). Its cross-DLL
 * entry points (ChiServerBootstrap{Hip,Sycl}Gpu) are defined in the GPU lib and
 * called from clio_run_cxx. CMake's WINDOWS_EXPORT_ALL_SYMBOLS does not reliably
 * export symbols from the nvcc-built (separable-compilation) GPU lib, so mark
 * them explicitly. CMake defines clio_run_cxx_gpu_EXPORTS when building that
 * target. Non-Windows: expands to nothing (default visibility handles it).
 */
#if defined(_WIN32)
#  ifdef clio_run_cxx_gpu_EXPORTS
#    define CLIO_RUN_GPU_API __declspec(dllexport)
#  else
#    define CLIO_RUN_GPU_API __declspec(dllimport)
#  endif
#else
#  define CLIO_RUN_GPU_API
#endif

#include "clio_ctp/util/singleton.h"

/** Same as CTP_DEFINE_GLOBAL_PTR_VAR_{H,CC} but with CLIO_RUN_API decoration
 *  so the symbol is exported from clio_run_cxx.dll and imported elsewhere. */
#define CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(T, NAME) extern CLIO_RUN_API __TU(T) * NAME;
#define CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_CC(T, NAME) CLIO_RUN_API __TU(T) *NAME = nullptr;

#endif  // CLIO_RUN_API_H_
