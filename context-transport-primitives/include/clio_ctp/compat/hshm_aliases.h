/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core (BSD 3-Clause license, see COPYING).
 */
#ifndef CLIO_CTP_COMPAT_HSHM_ALIASES_H_
#define CLIO_CTP_COMPAT_HSHM_ALIASES_H_

// ============================================================================
// Backward-compat aliases for the hermes_shm / HSHM era.
//
// Makes the legacy ctp::, ctp::ipc::, and HSHM_* identifiers remain valid for
// external code that has not migrated to the new ctp::, ctp::ipc::, and
// CTP_* names. Both forms work simultaneously.
//
// Included automatically by <clio_ctp/clio_ctp.h>, so every TU that pulls
// in the umbrella header sees these aliases. The macro aliases are
// generated mechanically from `grep '^#define CTP_[A-Z]'` across the
// context-transport-primitives tree, with include-guard-style symbols
// excluded (they have no client-facing role).
// ============================================================================

// --- Namespace aliases ---
// ctp::X   resolves to ctp::X
// ctp::ipc::X / ctp::ipc::X resolves to ctp::ipc::X
//
// These rely on the `ctp` and `ctp::ipc` namespaces being declared somewhere
// (the rest of <clio_ctp/clio_ctp.h>); forward-declare them so this header
// is self-contained.
namespace ctp { namespace ipc {} }
namespace hshm = ctp;
namespace hipc = ctp::ipc;  // shorthand alias (matches the historical ctp::ipc:: usage)

// --- Macro aliases ---
#define HSHM_COMPILING_DLL                   CTP_COMPILING_DLL
#define HSHM_CROSS_FUN                       CTP_CROSS_FUN
#define HSHM_CROSS_FUN_SEL                   CTP_CROSS_FUN_SEL
#define HSHM_DEFAULT_ALLOC                   CTP_DEFAULT_ALLOC
#define HSHM_DEFAULT_ALLOC_GPU_T             CTP_DEFAULT_ALLOC_GPU_T
#define HSHM_DEFAULT_SORT_CMP                CTP_DEFAULT_SORT_CMP
#define HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_CC  CTP_DEFINE_GLOBAL_CROSS_PTR_VAR_CC
#define HSHM_DEFINE_GLOBAL_CROSS_VAR_CC      CTP_DEFINE_GLOBAL_CROSS_VAR_CC
#define HSHM_DEFINE_GLOBAL_PTR_VAR_CC        CTP_DEFINE_GLOBAL_PTR_VAR_CC
#define HSHM_DEFINE_GLOBAL_VAR_CC            CTP_DEFINE_GLOBAL_VAR_CC
#define HSHM_DEVICE_ATOMIC_ADD_U32_DEVICE    CTP_DEVICE_ATOMIC_ADD_U32_DEVICE
#define HSHM_DEVICE_ATOMIC_ADD_U64_DEVICE    CTP_DEVICE_ATOMIC_ADD_U64_DEVICE
#define HSHM_DEVICE_ATOMIC_OR_U32_DEVICE     CTP_DEVICE_ATOMIC_OR_U32_DEVICE
#define HSHM_DEVICE_ATOMIC_OR_U32_SYSTEM     CTP_DEVICE_ATOMIC_OR_U32_SYSTEM
#define HSHM_DEVICE_CLOCK64                  CTP_DEVICE_CLOCK64
#define HSHM_DEVICE_EXTERN                   CTP_DEVICE_EXTERN
#define HSHM_DEVICE_FENCE_DEVICE             CTP_DEVICE_FENCE_DEVICE
#define HSHM_DEVICE_FENCE_SYSTEM             CTP_DEVICE_FENCE_SYSTEM
#define HSHM_DEVICE_PRINTF                   CTP_DEVICE_PRINTF
#define HSHM_DEVICE_SYNCWARP                 CTP_DEVICE_SYNCWARP
#define HSHM_DEV_TYPE_CPU                    CTP_DEV_TYPE_CPU
#define HSHM_DEV_TYPE_GPU                    CTP_DEV_TYPE_GPU
#define HSHM_DLL                             CTP_DLL
#define HSHM_DLL_EXPORT                      CTP_DLL_EXPORT
#define HSHM_DLL_IMPORT                      CTP_DLL_IMPORT
#define HSHM_DLL_SINGLETON                   CTP_DLL_SINGLETON
#define HSHM_ENABLE_CUDA_OR_ROCM             CTP_ENABLE_CUDA_OR_ROCM
#define HSHM_ENABLE_GPU                      CTP_ENABLE_GPU
#define HSHM_ERROR_HANDLE_CATCH              CTP_ERROR_HANDLE_CATCH
#define HSHM_ERROR_HANDLE_END                CTP_ERROR_HANDLE_END
#define HSHM_ERROR_HANDLE_START              CTP_ERROR_HANDLE_START
#define HSHM_ERROR_HANDLE_TRY                CTP_ERROR_HANDLE_TRY
#define HSHM_ERROR_IS                        CTP_ERROR_IS
#define HSHM_ERROR_PTR                       CTP_ERROR_PTR
#define HSHM_ERROR_TYPE                      CTP_ERROR_TYPE
#define HSHM_FUNC_IS_USED                    CTP_FUNC_IS_USED
#define HSHM_GET_GLOBAL_CROSS_PTR_VAR        CTP_GET_GLOBAL_CROSS_PTR_VAR
#define HSHM_GET_GLOBAL_CROSS_VAR            CTP_GET_GLOBAL_CROSS_VAR
#define HSHM_GET_GLOBAL_PTR_VAR              CTP_GET_GLOBAL_PTR_VAR
#define HSHM_GET_GLOBAL_VAR                  CTP_GET_GLOBAL_VAR
#define HSHM_GPU_FUN                         CTP_GPU_FUN
#define HSHM_GPU_KERNEL                      CTP_GPU_KERNEL
#define HSHM_GPU_OR_HOST                     CTP_GPU_OR_HOST
#define HSHM_GPU_VAR                         CTP_GPU_VAR
#define HSHM_HOST_FUN                        CTP_HOST_FUN
#define HSHM_HOST_VAR                        CTP_HOST_VAR
#define HSHM_INDIRECTLY_CALLABLE             CTP_INDIRECTLY_CALLABLE
#define HSHM_INLINE                          CTP_INLINE
#define HSHM_INLINE_CROSS_FUN                CTP_INLINE_CROSS_FUN
#define HSHM_INLINE_CROSS_FUN_SEL            CTP_INLINE_CROSS_FUN_SEL
#define HSHM_INLINE_CROSS_VAR                CTP_INLINE_CROSS_VAR
#define HSHM_INLINE_FLAG                     CTP_INLINE_FLAG
#define HSHM_INLINE_GPU_FUN                  CTP_INLINE_GPU_FUN
#define HSHM_INLINE_GPU_VAR                  CTP_INLINE_GPU_VAR
#define HSHM_INLINE_HOST_FUN                 CTP_INLINE_HOST_FUN
#define HSHM_INLINE_HOST_VAR                 CTP_INLINE_HOST_VAR
#define HSHM_IS_CUDA_COMPILER                CTP_IS_CUDA_COMPILER
#define HSHM_IS_CUDA_GPU                     CTP_IS_CUDA_GPU
#define HSHM_IS_DEVICE_PASS                  CTP_IS_DEVICE_PASS
#define HSHM_IS_GPU                          CTP_IS_GPU
#define HSHM_IS_GPU_COMPILER                 CTP_IS_GPU_COMPILER
#define HSHM_IS_HOST                         CTP_IS_HOST
#define HSHM_IS_ROCM_COMPILER                CTP_IS_ROCM_COMPILER
#define HSHM_IS_ROCM_GPU                     CTP_IS_ROCM_GPU
#define HSHM_IS_SYCL_COMPILER                CTP_IS_SYCL_COMPILER
#define HSHM_IS_SYCL_DEVICE                  CTP_IS_SYCL_DEVICE
#define HSHM_LOG                             CTP_LOG
#define HSHM_LOG_LEVEL                       CTP_LOG_LEVEL
#define HSHM_MALLOC                          CTP_MALLOC
#define HSHM_MCTX                            CTP_MCTX
#define HSHM_MSAN_UNPOISON                   CTP_MSAN_UNPOISON
#define HSHM_MSAN_UNPOISON_STRING            CTP_MSAN_UNPOISON_STRING
#define HSHM_NO_INLINE                       CTP_NO_INLINE
#define HSHM_NO_INLINE_CROSS_FUN             CTP_NO_INLINE_CROSS_FUN
#define HSHM_NO_INLINE_FLAG                  CTP_NO_INLINE_FLAG
#define HSHM_PERIODIC                        CTP_PERIODIC
#define HSHM_PRIV_INSERT_RESULT_DEFINED_     CTP_PRIV_INSERT_RESULT_DEFINED_
#define HSHM_ROOT_ALLOC                      CTP_ROOT_ALLOC
#define HSHM_ROOT_ALLOC_T                    CTP_ROOT_ALLOC_T
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_B CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_B
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_L CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_L
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_S CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_S
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Z CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Z
#define HSHM_SYSTEM_INFO                     CTP_SYSTEM_INFO
#define HSHM_SYSTEM_INFO_T                   CTP_SYSTEM_INFO_T
#define HSHM_THREAD_MODEL                    CTP_THREAD_MODEL
#define HSHM_THREAD_MODEL_T                  CTP_THREAD_MODEL_T
#define HSHM_THROW_ERROR                     CTP_THROW_ERROR
#define HSHM_THROW_STD_ERROR                 CTP_THROW_STD_ERROR

#endif  // CLIO_CTP_COMPAT_HSHM_ALIASES_H_
