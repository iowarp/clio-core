/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_API_H_
#define CLIO_CTE_FILESYSTEM_API_H_

#include "clio_cte/api.h"

/**
 * Per-DLL export macro for clio_cte_filesystem_client.
 *
 * Mirrors clio_cte/api.h's CLIO_CTE_API and clio_runtime/api.h's CLIO_RUN_API.
 * CMake sets CLIO_CTE_FS_BUILDING_DLL=1 PRIVATE on the
 * clio_cte_filesystem_client target; consumers see the import form.
 *
 * This exists rather than reusing CLIO_CTE_API because that macro answers a
 * different question. CLIO_CTE_API means "is this translation unit building
 * clio_cte_core_client?", so defining CLIO_CTE_CORE_BUILDING_DLL here to
 * export g_fs_client also told every declaration in the *core's* headers that
 * it was being built locally -- including g_cte_client, which then resolved to
 * a definition this DLL does not contain:
 *
 *   filesystem_client.obj : error LNK2019: unresolved external symbol
 *   "struct std::atomic<class clio::cte::core::Client *> clio::cte::core::g_cte_client"
 *
 * That stayed latent only because nothing in this client referenced the core
 * singleton on Windows: the descriptor layer, which does, was excluded there.
 */
#if defined(_WIN32)
#ifdef CLIO_CTE_FS_BUILDING_DLL
#define CLIO_CTE_FS_API __declspec(dllexport)
#else
#define CLIO_CTE_FS_API __declspec(dllimport)
#endif
#else
#define CLIO_CTE_FS_API
#endif

/** As CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_{H,CC}, decorated for this DLL. */
#define CLIO_CTE_FS_DEFINE_GLOBAL_PTR_VAR_H(T, NAME) \
  extern CLIO_CTE_FS_API ::std::atomic<__TU(T) *> NAME;
#define CLIO_CTE_FS_DEFINE_GLOBAL_PTR_VAR_CC(T, NAME) \
  CLIO_CTE_FS_API ::std::atomic<__TU(T) *> NAME{nullptr};

#endif  // CLIO_CTE_FILESYSTEM_API_H_
