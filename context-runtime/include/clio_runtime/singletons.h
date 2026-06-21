/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Central header for CLIO Runtime singleton access macros
 * 
 * This header provides convenient macros for accessing all CLIO Runtime singletons
 * using CTP's global cross pointer variable pattern. Include this header to
 * get access to all singleton macros in one place.
 */

#ifndef CLIO_RUNTIME_INCLUDE_SINGLETONS_H_
#define CLIO_RUNTIME_INCLUDE_SINGLETONS_H_

#include "clio_runtime/manager.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/module_manager.h"
#include "clio_runtime/work_orchestrator.h"
#include "clio_runtime/admin.h"

/**
 * Convenience macros for accessing CLIO Runtime singletons
 * 
 * These macros provide easy access to all CLIO Runtime singleton managers
 * using CTP's global cross pointer variable pattern.
 */

// Core Framework Singleton Access
// CLIO_RUNTIME_MANAGER - Main CLIO Runtime framework coordinator
// CLIO_CONFIG_MANAGER   - Configuration manager for YAML parsing
// CLIO_IPC              - IPC manager for shared memory and networking
// CLIO_POOL_MANAGER     - Pool manager for ChiPools and ChiContainers
// CLIO_MODULE_MANAGER   - Module manager for dynamic loading
// CLIO_WORK_ORCHESTRATOR - Work orchestrator for thread management
// CLIO_ADMIN            - Admin ChiMod client singleton

// All macros are defined in their respective header files:
// - CLIO_RUNTIME_MANAGER defined in clio_runtime/manager.h
// - CLIO_CONFIG_MANAGER defined in clio/config_manager.h
// - CLIO_IPC defined in clio/ipc_manager.h
// - CLIO_POOL_MANAGER defined in clio/pool_manager.h
// - CLIO_MODULE_MANAGER defined in clio/module_manager.h
// - CLIO_WORK_ORCHESTRATOR defined in clio/work_orchestrator.h
// - CLIO_ADMIN defined in clio/admin.h

/**
 * Example usage:
 * 
 * // Initialize the configuration manager
 * CLIO_CONFIG_MANAGER->Init();
 *
 * // Get worker thread count from config
 * u32 workers = CLIO_CONFIG_MANAGER->GetNumThreads();
 *
 * // Initialize IPC components
 * CLIO_IPC->ServerInit();
 * 
 * // Start worker threads
 * CLIO_WORK_ORCHESTRATOR->Init();
 * CLIO_WORK_ORCHESTRATOR->StartWorkers();
 * 
 * // Register a pool
 * CLIO_POOL_MANAGER->RegisterContainer(pool_id, container);
 */

#endif  // CLIO_RUNTIME_INCLUDE_SINGLETONS_H_