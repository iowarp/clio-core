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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_MODULE_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_MODULE_MANAGER_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "chimaera/types.h"

namespace chi {

// Forward declarations for ChiMod system
// Container is always a class forward declaration (defined in container.h)
class Container;

// ChiMod function types
typedef Container* (*alloc_chimod_t)();
typedef Container* (*new_chimod_t)(const PoolId* pool_id, const char* pool_name);
typedef const char* (*get_chimod_name_t)(void);
typedef void (*destroy_chimod_t)(Container* container);

/**
 * ChiMod metadata and shared library wrapper
 */
struct ChiModInfo {
  std::string name;
  std::string lib_path;
  hshm::SharedLibrary lib;
  
  // Function pointers
  alloc_chimod_t alloc_func;
  new_chimod_t new_func;
  get_chimod_name_t name_func;
  destroy_chimod_t destroy_func;
  
  ChiModInfo() : alloc_func(nullptr), new_func(nullptr), 
                 name_func(nullptr), destroy_func(nullptr) {}
};

/**
 * Module Manager singleton for dynamic loading of ChiMods
 * 
 * Handles discovery and loading of ChiMod shared libraries from:
 * - LD_LIBRARY_PATH directories
 * - CHI_REPO_PATH directories
 * 
 * Each ChiMod provides functions to query name and allocate ChiContainers.
 * Uses HSHM SharedLibrary for cross-platform dynamic loading.
 */
class ModuleManager {
 public:
  /**
   * Initialize module manager (generic wrapper)
   * Scans for and loads all available ChiMods
   * @return true if initialization successful, false otherwise
   */
  bool Init() { return ServerInit(); }

  /**
   * Initialize module manager (server/runtime only)
   * Scans for and loads all available ChiMods
   * @return true if initialization successful, false otherwise
   */
  bool ServerInit();

  /**
   * Finalize and cleanup module resources
   */
  void Finalize();

  /**
   * Load a specific ChiMod from path
   * @param lib_path Path to shared library
   * @return true if loaded successfully, false otherwise
   */
  bool LoadChiMod(const std::string& lib_path);

  /**
   * Get ChiMod by name
   * @param chimod_name Name of ChiMod
   * @return Pointer to ChiModInfo or nullptr if not found
   */
  ChiModInfo* GetChiMod(const std::string& chimod_name);

  /**
   * Get ChiMod by library name
   * @param chimod_lib_name Library name (e.g., "chimods_admin_runtime")
   * @return Pointer to ChiModInfo or nullptr if not found
   */
  ChiModInfo* GetChiModByLibName(const std::string& chimod_lib_name);

  /**
   * Create Container instance from ChiMod
   * @param chimod_name Name of ChiMod to instantiate
   * @param pool_id Pool identifier for the container
   * @param pool_name Pool name for the container
   * @return Pointer to Container or nullptr if failed
   */
  Container* CreateContainer(const std::string& chimod_name,
                                const PoolId& pool_id,
                                const std::string& pool_name);

  /**
   * Create Container instance using library name from CreateTask
   * @param chimod_lib_name Library name from CreateTask (e.g., "chimods_admin_runtime")
   * @param pool_id Pool identifier for the container
   * @param pool_name Pool name for the container
   * @return Pointer to Container or nullptr if failed
   */
  Container* CreateContainerByLibName(const std::string& chimod_lib_name,
                                         const PoolId& pool_id, 
                                         const std::string& pool_name);

  /**
   * Destroy Container instance
   * @param chimod_name Name of ChiMod that created the container
   * @param container Pointer to container to destroy
   */
  void DestroyContainer(const std::string& chimod_name, Container* container);

  /**
   * Get list of loaded ChiMod names
   * @return Vector of ChiMod names
   */
  std::vector<std::string> GetLoadedChiMods() const;

  /**
   * Check if module manager is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const;

 private:
  /**
   * Scan directories for ChiMod libraries
   */
  void ScanForChiMods();

  /**
   * Get list of directories to scan from environment
   * @return Vector of directory paths
   */
  std::vector<std::string> GetScanDirectories() const;

  /**
   * Get the directory path where the runtime shared object is located
   *
   * Uses dladdr() to discover the runtime location of the shared object,
   * allowing dynamic loading of sibling modules in the same directory.
   *
   * @return Absolute directory path, or empty string on failure
   */
  std::string GetModuleDirectory() const;

  /**
   * Check if file is a potential ChiMod library
   * @param file_path Path to file
   * @return true if file looks like a shared library
   */
  bool IsSharedLibrary(const std::string& file_path) const;

  /**
   * Check if shared object follows ChiMod naming convention
   * ChiMod libraries must contain "_runtime" in their name
   * @param file_path Path to shared object file
   * @return true if file follows ChiMod naming convention
   */
  bool HasModuleNamingConvention(const std::string& file_path) const;

  /**
   * Validate that library has required ChiMod entry points
   * @param lib Shared library to validate
   * @return true if library has required functions
   */
  bool ValidateChiMod(hshm::SharedLibrary& lib) const;

  bool is_initialized_ = false;
  
  // Map ChiMod name to ChiModInfo
  std::map<std::string, std::unique_ptr<ChiModInfo>> chimods_;
};

}  // namespace chi

// Global pointer variable declaration for Module manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H_DLL(CHI_DLL, chi::ModuleManager, g_module_manager);

// Macro for accessing the Module manager singleton using global pointer variable
#define CHI_MODULE_MANAGER HSHM_GET_GLOBAL_PTR_VAR(::chi::ModuleManager, g_module_manager)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_MODULE_MANAGER_H_