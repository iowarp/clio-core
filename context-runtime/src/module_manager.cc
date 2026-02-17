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
 * Module manager implementation with dynamic ChiMod loading
 */

#include "chimaera/module_manager.h"

#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>

#include <cstring>
#include <filesystem>

#include "chimaera/container.h"

// Global pointer variable definition for Module manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::ModuleManager, g_module_manager);

namespace chi {

// Helper function to get a symbol address for dladdr
// This avoids issues with member function pointer casts
static void *GetSymbolForDlAddr() {
  return reinterpret_cast<void *>(&GetSymbolForDlAddr);
}

// Constructor and destructor removed - handled by HSHM singleton pattern

bool ModuleManager::ServerInit() {
  if (is_initialized_) {
    return true;
  }

  HLOG(kDebug, "Initializing Module Manager...");

  // Scan for and load ChiMods
  ScanForChiMods();

  HLOG(kDebug, "Module Manager initialized with {} ChiMods loaded",
       chimods_.size());

  is_initialized_ = true;
  return true;
}

void ModuleManager::Finalize() {
  if (!is_initialized_) {
    return;
  }

  HLOG(kDebug, "Finalizing Module Manager...");

  // Clear all loaded ChiMods - SharedLibrary destructor handles cleanup
  chimods_.clear();

  is_initialized_ = false;
}

bool ModuleManager::LoadChiMod(const std::string &lib_path) {
  auto chimod_info = std::make_unique<ChiModInfo>();
  chimod_info->lib_path = lib_path;

  // Load the shared library
  chimod_info->lib.Load(lib_path);
  if (chimod_info->lib.IsNull()) {
    HLOG(kDebug, "Skipping library {} (failed to load): {}", lib_path,
         chimod_info->lib.GetError());
    return false;
  }

  // Validate ChiMod entry points
  if (!ValidateChiMod(chimod_info->lib)) {
    HLOG(kDebug,
         "Skipping library {} (invalid ChiMod - missing required symbols)",
         lib_path);
    return false;
  }

  // Get function pointers
  chimod_info->alloc_func =
      (alloc_chimod_t)chimod_info->lib.GetSymbol("alloc_chimod");
  chimod_info->new_func =
      (new_chimod_t)chimod_info->lib.GetSymbol("new_chimod");
  chimod_info->name_func =
      (get_chimod_name_t)chimod_info->lib.GetSymbol("get_chimod_name");
  chimod_info->destroy_func =
      (destroy_chimod_t)chimod_info->lib.GetSymbol("destroy_chimod");

  // Get ChiMod name
  if (chimod_info->name_func) {
    chimod_info->name = chimod_info->name_func();
    HLOG(kInfo, "Loaded ChiMod: {} from {}", chimod_info->name, lib_path);
  } else {
    return false;
  }

  // Store in map
  chimods_[chimod_info->name] = std::move(chimod_info);
  return true;
}

ChiModInfo *ModuleManager::GetChiMod(const std::string &chimod_name) {
  auto it = chimods_.find(chimod_name);
  return (it != chimods_.end()) ? it->second.get() : nullptr;
}

Container *ModuleManager::CreateContainer(const std::string &chimod_name,
                                          const PoolId &pool_id,
                                          const std::string &pool_name) {
  ChiModInfo *chimod = GetChiMod(chimod_name);
  if (!chimod || !chimod->new_func) {
    return nullptr;
  }

  return chimod->new_func(&pool_id, pool_name.c_str());
}

void ModuleManager::DestroyContainer(const std::string &chimod_name,
                                     Container *container) {
  ChiModInfo *chimod = GetChiMod(chimod_name);
  if (chimod && chimod->destroy_func && container) {
    chimod->destroy_func(container);
  }
}

std::vector<std::string> ModuleManager::GetLoadedChiMods() const {
  std::vector<std::string> names;
  for (const auto &pair : chimods_) {
    names.push_back(pair.first);
  }
  return names;
}

bool ModuleManager::IsInitialized() const { return is_initialized_; }

void ModuleManager::ScanForChiMods() {
  std::vector<std::string> scan_dirs = GetScanDirectories();

  for (const std::string &dir : scan_dirs) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
      continue;
    }

    try {
      for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
          std::string file_path = entry.path().string();
          if (IsSharedLibrary(file_path) &&
              HasModuleNamingConvention(file_path)) {
            HLOG(kDebug, "Attempting to load ChiMod from: {}", file_path);
            LoadChiMod(file_path);
          }
        }
      }
    } catch (const std::exception &e) {
      HLOG(kError, "Error scanning directory {}: {}", dir, e.what());
    }
  }
}

std::vector<std::string> ModuleManager::GetScanDirectories() const {
  std::vector<std::string> directories;

  // Add module directory first (highest priority)
  std::string module_dir = GetModuleDirectory();
  if (!module_dir.empty()) {
    directories.push_back(module_dir);
  }

  // Get CHI_REPO_PATH
  const char *chi_repo_path = std::getenv("CHI_REPO_PATH");
  if (chi_repo_path) {
    std::string path_str(chi_repo_path);
    // Split by colon (Unix) or semicolon (Windows)
    char delimiter = ':';
#ifdef _WIN32
    delimiter = ';';
#endif

    size_t start = 0;
    size_t end = path_str.find(delimiter);
    while (end != std::string::npos) {
      directories.push_back(path_str.substr(start, end - start));
      start = end + 1;
      end = path_str.find(delimiter, start);
    }
    directories.push_back(path_str.substr(start));
  }

  // Get LD_LIBRARY_PATH
  const char *ld_path = std::getenv("LD_LIBRARY_PATH");
  if (ld_path) {
    std::string path_str(ld_path);
    size_t start = 0;
    size_t end = path_str.find(':');
    while (end != std::string::npos) {
      directories.push_back(path_str.substr(start, end - start));
      start = end + 1;
      end = path_str.find(':', start);
    }
    directories.push_back(path_str.substr(start));
  }

  // Add default directories
  directories.push_back("./lib");
  directories.push_back("../lib");
  directories.push_back("/usr/local/lib");

  // Print all scan directories
  HLOG(kDebug, "ChiMod scan directories:");
  for (const auto &dir : directories) {
    HLOG(kDebug, "  {}", dir);
  }

  return directories;
}

std::string ModuleManager::GetModuleDirectory() const {
  Dl_info dl_info;
  // Use address of a function in this shared object to identify it
  void *symbol_addr = GetSymbolForDlAddr();

  if (dladdr(symbol_addr, &dl_info) == 0) {
    return "";
  }

  char resolved_path[PATH_MAX];
  if (realpath(dl_info.dli_fname, resolved_path) == nullptr) {
    return "";
  }

  char path_copy[PATH_MAX];
  strncpy(path_copy, resolved_path, PATH_MAX);
  path_copy[PATH_MAX - 1] = '\0';  // Ensure null termination
  return std::string(dirname(path_copy));
}

bool ModuleManager::IsSharedLibrary(const std::string &file_path) const {
  // Check file extension
#ifdef _WIN32
  const std::string ext = ".dll";
#elif __APPLE__
  const std::string ext = ".dylib";
#else
  const std::string ext = ".so";
#endif
  return file_path.length() >= ext.length() &&
         file_path.compare(file_path.length() - ext.length(), ext.length(),
                           ext) == 0;
}

bool ModuleManager::HasModuleNamingConvention(
    const std::string &file_path) const {
  // ChiMod libraries must contain "_runtime" in their name
  return file_path.find("_runtime") != std::string::npos;
}

bool ModuleManager::ValidateChiMod(hshm::SharedLibrary &lib) const {
  // Check for required ChiMod functions
  void *alloc_func = lib.GetSymbol("alloc_chimod");
  void *new_func = lib.GetSymbol("new_chimod");
  void *name_func = lib.GetSymbol("get_chimod_name");
  void *destroy_func = lib.GetSymbol("destroy_chimod");

  return (alloc_func != nullptr && new_func != nullptr &&
          name_func != nullptr && destroy_func != nullptr);
}

}  // namespace chi