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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CHIMAERA_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CHIMAERA_MANAGER_H_

#include "chimaera/types.h"
#include <memory>
#include <string>

namespace chi {

/**
 * Main Chimaera manager singleton class
 * 
 * Central coordinator for the distributed task execution framework.
 * Manages initialization and coordination between client and runtime modes.
 * Uses HSHM global cross pointer variable singleton pattern.
 */
class Chimaera {
 public:
  /**
   * Destructor - handles automatic finalization
   */
  ~Chimaera();

  /**
   * Initialize client components
   * @return true if initialization successful, false otherwise
   */
  bool ClientInit();

  /**
   * Initialize server/runtime components
   * @return true if initialization successful, false otherwise
   */
  bool ServerInit();


  /**
   * Finalize client components only
   */
  void ClientFinalize();

  /**
   * Finalize server/runtime components only
   */
  void ServerFinalize();

  /**
   * Check if Chimaera is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const;

  /**
   * Check if running in client mode
   * @return true if client mode, false otherwise
   */
  bool IsClient() const;

  /**
   * Check if running in runtime/server mode
   * @return true if runtime mode, false otherwise
   */
  bool IsRuntime() const;

  /**
   * Get the current hostname identified during initialization
   * @return hostname string
   */
  const std::string& GetCurrentHostname() const;

  /**
   * Get the 64-bit node ID stored in shared memory
   * @return 64-bit node ID, 0 if not available
   */
  u64 GetNodeId() const;

  /**
   * Check if Chimaera is currently in the process of initializing
   * @return true if either client or runtime initialization is in progress, false otherwise
   */
  bool IsInitializing() const;

 private:
  bool is_initialized_ = false;
  bool is_client_mode_ = false;
  bool is_runtime_mode_ = false;
  bool is_client_initialized_ = false;
  bool is_runtime_initialized_ = false;
  bool client_is_initializing_ = false;
  bool runtime_is_initializing_ = false;


};

}  // namespace chi

// Global pointer variable declaration for Chimaera manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H_DLL(CHI_DLL, chi::Chimaera, g_chimaera_manager);

// Macro for accessing the Chimaera manager singleton using global pointer variable
#define CHI_CHIMAERA_MANAGER HSHM_GET_GLOBAL_PTR_VAR(::chi::Chimaera, g_chimaera_manager)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CHIMAERA_MANAGER_H_