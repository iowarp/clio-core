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
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::Chimaera, g_chimaera_manager);

// Macro for accessing the Chimaera manager singleton using global pointer variable
#define CHI_CHIMAERA_MANAGER HSHM_GET_GLOBAL_PTR_VAR(::chi::Chimaera, g_chimaera_manager)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CHIMAERA_MANAGER_H_