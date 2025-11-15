#ifndef WRP_CEE_API_CONTEXT_INTERFACE_H_
#define WRP_CEE_API_CONTEXT_INTERFACE_H_

#include <string>
#include <vector>
#include <wrp_cae/core/factory/assimilation_ctx.h>

namespace iowarp {

/**
 * ContextInterface - High-level API for context exploration and management
 *
 * Provides a unified interface for bundling, querying, retrieving, and destroying
 * contexts in the IOWarp system. Integrates with the Context Assimilation Engine (CAE)
 * and Context Transfer Engine (CTE).
 */
class ContextInterface {
public:
  /**
   * Default constructor
   */
  ContextInterface();

  /**
   * Destructor
   */
  ~ContextInterface();

  /**
   * Bundle a group of related objects together and assimilate them
   *
   * This method takes a vector of AssimilationCtx objects and calls the CAE
   * ParseOmni function to schedule assimilation tasks for each context.
   *
   * @param bundle Vector of AssimilationCtx objects to assimilate
   * @return 0 on success, non-zero error code on failure
   */
  int ContextBundle(const std::vector<wrp_cae::core::AssimilationCtx> &bundle);

  /**
   * Retrieve the identities of objects matching tag and blob patterns
   *
   * Queries the CTE system for blobs matching the specified regex patterns.
   * This is useful for discovering what objects are available in the system.
   *
   * @param tag_re Tag regex pattern to match
   * @param blob_re Blob regex pattern to match
   * @param max_results Maximum number of results to return (0 = unlimited)
   * @return Vector of matching blob names
   */
  std::vector<std::string> ContextQuery(const std::string &tag_re,
                                         const std::string &blob_re,
                                         unsigned int max_results = 0);

  /**
   * Retrieve the identities and data of objects matching patterns
   *
   * This method queries for blobs matching the specified patterns and retrieves their
   * data into a packed binary buffer. Blobs are retrieved in batches for efficiency.
   *
   * The retrieved data is packed sequentially into a buffer and returned as a single
   * packed string. Buffer is automatically allocated and freed.
   *
   * @param tag_re Tag regex pattern to match
   * @param blob_re Blob regex pattern to match
   * @param max_results Maximum number of blobs to retrieve (0 = unlimited, default: 1024)
   * @param max_context_size Maximum total context size in bytes (default: 256MB)
   * @param batch_size Number of concurrent AsyncGetBlob operations (default: 32)
   * @return Vector containing one string with packed binary context data (empty if no data)
   */
  std::vector<std::string> ContextRetrieve(const std::string &tag_re,
                                            const std::string &blob_re,
                                            unsigned int max_results = 1024,
                                            size_t max_context_size = 256 * 1024 * 1024,
                                            unsigned int batch_size = 32);

  /**
   * Split/splice objects into a new context
   *
   * NOTE: This functionality is not yet implemented and will be added in a future version.
   *
   * @param new_ctx Name of the new context to create
   * @param tag_re Tag regex pattern to match for source objects
   * @param blob_re Blob regex pattern to match for source objects
   * @return 0 on success, non-zero error code on failure
   */
  int ContextSplice(const std::string &new_ctx,
                    const std::string &tag_re,
                    const std::string &blob_re);

  /**
   * Destroy contexts by name
   *
   * Deletes the specified contexts (tags) from the CTE system. Each context name
   * is treated as a tag name and deleted using the CTE DelTag API.
   *
   * @param context_names Vector of context names to destroy
   * @return 0 on success, non-zero error code on failure
   */
  int ContextDestroy(const std::vector<std::string> &context_names);

private:
  bool is_initialized_;  /**< Flag indicating whether the interface is initialized */
};

}  // namespace iowarp

#endif  // WRP_CEE_API_CONTEXT_INTERFACE_H_
