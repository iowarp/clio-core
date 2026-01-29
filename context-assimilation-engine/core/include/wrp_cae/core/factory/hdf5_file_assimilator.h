#ifndef WRP_CAE_CORE_HDF5_FILE_ASSIMILATOR_H_
#define WRP_CAE_CORE_HDF5_FILE_ASSIMILATOR_H_

#include <wrp_cae/core/factory/base_assimilator.h>
#include <hdf5.h>
#include <string>
#include <vector>
#include <memory>

// Forward declaration
namespace wrp_cte::core {
class Client;
}  // namespace wrp_cte::core

namespace wrp_cae::core {

/**
 * Hdf5FileAssimilator - Handles assimilation of HDF5 files
 * Discovers all datasets in an HDF5 file and transfers them to CTE
 * Each dataset is tagged with a unique identifier and includes:
 * - A description blob with tensor metadata (type and dimensions)
 * - Data chunks for efficient transfer
 */
class Hdf5FileAssimilator : public BaseAssimilator {
 public:
  /**
   * Constructor with CTE client
   * @param cte_client Shared pointer to initialized CTE client
   */
  explicit Hdf5FileAssimilator(std::shared_ptr<wrp_cte::core::Client> cte_client);

  /**
   * Schedule assimilation tasks for an HDF5 file
   * This is a coroutine that uses co_await for async CTE operations.
   * @param ctx Assimilation context with source, destination, and metadata
   * @param error_code Output: 0 on success, non-zero error code on failure
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume Schedule(const AssimilationCtx& ctx, int& error_code) override;

 private:
  /**
   * Check if a dataset path matches the filter patterns
   * @param dataset_path Dataset path to check (e.g., "/data/temperature")
   * @param include_patterns Glob patterns to include (empty means include all)
   * @param exclude_patterns Glob patterns to exclude
   * @return true if dataset should be included, false otherwise
   */
  bool MatchesFilter(const std::string& dataset_path,
                     const std::vector<std::string>& include_patterns,
                     const std::vector<std::string>& exclude_patterns);

  /**
   * Check if a string matches a glob pattern
   * @param str String to check
   * @param pattern Glob pattern (supports *, ?, [])
   * @return true if string matches pattern
   */
  bool MatchGlobPattern(const std::string& str, const std::string& pattern);

  /**
   * Open HDF5 file in read-only mode (serial)
   * @param file_path Path to the HDF5 file
   * @return HDF5 file ID, or negative value on error
   */
  hid_t OpenHdf5File(const std::string& file_path);

  /**
   * Close HDF5 file
   * @param file_id HDF5 file ID to close
   */
  void CloseHdf5File(hid_t file_id);

  /**
   * Discover all datasets in HDF5 file using visitor pattern
   * @param file_id HDF5 file ID
   * @param dataset_paths Output vector of dataset paths
   * @return 0 on success, non-zero error code on failure
   */
  int DiscoverDatasets(hid_t file_id, std::vector<std::string>& dataset_paths);

 public:
  /**
   * Process a single dataset: create tag, store description, transfer chunks
   * This is a coroutine that uses co_await for async CTE operations.
   * Made public for use by Runtime::ProcessHdf5Dataset for distributed processing.
   * @param file_id HDF5 file ID
   * @param dataset_path Path to dataset within file (e.g., "/data/temperature")
   * @param tag_prefix Prefix for tag name (destination path without protocol)
   * @param error_code Output: 0 on success, non-zero error code on failure
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume ProcessDataset(hid_t file_id, const std::string& dataset_path,
                                  const std::string& tag_prefix, int& error_code);

 private:

  /**
   * Get human-readable type name for HDF5 datatype
   * @param datatype HDF5 datatype ID
   * @return Type name string (int32, float64, etc.) or "unknown"
   */
  std::string GetTypeName(hid_t datatype);

  /**
   * Format tensor description string for dataset
   * @param datatype HDF5 datatype ID
   * @param dims Vector of dimension sizes
   * @return Formatted string like "tensor<float64, 100, 200>"
   */
  std::string FormatTensorDescription(hid_t datatype,
                                      const std::vector<hsize_t>& dims);

  /**
   * Extract protocol from URL (part before ::)
   * @param url URL in format protocol::path
   * @return Protocol string, or empty string if no protocol found
   */
  std::string GetUrlProtocol(const std::string& url);

  /**
   * Extract path from URL (part after ::)
   * @param url URL in format protocol::path
   * @return Path string, or empty string if no protocol found
   */
  std::string GetUrlPath(const std::string& url);

  /**
   * Callback for HDF5 link iteration (static wrapper)
   * @param loc_id Location ID
   * @param name Object name
   * @param info Object info
   * @param operator_data User data (vector of dataset paths)
   * @return 0 to continue iteration, non-zero to stop
   */
  static herr_t VisitCallback(hid_t loc_id, const char* name,
                              const H5L_info_t* info, void* operator_data);

  std::shared_ptr<wrp_cte::core::Client> cte_client_;
};

}  // namespace wrp_cae::core

#endif  // WRP_CAE_CORE_HDF5_FILE_ASSIMILATOR_H_
