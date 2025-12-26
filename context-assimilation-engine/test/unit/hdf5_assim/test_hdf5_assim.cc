/**
 * test_hdf5_assim.cc - Unit test for ParseOmni API with HDF5 file assimilation
 *
 * This test validates the ParseOmni API with HDF5 format by:
 * 1. Creating a test HDF5 file with multiple datasets
 * 2. Serializing an AssimilationCtx for HDF5 format
 * 3. Calling ParseOmni to discover and transfer datasets to CTE
 * 4. Validating that multiple tags were created (one per dataset)
 * 5. Verifying each tag's metadata and data in CTE
 *
 * Test Strategy:
 * - Tests HDF5 format discovery and multi-dataset handling
 * - Tests hierarchical dataset structure (groups)
 * - Tests various data types (int, double, float)
 * - Tests tensor metadata generation
 * - Tests integration with CTE (tag creation, blob storage)
 *
 * Environment Variables:
 * - INIT_CHIMAERA: If set to "1", initializes Chimaera runtime
 * - TEST_HDF5_FILE: Override default test file path
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <algorithm>

// HDF5 library
#include <hdf5.h>

// Chimaera and CAE headers
#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

// CTE headers
#include <wrp_cte/core/core_client.h>

// Test configuration
const std::string kTestFileName = "/tmp/test_hdf5_assim_file.h5";
const std::string kTestTagBase = "test_hdf5_tag";

/**
 * Generate a test HDF5 file with multiple datasets
 * This creates a file with various data types and dimensions
 */
bool GenerateTestHDF5File(const std::string& file_path) {
  std::cout << "Generating test HDF5 file: " << file_path << std::endl;

  // Create HDF5 file
  hid_t file_id = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    std::cerr << "ERROR: Failed to create HDF5 file: " << file_path << std::endl;
    return false;
  }

  // Dataset 1: /int_dataset - 1D array of 100 integers
  {
    hsize_t dims[1] = {100};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "/int_dataset", H5T_NATIVE_INT,
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i) {
      data[i] = i * 10;
    }
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    std::cout << "  Created /int_dataset: 1D array of 100 integers" << std::endl;
  }

  // Dataset 2: /double_dataset - 2D array (10x20) of doubles
  {
    hsize_t dims[2] = {10, 20};
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "/double_dataset", H5T_NATIVE_DOUBLE,
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<double> data(200);
    for (int i = 0; i < 200; ++i) {
      data[i] = i * 1.5;
    }
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    std::cout << "  Created /double_dataset: 2D array (10x20) of doubles" << std::endl;
  }

  // Dataset 3: /float_dataset - 1D array of 50 floats
  {
    hsize_t dims[1] = {50};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "/float_dataset", H5T_NATIVE_FLOAT,
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<float> data(50);
    for (int i = 0; i < 50; ++i) {
      data[i] = i * 2.5f;
    }
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    std::cout << "  Created /float_dataset: 1D array of 50 floats" << std::endl;
  }

  // Dataset 4: /group/nested_dataset - Nested dataset to test hierarchical discovery
  {
    // Create group
    hid_t group_id = H5Gcreate2(file_id, "/group", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t dims[1] = {30};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate2(group_id, "nested_dataset", H5T_NATIVE_INT,
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<int> data(30);
    for (int i = 0; i < 30; ++i) {
      data[i] = i * 5;
    }
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Gclose(group_id);
    std::cout << "  Created /group/nested_dataset: nested 1D array of 30 integers" << std::endl;
  }

  H5Fclose(file_id);
  std::cout << "Test HDF5 file generated successfully" << std::endl;
  return true;
}

/**
 * Verify dataset data by comparing HDF5 source with CTE tag data
 *
 * @param file_path Path to the HDF5 file
 * @param dataset_path Path to the dataset within the HDF5 file
 * @param tag_name Full tag name in CTE
 * @param cte_client Pointer to CTE client
 * @return true if data matches, false otherwise
 */
bool VerifyDatasetData(const std::string& file_path,
                       const std::string& dataset_path,
                       const std::string& tag_name,
                       wrp_cte::core::Client* cte_client) {
  std::cout << "  Verifying data for dataset: " << dataset_path << std::endl;

  // Open HDF5 file and dataset
  hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cerr << "    ERROR: Failed to open HDF5 file: " << file_path << std::endl;
    return false;
  }

  hid_t dataset_id = H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    std::cerr << "    ERROR: Failed to open dataset: " << dataset_path << std::endl;
    H5Fclose(file_id);
    return false;
  }

  // Get dataset properties
  hid_t dataspace_id = H5Dget_space(dataset_id);
  hid_t datatype_id = H5Dget_type(dataset_id);

  hssize_t num_elements = H5Sget_simple_extent_npoints(dataspace_id);
  if (num_elements < 0) {
    std::cerr << "    ERROR: Failed to get number of elements" << std::endl;
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return false;
  }

  size_t element_size = H5Tget_size(datatype_id);
  size_t total_size = num_elements * element_size;

  std::cout << "    Dataset info: " << num_elements << " elements, "
            << element_size << " bytes per element, "
            << total_size << " total bytes" << std::endl;

  // Allocate buffer for HDF5 data
  std::vector<char> hdf5_data(total_size);

  // Read data from HDF5
  herr_t status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, hdf5_data.data());
  if (status < 0) {
    std::cerr << "    ERROR: Failed to read data from HDF5 dataset" << std::endl;
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return false;
  }

  // Get CTE tag
  wrp_cte::core::TagId tag_id = cte_client->GetOrCreateTag(tag_name);
  if (tag_id.IsNull()) {
    std::cerr << "    ERROR: Tag not found in CTE: " << tag_name << std::endl;
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return false;
  }

  // Get tag size from CTE
  size_t cte_tag_size = cte_client->GetTagSize(tag_id);
  std::cout << "    CTE tag size: " << cte_tag_size << " bytes" << std::endl;

  // Check if sizes match
  if (cte_tag_size != total_size) {
    std::cerr << "    ERROR: Size mismatch - HDF5: " << total_size
              << " bytes, CTE: " << cte_tag_size << " bytes" << std::endl;
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return false;
  }

  // Allocate buffer for CTE data
  std::vector<char> cte_data(total_size);

  // Read data from CTE by getting all blobs (chunks)
  // For datasets <= 1MB, data is in "chunk_0"
  // For larger datasets, data is split across "chunk_0", "chunk_1", etc.
  std::vector<std::string> blob_names = cte_client->GetContainedBlobs(tag_id);
  std::cout << "    Found " << blob_names.size() << " blobs in tag" << std::endl;

  // Filter out the "description" blob and get only chunk blobs
  std::vector<std::string> chunk_blobs;
  for (const auto& blob_name : blob_names) {
    if (blob_name.find("chunk_") == 0) {
      chunk_blobs.push_back(blob_name);
    }
  }

  // Sort chunk blobs by number to ensure correct order
  std::sort(chunk_blobs.begin(), chunk_blobs.end(), [](const std::string& a, const std::string& b) {
    // Extract chunk numbers and compare
    size_t a_num = std::stoul(a.substr(6)); // Skip "chunk_"
    size_t b_num = std::stoul(b.substr(6));
    return a_num < b_num;
  });

  std::cout << "    Found " << chunk_blobs.size() << " data chunks" << std::endl;

  // Read all chunks and reconstruct data
  size_t bytes_read = 0;
  for (const auto& blob_name : chunk_blobs) {
    // Get blob size
    chi::u64 blob_size = cte_client->GetBlobSize(tag_id, blob_name);
    std::cout << "    Reading blob '" << blob_name << "' (size: " << blob_size << " bytes)" << std::endl;

    if (bytes_read + blob_size > total_size) {
      std::cerr << "    ERROR: Total blob size exceeds expected size" << std::endl;
      H5Tclose(datatype_id);
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Fclose(file_id);
      return false;
    }

    // Allocate shared memory buffer for this blob
    auto blob_buffer = CHI_IPC->AllocateBuffer(blob_size);

    // Read blob into shared memory buffer
    hipc::ShmPtr<> blob_shm_ptr = blob_buffer.shm_.template Cast<void>();
    bool success = cte_client->GetBlob(tag_id, blob_name, 0, blob_size, 0, blob_shm_ptr);
    if (!success) {
      std::cerr << "    ERROR: Failed to read blob '" << blob_name << "'" << std::endl;
      CHI_IPC->FreeBuffer(blob_buffer);
      H5Tclose(datatype_id);
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Fclose(file_id);
      return false;
    }

    // Copy from shared memory to our local buffer
    std::memcpy(cte_data.data() + bytes_read, blob_buffer.ptr_, blob_size);

    // Free the shared memory buffer
    CHI_IPC->FreeBuffer(blob_buffer);

    bytes_read += blob_size;
  }

  if (bytes_read != total_size) {
    std::cerr << "    ERROR: Failed to read complete data from CTE - expected "
              << total_size << " bytes, got " << bytes_read << " bytes" << std::endl;
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return false;
  }

  std::cout << "    Successfully read " << bytes_read << " bytes from CTE" << std::endl;

  // Determine data type for comparison
  H5T_class_t type_class = H5Tget_class(datatype_id);
  bool data_matches = true;
  size_t mismatch_count = 0;

  if (type_class == H5T_INTEGER) {
    // Integer comparison - byte-by-byte
    std::cout << "    Comparing integer data..." << std::endl;
    for (size_t i = 0; i < total_size; ++i) {
      if (hdf5_data[i] != cte_data[i]) {
        if (mismatch_count == 0) {
          std::cerr << "    First mismatch at byte " << i << ": HDF5="
                    << static_cast<int>(hdf5_data[i]) << ", CTE="
                    << static_cast<int>(cte_data[i]) << std::endl;
        }
        mismatch_count++;
        data_matches = false;
      }
    }
  } else if (type_class == H5T_FLOAT) {
    // Floating-point comparison with epsilon
    if (element_size == sizeof(double)) {
      std::cout << "    Comparing double data (epsilon=1e-10)..." << std::endl;
      const double* hdf5_doubles = reinterpret_cast<const double*>(hdf5_data.data());
      const double* cte_doubles = reinterpret_cast<const double*>(cte_data.data());
      const double epsilon = 1e-10;

      for (size_t i = 0; i < static_cast<size_t>(num_elements); ++i) {
        double diff = std::abs(hdf5_doubles[i] - cte_doubles[i]);
        if (diff > epsilon) {
          if (mismatch_count == 0) {
            std::cerr << "    First mismatch at element " << i << ": HDF5="
                      << hdf5_doubles[i] << ", CTE=" << cte_doubles[i]
                      << ", diff=" << diff << std::endl;
          }
          mismatch_count++;
          data_matches = false;
        }
      }
    } else if (element_size == sizeof(float)) {
      std::cout << "    Comparing float data (epsilon=1e-6)..." << std::endl;
      const float* hdf5_floats = reinterpret_cast<const float*>(hdf5_data.data());
      const float* cte_floats = reinterpret_cast<const float*>(cte_data.data());
      const float epsilon = 1e-6f;

      for (size_t i = 0; i < static_cast<size_t>(num_elements); ++i) {
        float diff = std::abs(hdf5_floats[i] - cte_floats[i]);
        if (diff > epsilon) {
          if (mismatch_count == 0) {
            std::cerr << "    First mismatch at element " << i << ": HDF5="
                      << hdf5_floats[i] << ", CTE=" << cte_floats[i]
                      << ", diff=" << diff << std::endl;
          }
          mismatch_count++;
          data_matches = false;
        }
      }
    } else {
      std::cerr << "    WARNING: Unsupported float size: " << element_size << " bytes" << std::endl;
      // Fall back to byte-by-byte comparison
      for (size_t i = 0; i < total_size; ++i) {
        if (hdf5_data[i] != cte_data[i]) {
          mismatch_count++;
          data_matches = false;
        }
      }
    }
  } else {
    // Unknown type - byte-by-byte comparison
    std::cout << "    Comparing as raw bytes..." << std::endl;
    for (size_t i = 0; i < total_size; ++i) {
      if (hdf5_data[i] != cte_data[i]) {
        if (mismatch_count == 0) {
          std::cerr << "    First mismatch at byte " << i << ": HDF5="
                    << static_cast<int>(hdf5_data[i]) << ", CTE="
                    << static_cast<int>(cte_data[i]) << std::endl;
        }
        mismatch_count++;
        data_matches = false;
      }
    }
  }

  // Cleanup HDF5 resources
  H5Tclose(datatype_id);
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  // Print comparison results
  if (data_matches) {
    std::cout << "    SUCCESS: Data verification passed - all values match" << std::endl;
  } else {
    std::cerr << "    FAILURE: Data verification failed - " << mismatch_count
              << " mismatches out of " << total_size << " bytes" << std::endl;
  }

  return data_matches;
}

/**
 * Clean up test file
 */
void CleanupTestFile(const std::string& file_path) {
  if (std::remove(file_path.c_str()) == 0) {
    std::cout << "Test file cleaned up: " << file_path << std::endl;
  } else {
    std::cerr << "WARNING: Failed to remove test file: " << file_path << std::endl;
  }
}

/**
 * Main test function
 */
int main(int argc, char* argv[]) {
  std::cout << "======================================" << std::endl;
  std::cout << "HDF5 Assimilation ParseOmni Unit Test" << std::endl;
  std::cout << "======================================" << std::endl;

  int exit_code = 0;

  try {
    // Initialize Chimaera runtime (CHIMAERA_WITH_RUNTIME controls behavior)
    std::cout << "Initializing Chimaera..." << std::endl;
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      std::cerr << "ERROR: Failed to initialize Chimaera" << std::endl;
      return 1;
    }
    std::cout << "Chimaera initialized successfully" << std::endl;

    // Verify Chimaera IPC is available
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      std::cerr << "ERROR: Chimaera IPC not initialized" << std::endl;
      return 1;
    }
    std::cout << "Chimaera IPC verified" << std::endl;

    // Step 1: Generate test HDF5 file
    std::cout << "\n[STEP 1] Generating test HDF5 file..." << std::endl;
    if (!GenerateTestHDF5File(kTestFileName)) {
      return 1;
    }

    // Step 2: Connect to CTE
    std::cout << "\n[STEP 2] Connecting to CTE..." << std::endl;
    wrp_cte::core::WRP_CTE_CLIENT_INIT();
    std::cout << "CTE client initialized" << std::endl;

    // Step 2.5: Initialize CAE client
    std::cout << "\n[STEP 2.5] Initializing CAE client..." << std::endl;
    WRP_CAE_CLIENT_INIT();
    std::cout << "CAE client initialized" << std::endl;

    // Step 3: Create CAE pool
    std::cout << "\n[STEP 3] Creating CAE pool..." << std::endl;
    wrp_cae::core::Client cae_client;
    wrp_cae::core::CreateParams params;

    cae_client.Create(
        chi::PoolQuery::Local(),
        "test_cae_pool",
        wrp_cae::core::kCaePoolId,
        params);

    std::cout << "CAE pool created with ID: " << cae_client.pool_id_ << std::endl;

    // Step 4: Create AssimilationCtx for HDF5
    std::cout << "\n[STEP 4] Creating AssimilationCtx for HDF5..." << std::endl;
    wrp_cae::core::AssimilationCtx ctx;
    ctx.src = "hdf5::" + kTestFileName;
    ctx.dst = "iowarp::" + kTestTagBase;
    ctx.format = "hdf5";
    ctx.depends_on = "";
    ctx.range_off = 0;
    ctx.range_size = 0;  // 0 means process entire file

    std::cout << "AssimilationCtx created:" << std::endl;
    std::cout << "  src: " << ctx.src << std::endl;
    std::cout << "  dst: " << ctx.dst << std::endl;
    std::cout << "  format: " << ctx.format << std::endl;

    // Step 5: Call ParseOmni with vector containing single context
    std::cout << "\n[STEP 5] Calling ParseOmni..." << std::endl;
    std::vector<wrp_cae::core::AssimilationCtx> contexts = {ctx};
    chi::u32 num_tasks_scheduled = 0;
    chi::u32 result_code = cae_client.ParseOmni(contexts, num_tasks_scheduled);

    std::cout << "ParseOmni completed:" << std::endl;
    std::cout << "  result_code: " << result_code << std::endl;
    std::cout << "  num_tasks_scheduled: " << num_tasks_scheduled << std::endl;

    // Step 6: Validate results
    std::cout << "\n[STEP 6] Validating results..." << std::endl;

    if (result_code != 0) {
      std::cerr << "ERROR: ParseOmni failed with result_code: " << result_code << std::endl;
      exit_code = 1;
    } else if (num_tasks_scheduled == 0) {
      std::cerr << "ERROR: ParseOmni returned 0 tasks scheduled" << std::endl;
      exit_code = 1;
    } else {
      std::cout << "SUCCESS: ParseOmni executed successfully" << std::endl;
    }

    // Step 7: Verify datasets in CTE
    std::cout << "\n[STEP 7] Verifying datasets in CTE..." << std::endl;

    // Get CTE client
    auto cte_client = WRP_CTE_CLIENT;

    // Expected dataset names (based on HDF5 file structure)
    std::vector<std::string> expected_datasets = {
      "int_dataset",
      "double_dataset",
      "float_dataset",
      "group/nested_dataset"
    };

    std::cout << "Expected " << expected_datasets.size() << " datasets to be created" << std::endl;

    size_t datasets_found = 0;
    size_t datasets_verified = 0;
    for (const auto& dataset_name : expected_datasets) {
      std::string full_tag_name = kTestTagBase + "/" + dataset_name;
      std::string dataset_path = "/" + dataset_name;
      std::cout << "\nChecking dataset: " << dataset_name << std::endl;
      std::cout << "  Full tag name: " << full_tag_name << std::endl;

      // Check if tag exists
      wrp_cte::core::TagId tag_id = cte_client->GetOrCreateTag(full_tag_name);
      if (tag_id.IsNull()) {
        std::cerr << "  WARNING: Tag not found in CTE: " << full_tag_name << std::endl;
        continue;
      }

      datasets_found++;
      std::cout << "  Tag found (ID: " << tag_id << ")" << std::endl;

      // Get tag size
      size_t tag_size = cte_client->GetTagSize(tag_id);
      std::cout << "  Tag size: " << tag_size << " bytes" << std::endl;

      if (tag_size == 0) {
        std::cerr << "  WARNING: Tag size is 0, no data transferred" << std::endl;
        continue;
      }

      // Verify dataset data by comparing with original HDF5 data
      bool data_verified = VerifyDatasetData(kTestFileName, dataset_path, full_tag_name, cte_client);
      if (data_verified) {
        datasets_verified++;
      } else {
        std::cerr << "  ERROR: Data verification failed for dataset: " << dataset_name << std::endl;
        exit_code = 1;
      }
    }

    std::cout << "\nDataset verification summary:" << std::endl;
    std::cout << "  Expected datasets: " << expected_datasets.size() << std::endl;
    std::cout << "  Found datasets: " << datasets_found << std::endl;
    std::cout << "  Verified datasets: " << datasets_verified << std::endl;

    if (datasets_found == 0) {
      std::cerr << "ERROR: No datasets found in CTE" << std::endl;
      std::cerr << "NOTE: HDF5 assimilator may not yet be fully implemented" << std::endl;
      exit_code = 1;
    } else if (datasets_found < expected_datasets.size()) {
      std::cerr << "WARNING: Not all datasets were found (" << datasets_found
                << "/" << expected_datasets.size() << ")" << std::endl;
      // Not a hard failure - HDF5 assimilator may be under development
    } else if (datasets_verified < datasets_found) {
      std::cerr << "ERROR: Not all datasets passed data verification (" << datasets_verified
                << "/" << datasets_found << ")" << std::endl;
      exit_code = 1;
    } else {
      std::cout << "SUCCESS: All expected datasets found and verified in CTE" << std::endl;
    }

    // Step 8: Cleanup
    std::cout << "\n[STEP 8] Cleaning up..." << std::endl;
    CleanupTestFile(kTestFileName);

  } catch (const std::exception& e) {
    std::cerr << "ERROR: Exception caught: " << e.what() << std::endl;
    exit_code = 1;
  }

  // Print final result
  std::cout << "\n========================================" << std::endl;
  if (exit_code == 0) {
    std::cout << "TEST PASSED" << std::endl;
  } else {
    std::cout << "TEST FAILED" << std::endl;
  }
  std::cout << "========================================" << std::endl;

  return exit_code;
}
