# CTE Core API Documentation

## Overview

The Content Transfer Engine (CTE) Core is a high-performance distributed storage middleware system built on the Chimaera framework. It provides a flexible blob storage API with advanced features including:

- **Multi-target Storage Management**: Register and manage multiple storage backends (file, RAM, NVMe)
- **Blob Storage with Tags**: Store and retrieve data blobs with tag-based organization
- **Block-based Data Management**: Efficient block-level data placement across multiple targets
- **Performance Monitoring**: Built-in telemetry and performance metrics collection
- **Configurable Data Placement**: Multiple data placement algorithms (random, round-robin, max bandwidth)
- **Asynchronous Operations**: Async-only API with C++20 coroutine support

CTE Core implements a ChiMod (Chimaera Module) that integrates with the Chimaera distributed runtime system, providing scalable data management across multiple nodes in a cluster.

## Installation & Linking

### Prerequisites

- CMake 3.20 or higher
- C++17 compatible compiler
- Chimaera framework (chimaera and chimaera_admin packages)
- yaml-cpp library
- Python 3.7+ (for Python bindings)
- nanobind (for Python bindings)

### Building CTE Core

```bash
# Clone the repository
git clone <repository-url>
cd content-transfer-engine

# Create build directory
mkdir build && cd build

# Configure with CMake (using debug preset as recommended)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build the project
make -j

# Install (optional)
sudo make install
```

### Linking to CTE Core in CMake Projects

To use CTE Core in your CMake project, follow the patterns established in the MODULE_DEVELOPMENT_GUIDE.md. Add the following to your `CMakeLists.txt`:

```cmake
# Find required Chimaera framework packages
find_package(chimaera REQUIRED)              # Core Chimaera framework
find_package(chimaera_admin REQUIRED)        # Admin ChiMod (required)

# Find CTE Core ChiMod package
find_package(wrp_cte_core REQUIRED)          # CTE Core ChiMod

# Create your executable or library
add_executable(my_app main.cpp)

# Link against CTE Core libraries using modern target aliases
target_link_libraries(my_app 
  PRIVATE 
    wrp_cte::core_client                     # CTE Core client library
    # wrp_cte::core_runtime                  # Optional - if you need runtime functionality
    # chimaera::admin_client                 # Optional - if you need admin functionality
)

# Note: Include directories are handled automatically by the ChiMod targets
# No manual target_include_directories() call needed
```

#### Package and Target Naming

CTE Core follows the Chimaera ChiMod naming conventions:

- **Package Name**: `wrp_cte_core` (for `find_package(wrp_cte_core REQUIRED)`)
- **Target Aliases**: `wrp_cte::core_client`, `wrp_cte::core_runtime` (recommended for linking)
- **Actual Targets**: `wrp_cte_core_client`, `wrp_cte_core_runtime`
- **Library Files**: `libwrp_cte_core_client.so`, `libwrp_cte_core_runtime.so`
- **Include Path**: `wrp_cte/core/` (e.g., `#include <wrp_cte/core/core_client.h>`)

#### Dependency Management

The CTE Core ChiMod targets automatically include all required dependencies:

- **Core Chimaera Framework**: Automatically linked via `wrp_cte::core_client` target
- **Admin ChiMod**: Available via `chimaera::admin_client` if needed
- **Include Paths**: Automatically configured by ChiMod targets
- **System Dependencies**: Handled by the build system (threading, YAML, etc.)

External applications only need to link against the CTE Core targets - all framework dependencies are resolved automatically.

### Runtime Dependencies

The CTE Core runtime library (`libwrp_cte_core_runtime.so`) must be available at runtime. It will be automatically loaded by the Chimaera framework when the CTE Core container is created.

### External Application Example

For external applications using CTE Core, follow these patterns (based on the MODULE_DEVELOPMENT_GUIDE.md):

```cmake
# External application CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(my_cte_application)

set(CMAKE_CXX_STANDARD_20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(chimaera REQUIRED)              # Core Chimaera framework
find_package(chimaera_admin REQUIRED)        # Admin ChiMod
find_package(wrp_cte_core REQUIRED)          # CTE Core ChiMod

# Find additional dependencies
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# Create your application
add_executable(my_cte_app main.cpp)

# Link with CTE Core - dependencies are automatically included
target_link_libraries(my_cte_app
  wrp_cte::core_client                       # CTE Core client (required)
  # wrp_cte::core_runtime                    # Optional - if needed
  # chimaera::admin_client                   # Optional - if needed
  ${CMAKE_THREAD_LIBS_INIT}                 # Threading support
)
```

## API Reference

### Core Client Class

The main entry point for CTE Core functionality is the `wrp_cte::core::Client` class.

#### Class Definition

The CTE client provides an **async-only API**. All methods return `chi::Future<TaskType>` for asynchronous completion.

```cpp
namespace wrp_cte::core {

class Client : public chi::ContainerClient {
public:
  // Constructors
  Client();
  explicit Client(const chi::PoolId &pool_id);

  // Container lifecycle
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery &pool_query,
      const std::string &pool_name,
      const chi::PoolId &custom_pool_id,
      const CreateParams &params = CreateParams());

  // Target management
  chi::Future<RegisterTargetTask> AsyncRegisterTarget(
      const std::string &target_name,
      chimaera::bdev::BdevType bdev_type,
      chi::u64 total_size,
      const chi::PoolQuery &target_query = chi::PoolQuery::Local(),
      const chi::PoolId &bdev_id = chi::PoolId::GetNull());

  chi::Future<UnregisterTargetTask> AsyncUnregisterTarget(
      const std::string &target_name);

  chi::Future<ListTargetsTask> AsyncListTargets();

  chi::Future<StatTargetsTask> AsyncStatTargets();

  // Tag management
  chi::Future<GetOrCreateTagTask<CreateParams>> AsyncGetOrCreateTag(
      const std::string &tag_name,
      const TagId &tag_id = TagId::GetNull());

  chi::Future<DelTagTask> AsyncDelTag(const TagId &tag_id);
  chi::Future<DelTagTask> AsyncDelTag(const std::string &tag_name);

  chi::Future<GetTagSizeTask> AsyncGetTagSize(const TagId &tag_id);

  // Blob operations
  chi::Future<PutBlobTask> AsyncPutBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      hipc::ShmPtr<> blob_data,
      float score, chi::u32 flags);

  chi::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      chi::u32 flags,
      hipc::ShmPtr<> blob_data);

  chi::Future<DelBlobTask> AsyncDelBlob(
      const TagId &tag_id,
      const std::string &blob_name);

  chi::Future<ReorganizeBlobTask> AsyncReorganizeBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      float new_score);

  // Blob metadata operations
  chi::Future<GetBlobScoreTask> AsyncGetBlobScore(
      const TagId &tag_id,
      const std::string &blob_name);

  chi::Future<GetBlobSizeTask> AsyncGetBlobSize(
      const TagId &tag_id,
      const std::string &blob_name);

  chi::Future<GetContainedBlobsTask> AsyncGetContainedBlobs(
      const TagId &tag_id);

  // Telemetry
  chi::Future<PollTelemetryLogTask> AsyncPollTelemetryLog(
      std::uint64_t minimum_logical_time);

  // Query operations
  chi::Future<TagQueryTask> AsyncTagQuery(
      const std::string &tag_regex,
      chi::u32 max_tags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast());

  chi::Future<BlobQueryTask> AsyncBlobQuery(
      const std::string &tag_regex,
      const std::string &blob_regex,
      chi::u32 max_blobs = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast());
};

}  // namespace wrp_cte::core
```

### Tag Wrapper Class

The `wrp_cte::core::Tag` class provides a simplified, object-oriented interface for blob operations within a specific tag. This wrapper class eliminates the need to pass `TagId` parameters for each operation, making the API more convenient and less error-prone.

#### Class Definition

```cpp
namespace wrp_cte::core {

class Tag {
private:
  TagId tag_id_;
  std::string tag_name_;

public:
  // Constructors
  explicit Tag(const std::string &tag_name);  // Creates or gets existing tag
  explicit Tag(const TagId &tag_id);          // Uses existing TagId directly

  // Blob storage operations (synchronous wrappers)
  void PutBlob(const std::string &blob_name, const char *data, size_t data_size, size_t off = 0);
  void PutBlob(const std::string &blob_name, const hipc::ShmPtr<> &data, size_t data_size,
               size_t off = 0, float score = 1.0f);

  // Asynchronous blob storage
  chi::Future<PutBlobTask> AsyncPutBlob(const std::string &blob_name, const hipc::ShmPtr<> &data,
                                        size_t data_size, size_t off = 0, float score = 1.0f);

  // Blob retrieval operations (synchronous wrappers)
  void GetBlob(const std::string &blob_name, char *data, size_t data_size, size_t off = 0);
  void GetBlob(const std::string &blob_name, hipc::ShmPtr<> data, size_t data_size, size_t off = 0);

  // Blob metadata operations (synchronous wrappers)
  float GetBlobScore(const std::string &blob_name);
  chi::u64 GetBlobSize(const std::string &blob_name);
  std::vector<std::string> GetContainedBlobs();

  // Blob reorganization
  void ReorganizeBlob(const std::string &blob_name, float new_score);

  // Tag accessor
  const TagId& GetTagId() const { return tag_id_; }
};

}  // namespace wrp_cte::core
```

#### Key Features

- **Automatic Tag Management**: Constructor with tag name automatically creates or retrieves existing tags
- **Simplified API**: No need to pass TagId for each operation
- **Memory Management**: Raw data variant automatically handles shared memory allocation and cleanup
- **Exception Safety**: Operations throw exceptions on failure for clear error handling
- **Score Support**: Blob scoring for intelligent data placement across storage targets
- **Blob Enumeration**: `GetContainedBlobs()` method returns all blob names in the tag
- **Reorganization Support**: `ReorganizeBlob()` method for data tier migration

#### Memory Management Guidelines

**For Synchronous Tag Wrapper Operations:**
- Raw data variant (`const char*`) automatically manages shared memory lifecycle
- Shared memory variant requires caller to manage `hipc::ShmPtr<>` lifecycle

**For Asynchronous Operations:**
- Only shared memory variant available to avoid memory lifecycle issues
- Caller must keep shared memory buffers alive until async task completes
- See usage examples below for proper async memory management patterns

### Data Structures

#### CreateParams

Configuration parameters for CTE container creation:

```cpp
struct CreateParams {
  chi::string config_file_path_;  // YAML config file path
  chi::u32 worker_count_;         // Number of worker threads (default: 4)
  
  CreateParams();
  CreateParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc, 
               const std::string& config_file_path = "", 
               chi::u32 worker_count = 4);
};
```

#### ListTargets Return Type

The `AsyncListTargets` method returns a Future. Access target names via `task->target_names_` after `Wait()`:

```cpp
chi::Future<ListTargetsTask> AsyncListTargets();
```

Example usage:
```cpp
auto task = cte_client->AsyncListTargets();
task.Wait();
for (const auto& target_name : task->target_names_) {
    std::cout << "Target: " << target_name << "\n";
}
```

#### GetOrCreateTag Return Type

The `AsyncGetOrCreateTag` method returns a Future. Access the TagId via `task->tag_id_` after `Wait()`:

```cpp
chi::Future<GetOrCreateTagTask<CreateParams>> AsyncGetOrCreateTag(
    const std::string &tag_name,
    const TagId &tag_id = TagId::GetNull());
```

Example usage:
```cpp
auto task = cte_client->AsyncGetOrCreateTag("my_dataset");
task.Wait();
TagId tag_id = task->tag_id_;
```

#### BlobInfo

Blob metadata and block management:

```cpp
struct BlobInfo {
  BlobId blob_id_;
  std::string blob_name_;
  std::vector<BlobBlock> blocks_;  // Ordered blocks making up the blob
  float score_;                    // 0-1 score for reorganization
  Timestamp last_modified_;
  Timestamp last_read_;
  
  chi::u64 GetTotalSize() const;   // Total size from all blocks
};
```

**Note**: Individual blob sizes can be queried efficiently using `Client::GetBlobSize()` or `Tag::GetBlobSize()` without needing to retrieve full BlobInfo.

#### BlobBlock

Individual block within a blob:

```cpp
struct BlobBlock {
  chimaera::bdev::Client bdev_client_;  // Target client for this block
  chi::u64 target_offset_;             // Offset within target
  chi::u64 size_;                      // Size of this block
};
```

#### CteTelemetry

Telemetry data for performance monitoring:

```cpp
struct CteTelemetry {
  CteOp op_;                    // Operation type
  size_t off_;                  // Offset within blob
  size_t size_;                 // Size of operation
  BlobId blob_id_;
  TagId tag_id_;
  Timestamp mod_time_;
  Timestamp read_time_;
  std::uint64_t logical_time_;  // For ordering entries
};

enum class CteOp : chi::u32 {
  kPutBlob = 0,
  kGetBlob = 1,
  kDelBlob = 2,
  kGetOrCreateTag = 3,
  kDelTag = 4,
  kGetTagSize = 5,
  kGetBlobScore = 6,
  kGetBlobSize = 7
};
```

### Global Access

CTE Core provides singleton access patterns:

```cpp
// Initialize CTE client (must be called once)
// NOTE: This automatically calls chi::CHIMAERA_INIT internally
// config_path: Optional path to YAML configuration file
// pool_query: Pool query type for CTE container creation (default: Dynamic)
bool WRP_CTE_CLIENT_INIT(const std::string &config_path = "",
                         const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic());

// Access global CTE client instance
auto *client = WRP_CTE_CLIENT;
```

**Important Notes:**
- `WRP_CTE_CLIENT_INIT` automatically calls `chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)` internally
- You do NOT need to call `chi::CHIMAERA_INIT` separately when using CTE Core
- Configuration is managed per-Runtime instance (no global ConfigManager singleton)
- The config file path can also be specified via the `WRP_RUNTIME_CONF` environment variable

## Usage Examples

### Basic Initialization

```cpp
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

int main() {
  // Initialize CTE subsystem
  // NOTE: WRP_CTE_CLIENT_INIT automatically calls chi::CHIMAERA_INIT internally
  // You do NOT need to call chi::CHIMAERA_INIT separately
  wrp_cte::core::WRP_CTE_CLIENT_INIT("/path/to/config.yaml");

  // Get global CTE client instance (created during initialization)
  auto *cte_client = WRP_CTE_CLIENT;

  // The CTE client is now ready to use - no need to call Create() again
  // The client is automatically initialized with the pool specified during WRP_CTE_CLIENT_INIT

  return 0;
}
```

### Registering Storage Targets

```cpp
// Get global CTE client
auto *cte_client = WRP_CTE_CLIENT;

// Register a file-based storage target
std::string target_path = "/mnt/nvme/cte_storage.dat";
chi::u64 target_size = 100ULL * 1024 * 1024 * 1024;  // 100GB

auto reg_task = cte_client->AsyncRegisterTarget(
    target_path,
    chimaera::bdev::BdevType::kFile,
    target_size
);
reg_task.Wait();

if (reg_task->return_code_ == 0) {
    std::cout << "Target registered successfully\n";
}

// Register a RAM-based cache target
auto cache_task = cte_client->AsyncRegisterTarget(
    "ram::cte_cache",
    chimaera::bdev::BdevType::kRam,
    8ULL * 1024 * 1024 * 1024  // 8GB
);
cache_task.Wait();

// List all registered targets
auto list_task = cte_client->AsyncListTargets();
list_task.Wait();
for (const auto& target_name : list_task->target_names_) {
    std::cout << "Target: " << target_name << "\n";
}
```

### Working with Tags and Blobs

#### Using the Core Client Directly

```cpp
// Get global CTE client
auto *cte_client = WRP_CTE_CLIENT;

// Create or get a tag for grouping related blobs
auto tag_task = cte_client->AsyncGetOrCreateTag("dataset_v1");
tag_task.Wait();
TagId tag_id = tag_task->tag_id_;

// Prepare data for storage
std::vector<char> data(1024 * 1024);  // 1MB of data
std::fill(data.begin(), data.end(), 'A');

// Allocate shared memory for the data
hipc::FullPtr<char> shm_buffer = CHI_IPC->AllocateBuffer(data.size());
memcpy(shm_buffer.ptr_, data.data(), data.size());

auto put_task = cte_client->AsyncPutBlob(
    tag_id,
    "blob_001",           // Blob name
    0,                    // Offset
    data.size(),          // Size
    shm_buffer.shm_,      // Shared memory pointer
    0.8f,                 // Score (0-1, higher = hotter data)
    0                     // Flags
);
put_task.Wait();

if (put_task->return_code_ == 0) {
    std::cout << "Blob stored successfully\n";

    // Get blob size
    auto size_task = cte_client->AsyncGetBlobSize(tag_id, "blob_001");
    size_task.Wait();
    std::cout << "Stored blob size: " << size_task->size_ << " bytes\n";

    // Get blob score
    auto score_task = cte_client->AsyncGetBlobScore(tag_id, "blob_001");
    score_task.Wait();
    std::cout << "Blob score: " << score_task->score_ << "\n";
}

// Retrieve the blob
auto retrieve_buffer = CHI_IPC->AllocateBuffer(data.size());
auto get_task = cte_client->AsyncGetBlob(
    tag_id,
    "blob_001",           // Blob name for lookup
    0,                    // Offset
    data.size(),          // Size to read
    0,                    // Flags
    retrieve_buffer.shm_  // Buffer for data
);
get_task.Wait();

// Get all blob names in the tag
auto blobs_task = cte_client->AsyncGetContainedBlobs(tag_id);
blobs_task.Wait();
std::cout << "Tag contains " << blobs_task->blob_names_.size() << " blobs\n";
for (const auto& name : blobs_task->blob_names_) {
    std::cout << "  - " << name << "\n";
}

// Get total size of all blobs in tag
auto tag_size_task = cte_client->AsyncGetTagSize(tag_id);
tag_size_task.Wait();
std::cout << "Tag total size: " << tag_size_task->size_ << " bytes\n";

// Delete a specific blob
auto del_blob_task = cte_client->AsyncDelBlob(tag_id, "blob_001");
del_blob_task.Wait();

// Delete entire tag (removes all blobs)
auto del_tag_task = cte_client->AsyncDelTag(tag_id);
del_tag_task.Wait();
```

#### Using the Tag Wrapper (Recommended for Convenience)

```cpp
// Create tag wrapper - automatically creates or gets existing tag
wrp_cte::core::Tag dataset_tag("dataset_v1");

// Prepare data for storage
std::vector<char> data(1024 * 1024);  // 1MB of data
std::fill(data.begin(), data.end(), 'A');

try {
    // Store blob - automatically handles shared memory management
    dataset_tag.PutBlob("blob_001", data.data(), data.size());
    std::cout << "Blob stored successfully\n";

    // Get blob size
    chi::u64 blob_size = dataset_tag.GetBlobSize("blob_001");
    std::cout << "Stored blob size: " << blob_size << " bytes\n";

    // Get blob score
    float blob_score = dataset_tag.GetBlobScore("blob_001");
    std::cout << "Blob score: " << blob_score << "\n";

    // Retrieve the blob using automatic memory management (recommended)
    std::vector<char> retrieve_data(blob_size);
    dataset_tag.GetBlob("blob_001", retrieve_data.data(), blob_size);

    // Alternative: Retrieve using manual shared memory management
    // auto retrieve_buffer = CHI_IPC->AllocateBuffer(blob_size);
    // dataset_tag.GetBlob("blob_001", retrieve_buffer.shm_, blob_size);

    std::cout << "Blob retrieved successfully\n";

    // Get all blobs in the tag
    std::vector<std::string> blob_names = dataset_tag.GetContainedBlobs();
    std::cout << "Tag contains " << blob_names.size() << " blobs\n";

    // Reorganize blob with new score
    dataset_tag.ReorganizeBlob("blob_001", 0.95f);  // Move to hot tier

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
}

// For tag-level operations, you can use the core client:
auto *cte_client = WRP_CTE_CLIENT;

// Get total size of all blobs in tag
auto tag_size_task = cte_client->AsyncGetTagSize(dataset_tag.GetTagId());
tag_size_task.Wait();
std::cout << "Tag total size: " << tag_size_task->size_ << " bytes\n";

// Delete entire tag (removes all blobs)
auto del_task = cte_client->AsyncDelTag(dataset_tag.GetTagId());
del_task.Wait();
```

### Tag Wrapper Usage Examples

The Tag wrapper class provides a more convenient interface for blob operations within a specific tag. Here are comprehensive examples showing different usage patterns:

#### Basic Tag Wrapper Operations

```cpp
#include <wrp_cte/core/core_client.h>
#include <iostream>
#include <vector>

// Initialize CTE system (same as before)
// ... initialization code ...

// Create a tag wrapper - automatically creates or gets existing tag
wrp_cte::core::Tag dataset_tag("dataset_v1");

// Store data using the simple raw data interface
std::vector<char> data(1024 * 1024);  // 1MB of data
std::fill(data.begin(), data.end(), 'X');

try {
    // Simple PutBlob - automatically manages shared memory
    dataset_tag.PutBlob("sample_blob", data.data(), data.size());
    std::cout << "Blob stored successfully\n";
    
    // Get blob size without retrieving data
    chi::u64 blob_size = dataset_tag.GetBlobSize("sample_blob");
    std::cout << "Blob size: " << blob_size << " bytes\n";
    
    // Get blob score (data temperature)
    float blob_score = dataset_tag.GetBlobScore("sample_blob");
    std::cout << "Blob score: " << blob_score << "\n";
    
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
}
```

#### Memory Management: Automatic vs Manual

The Tag class provides two GetBlob variants to suit different memory management preferences:

```cpp
wrp_cte::core::Tag data_tag("performance_data");

try {
    // Store some test data
    std::string test_data = "Sample blob data for retrieval testing";
    data_tag.PutBlob("test_blob", test_data.c_str(), test_data.size());
    
    chi::u64 blob_size = data_tag.GetBlobSize("test_blob");
    std::cout << "Blob size: " << blob_size << " bytes\n";
    
    // Method 1: Automatic memory management (recommended for most use cases)
    std::vector<char> auto_buffer(blob_size);
    data_tag.GetBlob("test_blob", auto_buffer.data(), blob_size);
    std::cout << "Retrieved with automatic memory management\n";
    
    // Method 2: Manual shared memory management (for advanced use cases)
    auto shm_buffer = CHI_IPC->AllocateBuffer(blob_size);
    if (!shm_buffer.IsNull()) {
        data_tag.GetBlob("test_blob", shm_buffer.shm_, blob_size);
        std::cout << "Retrieved with manual shared memory management\n";
        // shm_buffer automatically freed when it goes out of scope
    }
    
    // Method 1 is preferred because:
    // - No shared memory allocation required
    // - Automatic cleanup via RAII
    // - Works with standard C++ containers
    // - Simpler error handling
    
} catch (const std::exception& e) {
    std::cerr << "Memory management example error: " << e.what() << "\n";
}
```

#### Advanced Tag Wrapper with Scoring

```cpp
// Create tag wrapper for time-series data
wrp_cte::core::Tag timeseries_tag("timeseries_2024");

// Store multiple data chunks with different scores (data temperatures)
std::vector<std::vector<char>> chunks;
std::vector<float> scores = {0.9f, 0.7f, 0.5f, 0.2f};  // Hot to cold data
std::vector<std::string> chunk_names = {"latest", "recent", "old", "archived"};

for (size_t i = 0; i < 4; ++i) {
    chunks.emplace_back(1024 * 512);  // 512KB chunks
    std::fill(chunks[i].begin(), chunks[i].end(), 'A' + i);
    
    try {
        // For custom scoring, use shared memory version:
        auto shm_ptr = CHI_IPC->AllocateBuffer(chunks[i].size());
        memcpy(shm_ptr.ptr_, chunks[i].data(), chunks[i].size());
        timeseries_tag.PutBlob(chunk_names[i], shm_ptr.shm_, chunks[i].size(), 0, scores[i]);

        std::cout << "Stored chunk '" << chunk_names[i] << "' with score " << scores[i] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Failed to store chunk " << chunk_names[i] << ": " << e.what() << "\n";
    }
}
```

#### Blob Retrieval with Tag Wrapper

```cpp
// Create tag wrapper from existing TagId
TagId existing_tag_id = /* ... get from somewhere ... */;
wrp_cte::core::Tag existing_tag(existing_tag_id);

try {
    // First, check if blob exists and get its size
    chi::u64 blob_size = existing_tag.GetBlobSize("target_blob");
    if (blob_size == 0) {
        std::cout << "Blob 'target_blob' not found or empty\n";
        return;
    }
    
    std::cout << "Blob size: " << blob_size << " bytes\n";

    // Allocate shared memory buffer for retrieval
    auto retrieve_buffer = CHI_IPC->AllocateBuffer(blob_size);
    if (retrieve_buffer.IsNull()) {
        throw std::runtime_error("Failed to allocate retrieval buffer");
    }

    // Retrieve the blob
    existing_tag.GetBlob("target_blob", retrieve_buffer.shm_, blob_size);

    // Process the retrieved data
    ProcessBlobData(retrieve_buffer.ptr_, blob_size);

    std::cout << "Successfully retrieved and processed blob\n";

} catch (const std::exception& e) {
    std::cerr << "Blob retrieval error: " << e.what() << "\n";
}
```

#### Asynchronous Operations with Tag Wrapper

```cpp
wrp_cte::core::Tag async_tag("async_operations");

// Prepare data for async operations
std::vector<std::vector<char>> async_data;
std::vector<hipc::FullPtr<char>> shm_buffers;
std::vector<chi::Future<PutBlobTask>> async_tasks;

for (int i = 0; i < 5; ++i) {
    // Prepare data
    async_data.emplace_back(1024 * 256);  // 256KB each
    std::fill(async_data[i].begin(), async_data[i].end(), 'Z' - i);

    // Allocate shared memory (must keep alive until async operation completes)
    auto shm_buffer = CHI_IPC->AllocateBuffer(async_data[i].size());
    if (shm_buffer.IsNull()) {
        std::cerr << "Failed to allocate shared memory for async operation " << i << "\n";
        continue;
    }

    // Copy data to shared memory
    memcpy(shm_buffer.ptr_, async_data[i].data(), async_data[i].size());

    try {
        // Start async operation (returns immediately)
        auto task = async_tag.AsyncPutBlob(
            "async_blob_" + std::to_string(i),
            shm_buffer.shm_,
            async_data[i].size(),
            0,    // offset
            0.6f  // score
        );

        // Store references to keep alive
        shm_buffers.push_back(std::move(shm_buffer));
        async_tasks.push_back(std::move(task));

        std::cout << "Started async put for blob " << i << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Failed to start async put " << i << ": " << e.what() << "\n";
    }
}

// Wait for all async operations to complete
std::cout << "Waiting for async operations to complete...\n";
for (size_t i = 0; i < async_tasks.size(); ++i) {
    try {
        async_tasks[i].Wait();  // Note: Wait() on Future, not pointer
        if (async_tasks[i]->return_code_ == 0) {
            std::cout << "Async operation " << i << " completed successfully\n";
        } else {
            std::cout << "Async operation " << i << " failed with code "
                      << async_tasks[i]->return_code_ << "\n";
        }
        // Task is automatically cleaned up when Future goes out of scope

    } catch (const std::exception& e) {
        std::cerr << "Error waiting for async operation " << i << ": " << e.what() << "\n";
    }
}

// Note: shm_buffers will be automatically cleaned up when they go out of scope
```

### Asynchronous Operations

```cpp
// Get global CTE client
auto *cte_client = WRP_CTE_CLIENT;

// Allocate shared memory for the data
hipc::FullPtr<char> shm_buffer = CHI_IPC->AllocateBuffer(data.size());
memcpy(shm_buffer.ptr_, data.data(), data.size());

// Asynchronous blob operations for better performance
auto put_task = cte_client->AsyncPutBlob(
    tag_id, "async_blob",
    0, data.size(), shm_buffer.shm_, 0.5f, 0
);

// Do other work while blob is being stored
ProcessOtherData();

// Wait for completion
put_task.Wait();
if (put_task->return_code_ == 0) {
    std::cout << "Async put completed successfully\n";
}
// Task is automatically cleaned up when Future goes out of scope

// Multiple async operations
std::vector<chi::Future<PutBlobTask>> tasks;
std::vector<hipc::FullPtr<char>> buffers;  // Keep buffers alive

for (int i = 0; i < 10; ++i) {
    // Allocate buffer for each task
    auto buffer = CHI_IPC->AllocateBuffer(data.size());
    memcpy(buffer.ptr_, data.data(), data.size());

    auto task = cte_client->AsyncPutBlob(
        tag_id,
        "blob_" + std::to_string(i),
        0, data.size(), buffer.shm_, 0.5f, 0
    );

    buffers.push_back(std::move(buffer));  // Keep buffer alive
    tasks.push_back(std::move(task));
}

// Wait for all to complete
for (auto& task : tasks) {
    task.Wait();
}
// buffers and tasks automatically cleaned up here
```

### Performance Monitoring

```cpp
// Get global CTE client
auto *cte_client = WRP_CTE_CLIENT;

// Poll telemetry log for performance analysis
std::uint64_t last_logical_time = 0;

auto telemetry_task = cte_client->AsyncPollTelemetryLog(last_logical_time);
telemetry_task.Wait();

for (const auto& entry : telemetry_task->telemetry_) {
    std::cout << "Operation: ";
    switch (entry.op_) {
        case CteOp::kPutBlob: std::cout << "PUT"; break;
        case CteOp::kGetBlob: std::cout << "GET"; break;
        case CteOp::kDelBlob: std::cout << "DEL"; break;
        case CteOp::kGetBlobScore: std::cout << "GET_SCORE"; break;
        case CteOp::kGetBlobSize: std::cout << "GET_SIZE"; break;
        case CteOp::kGetOrCreateTag: std::cout << "GET_TAG"; break;
        case CteOp::kDelTag: std::cout << "DEL_TAG"; break;
        case CteOp::kGetTagSize: std::cout << "TAG_SIZE"; break;
        default: std::cout << "OTHER"; break;
    }
    std::cout << " Size: " << entry.size_
              << " Offset: " << entry.off_
              << " LogicalTime: " << entry.logical_time_ << "\n";

    // Update last_logical_time for next poll
    if (entry.logical_time_ > last_logical_time) {
        last_logical_time = entry.logical_time_;
    }
}

// Update target statistics
auto stat_task = cte_client->AsyncStatTargets();
stat_task.Wait();

// List all targets
auto list_task = cte_client->AsyncListTargets();
list_task.Wait();
for (const auto& target_name : list_task->target_names_) {
    std::cout << "Target: " << target_name << "\n";
}
```

### Blob Reorganization

```cpp
// Reorganize blobs based on new access patterns
// Higher scores (closer to 1.0) indicate hotter data

auto *cte_client = WRP_CTE_CLIENT;

// Reorganize multiple blobs by calling AsyncReorganizeBlob once per blob
std::vector<std::string> blob_names = {"blob_001", "blob_002", "blob_003"};
std::vector<float> new_scores = {0.95f, 0.7f, 0.3f};  // Hot, warm, cold

for (size_t i = 0; i < blob_names.size(); ++i) {
    auto reorg_task = cte_client->AsyncReorganizeBlob(tag_id, blob_names[i], new_scores[i]);
    reorg_task.Wait();
    if (reorg_task->return_code_ == 0) {
        std::cout << "Blob " << blob_names[i] << " reorganized successfully\n";
    }
}

// Example: Reorganize single blob
auto single_task = cte_client->AsyncReorganizeBlob(tag_id, "important_blob", 0.95f);
single_task.Wait();
if (single_task->return_code_ == 0) {
    std::cout << "Single blob reorganized successfully\n";
}

// Using Tag wrapper (simpler API)
wrp_cte::core::Tag my_tag("my_dataset");
my_tag.ReorganizeBlob("hot_data", 0.95f);  // Move to hot tier
my_tag.ReorganizeBlob("cold_archive", 0.2f);  // Move to cold tier
```

## Configuration

CTE Core uses YAML configuration files for runtime parameters. Configuration can be loaded from:
1. A file path specified during initialization
2. Environment variable `WRP_RUNTIME_CONF`
3. Programmatically via the Config API

### Configuration File Format

```yaml
# Worker thread configuration
worker_count: 4

# Target management settings
targets:
  max_targets: 100
  default_target_timeout_ms: 30000
  auto_unregister_failed: true

# Performance tuning
performance:
  target_stat_interval_ms: 5000      # Target statistics update interval
  blob_cache_size_mb: 512            # Cache size for blob operations
  max_concurrent_operations: 64      # Max concurrent I/O operations
  score_threshold: 0.7               # Threshold for blob reorganization

# Queue configuration for different operation types
queues:
  target_management:
    lane_count: 2
    priority: "low_latency"
  
  tag_management:
    lane_count: 2
    priority: "low_latency"
  
  blob_operations:
    lane_count: 4
    priority: "high_latency"
  
  stats:
    lane_count: 1
    priority: "high_latency"

# Storage device configuration
storage:
  # Primary high-performance storage with manual tier score
  - path: "/mnt/nvme/cte_primary"
    bdev_type: "file"
    capacity_limit: "1TB"
    score: 0.9                # Optional: Manual tier score (0.0-1.0)
  
  # RAM-based cache (highest tier)
  - path: "/tmp/cte_cache"
    bdev_type: "ram"
    capacity_limit: "8GB"
    score: 1.0                # Optional: Manual tier score for fastest access
  
  # Secondary storage (uses automatic scoring)
  - path: "/mnt/ssd/cte_secondary"
    bdev_type: "file"
    capacity_limit: "500GB"
    # No score specified - uses automatic bandwidth-based scoring

# Data Placement Engine configuration
dpe:
  dpe_type: "max_bw"  # Options: "random", "round_robin", "max_bw"
```

### Programmatic Configuration

Configuration in CTE Core is now managed per-Runtime instance, not through a global singleton. Configuration is loaded during initialization through the `WRP_CTE_CLIENT_INIT` function.

```cpp
#include <wrp_cte/core/core_client.h>

// Initialize CTE with configuration file
// Configuration is passed to the Runtime during creation
bool success = wrp_cte::core::WRP_CTE_CLIENT_INIT("/path/to/config.yaml");

// Or use environment variable WRP_RUNTIME_CONF
// export WRP_RUNTIME_CONF=/path/to/config.yaml
success = wrp_cte::core::WRP_CTE_CLIENT_INIT();

// Configuration is now embedded in the Runtime instance
// and cannot be modified after initialization
```

**Note:** The ConfigManager singleton has been removed. Configuration is now:
- Loaded once during `WRP_CTE_CLIENT_INIT`
- Embedded in the CTE Runtime instance via `CreateParams`
- Immutable after initialization
- Can be specified via file path parameter or `WRP_RUNTIME_CONF` environment variable

### Queue Priority Options

- `"low_latency"` - Optimized for minimal latency (chi::kLowLatency)
- `"high_latency"` - Optimized for throughput (chi::kHighLatency)

### Storage Device Types

- `"file"` - File-based block device
- `"ram"` - RAM-based block device (for caching)
- `"dev_dax"` - Persistent memory device
- `"posix"` - POSIX file system interface

### Manual Tier Scoring

Storage devices support optional manual tier scoring to override automatic bandwidth-based tier assignment:

#### Configuration Parameters

- **`score`** *(optional, float 0.0-1.0)*: Manual tier score for the storage device
  - **1.0**: Highest tier (fastest access, e.g., RAM, high-end NVMe)
  - **0.8-0.9**: High-performance tier (e.g., NVMe SSDs)
  - **0.5-0.7**: Medium-performance tier (e.g., SATA SSDs)
  - **0.1-0.4**: Low-performance tier (e.g., HDDs, network storage)
  - **Not specified**: Uses automatic bandwidth-based scoring

#### Behavior

- Manual scores are preserved during target statistics updates
- Targets with manual scores will not be overwritten by automatic scoring algorithms
- Data placement engines use these scores for intelligent tier selection
- Mixed configurations (some manual, some automatic) are fully supported

#### Example Configuration

```yaml
storage:
  # Fastest tier - manual score
  - path: "/mnt/ram/cache"
    bdev_type: "ram"
    capacity_limit: "16GB"
    score: 1.0
  
  # High-performance tier - manual score  
  - path: "/mnt/nvme/primary"
    bdev_type: "file"
    capacity_limit: "1TB"
    score: 0.85
  
  # Medium tier - automatic scoring
  - path: "/mnt/ssd/secondary"
    bdev_type: "file"
    capacity_limit: "2TB"
    # Uses automatic bandwidth measurement
```

### Data Placement Engine Types

- `"random"` - Random placement across targets
- `"round_robin"` - Round-robin placement
- `"max_bw"` - Place on target with maximum available bandwidth

## Python Bindings

CTE Core provides Python bindings for easy integration with Python applications.

### Installation

```bash
# Build Python bindings
cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make

# Install Python module
pip install ./wrapper/python
```

### Python API Usage

```python
import wrp_cte_core_ext as cte

# Initialize CTE
# NOTE: This automatically calls chi::CHIMAERA_INIT() internally
cte.initialize_cte("/path/to/config.yaml")

# Get global CTE client
client = cte.get_cte_client()

# Create or get a tag
tag_task = client.async_get_or_create_tag("my_dataset")
tag_task.wait()
tag_id = tag_task.tag_id

# Poll telemetry log
minimum_logical_time = 0
telemetry_task = client.async_poll_telemetry_log(minimum_logical_time)
telemetry_task.wait()

for entry in telemetry_task.telemetry:
    print(f"Operation: {entry.op}")
    print(f"Size: {entry.size}")
    print(f"Offset: {entry.off}")
    print(f"Logical Time: {entry.logical_time}")

# Reorganize blobs with new scores
blob_names = ["blob_001", "blob_002", "blob_003"]
new_scores = [0.95, 0.85, 0.75]  # Different tier assignments

# Call async_reorganize_blob once per blob
for blob_name, new_score in zip(blob_names, new_scores):
    task = client.async_reorganize_blob(tag_id, blob_name, new_score)
    task.wait()
    if task.return_code == 0:
        print(f"Blob {blob_name} reorganized successfully")
    else:
        print(f"Reorganization of {blob_name} failed with error code: {task.return_code}")
```

### Python Data Types

```python
# Create unique IDs
tag_id = cte.TagId.GetNull()
blob_id = cte.BlobId.GetNull()

# Check if ID is null
if tag_id.IsNull():
    print("Tag ID is null")

# Access ID components
print(f"Major: {tag_id.major_}, Minor: {tag_id.minor_}")

# Operation types
print(cte.CteOp.kPutBlob)    # Put blob operation
print(cte.CteOp.kGetBlob)    # Get blob operation
print(cte.CteOp.kDelBlob)    # Delete blob operation
```

### Python Blob Reorganization

The Python bindings support blob reorganization for dynamic data placement optimization using the async API:

```python
import wrp_cte_core_ext as cte

# Initialize CTE system (as shown in previous examples)
cte.initialize_cte("/path/to/config.yaml")
client = cte.get_cte_client()

# Get or create tag for the blobs
tag_task = client.async_get_or_create_tag("my_dataset")
tag_task.wait()
tag_id = tag_task.tag_id

# Example 1: Reorganize multiple blobs to different tiers
blob_names = ["hot_data", "warm_data", "cold_archive"]
new_scores = [0.95, 0.6, 0.2]  # Hot, warm, and cold tiers

# Call async_reorganize_blob once per blob
for blob_name, new_score in zip(blob_names, new_scores):
    task = client.async_reorganize_blob(tag_id, blob_name, new_score)
    task.wait()
    if task.return_code == 0:
        print(f"Blob {blob_name} reorganized successfully")
    else:
        print(f"Reorganization of {blob_name} failed with error code: {task.return_code}")

# Example 2: Promote frequently accessed blobs based on telemetry
telemetry_task = client.async_poll_telemetry_log(0)
telemetry_task.wait()
access_counts = {}

# Count accesses per blob name (requires tracking blob names from telemetry)
for entry in telemetry_task.telemetry:
    if entry.op == cte.CteOp.kGetBlob:
        # Track access patterns
        blob_key = (entry.blob_id.major, entry.blob_id.minor)
        access_counts[blob_key] = access_counts.get(blob_key, 0) + 1

# Batch reorganize based on access frequency
# Assuming you have a mapping of blob IDs to names
blob_id_to_name = {
    (0, 1): "dataset_001",
    (0, 2): "dataset_002",
    (0, 3): "dataset_003"
}

blobs_to_reorganize = []
new_scores_list = []

for blob_key, count in access_counts.items():
    if blob_key in blob_id_to_name and count > 10:
        blob_name = blob_id_to_name[blob_key]
        blobs_to_reorganize.append(blob_name)

        # Calculate score based on access frequency
        score = min(0.5 + (count / 100.0), 1.0)
        new_scores_list.append(score)

# Perform reorganization for each blob
if blobs_to_reorganize:
    for blob_name, new_score in zip(blobs_to_reorganize, new_scores_list):
        task = client.async_reorganize_blob(tag_id, blob_name, new_score)
        task.wait()
        if task.return_code == 0:
            print(f"Reorganized blob {blob_name} successfully")

# Example 3: Tier-based reorganization strategy
# Organize blobs into three tiers based on size and access patterns

# Small, frequently accessed -> Hot tier (0.9)
small_hot_blobs = ["config", "index", "metadata"]
for blob_name in small_hot_blobs:
    task = client.async_reorganize_blob(tag_id, blob_name, 0.9)
    task.wait()
    if task.return_code == 0:
        print(f"Hot tier blob {blob_name} reorganized")

# Medium, occasionally accessed -> Warm tier (0.5-0.7)
warm_blobs = ["dataset_recent_01", "dataset_recent_02"]
warm_scores = [0.6, 0.5]
for blob_name, score in zip(warm_blobs, warm_scores):
    task = client.async_reorganize_blob(tag_id, blob_name, score)
    task.wait()
    if task.return_code == 0:
        print(f"Warm tier blob {blob_name} reorganized")

# Large, rarely accessed -> Cold tier (0.1-0.3)
cold_blobs = ["archive_2023", "backup_full"]
cold_scores = [0.2, 0.1]
for blob_name, score in zip(cold_blobs, cold_scores):
    task = client.async_reorganize_blob(tag_id, blob_name, score)
    task.wait()
    if task.return_code == 0:
        print(f"Cold tier blob {blob_name} reorganized")
```

**Score Guidelines for Python:**
- `0.9 - 1.0`: Highest tier (RAM cache, frequently accessed)
- `0.7 - 0.8`: High tier (NVMe, recently accessed)
- `0.4 - 0.6`: Medium tier (SSD, occasionally accessed)
- `0.1 - 0.3`: Low tier (HDD, archival data)
- `0.0`: Lowest tier (cold storage, rarely accessed)

**Method Pattern:**
```python
# All Python client methods use async pattern
task = client.async_reorganize_blob(
    tag_id,         # Tag ID containing the blob
    blob_name,      # Blob name (string)
    new_score       # New score (float, 0.0 to 1.0)
)
task.wait()
result = task.return_code  # 0 = success
```

**Return Codes:**
- `0`: Success - blob reorganized successfully
- `Non-zero`: Error - reorganization failed (tag not found, blob not found, insufficient space, etc.)

**Important Notes:**
- All Python client methods follow the async pattern with `.wait()` completion
- All blobs must belong to the specified `tag_id`
- Scores must be in the range `[0.0, 1.0]`
- Higher scores indicate hotter data that should be placed on faster storage tiers

## Advanced Topics

### Best Practices

#### Choosing Between Tag Wrapper and Direct Client API

Generally, the tag wrapper class is preferred over the direct API. 

#### Memory Management Best Practices

**For Raw Data Operations:**
```cpp
// Tag wrapper automatically manages shared memory for sync operations
wrp_cte::core::Tag tag("my_data");
std::vector<char> data = LoadData();
tag.PutBlob("item", data.data(), data.size());  // Safe - automatic cleanup
```

**For Shared Memory Operations:**
```cpp
// Manual shared memory management - more control
// NOTE: AllocateBuffer is NOT templated - it returns hipc::FullPtr<char>
auto shm_buffer = CHI_IPC->AllocateBuffer(data_size);
memcpy(shm_buffer.ptr_, raw_data, data_size);
tag.PutBlob("item", shm_buffer.shm_, data_size, 0, score);
// shm_buffer automatically cleaned up when it goes out of scope
```

**For Asynchronous Operations:**
```cpp
// Always keep shared memory alive until async task completes
std::vector<hipc::FullPtr<char>> buffers;  // Keep alive
std::vector<chi::Future<PutBlobTask>> tasks;

for (auto& data_chunk : data_chunks) {
    auto buffer = CHI_IPC->AllocateBuffer(data_chunk.size());
    memcpy(buffer.ptr_, data_chunk.data(), data_chunk.size());

    auto task = tag.AsyncPutBlob("chunk", buffer.shm_, data_chunk.size());

    buffers.push_back(std::move(buffer));  // Keep alive!
    tasks.push_back(std::move(task));
}

// Wait for completion and cleanup
for (auto& task : tasks) {
    task.Wait();  // Note: Wait() on Future, not pointer
}
// buffers and tasks automatically cleaned up here
```

#### Performance Optimization

**Blob Scoring Guidelines:**
- Use scores 0.8-1.0 for frequently accessed "hot" data
- Use scores 0.4-0.7 for occasionally accessed "warm" data  
- Use scores 0.0-0.3 for archival "cold" data
- CTE uses scores for intelligent placement across storage tiers

**Batch Operations:**
```cpp
// Efficient: Group related operations
wrp_cte::core::Tag batch_tag("batch_job");
for (const auto& item : batch_items) {
    batch_tag.PutBlob(item.name, item.data, item.size);
}

// Less efficient: Multiple tags with few operations each
// Creates overhead for tag lookup and context switching
```

**Size Queries:**
```cpp
// Efficient: Check size before allocating retrieval buffer
chi::u64 blob_size = tag.GetBlobSize("large_blob");
if (blob_size > 0) {
    auto buffer = CHI_IPC->AllocateBuffer(blob_size);
    tag.GetBlob("large_blob", buffer.shm_, blob_size);
}

// Less efficient: Allocate maximum possible size
// auto buffer = CHI_IPC->AllocateBuffer(MAX_SIZE);  // Wasteful
```

#### Error Handling Patterns

**Tag Wrapper (Exception-based):**
```cpp
try {
    wrp_cte::core::Tag tag("dataset");
    tag.PutBlob("data", buffer, size);
    
    chi::u64 stored_size = tag.GetBlobSize("data");
    if (stored_size != size) {
        throw std::runtime_error("Size mismatch after storage");
    }
    
} catch (const std::exception& e) {
    std::cerr << "Storage operation failed: " << e.what() << "\n";
    // Automatic cleanup via RAII
}
```

**Direct Client (Async with Return Code):**
```cpp
auto *client = WRP_CTE_CLIENT;

auto tag_task = client->AsyncGetOrCreateTag("dataset");
tag_task.Wait();
TagId tag_id = tag_task->tag_id_;

auto put_task = client->AsyncPutBlob(tag_id, "data",
                                     0, size, buffer.shm_, 0.5f, 0);
put_task.Wait();

if (put_task->return_code_ != 0) {
    std::cerr << "PutBlob failed with code: " << put_task->return_code_ << "\n";
    return false;
}

auto size_task = client->AsyncGetBlobSize(tag_id, "data");
size_task.Wait();
if (size_task->size_ != size) {
    std::cerr << "Size mismatch: expected " << size << ", got " << size_task->size_ << "\n";
    return false;
}
```

#### Thread Safety Considerations

- Both Tag wrapper and Client are thread-safe
- Multiple threads can safely share the same Tag or Client instance
- Shared memory buffers (`hipc::FullPtr`) should not be shared between threads
- Each thread should maintain its own buffer allocations for optimal performance

### Multi-Node Deployment

CTE Core supports distributed deployment across multiple nodes:

1. Configure Chimaera for multi-node operation
2. Use appropriate PoolQuery values:
   - `chi::PoolQuery::Local()` - Local node only
   - `chi::PoolQuery::Global()` - All nodes
   - Custom pool queries for specific node groups

### Custom Data Placement Algorithms

Extend the DPE (Data Placement Engine) by implementing custom placement strategies:

1. Inherit from the base DPE class
2. Implement placement logic based on target metrics
3. Register the new DPE type in configuration

### Performance Optimization

1. **Batch Operations**: Use async APIs for multiple operations
2. **Score-based Placement**: Set appropriate scores (0-1) for data temperature
3. **Target Balancing**: Monitor and rebalance based on target metrics
4. **Queue Tuning**: Adjust lane counts and priorities based on workload

### Error Handling

All async operations return a Future. After calling `Wait()`, check the `return_code_` field:
- `0`: Success
- Non-zero: Error (specific codes depend on operation)

Always check return values and handle errors appropriately:

```cpp
auto task = cte_client->AsyncRegisterTarget(target_name, bdev_type, size);
task.Wait();
if (task->return_code_ != 0) {
    // Handle error
    std::cerr << "Failed to register target, error code: " << task->return_code_ << "\n";
}
```

### Thread Safety

- CTE Core client operations are thread-safe
- Multiple threads can share a client instance
- Async operations are particularly suitable for multi-threaded usage

### Memory Management

- CTE Core uses shared memory for zero-copy data transfer
- The `hipc::ShmPtr<>` type represents shared memory locations
- `hipc::FullPtr<char>` manages allocation lifecycle with RAII cleanup
- Use `CHI_IPC->AllocateBuffer(size)` to allocate shared memory buffers

## Troubleshooting

### Common Issues

1. **Initialization Failures**
   - Ensure Chimaera runtime is initialized first
   - Check configuration file path and format
   - Verify storage paths have appropriate permissions

2. **Target Registration Errors**
   - Confirm target path exists and is writable
   - Check available disk space
   - Verify bdev type matches storage medium

3. **Blob Operations Failing**
   - Ensure tag exists before blob operations
   - Check target has sufficient space
   - Verify data pointers are valid shared memory

4. **Performance Issues**
   - Monitor target statistics regularly
   - Adjust worker count based on workload
   - Tune queue configurations
   - Consider data placement strategy

### Debug Logging

Enable debug logging by setting environment variables:

```bash
export CHIMAERA_LOG_LEVEL=DEBUG
export CTE_LOG_LEVEL=DEBUG
```

### Metrics Collection

Use the telemetry API to collect performance metrics:

```cpp
auto *cte_client = WRP_CTE_CLIENT;

// Continuous monitoring loop
while (running) {
    auto telemetry_task = cte_client->AsyncPollTelemetryLog(last_logical_time);
    telemetry_task.Wait();

    ProcessTelemetry(telemetry_task->telemetry_);

    if (!telemetry_task->telemetry_.empty()) {
        last_logical_time = telemetry_task->telemetry_.back().logical_time_;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
}
```

## API Stability and Versioning

CTE Core follows semantic versioning:
- Major version: Breaking API changes
- Minor version: New features, backward compatible
- Patch version: Bug fixes

Check version compatibility:

```cpp
// Version macros (defined in headers)
#if CTE_CORE_VERSION_MAJOR >= 1 && CTE_CORE_VERSION_MINOR >= 0
    // Use newer API features
#endif
```

## Support and Resources

- **Documentation**: This document and inline API documentation
- **Examples**: See `test/unit/` directory for comprehensive examples
- **Configuration**: Example configs in `config/` directory
- **Issues**: Report bugs via project issue tracker