# Bdev ChiMod Documentation

## Overview

The Bdev (Block Device) ChiMod provides a high-performance interface for block device operations supporting both file-based and RAM-based storage backends. It manages block allocation, read/write operations, and performance monitoring with flexible storage options.

**Key Features:**
- **Dual Backend Support**: File-based storage (using libaio) and RAM-based storage (using malloc)
- **Asynchronous I/O**: For file-based storage using libaio, synchronous operations for RAM-based storage
- **Hierarchical block allocation** with multiple size categories (4KB, 64KB, 256KB, 1MB)
- **Performance monitoring** and statistics collection for both backends
- **Memory-aligned I/O operations** for optimal file-based performance
- **Block allocation and deallocation management** with unified API

## CMake Integration

### External Projects

To use the Bdev ChiMod in external projects:

```cmake
find_package(chimaera_bdev REQUIRED)       # BDev ChiMod package
find_package(chimaera_admin REQUIRED)      # Admin ChiMod (always required)
find_package(chimaera REQUIRED)            # Core Chimaera (automatically includes ChimaeraCommon.cmake)

target_link_libraries(your_application
  chimaera::bdev_client         # Bdev client library
  chimaera::admin_client        # Admin client (required)
  ${CMAKE_THREAD_LIBS_INIT}     # Threading support
)
# Core Chimaera library dependencies are automatically included by ChiMod libraries
```

### Required Headers

```cpp
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <chimaera/admin/admin_client.h>  // Required for CreateTask
```

## API Reference

### Client Class: `chimaera::bdev::Client`

The Bdev client provides the primary interface for block device operations.

#### Constructor

```cpp
// Default constructor
Client()

// Constructor with pool ID
explicit Client(const chi::PoolId& pool_id)
```

#### Container Management

##### `AsyncCreate()`
Creates and initializes the bdev container asynchronously with specified backend type.

```cpp
chi::Future<chimaera::bdev::CreateTask> AsyncCreate(
    const chi::PoolQuery& pool_query,
    const std::string& pool_name, const chi::PoolId& custom_pool_id,
    BdevType bdev_type, chi::u64 total_size = 0, chi::u32 io_depth = 32,
    chi::u32 alignment = 4096, const PerfMetrics* perf_metrics = nullptr)
```

**Parameters:**
- `pool_query`: Pool domain query (typically `chi::PoolQuery::Dynamic()` for automatic caching)
- `pool_name`: Pool name (serves as file path for kFile, unique identifier for kRam)
- `custom_pool_id`: Explicit pool ID to create for this container
- `bdev_type`: Backend type (`BdevType::kFile` or `BdevType::kRam`)
- `total_size`: Total size available for allocation (0 = use file size for kFile, required for kRam)
- `io_depth`: libaio queue depth for asynchronous operations (ignored for kRam, default: 32)
- `alignment`: I/O alignment in bytes for optimal performance (default: 4096)
- `perf_metrics`: **Optional** user-defined performance characteristics (nullptr = use defaults)

**Returns:** Future for asynchronous completion checking

**Performance Characteristics Definition:**
Instead of automatic benchmarking during container creation, users can optionally specify the expected performance characteristics of their storage device. This allows for:
- **Faster container initialization** (no benchmarking delay)
- **Predictable performance modeling** for different storage types
- **Custom device profiling** based on external testing
- **Flexible usage** - defaults used when not specified

**Example with Default Performance (recommended for most users):**
```cpp
// Create container with default performance characteristics
const chi::PoolId pool_id = chi::PoolId(8000, 0);
auto create_task = bdev_client.AsyncCreate(pool_query, "/dev/nvme0n1", pool_id, BdevType::kFile);
create_task.Wait();

if (create_task->GetReturnCode() != 0) {
  std::cerr << "BDev creation failed" << std::endl;
  return;
}
```

**Example with Custom Performance (for advanced users):**
```cpp
// Define performance characteristics for a high-end NVMe SSD
PerfMetrics nvme_perf;
nvme_perf.read_bandwidth_mbps_ = 3500.0;   // 3.5 GB/s read
nvme_perf.write_bandwidth_mbps_ = 3000.0;  // 3.0 GB/s write
nvme_perf.read_latency_us_ = 50.0;         // 50μs read latency
nvme_perf.write_latency_us_ = 80.0;        // 80μs write latency
nvme_perf.iops_ = 500000.0;                // 500K IOPS

// Create container with custom performance profile
const chi::PoolId pool_id = chi::PoolId(8000, 0);
auto create_task = bdev_client.AsyncCreate(pool_query, "/dev/nvme0n1", pool_id, BdevType::kFile,
                                           0, 64, 4096, &nvme_perf);
create_task.Wait();
```

**Usage Examples:**

*File-based storage:*
```cpp
chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
const chi::PoolId pool_id = chi::PoolId(8000, 0);
chimaera::bdev::Client bdev_client(pool_id);

auto pool_query = chi::PoolQuery::Dynamic();  // Recommended for automatic caching
// File-based storage (pool_name IS the file path)
auto task = bdev_client.AsyncCreate(pool_query, "/dev/nvme0n1", pool_id, BdevType::kFile, 0, 64, 4096);
task.Wait();
```

*RAM-based storage:*
```cpp
// RAM-based storage (1GB, pool_name is unique identifier)
const chi::PoolId pool_id = chi::PoolId(8001, 0);
auto task = bdev_client.AsyncCreate(pool_query, "my_ram_device", pool_id, BdevType::kRam, 1024*1024*1024);
task.Wait();
```

**Note:** The `perf_metrics` parameter is optional and positioned last for convenience. Pass `nullptr` (default) to use conservative default performance characteristics, or provide a pointer to custom metrics for specific device modeling.

#### Block Management Operations

##### `AsyncAllocateBlocks()`
Allocates multiple blocks with the specified total size asynchronously. The system automatically determines the optimal block configuration based on the requested size.

```cpp
chi::Future<chimaera::bdev::AllocateBlocksTask> AsyncAllocateBlocks(
    const chi::PoolQuery& pool_query,
    chi::u64 size)
```

**Parameters:**
- `pool_query`: Pool domain query for routing (typically `chi::PoolQuery::Local()`)
- `size`: Total size to allocate in bytes

**Returns:** Future for asynchronous completion checking. Access allocated blocks via `task->blocks_` after calling `Wait()`.

**Block Allocation Algorithm:**
- **Size < 1MB**: Allocates a single block of the next largest size category (4KB, 64KB, 256KB, or 1MB)
- **Size >= 1MB**: Allocates only 1MB blocks to meet the requested size

**Usage:**
```cpp
auto pool_query = chi::PoolQuery::Local();
auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query, 512*1024);  // Allocate 512KB
alloc_task.Wait();

if (alloc_task->return_code_ == 0) {
  auto& blocks = alloc_task->blocks_;
  std::cout << "Allocated " << blocks.size() << " block(s)" << std::endl;
  for (const auto& block : blocks) {
    std::cout << "  Block at offset " << block.offset_ << " with size " << block.size_ << std::endl;
  }
}
```

##### `AsyncFreeBlocks()`
Frees multiple previously allocated blocks asynchronously.

```cpp
chi::Future<chimaera::bdev::FreeBlocksTask> AsyncFreeBlocks(
    const chi::PoolQuery& pool_query,
    const std::vector<Block>& blocks)
```

**Parameters:**
- `pool_query`: Pool domain query for routing (typically `chi::PoolQuery::Local()`)
- `blocks`: Vector of block structures to free

**Returns:** Future for asynchronous completion checking

**Usage:**
```cpp
auto pool_query = chi::PoolQuery::Local();
auto free_task = bdev_client.AsyncFreeBlocks(pool_query, blocks);
free_task.Wait();

if (free_task->return_code_ == 0) {
  std::cout << "Successfully freed " << blocks.size() << " block(s)" << std::endl;
}
```

#### I/O Operations

##### `AsyncWrite()`
Writes data to previously allocated blocks asynchronously.

```cpp
chi::Future<chimaera::bdev::WriteTask> AsyncWrite(
    const chi::PoolQuery& pool_query,
    const ArrayVector<Block, 128>& blocks, hipc::ShmPtr<> data, size_t length)
```

**Parameters:**
- `pool_query`: Pool domain query for routing (typically `chi::PoolQuery::Local()`)
- `blocks`: Target blocks for writing
- `data`: Pointer to data to write (hipc::ShmPtr<>)
- `length`: Size of data to write in bytes

**Returns:** Future for asynchronous completion checking

**Usage:**
```cpp
// Prepare data
size_t data_size = 4096;
auto* ipc_manager = CHI_IPC;
hipc::ShmPtr<> write_ptr = ipc_manager->AllocateBuffer(data_size);
hipc::FullPtr<char> write_data(write_ptr);
memset(write_data.ptr_, 0xAB, data_size);  // Fill with pattern

// Write to block
auto pool_query = chi::PoolQuery::Local();
auto write_task = bdev_client.AsyncWrite(pool_query, blocks, write_ptr, data_size);
write_task.Wait();

if (write_task->return_code_ == 0) {
  std::cout << "Wrote data successfully" << std::endl;
}

// Free buffer when done
ipc_manager->FreeBuffer(write_ptr);
```

##### `AsyncRead()`
Reads data from previously allocated and written blocks asynchronously.

```cpp
chi::Future<chimaera::bdev::ReadTask> AsyncRead(
    const chi::PoolQuery& pool_query,
    const ArrayVector<Block, 128>& blocks, hipc::ShmPtr<> data, size_t buffer_size)
```

**Parameters:**
- `pool_query`: Pool domain query for routing (typically `chi::PoolQuery::Local()`)
- `blocks`: Source blocks for reading
- `data`: Output buffer pointer (allocated by caller)
- `buffer_size`: Size of the buffer in bytes

**Returns:** Future for asynchronous completion checking

**Usage:**
```cpp
// Allocate read buffer
size_t buffer_size = blocks[0].size_;
auto* ipc_manager = CHI_IPC;
hipc::ShmPtr<> read_ptr = ipc_manager->AllocateBuffer(buffer_size);

// Read data back
auto pool_query = chi::PoolQuery::Local();
auto read_task = bdev_client.AsyncRead(pool_query, blocks, read_ptr, buffer_size);
read_task.Wait();

if (read_task->return_code_ == 0) {
  std::cout << "Read data successfully" << std::endl;

  // Access the data
  hipc::FullPtr<char> read_data(read_ptr);
  // Verify data integrity
  bool data_matches = (memcmp(write_data.ptr_, read_data.ptr_, buffer_size) == 0);
  std::cout << "Data integrity check: " << (data_matches ? "PASS" : "FAIL") << std::endl;
}

// Free buffer when done
ipc_manager->FreeBuffer(read_ptr);
```

#### Performance Monitoring

##### `AsyncGetStats()`
Retrieves performance statistics asynchronously.

```cpp
chi::Future<chimaera::bdev::GetStatsTask> AsyncGetStats()
```

**Returns:** Future for asynchronous completion checking. Access performance metrics via `task->metrics_` and remaining space via `task->remaining_size_` after calling `Wait()`.

**Important Note:** GetStats returns the performance characteristics that were specified during container creation (either default values or user-provided custom metrics), not calculated runtime statistics.

**Usage:**
```cpp
auto stats_task = bdev_client.AsyncGetStats();
stats_task.Wait();

if (stats_task->return_code_ == 0) {
  auto& metrics = stats_task->metrics_;
  chi::u64 remaining_space = stats_task->remaining_size_;

  std::cout << "Performance Statistics:" << std::endl;
  std::cout << "  Read bandwidth: " << metrics.read_bandwidth_mbps_ << " MB/s" << std::endl;
  std::cout << "  Write bandwidth: " << metrics.write_bandwidth_mbps_ << " MB/s" << std::endl;
  std::cout << "  Read latency: " << metrics.read_latency_us_ << " μs" << std::endl;
  std::cout << "  Write latency: " << metrics.write_latency_us_ << " μs" << std::endl;
  std::cout << "  IOPS: " << metrics.iops_ << std::endl;
  std::cout << "  Remaining space: " << remaining_space << " bytes" << std::endl;
}
```

## Data Structures

### BdevType Enum
Specifies the storage backend type.

```cpp
enum class BdevType : chi::u32 {
  kFile = 0,  // File-based block device (default)
  kRam = 1    // RAM-based block device
};
```

**Backend Characteristics:**
- **kFile**: Uses file-based storage with libaio for asynchronous I/O, supports alignment requirements, persistent data
- **kRam**: Uses malloc-allocated RAM buffer, synchronous operations, volatile data (lost on restart)

### Block Structure
Represents an allocated block of storage.

```cpp
struct Block {
  chi::u64 offset_;     // Offset within file/device
  chi::u64 size_;       // Size of block in bytes
  chi::u32 block_type_; // Block size category (0=4KB, 1=64KB, 2=256KB, 3=1MB)
}
```

**Block Type Categories:**
- `0`: 4KB blocks - for small, frequent I/O operations
- `1`: 64KB blocks - for medium-sized operations
- `2`: 256KB blocks - for large sequential operations  
- `3`: 1MB blocks - for very large bulk operations

### PerfMetrics Structure
Contains performance monitoring data.

```cpp
struct PerfMetrics {
  double read_bandwidth_mbps_;   // Read bandwidth in MB/s
  double write_bandwidth_mbps_;  // Write bandwidth in MB/s
  double read_latency_us_;       // Average read latency in microseconds
  double write_latency_us_;      // Average write latency in microseconds
  double iops_;                  // I/O operations per second
}
```

## Task Types

### CreateTask
Container creation task for the bdev module. This is an alias for `chimaera::admin::GetOrCreatePoolTask<CreateParams>`.

**Key Fields:**
- Inherits from `BaseCreateTask` with bdev-specific `CreateParams`
- Processed by admin module for pool creation
- Contains serialized bdev configuration parameters

### AllocateBlocksTask
Block allocation task for multiple blocks.

**Key Fields:**
- `size_`: Requested total size in bytes (IN)
- `blocks_`: Allocated blocks information vector (OUT)
- `return_code_`: Operation result (0 = success)

### FreeBlocksTask
Block deallocation task for multiple blocks.

**Key Fields:**
- `blocks_`: Vector of blocks to free (IN)
- `return_code_`: Operation result (0 = success)

### WriteTask
Block write operation task.

**Key Fields:**
- `block_`: Target block for writing (IN)
- `data_`: Pointer to data to write (IN)
- `length_`: Size of data to write (IN)
- `bytes_written_`: Number of bytes actually written (OUT)
- `return_code_`: Operation result (0 = success)

### ReadTask
Block read operation task.

**Key Fields:**
- `block_`: Source block for reading (IN)
- `data_`: Pointer to buffer for read data (OUT)
- `length_`: Size of buffer / actual bytes read (INOUT)
- `bytes_read_`: Number of bytes actually read (OUT)
- `return_code_`: Operation result (0 = success)

### GetStatsTask
Performance statistics retrieval task.

**Key Fields:**
- `metrics_`: Performance metrics (OUT)
- `remaining_size_`: Remaining allocatable space (OUT)
- `return_code_`: Operation result (0 = success)

## Configuration

### CreateParams Structure
Configuration parameters for bdev container creation:

```cpp
struct CreateParams {
  BdevType bdev_type_;         // Block device type (file or RAM)
  chi::u64 total_size_;        // Total size for allocation (0 = file size for kFile, required for kRam)
  chi::u32 io_depth_;          // libaio queue depth (ignored for kRam, default: 32)
  chi::u32 alignment_;         // I/O alignment in bytes (default: 4096)
  PerfMetrics perf_metrics_;   // User-defined performance characteristics
  
  // Required: chimod library name for module manager
  static constexpr const char* chimod_lib_name = "chimaera_bdev";
}
```

**Note**: The `file_path_` field has been removed. The pool name (passed to Create/AsyncCreate) now serves as the file path for file-based BDevs.

**Parameter Guidelines:**
- **bdev_type_**: Choose `BdevType::kFile` for persistent storage or `BdevType::kRam` for high-speed volatile storage
- **pool_name**: 
  - For kFile: **IS the file path** (can be block device `/dev/nvme0n1` or regular file)
  - For kRam: Unique identifier for the RAM device
- **total_size_**: 
  - For kFile: Set to 0 to use full file/device size, or specify limit
  - For kRam: **Required** - specifies the RAM buffer size to allocate
- **io_depth_**: Higher values improve parallelism for kFile but use more memory (typical: 16-128), ignored for kRam
- **alignment_**: Must match device requirements for kFile (typically 512 or 4096 bytes), less critical for kRam

**Important:** The `chimod_lib_name` does NOT include the `_runtime` suffix as it is automatically appended by the module manager.

## Usage Examples

### File-based Block Device Workflow
```cpp
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/admin/admin_client.h>

int main() {
  // Initialize Chimaera (client mode with embedded runtime)
  chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

  // Create admin client first (always required)
  const chi::PoolId admin_pool_id = chi::kAdminPoolId;
  chimaera::admin::Client admin_client(admin_pool_id);
  auto admin_task = admin_client.AsyncCreate(chi::PoolQuery::Local(), "admin", admin_pool_id);
  admin_task.Wait();

  // Create bdev client
  const chi::PoolId bdev_pool_id = chi::PoolId(8000, 0);
  chimaera::bdev::Client bdev_client(bdev_pool_id);

  auto pool_query = chi::PoolQuery::Dynamic();  // Recommended for automatic caching

  // Initialize with default performance characteristics (recommended)
  auto create_task = bdev_client.AsyncCreate(pool_query, "/dev/nvme0n1", bdev_pool_id,
                                              BdevType::kFile, 0, 64, 4096);
  create_task.Wait();

  if (create_task->GetReturnCode() != 0) {
    std::cerr << "BDev creation failed" << std::endl;
    return 1;
  }

  // Allocate blocks for 1MB of data
  auto pool_query_local = chi::PoolQuery::Local();
  auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query_local, 1024 * 1024);
  alloc_task.Wait();

  if (alloc_task->return_code_ != 0) {
    std::cerr << "Block allocation failed" << std::endl;
    return 1;
  }

  auto& blocks = alloc_task->blocks_;
  std::cout << "Allocated " << blocks.size() << " block(s)" << std::endl;

  // Prepare test data
  auto* ipc_manager = CHI_IPC;
  size_t data_size = blocks[0].size_;
  hipc::ShmPtr<> write_ptr = ipc_manager->AllocateBuffer(data_size);
  hipc::FullPtr<char> test_data(write_ptr);
  memset(test_data.ptr_, 0xDE, data_size);
  for (size_t i = 0; i < data_size; i += 4096) {
    // Add pattern to verify data integrity
    test_data.ptr_[i] = static_cast<char>(i % 256);
  }

  // Write data
  auto write_task = bdev_client.AsyncWrite(pool_query_local, blocks, write_ptr, data_size);
  write_task.Wait();
  std::cout << "Write completed" << std::endl;

  // Read data back
  hipc::ShmPtr<> read_ptr = ipc_manager->AllocateBuffer(data_size);
  auto read_task = bdev_client.AsyncRead(pool_query_local, blocks, read_ptr, data_size);
  read_task.Wait();
  hipc::FullPtr<char> read_data(read_ptr);

  // Verify data integrity
  bool integrity_ok = (read_task->return_code_ == 0) &&
                     (memcmp(test_data.ptr_, read_data.ptr_, data_size) == 0);
  std::cout << "Data integrity: " << (integrity_ok ? "PASS" : "FAIL") << std::endl;

  // Get performance characteristics (user-defined, not runtime measured)
  auto stats_task = bdev_client.AsyncGetStats();
  stats_task.Wait();

  if (stats_task->return_code_ == 0) {
    auto& perf = stats_task->metrics_;
    std::cout << "\nDevice Performance Profile:" << std::endl;
    std::cout << "  Read: " << perf.read_bandwidth_mbps_ << " MB/s" << std::endl;
    std::cout << "  Write: " << perf.write_bandwidth_mbps_ << " MB/s" << std::endl;
    std::cout << "  IOPS: " << perf.iops_ << std::endl;
    std::cout << "  Note: Values reflect user-defined characteristics, not runtime measurements" << std::endl;
  }

  // Free the allocated blocks
  auto free_task = bdev_client.AsyncFreeBlocks(pool_query_local, std::vector<Block>(blocks.begin(), blocks.end()));
  free_task.Wait();
  std::cout << "Blocks freed: " << (free_task->return_code_ == 0 ? "SUCCESS" : "FAILED") << std::endl;

  // Clean up buffers
  ipc_manager->FreeBuffer(write_ptr);
  ipc_manager->FreeBuffer(read_ptr);

  return 0;
}
```

### RAM-based Block Device Workflow
```cpp
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/admin/admin_client.h>

int main() {
  // Initialize Chimaera (client mode with embedded runtime)
  chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

  // Create admin client first (always required)
  const chi::PoolId admin_pool_id = chi::kAdminPoolId;
  chimaera::admin::Client admin_client(admin_pool_id);
  auto admin_task = admin_client.AsyncCreate(chi::PoolQuery::Local(), "admin", admin_pool_id);
  admin_task.Wait();

  // Create bdev client
  const chi::PoolId bdev_pool_id = chi::PoolId(8001, 0);
  chimaera::bdev::Client bdev_client(bdev_pool_id);

  auto pool_query = chi::PoolQuery::Dynamic();  // Recommended for automatic caching

  // Initialize with default RAM performance characteristics (recommended)
  auto create_task = bdev_client.AsyncCreate(pool_query, "my_ram_device", bdev_pool_id,
                                              BdevType::kRam, 1024*1024*1024);
  create_task.Wait();

  if (create_task->GetReturnCode() != 0) {
    std::cerr << "BDev creation failed" << std::endl;
    return 1;
  }

  // Allocate blocks for 1MB of data (from RAM)
  auto pool_query_local = chi::PoolQuery::Local();
  auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query_local, 1024 * 1024);
  alloc_task.Wait();

  if (alloc_task->return_code_ != 0) {
    std::cerr << "Block allocation failed" << std::endl;
    return 1;
  }

  auto& blocks = alloc_task->blocks_;

  // Prepare test data
  auto* ipc_manager = CHI_IPC;
  size_t data_size = blocks[0].size_;
  hipc::ShmPtr<> write_ptr = ipc_manager->AllocateBuffer(data_size);
  hipc::FullPtr<char> test_data(write_ptr);
  memset(test_data.ptr_, 0xAB, data_size);

  // Write data to RAM (very fast)
  auto start = std::chrono::high_resolution_clock::now();
  auto write_task = bdev_client.AsyncWrite(pool_query_local, blocks, write_ptr, data_size);
  write_task.Wait();
  auto write_end = std::chrono::high_resolution_clock::now();

  // Read data from RAM (very fast)
  hipc::ShmPtr<> read_ptr = ipc_manager->AllocateBuffer(data_size);
  auto read_task = bdev_client.AsyncRead(pool_query_local, blocks, read_ptr, data_size);
  read_task.Wait();
  auto read_end = std::chrono::high_resolution_clock::now();
  hipc::FullPtr<char> read_data(read_ptr);

  // Calculate performance
  double write_time_ms = std::chrono::duration<double, std::milli>(write_end - start).count();
  double read_time_ms = std::chrono::duration<double, std::milli>(read_end - write_end).count();

  std::cout << "RAM Backend Performance:" << std::endl;
  std::cout << "  Write time: " << write_time_ms << " ms" << std::endl;
  std::cout << "  Read time: " << read_time_ms << " ms" << std::endl;
  std::cout << "  Write bandwidth: " << (data_size / 1024.0 / 1024.0) / (write_time_ms / 1000.0) << " MB/s" << std::endl;

  // Verify data integrity
  bool integrity_ok = (read_task->return_code_ == 0) &&
                     (memcmp(test_data.ptr_, read_data.ptr_, data_size) == 0);
  std::cout << "Data integrity: " << (integrity_ok ? "PASS" : "FAIL") << std::endl;

  // Free the allocated blocks
  auto free_task = bdev_client.AsyncFreeBlocks(pool_query_local, std::vector<Block>(blocks.begin(), blocks.end()));
  free_task.Wait();
  std::cout << "Blocks freed: " << (free_task->return_code_ == 0 ? "SUCCESS" : "FAILED") << std::endl;

  // Clean up buffers
  ipc_manager->FreeBuffer(write_ptr);
  ipc_manager->FreeBuffer(read_ptr);

  return 0;
}
```

### Basic Async Operations Example
```cpp
// Example of async block allocation and I/O
auto pool_query = chi::PoolQuery::Local();
auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query, 65536);  // 64KB
alloc_task.Wait();

if (alloc_task->return_code_ == 0) {
  auto& blocks = alloc_task->blocks_;

  // Prepare data buffer
  auto* ipc_manager = CHI_IPC;
  size_t data_size = blocks[0].size_;
  hipc::ShmPtr<> write_ptr = ipc_manager->AllocateBuffer(data_size);
  hipc::FullPtr<char> data(write_ptr);
  memset(data.ptr_, 0xFF, data_size);

  // Write
  auto write_task = bdev_client.AsyncWrite(pool_query, blocks, write_ptr, data_size);
  write_task.Wait();

  std::cout << "Write completed: " << (write_task->return_code_ == 0 ? "SUCCESS" : "FAILED") << std::endl;

  // Read
  hipc::ShmPtr<> read_ptr = ipc_manager->AllocateBuffer(data_size);
  auto read_task = bdev_client.AsyncRead(pool_query, blocks, read_ptr, data_size);
  read_task.Wait();

  std::cout << "Read completed: " << (read_task->return_code_ == 0 ? "SUCCESS" : "FAILED") << std::endl;

  // Free blocks
  auto free_task = bdev_client.AsyncFreeBlocks(pool_query, std::vector<Block>(blocks.begin(), blocks.end()));
  free_task.Wait();

  // Clean up buffers
  ipc_manager->FreeBuffer(write_ptr);
  ipc_manager->FreeBuffer(read_ptr);
}
```

### Performance Benchmarking
```cpp
// Benchmark different block sizes
const std::vector<chi::u64> block_sizes = {4096, 65536, 262144, 1048576};
const size_t num_operations = 1000;

auto* ipc_manager = CHI_IPC;
auto pool_query = chi::PoolQuery::Local();

for (chi::u64 block_size : block_sizes) {
  auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_operations; ++i) {
    auto alloc_task = bdev_client.AsyncAllocateBlocks(pool_query, block_size);
    alloc_task.Wait();
    auto& blocks = alloc_task->blocks_;

    // Prepare data
    hipc::ShmPtr<> write_ptr = ipc_manager->AllocateBuffer(block_size);
    hipc::FullPtr<char> data(write_ptr);
    memset(data.ptr_, static_cast<char>(i % 256), block_size);

    auto write_task = bdev_client.AsyncWrite(pool_query, blocks, write_ptr, block_size);
    write_task.Wait();

    // Read data back
    hipc::ShmPtr<> read_ptr = ipc_manager->AllocateBuffer(block_size);
    auto read_task = bdev_client.AsyncRead(pool_query, blocks, read_ptr, block_size);
    read_task.Wait();

    auto free_task = bdev_client.AsyncFreeBlocks(pool_query, std::vector<Block>(blocks.begin(), blocks.end()));
    free_task.Wait();

    // Clean up buffers
    ipc_manager->FreeBuffer(write_ptr);
    ipc_manager->FreeBuffer(read_ptr);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

  double throughput_mbps = (block_size * num_operations) /
                          (duration.count() * 1024.0);

  std::cout << "Block size " << block_size << " bytes: "
            << throughput_mbps << " MB/s" << std::endl;
}
```

## Dependencies

- **HermesShm**: Shared memory framework and IPC
- **Chimaera core runtime**: Base runtime objects and task framework
- **Admin ChiMod**: Required for pool creation and management
- **cereal**: Serialization library for network communication
- **libaio**: Linux asynchronous I/O library for high-performance block operations
- **Boost.Fiber** and **Boost.Context**: Coroutine support

## Installation

1. Ensure libaio is installed on your system:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libaio-dev
   
   # RHEL/CentOS
   sudo yum install libaio-devel
   ```

2. Build Chimaera with the bdev module:
   ```bash
   cmake --preset debug
   cmake --build build
   ```

3. Install to system or custom prefix:
   ```bash
   cmake --install build --prefix /usr/local
   ```

4. For external projects, set CMAKE_PREFIX_PATH:
   ```bash
   export CMAKE_PREFIX_PATH="/usr/local:/path/to/hermes-shm:/path/to/other/deps"
   ```

## Error Handling

All operations are asynchronous and return `chi::Future<TaskType>`. Check `return_code_` after calling `Wait()`:

```cpp
auto pool_query = chi::PoolQuery::Local();
auto task = bdev_client.AsyncAllocateBlocks(pool_query, 65536);
task.Wait();

if (task->return_code_ != 0) {
  std::cerr << "Block allocation failed with code: " << task->return_code_ << std::endl;
}
```

**Common Error Scenarios:**
- Insufficient storage space for allocation
- I/O alignment violations
- Device access permissions
- Corrupted block metadata
- Network failures in distributed setups

## Performance Management

### Performance Characteristics Definition

**User-Defined Performance Model**: The BDev module now uses user-provided performance characteristics instead of automatic benchmarking. This approach offers several advantages:

1. **No Benchmarking Overhead**: Container creation is faster without benchmark delays
2. **Predictable Performance Modeling**: Consistent performance reporting across restarts
3. **Custom Device Profiling**: Model specific storage devices based on external testing
4. **Flexible Performance Profiles**: Switch between different performance profiles for testing

**Setting Performance Characteristics:**
```cpp
// Example: High-end NVMe SSD profile
PerfMetrics nvme_perf;
nvme_perf.read_bandwidth_mbps_ = 7000.0;   // 7 GB/s sequential read
nvme_perf.write_bandwidth_mbps_ = 5000.0;  // 5 GB/s sequential write
nvme_perf.read_latency_us_ = 30.0;         // 30μs random read
nvme_perf.write_latency_us_ = 50.0;        // 50μs random write
nvme_perf.iops_ = 1000000.0;               // 1M random IOPS

// Example: SATA SSD profile
PerfMetrics sata_perf;
sata_perf.read_bandwidth_mbps_ = 550.0;    // 550 MB/s
sata_perf.write_bandwidth_mbps_ = 500.0;   // 500 MB/s
sata_perf.read_latency_us_ = 100.0;        // 100μs
sata_perf.write_latency_us_ = 200.0;       // 200μs
sata_perf.iops_ = 95000.0;                 // 95K IOPS

// Example: Mechanical HDD profile
PerfMetrics hdd_perf;
hdd_perf.read_bandwidth_mbps_ = 180.0;     // 180 MB/s
hdd_perf.write_bandwidth_mbps_ = 160.0;    // 160 MB/s
hdd_perf.read_latency_us_ = 8000.0;        // 8ms seek time
hdd_perf.write_latency_us_ = 10000.0;      // 10ms seek time
hdd_perf.iops_ = 150.0;                    // 150 IOPS
```

### Backend Selection

**Use RAM Backend (`BdevType::kRam`) when:**
- Maximum performance is critical
- Data persistence is not required
- Working with temporary data or caching
- Testing and benchmarking scenarios
- Sufficient system RAM is available

**Use File Backend (`BdevType::kFile`) when:**
- Data persistence is required
- Working with datasets larger than available RAM
- Integration with existing storage infrastructure
- Need for data durability across restarts

### Performance Tuning

1. **Block Size Selection**: Choose appropriate block sizes based on I/O patterns
   - Small blocks (4KB): Random access patterns
   - Large blocks (1MB): Sequential operations

2. **I/O Depth** (File backend only): Higher io_depth values improve parallelism but consume more memory

3. **Alignment** (File backend): Ensure data is properly aligned to device boundaries (typically 4096 bytes)

4. **Async Operations**: Use async methods for better parallelism in I/O-intensive applications

5. **Batch Operations**: Group multiple allocations/deallocations when possible to reduce overhead

6. **Performance Profile Selection**: Choose appropriate performance characteristics that match your storage device

### Typical Performance Profiles

**RAM Backend (DDR4-3200):**
- **Latency**: ~0.1 microseconds
- **Bandwidth**: ~20-25 GB/s
- **IOPS**: ~10M IOPS
- **Scalability**: Excellent for concurrent access

**High-End NVMe SSD:**
- **Latency**: ~30-50 microseconds
- **Bandwidth**: ~5-7 GB/s sequential
- **IOPS**: ~500K-1M random IOPS
- **Scalability**: Excellent with proper io_depth

**SATA SSD:**
- **Latency**: ~100-200 microseconds
- **Bandwidth**: ~500-550 MB/s
- **IOPS**: ~80K-100K IOPS
- **Scalability**: Good

**Mechanical HDD:**
- **Latency**: ~8-12 milliseconds (seek time)
- **Bandwidth**: ~150-200 MB/s sequential
- **IOPS**: ~100-200 IOPS
- **Scalability**: Limited by mechanical constraints

## Important Notes

1. **Admin Dependency**: The bdev module requires the admin module to be initialized first for pool creation.

2. **Block Lifecycle**: Always free allocated blocks to prevent memory leaks and fragmentation.

3. **Thread Safety**: Operations are designed for single-threaded access. Use external synchronization for multi-threaded environments.

4. **Device Permissions**: Ensure the application has appropriate permissions to access block devices.

5. **Data Persistence**: Data written to blocks persists across container restarts if backed by persistent storage.

6. **Performance Characteristics**: Performance metrics returned by GetStats() reflect the user-defined values specified during container creation, not runtime measurements. For actual performance monitoring, implement separate benchmarking tools.

7. **Default Performance Values**: If no custom performance characteristics are provided (perf_metrics = nullptr), the container uses conservative default values (100 MB/s read/write, 1ms latency, 1000 IOPS) suitable for basic operations.

8. **Optional Performance Parameter**: The performance metrics parameter is optional and positioned last in all Create methods for convenience. Most users can omit this parameter and use the defaults.