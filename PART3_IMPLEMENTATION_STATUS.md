# Part 3: Submitting Tasks From The GPU - Implementation Status

## ✅ Completed

### 1. MakeFuture Split into Sub-functions (Task #4)

**Implemented:**
- `MakeCopyFuture()` - GPU-compatible function (HSHM_CROSS_FUN) that always serializes tasks
  - Serializes task into FutureShm's copy_space
  - Sets FUTURE_COPY_FROM_CLIENT flag
  - Used by clients and GPU kernels

- `MakePointerFuture()` - Runtime-only function that wraps task pointer without serialization
  - Creates FutureShm without copy_space
  - Used by runtime workers for zero-copy task submission

- `MakeFuture()` - Updated to delegate to appropriate sub-function
  - Client path: calls MakeCopyFuture()
  - Runtime path: calls MakePointerFuture()

**Files Modified:**
- `/workspace/context-runtime/include/chimaera/ipc_manager.h`
  - Added MakeCopyFuture() template method (lines ~285-350)
  - Added MakePointerFuture() template method (lines ~352-376)
  - Simplified MakeFuture() to call sub-functions (lines ~378-410)

**Usage:**
```cpp
// GPU kernel can now call MakeCopyFuture directly
__global__ void submit_task_kernel(...) {
  CHIMAERA_GPU_INIT(backend);

  // Create and serialize task
  Future<MyTask> future = (&g_ipc_manager)->MakeCopyFuture(task_ptr);

  // Enqueue to GPU queue...
}
```

### 2. GPU Queue Infrastructure (Task #2)

**Implemented:**
- `ServerInitGpuQueues()` - Initializes one ring buffer per GPU device
  - Uses `GpuApi::GetDeviceCount()` to detect available GPUs
  - Creates pinned host memory segments using `GpuShmMmap`
  - Allocates one TaskQueue (ring buffer) per GPU
  - Stores backends in `gpu_backends_` vector
  - Stores queues in `gpu_queues_` vector

**Infrastructure Added:**
- GPU backend storage: `std::vector<std::unique_ptr<hipc::GpuShmMmap>> gpu_backends_`
- GPU queue storage: `std::vector<hipc::FullPtr<TaskQueue>> gpu_queues_`
- ServerInitGpuQueues() method for queue initialization
- Called from ServerInit() during runtime startup

**Features:**
- Configurable GPU segment size (default 64MB per GPU)
- Uses existing TaskQueue infrastructure (single lane, 2 priorities per GPU)
- Graceful handling when no GPUs are present (logs info, continues)
- Unique backend IDs (1000+gpu_id) to avoid conflicts with CPU backends
- Proper error handling and logging throughout

**Files Modified:**
- `/workspace/context-runtime/include/chimaera/ipc_manager.h`
  - Added `#include "hermes_shm/memory/backend/gpu_shm_mmap.h"`
  - Added gpu_backends_ and gpu_queues_ member variables (lines ~976-983)
  - Added ServerInitGpuQueues() declaration (lines ~936-943)

- `/workspace/context-runtime/src/ipc_manager.cc`
  - Implemented ServerInitGpuQueues() (lines ~443-524)
  - Called ServerInitGpuQueues() from ServerInit() (lines ~159-167)

**Configuration:**
```cpp
// In config file, can specify:
"gpu_segment_size": 67108864  // 64MB per GPU (default)
"queue_depth": 1024           // Ring buffer depth (shared with CPU queues)
```

## 🔄 In Progress / Remaining

### 3. NUMA Awareness for GPU Allocation (Task #3)

**Status:** Pending

**Requirements:**
- Query GPU's NUMA node affinity
- Modify GpuShmMmap::shm_init() to accept NUMA node parameter
- Use numa_alloc_onnode() or similar for NUMA-specific allocation
- Ensure pinned host memory is allocated from GPU's local NUMA node

**Approach:**
1. Add method to query GPU NUMA affinity (likely via CUDA/ROCm device properties)
2. Update GpuShmMmap to support NUMA node parameter in shm_init()
3. Use libnuma or similar to allocate from specific NUMA node
4. Update ServerInitGpuQueues() to pass NUMA node when creating backends

### 4. Worker GPU Queue Processing (Task #5)

**Status:** Pending

**Requirements:**
- Assign GPU queues to workers
- Split `ProcessNewTasks()` into `ProcessNewTask()` for single-task processing
- Add iteration logic to process both CPU and GPU queues
- Ensure workers can deserialize and execute GPU-submitted tasks

**Approach:**
1. Create ProcessNewTask() that processes a single task
2. Update ProcessNewTasks() to call ProcessNewTask() in a loop
3. Add logic to round-robin or prioritize between CPU and GPU queues
4. Handle GPU queue assignment (all workers? dedicated workers?)

## 📝 Implementation Notes

### GPU Queue Design
- Each GPU gets its own segment with pinned host memory
- Single ring buffer (TaskQueue) per GPU for now
- Tasks submitted from GPU kernels are serialized using MakeCopyFuture()
- Workers will eventually poll both CPU queues and GPU queues

### Memory Layout
```
GPU 0: [GpuShmMmap Backend] → [Allocator] → [TaskQueue (1 lane, 2 priorities)]
GPU 1: [GpuShmMmap Backend] → [Allocator] → [TaskQueue (1 lane, 2 priorities)]
...
```

### Future Enhancements
- NUMA-aware allocation (Task #3)
- Multiple lanes per GPU (if needed for higher concurrency)
- Dedicated GPU queue workers vs shared workers
- Direct GPU-to-GPU task submission (bypass host)
- GPU queue monitoring and statistics

## 🧪 Testing Needed

1. **GPU Queue Initialization**
   - Verify ServerInitGpuQueues() creates correct number of queues
   - Test with 0, 1, and multiple GPUs
   - Verify queue depths and priorities

2. **MakeCopyFuture from GPU**
   - Create kernel that calls MakeCopyFuture()
   - Verify task serialization works on GPU
   - Test with various task types and sizes

3. **NUMA Awareness** (after Task #3)
   - Verify GPU memory allocated from correct NUMA node
   - Performance testing with NUMA-aware vs NUMA-unaware

4. **Worker Processing** (after Task #5)
   - Verify workers process GPU queue tasks
   - Test task deserialization from GPU queues
   - Performance comparison CPU-only vs CPU+GPU queues

## 🎯 Next Steps

1. Complete Task #3: Add NUMA awareness to GpuShmMmap
2. Complete Task #5: Update Worker to process GPU queues
3. Create end-to-end test: GPU kernel submits task → Worker processes it
4. Add GPU queue monitoring/statistics
5. Performance optimization and tuning
