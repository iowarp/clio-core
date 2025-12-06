@CLAUDE.md 

Let's change the way MemoryBackend works. currently, it looks like this:
```
class MemoryBackend {
 public:
  MemoryBackendHeader *header_;
  union {
    char *data_; /** For CPU-only backends */
    char *md_;   /** For CPU+GPU backends */
  };
  union {
    size_t data_size_; /** For CPU-only backends */
    size_t md_size_;   /** For CPU+GPU backends */
  };
  bitfield64_t flags_;
  char *accel_data_;
  size_t accel_data_size_;
  int accel_id_;
}
```

I want it to be like this:
```
class MemoryBackend {
 public:
  MemoryBackendHeader *header_;
  char *md_;   //  metadata for how procesess (on CPU) connect to this guy. Not required for allocators.
  size_t md_size_;   // metadata size. Not required for allocators.
  bitfield64_t flags_;
  char *accel_data_;  // buffer_ in class Allocator
  size_t accel_data_size_;  // buffer_size_ in class Allocator
  int accel_id_;
}
```

Consequences:
1. Make it so gpu_malloc and gpu_shm_mmap call the SystemInfo::MapSharedMemory internally instead of inheriting for PosixShmMmap
2. Make it so malloc_backend.h, posix_mmap.h, and posix_shm_mmap.h first allocate to md_ and then, at alignment of 4KB, shift to the data_ segment.

The minimum backend size should be 1MB.


How does GPU allocation work? Two cases:
1. Private memory.
2. Shared memory (IPC mem handle).

Private memory:
1. We create the backend on the CPU. We may need to share the backend on the CPU across processes.
Requires a metadata payload. We should do this for all allocators. Separate 
2. We must create the allocator on the GPU. This requires copying the backend to the GPU and then 

Shared memory:
1. The data works on both CPU and GPU. Pinned host memory.
2. We can just do the traditional path.

Remove the unions from class Backend. We will assume there is a separation between 

