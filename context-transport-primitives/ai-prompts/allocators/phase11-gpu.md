@CLAUDE.md 

For this phase, let's use the cuda-debug preset to compile.
Fix any compilation issues that occur.
Try compiling immediately and then go on to the feature fixes.

# Augment MemoryBackend to handle GPU allocators
Add a flag called MEMORY_BACKEND_GPU_ONLY to flags.
Set this to true for the one GpuMalloc backend.
Add methods to set MEMORY_BACKEND_GPU_ONLY: SetGpuOnly, IsGpuOnly, UnsetGpuOnly.
Add another method called DoAccelPath that returns bool.
It returns true if IsGpuOnly is true and HSHM_IS_HOST is true.

MakeAlloc and AttachAlloc should have conditional logic.
If DoAccelPath is false, execute MakeAlloc as-is.
Otherwise, execute a kernel that takes the backend (and all other arguments) as input.
The else path should use the macros HSHM_ENABLE_CUDA and HSHM_ENABLE_ROCM internally to avoid compile errors for cases where we don't want cuda / rocm.
The kernel will then call backend.MakeAlloc(...) OR backend.AttachAlloc(...).
Please make use of the macros in macros.h to define kernels that are compatible across both cuda and rocm.

# Augment BaseAllocator to handle GPU allocators
If backend_.DoAccelPath is false, execute each 
Otherwise, execute a templated GPU kernel for the particular method.
We should have overrides for everything in BaseAllocator.

# GpuShmMmap
Should have a similar layout to PosixShmMmap.
Should also look at GpuMalloc's current implementation.
GPuMalloc does close to what I want GpuShmMmap to do.
GpuMalloc will be changed next. 
The main difference between the two are the APIs that cuda needs to register the memory with Cuda and enable IPC.
md_ and md_size_ should not exist anymore.

# GpuMalloc
The data and MemoryBackend header should be allocated differently.
The MemoryBackendHeader should be allocated with regular malloc
The data should be allocated with cudaMalloc.

# GpuShmMmap Test
1. Create a GpuShmMmap backend
2. Create an allocator on that backend
3. Allocate a ring_buffer on that backend
4. Pass the ring_buffer to the kernel
5. Verify that we can place 10 elements on the ring buffer
6. Verify the runtime can pop the 10 elements

# GpuMalloc Test
1. Create a GpuMalloc backend.
Then do the same as GpuShmMmap test.

Place both unit tests under a directory called test/unit/gpu