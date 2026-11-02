# CUDA Kernel Debugging in VSCode

## Quick Start

1. **Select the CUDA-GDB configuration**: In the VSCode debug panel, choose "Debug GPU Malloc Test (CUDA-GDB)"
2. **Set breakpoints**: Click in the gutter next to kernel code (e.g., `MakeAllocKernel`, `PushElementsKernel`)
3. **Press F5** to start debugging

## Setting Breakpoints in Kernels

You can set breakpoints in any of these kernels:
- `MakeAllocKernel` (line 36)
- `AllocateRingBufferKernel` (line 52)
- `PushElementsKernel` (line 73)
- `PopElementsKernel` (line 91)

## CUDA-GDB Commands in Debug Console

Once stopped at a breakpoint in a kernel, use these commands in the Debug Console:

### Thread and Block Information
```gdb
-exec info cuda threads     # Show all CUDA threads
-exec info cuda blocks      # Show all CUDA blocks
-exec cuda thread           # Show current thread coordinates
-exec cuda block            # Show current block coordinates
```

### Switch Between Threads/Blocks
```gdb
-exec cuda thread (0,0,0)   # Switch to thread (0,0,0)
-exec cuda block (1,0,0)    # Switch to block (1,0,0)
-exec cuda device 0         # Switch to GPU device 0
```

### Inspect Variables
```gdb
-exec print threadIdx.x     # Print thread X coordinate
-exec print blockIdx.x      # Print block X coordinate
-exec print values[0]       # Print array element
-exec print *ring           # Dereference pointer
```

### Execution Control
```gdb
-exec continue              # Continue execution
-exec step                  # Step into next line
-exec next                  # Step over next line
-exec finish                # Run until current function returns
```

## Common CUDA-GDB Issues

### Issue: Breakpoints not hitting
**Solution**: Ensure you compiled with `-G` flag (device debug symbols)
```cmake
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
```

### Issue: Can't see all threads
**Solution**: CUDA-GDB only shows active threads. Use `-exec info cuda threads` to list all.

### Issue: Variables show as optimized out
**Solution**: Recompile with `-O0` to disable optimizations:
```cmake
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
```

## Debugging Workflow

### Example: Debug PushElementsKernel

1. Set breakpoint at line 76 (inside the for loop)
2. Start debugging (F5)
3. When breakpoint hits, the debugger stops in the first thread
4. In Debug Console:
   ```gdb
   -exec print i              # Current loop iteration
   -exec print values[i]      # Value being pushed
   -exec cuda thread          # Which thread is this?
   -exec next                 # Step to next line
   ```

## Alternative: Printf Debugging

If CUDA-GDB doesn't work, add printf statements:

```cpp
__global__ void PushElementsKernel(...) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Pushing %zu elements\n", count);
  }
  for (size_t i = 0; i < count; ++i) {
    printf("Thread %d pushing value %d\n", threadIdx.x, values[i]);
    ring->Emplace(values[i]);
  }
}
```

Then run and check output:
```bash
./build/bin/test_gpu_malloc
```

## Compute Sanitizer (Memory Debugging)

For memory errors, use compute-sanitizer from terminal:

```bash
# Memory leak detection
compute-sanitizer --tool memcheck ./build/bin/test_gpu_malloc

# Race condition detection
compute-sanitizer --tool racecheck ./build/bin/test_gpu_malloc

# Uninitialized memory detection
compute-sanitizer --tool initcheck ./build/bin/test_gpu_malloc

# Synchronization errors
compute-sanitizer --tool synccheck ./build/bin/test_gpu_malloc
```

## Tips

1. **Single-thread debugging**: Launch kernels with `<<<1, 1>>>` for simpler debugging
2. **Focus on thread (0,0,0)**: Usually sufficient for logic bugs
3. **Use conditional breakpoints**: Right-click breakpoint → Edit → Add condition like `threadIdx.x == 0`
4. **Check for CUDA errors**: Always check `cudaGetLastError()` after kernel launches
5. **Synchronize before checking**: Call `cudaDeviceSynchronize()` to ensure kernel completion

## VSCode Extensions (Optional)

For enhanced CUDA debugging, install:
- **NVIDIA NSight Visual Studio Code Edition** (if available)
- **C/C++ Extension Pack** (already required)

## Configuration Files

- **Launch config**: `.vscode/launch.json` - "Debug GPU Malloc Test (CUDA-GDB)"
- **Test file**: `context-transport-primitives/base/test/gpu/test_gpu_malloc.cc`
