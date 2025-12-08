# Allocator Benchmark Suite

This directory contains a comprehensive benchmark suite for measuring the performance of different memory allocator implementations in the context-transport-primitives library.

## Supported Allocators

1. **MultiProcessAllocator (`mp`)**: A thread-safe, multi-process allocator with three-tier allocation strategy (thread-local, process-local, global)
2. **BuddyAllocator (`buddy`)**: A buddy system allocator for efficient memory management
3. **Standard malloc (`malloc`)**: The standard C library malloc/free for baseline comparison

## Building

The benchmark is built automatically when `HSHM_ENABLE_BENCHMARKS` is enabled:

```bash
cmake --preset=debug -DWRP_CORE_ENABLE_BENCHMARKS=ON
cmake --build build --target allocator_benchmark
```

The executable is installed to `build/bin/allocator_benchmark`.

## Usage

```bash
allocator_benchmark <allocator_type> <num_threads> <min_size> <max_size> [duration_sec]
```

### Parameters

- **allocator_type**: Allocator to benchmark
  - `mp`: MultiProcessAllocator
  - `buddy`: BuddyAllocator
  - `malloc`: Standard C malloc

- **num_threads**: Number of concurrent threads (1-64)

- **min_size**: Minimum allocation size
  - Supports size suffixes: `K` (kilobytes), `M` (megabytes), `G` (gigabytes)
  - Examples: `1K`, `4K`, `64K`, `1M`, `16M`

- **max_size**: Maximum allocation size
  - Same format as min_size
  - Must be >= min_size

- **duration_sec** (optional): Duration to run benchmark in seconds
  - Default: 10 seconds
  - Range: 1-3600 seconds

## Examples

### Benchmark MultiProcessAllocator with 8 threads
```bash
./build/bin/allocator_benchmark mp 8 4K 1M 30
```
Runs MultiProcessAllocator with 8 threads, allocation sizes between 4KB and 1MB, for 30 seconds.

### Benchmark BuddyAllocator with 4 threads
```bash
./build/bin/allocator_benchmark buddy 4 1K 64K
```
Runs BuddyAllocator with 4 threads, allocation sizes between 1KB and 64KB, for 10 seconds (default).

### Benchmark standard malloc with 16 threads
```bash
./build/bin/allocator_benchmark malloc 16 128 16K 60
```
Runs standard malloc with 16 threads, allocation sizes between 128 bytes and 16KB, for 60 seconds.

### Compare different allocators
```bash
# Run same workload on different allocators
./build/bin/allocator_benchmark mp 8 4K 1M 30
./build/bin/allocator_benchmark buddy 8 4K 1M 30
./build/bin/allocator_benchmark malloc 8 4K 1M 30
```

## Output

The benchmark outputs detailed results including:

- **Threads**: Number of concurrent threads used
- **Size Range**: Minimum and maximum allocation sizes
- **Duration**: Requested duration and actual elapsed time in milliseconds
- **Total Allocs**: Total number of allocation operations
- **Total Frees**: Total number of free operations
- **Total Operations**: Sum of allocations and frees
- **Operations/sec**: Throughput in operations per second

Example output:
```
=== MultiProcessAllocator Benchmark Results ===
Threads:          8
Size Range:       4KB - 1MB
Duration:         30 seconds (actual: 30042 ms)
Total Allocs:     1,234,567
Total Frees:      1,234,567
Total Operations: 2,469,134
Operations/sec:   82,203.45
============================================
```

## Benchmark Methodology

The benchmark uses a realistic allocation pattern:

1. Each thread maintains a pool of up to 1000 active allocations
2. On each iteration, the thread randomly decides to:
   - **Allocate**: Request a random size between min_size and max_size
   - **Free**: Release a randomly selected active allocation
3. All allocated memory is written to ensure it's valid and mapped
4. The benchmark runs for the specified duration
5. All remaining allocations are freed at the end

This pattern simulates real-world memory allocation behavior where allocations and deallocations are interleaved.

## Performance Tips

- **Thread Scaling**: Test with different thread counts (1, 2, 4, 8, 16, 32) to understand scaling characteristics
- **Size Distribution**: Try different size ranges to test small allocations (< 4KB), medium allocations (4KB-1MB), and large allocations (> 1MB)
- **Duration**: Use longer durations (30-60 seconds) for stable measurements
- **System Load**: Run benchmarks on an idle system for consistent results

## Comparison Script

A helper script is provided to easily compare all three allocators with the same workload:

```bash
./context-transport-primitives/benchmark/run_comparison.sh <num_threads> <min_size> <max_size> <duration_sec>
```

Example:
```bash
./context-transport-primitives/benchmark/run_comparison.sh 8 4K 1M 10
```

This runs all three allocators sequentially with identical parameters and displays results for easy comparison.

## Implementation Details

- **MultiProcessAllocator**: Uses PosixShmMmap backend with 2GB heap
- **BuddyAllocator**: Uses MallocBackend with 2GB heap
- **Standard malloc**: Uses system malloc directly

All HSHM allocators write to allocated memory to verify allocations are valid and properly mapped.

## Known Performance Characteristics

From testing on the development system:

- **malloc**: Fastest for most workloads (3-9 million ops/sec for 4KB allocations)
- **MultiProcessAllocator**: Good thread-local performance (1-7 million ops/sec for 4KB allocations)
- **BuddyAllocator**: Good single-size performance but slower with variable sizes

These numbers will vary based on hardware, system load, and workload characteristics.
