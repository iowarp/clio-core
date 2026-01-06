@CLAUDE.md Implement a benchmark for Put, Get, GetTagSize. The benchmark should take as input a test_case, depth, io_size, and io_count. Test case is the benchmark to conduct. Options should be Put, Get, PutGet. Depth should be the number of async requests to generate. For example, if the depth is 4, then generate 4 PutBlob operations using async, and then wait for all 4 to complete. io_size is the size of I/O operations. io_count is the number of I/O operations to generate per node.

You may use MPI for building the benchmark to support parallel I/O.

Implement the benchmarks under the benchmark directory.

Build a jarvis package for the benchmark under test/jarvis_iowarp/jarvis_iowarp/wrp_cte_bench. Read @docs/jarvis/package_dev_guide.md to see how to build a package properly. This is an application package.
