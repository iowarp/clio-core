# Dockerfile for Redis Benchmark
# Runs Redis benchmarks for performance comparison with CTE
#
# Usage:
#   docker run iowarp/cte-redis-bench:latest Set 4 1m 10000
#   docker run iowarp/cte-redis-bench:latest Get 8 4k 100000
#   docker run iowarp/cte-redis-bench:latest SetGet 4 1m 10000
#   docker run iowarp/cte-redis-bench:latest All 4 1m 10000

FROM iowarp/build-cpu:latest

# Install Redis server and tools
# Using redis from Ubuntu repositories for simplicity and stability
RUN sudo apt-get update && \
    sudo apt-get install -y redis-server redis-tools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Create benchmark directory
RUN sudo mkdir -p /benchmarks && \
    sudo chown -R iowarp:ubuntu /benchmarks

# Copy the Redis benchmark script
COPY context-transfer-engine/benchmark/redis_bench.sh /benchmarks/redis_bench.sh

# Make the script executable
RUN sudo chmod +x /benchmarks/redis_bench.sh

# Set working directory
WORKDIR /benchmarks

# Set default environment variables
ENV TEST_CASE=All
ENV NUM_CLIENTS=4
ENV IO_SIZE=1m
ENV IO_COUNT=10000

# Create a wrapper script that handles both direct arguments and environment variables
RUN echo '#!/bin/bash' > /benchmarks/run_bench.sh && \
    echo 'set -e' >> /benchmarks/run_bench.sh && \
    echo '# If arguments are provided, use them directly' >> /benchmarks/run_bench.sh && \
    echo 'if [ $# -eq 4 ]; then' >> /benchmarks/run_bench.sh && \
    echo '    exec /benchmarks/redis_bench.sh "$@"' >> /benchmarks/run_bench.sh && \
    echo '# Otherwise, use environment variables' >> /benchmarks/run_bench.sh && \
    echo 'else' >> /benchmarks/run_bench.sh && \
    echo '    exec /benchmarks/redis_bench.sh "$TEST_CASE" "$NUM_CLIENTS" "$IO_SIZE" "$IO_COUNT"' >> /benchmarks/run_bench.sh && \
    echo 'fi' >> /benchmarks/run_bench.sh && \
    sudo chmod +x /benchmarks/run_bench.sh

# Set the entrypoint to run the wrapper script
# Arguments: test_case, num_clients, io_size, io_count
# Or use environment variables: TEST_CASE, NUM_CLIENTS, IO_SIZE, IO_COUNT
ENTRYPOINT ["/benchmarks/run_bench.sh"]

# Default arguments
CMD []
