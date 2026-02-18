# Dockerfile for WRP CTE Benchmark
# Runs CTE Core benchmarks for Put, Get, and PutGet operations
#
# Usage with environment variables:
#   docker run -e TEST_CASE=Put -e NUM_PROCS=1 -e DEPTH=4 -e IO_SIZE=1m -e IO_COUNT=100 iowarp/cte-wrp-bench:latest
#   docker run -e TEST_CASE=Get -e NUM_PROCS=4 -e DEPTH=8 -e IO_SIZE=4k -e IO_COUNT=1000 iowarp/cte-wrp-bench:latest
#
# Note: This container uses environment variables only and automatically stops the Chimaera runtime after benchmark completion

FROM iowarp/deps-cpu:latest

WORKDIR /workspace
COPY . /workspace/

# Build and install IOWarp
RUN sudo chown -R $(whoami):$(whoami) /workspace && \
    git submodule update --init --recursive && \
    mkdir -p build && \
    cd build && \
    cmake --preset build-cpu-release ../ && \
    sudo make -j$(nproc) install && \
    sudo rm -rf /workspace

# Create benchmark directory
RUN sudo mkdir -p /benchmarks && \
    sudo chown -R iowarp:ubuntu /benchmarks

# Copy the benchmark script and configuration file
COPY context-transfer-engine/benchmark/wrp_cte_bench.sh /benchmarks/wrp_cte_bench.sh
COPY context-transfer-engine/benchmark/cte_config_ram.yaml /benchmarks/cte_config.yaml

# Make the script executable
RUN sudo chmod +x /benchmarks/wrp_cte_bench.sh

# Set working directory
WORKDIR /benchmarks

# Set default environment variables for CTE runtime
ENV CHIMAERA_WITH_RUNTIME=0
ENV CHI_SERVER_CONF=/benchmarks/cte_config.yaml

# Default benchmark parameters (can be overridden with environment variables)
ENV TEST_CASE=Put
ENV NUM_PROCS=1
ENV DEPTH=4
ENV IO_SIZE=1m
ENV IO_COUNT=100

# Create a wrapper script that uses environment variables and stops runtime after benchmark
RUN echo '#!/bin/bash' > /benchmarks/run_bench.sh && \
    echo 'set -e' >> /benchmarks/run_bench.sh && \
    echo '# Run benchmark using environment variables' >> /benchmarks/run_bench.sh && \
    echo '/benchmarks/wrp_cte_bench.sh "$TEST_CASE" "$NUM_PROCS" "$DEPTH" "$IO_SIZE" "$IO_COUNT"' >> /benchmarks/run_bench.sh && \
    echo 'BENCH_EXIT=$?' >> /benchmarks/run_bench.sh && \
    echo '# Stop Chimaera runtime after benchmark completes' >> /benchmarks/run_bench.sh && \
    echo 'chimaera runtime stop' >> /benchmarks/run_bench.sh && \
    echo 'exit $BENCH_EXIT' >> /benchmarks/run_bench.sh && \
    sudo chmod +x /benchmarks/run_bench.sh

# Set the entrypoint to run the wrapper script
# Use environment variables: TEST_CASE, NUM_PROCS, DEPTH, IO_SIZE, IO_COUNT
ENTRYPOINT ["/benchmarks/run_bench.sh"]
