# IOWarp CPU Deploy Container
# Minimal deployment container with only runtime binaries
#
# Builds IOWarp from source using deps-cpu, then copies only the
# installed binaries into a minimal Ubuntu image.
#
# Usage:
#   docker build -t iowarp/deploy-cpu:latest -f docker/deploy-cpu.Dockerfile .
#
FROM ubuntu:24.04 AS runtime-base
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="1.0"
LABEL description="IOWarp CPU deployment container"

# Disable prompt during packages installation
ARG DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libelf1 \
    openmpi-bin \
    libopenmpi3t64 \
    && rm -rf /var/lib/apt/lists/*

# Create iowarp user
RUN useradd -m -s /bin/bash iowarp && \
    mkdir -p /etc/iowarp && \
    chown -R iowarp:iowarp /etc/iowarp

# MPI environment
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

#------------------------------------------------------------
# Build from deps-cpu
#------------------------------------------------------------

FROM iowarp/deps-cpu:latest AS builder

WORKDIR /workspace
COPY . /workspace/

RUN sudo chown -R $(whoami):$(whoami) /workspace && \
    git submodule update --init --recursive && \
    cd /workspace/external/runtime-deployment && \
    source /home/iowarp/miniconda3/etc/profile.d/conda.sh && \
    conda activate base && \
    pip install -e . -r requirements.txt && \
    jarvis init && \
    jarvis rg build && \
    jarvis repo add /workspace/jarvis_iowarp && \
    cd /workspace && \
    mkdir -p build && \
    cd build && \
    cmake --preset build-cpu-release ../ && \
    sudo make -j$(nproc) install

# Install default runtime configuration
RUN sudo mkdir -p /etc/iowarp && \
    sudo cp /workspace/context-runtime/config/chimaera_default.yaml /etc/iowarp/wrp_conf.yaml && \
    sudo touch /etc/iowarp/hostfile

#------------------------------------------------------------
# Final Deploy Image
#------------------------------------------------------------

FROM runtime-base

# Copy conda environment (needed for runtime dependencies like hdf5, zeromq, etc.)
COPY --from=builder /home/iowarp/miniconda3 /home/iowarp/miniconda3

# Copy IOWarp installation from build container
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/include /usr/local/include
COPY --from=builder /usr/local/share /usr/local/share

# Copy IOWarp runtime configuration
COPY --from=builder /etc/iowarp /etc/iowarp

# Set up library paths
ENV LD_LIBRARY_PATH=/usr/local/lib:/home/iowarp/miniconda3/lib:/usr/lib/x86_64-linux-gnu
ENV PATH=/usr/local/bin:/home/iowarp/miniconda3/bin:${PATH}
ENV CONDA_PREFIX=/home/iowarp/miniconda3
ENV CMAKE_PREFIX_PATH=/home/iowarp/miniconda3:${CMAKE_PREFIX_PATH}

# Set runtime configuration environment variable
ENV WRP_RUNTIME_CONF=/etc/iowarp/wrp_conf.yaml

# Update library cache
RUN ldconfig

# Set ownership for iowarp user
RUN chown -R iowarp:iowarp /home/iowarp

# Switch to iowarp user
USER iowarp
WORKDIR /home/iowarp

# Initialize conda in bashrc
RUN echo 'eval "$(/home/iowarp/miniconda3/bin/conda shell.bash hook)"' >> /home/iowarp/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/home/iowarp/miniconda3/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc

CMD ["/bin/bash"]
