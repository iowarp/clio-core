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
LABEL version="2.0"
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
RUN useradd -m -s /bin/bash iowarp

# MPI environment
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

#------------------------------------------------------------
# Build from deps-cpu
#------------------------------------------------------------

FROM iowarp/deps-cpu:latest AS builder

WORKDIR /workspace
COPY . /workspace/

ENV VIRTUAL_ENV="/home/iowarp/venv"
ENV PATH="${VIRTUAL_ENV}/bin:/home/iowarp/.local/bin:${PATH}"

RUN sudo chown -R $(whoami):$(whoami) /workspace && \
    git submodule update --init --recursive && \
    cd /workspace/external/jarvis-cd && \
    pip install -r requirements.txt && \
    pip install -e . && \
    jarvis init && \
    jarvis rg build && \
    jarvis repo add /workspace/jarvis_iowarp && \
    cd /workspace && \
    mkdir -p build && \
    cd build && \
    cmake --preset build-cpu-release -DWRP_CORE_ENABLE_CONDA=OFF ../ && \
    sudo make -j$(nproc) install

# Seed default config at ~/.chimaera/chimaera.yaml (picked up automatically by runtime)
RUN mkdir -p /home/iowarp/.chimaera && \
    cp /workspace/context-runtime/config/chimaera_default.yaml \
       /home/iowarp/.chimaera/chimaera.yaml

#------------------------------------------------------------
# Final Deploy Image
#------------------------------------------------------------

FROM runtime-base

# Copy IOWarp installation from build container
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share /usr/local/share

# Copy default config for runtime auto-discovery
COPY --from=builder --chown=iowarp:iowarp /home/iowarp/.chimaera /home/iowarp/.chimaera

# Set up library paths
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu
ENV PATH=/usr/local/bin:${PATH}

# Update library cache
RUN ldconfig

# Switch to iowarp user
USER iowarp
WORKDIR /home/iowarp

# Set up environment in bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc

CMD ["/bin/bash"]
