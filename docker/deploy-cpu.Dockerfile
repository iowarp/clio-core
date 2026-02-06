# IOWarp CPU Deploy Container
# Minimal deployment container with only runtime binaries
#
# This container copies the installed binaries from build-cpu, reducing
# overall container size without sacrificing functionality.
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
# Copy from Build Container
#------------------------------------------------------------

FROM iowarp/build-cpu:latest AS builder

# The build container already has everything installed in /usr/local

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
