# IOWarp CPU Build Container
# Builds IOWarp from the current source tree
#
# Usage:
#   docker build -t iowarp/build-cpu:latest -f docker/build-cpu.Dockerfile .
#
FROM iowarp/deps-cpu:latest
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="1.0"
LABEL description="IOWarp CPU build container"

# Set working directory
WORKDIR /workspace

# Copy the entire source tree
COPY . /workspace/

# Initialize git submodules and build using the build-cpu-release preset
# Install to /usr/local
RUN sudo chown -R $(whoami):$(whoami) /workspace && \
    git submodule update --init --recursive && \
    mkdir -p build && \
    cd build && \
    cmake --preset build-cpu-release ../ && \
    sudo make -j$(nproc) install && \
    sudo rm -rf /workspace

# Add iowarp-core to Spack configuration
RUN echo "  iowarp-core:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: iowarp-core@main" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr/local" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml

# Create empty runtime configuration files
# Pre-create wrp_config.yaml and hostfile as files (not directories) to enable Docker volume mounting
RUN sudo mkdir -p /etc/iowarp && \
    sudo touch /etc/iowarp/wrp_conf.yaml && \
    sudo touch /etc/iowarp/wrp_config.yaml && \
    sudo touch /etc/iowarp/hostfile

# Set runtime configuration environment variable
ENV WRP_RUNTIME_CONF=/etc/iowarp/wrp_conf.yaml

WORKDIR /home/iowarp

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
