# Dockerfile for building the Content Transfer Engine (CTE)
# DEPRECATED: Use build-cpu.Dockerfile instead.
# This file is kept for backward compatibility.
#
# Inherits from iowarp/deps-cpu:latest which contains all build dependencies

FROM iowarp/deps-cpu:latest

# Set working directory
WORKDIR /workspace

# Copy the entire CTE source tree
COPY . /workspace/

# Initialize git submodules and build
# Install to /usr/local
RUN sudo chown -R $(whoami):$(whoami) /workspace && \
    git submodule update --init --recursive && \
    mkdir -p build && \
    cd build && \
    cmake --preset release ../ && \
    sudo make -j$(nproc) install && \
    sudo rm -rf /workspace


# Add iowarp-cte to Spack configuration
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
