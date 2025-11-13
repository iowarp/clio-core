FROM iowarp/iowarp-deps:latest

COPY . /workspace

WORKDIR /workspace

RUN cd build && sudo make -j$(nproc) install
RUN sudo rm -rf /workspace

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
