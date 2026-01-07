FROM iowarp/iowarp-deps:latest

# Switch to root for package installation
USER root

# Install Docker CLI and Docker-in-Docker dependencies
# Also install network diagnostic tools (netstat, lsof, ss, etc.)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    iptables \
    supervisor \
    net-tools \
    lsof \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*


# Add Docker's official GPG key and repository
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg

RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
RUN apt-get update && apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Add iowarp user to docker group
RUN usermod -aG docker iowarp

# Create docker group if it doesn't exist (it should from docker install)
RUN getent group docker || groupadd docker

# Set up Docker socket permissions script
RUN echo '#!/bin/bash\n\
    if [ -S /var/run/docker.sock ]; then\n\
    sudo chmod 666 /var/run/docker.sock\n\
    fi\n\
    exec "$@"' > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Allow iowarp user to manage docker socket permissions without password
RUN echo "iowarp ALL=(ALL) NOPASSWD: /bin/chmod 666 /var/run/docker.sock" >> /etc/sudoers.d/docker-socket \
    && chmod 0440 /etc/sudoers.d/docker-socket

# Install NVIDIA Container Toolkit repository
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install CUDA from NVIDIA's official repository
# This installs the complete CUDA toolkit + runtime libraries for actual GPU execution
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-6 \
    cuda-cudart-12-6 \
    cuda-libraries-12-6 \
    cuda-nvrtc-12-6 \
    cuda-nvml-dev-12-6 \
    libcublas-12-6 \
    libcufft-12-6 \
    libcurand-12-6 \
    libcusolver-12-6 \
    libcusparse-12-6 \
    libnpp-12-6 \
    libnvidia-container-tools \
    libnvidia-container1 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables for runtime execution
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:/usr/lib/x86_64-linux-gnu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Switch to iowarp user for venv setup
# Note: conda is already installed in the base image (iowarp/iowarp-deps:latest)
# with all development dependencies (boost, hdf5, yaml-cpp, zeromq, etc.)
USER iowarp
WORKDIR /home/iowarp

# Add CUDA paths to bashrc
# Note: conda is inherited from base image with all dev dependencies pre-installed
# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /home/iowarp/miniconda3 \
    && rm /tmp/miniconda.sh

# Initialize conda for bash
RUN /home/iowarp/miniconda3/bin/conda init bash \
    && /home/iowarp/miniconda3/bin/conda config --add channels conda-forge \
    && /home/iowarp/miniconda3/bin/conda config --set channel_priority strict

# Accept Anaconda Terms of Service and install all development dependencies via conda
# This avoids library conflicts between system packages and conda packages
# Dependencies installed:
#   - Build tools: cmake, ninja, conda-build, pkg-config
#   - Core libraries: boost, hdf5, yaml-cpp, zeromq, cppzmq, cereal
#   - Testing: catch2
#   - Network: libcurl, openssl
#   - Compression libraries: zlib, bzip2, lzo, zstd, lz4, xz (lzma), brotli, snappy
#   - Lossy compression: sz3, zfp (scientific compressors, available via conda)
#   - Optional: poco (for Globus support), nlohmann_json
#   - GPU: CUDA installed from NVIDIA repos (see below)
RUN /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /home/iowarp/miniconda3/bin/conda install -y \
        conda-build \
        cmake \
        ninja \
        pkg-config \
        boost \
        hdf5 \
        yaml-cpp \
        zeromq \
        cppzmq \
        cereal \
        catch2 \
        libcurl \
        openssl \
        zlib \
        bzip2 \
        lzo \
        zstd \
        lz4-c \
        xz \
        brotli \
        snappy \
        sz3 \
        zfp \
        poco \
        nlohmann_json \
    && /home/iowarp/miniconda3/bin/conda clean -ya

# Create Python virtual environment in user's home directory
# Note: Users can choose between conda environments or venv
# Install Python packages for data analysis and benchmarking
RUN python3 -m venv /home/iowarp/venv \
    && /home/iowarp/venv/bin/pip install --upgrade pip setuptools wheel \
    && /home/iowarp/venv/bin/pip install pyyaml nanobind \
    && /home/iowarp/venv/bin/pip install pandas matplotlib seaborn numpy

# Switch back to root for libpressio installation
USER root

# Install lossy compression libraries and LibPressio
# This provides scientific data compression capabilities for IOWarp Core
RUN cd /tmp \
    # Install FPZIP library (fast floating-point compressor)
    && git clone https://github.com/LLNL/fpzip.git fpzip_src \
    && cd fpzip_src \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX=/usr/local \
                -DCMAKE_POLICY_DEFAULT_CMP0025=NEW \
                -DBUILD_SHARED_LIBS=ON \
                -DBUILD_TESTING=OFF \
                -DBUILD_UTILITIES=OFF \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    # Install LibPressio with compression plugins enabled (only working compressors)
    && cd /tmp \
    && git clone https://github.com/robertu94/libpressio.git libpressio_src \
    && cd libpressio_src \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX=/usr/local \
                -DLIBPRESSIO_HAS_ZFP=ON \
                -DLIBPRESSIO_HAS_SZ3=ON \
                -DLIBPRESSIO_HAS_FPZIP=ON \
                -DBUILD_TESTING=OFF \
    && make -j$(nproc) libpressio \
    && make install \
    && ldconfig \
    # Cleanup temporary build files
    && cd /tmp \
    && rm -rf fpzip_src libpressio_src

# Switch back to iowarp user
USER iowarp

# Add conda activation and CUDA paths to bashrc
# Conda base environment is auto-activated to ensure all conda dependencies are available
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# CUDA environment variables for GPU execution' >> /home/iowarp/.bashrc \
    && echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> /home/iowarp/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/lib64/stubs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc \
    && echo 'export NVIDIA_VISIBLE_DEVICES=all' >> /home/iowarp/.bashrc \
    && echo 'export NVIDIA_DRIVER_CAPABILITIES=compute,utility' >> /home/iowarp/.bashrc

# Install libaio for Linux AIO support (required by bdev ChiMod)
SHELL ["/bin/bash", "-c"]
RUN source /home/iowarp/miniconda3/etc/profile.d/conda.sh \
    && conda activate base \
    && conda install -y libaio -c conda-forge

# Install Jarvis (IOWarp runtime deployment tool)
# Note: Must source conda in non-interactive shell since .bashrc isn't loaded during build
RUN cd /home/iowarp \
    && git clone https://github.com/iowarp/runtime-deployment.git \
    && cd runtime-deployment \
    && source /home/iowarp/miniconda3/etc/profile.d/conda.sh \
    && conda activate base \
    && pip install -e . -r requirements.txt  \
    && jarvis init \
    && jarvis rg build

    && echo 'export NVIDIA_DRIVER_CAPABILITIES=compute,utility' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# >>> conda initialize >>>' >> /home/iowarp/.bashrc \
    && echo '# Conda base environment is auto-activated with all dev dependencies' >> /home/iowarp/.bashrc \
    && echo '# This includes: boost, hdf5, yaml-cpp, zeromq, cereal, catch2, etc.' >> /home/iowarp/.bashrc \
    && echo '# Compression libraries: zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, sz3, zfp' >> /home/iowarp/.bashrc \
    && echo '# Lossy compression: ZFP, SZ3, FPZIP (via LibPressio)' >> /home/iowarp/.bashrc \
    && echo '# Create custom environments if needed: conda create -n myenv' >> /home/iowarp/.bashrc \
    && echo 'eval "$(/home/iowarp/miniconda3/bin/conda shell.bash hook)"' >> /home/iowarp/.bashrc \
    && echo '# <<< conda initialize <<<' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# Python virtual environment (alternative to conda)' >> /home/iowarp/.bashrc \
    && echo '# Note: venv is NOT recommended when using conda dependencies' >> /home/iowarp/.bashrc \
    && echo '# Uncomment to auto-activate venv instead of conda (use at your own risk):' >> /home/iowarp/.bashrc \
    && echo '# if [ -f /home/iowarp/venv/bin/activate ]; then' >> /home/iowarp/.bashrc \
    && echo '#     source /home/iowarp/venv/bin/activate' >> /home/iowarp/.bashrc \
    && echo '# fi' >> /home/iowarp/.bashrc

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
