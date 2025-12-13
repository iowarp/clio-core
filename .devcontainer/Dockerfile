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

# Remove system HDF5 packages to prevent conflicts with conda HDF5
# System HDF5 was built against different libcurl than conda, causing linker errors
RUN apt-get update && apt-get remove -y \
    libhdf5-dev \
    libhdf5-103-1t64 \
    libhdf5-cpp-103-1t64 \
    libhdf5-fortran-102t64 \
    libhdf5-hl-100t64 \
    libhdf5-hl-cpp-100t64 \
    libhdf5-hl-fortran-100t64 \
    hdf5-helpers \
    hdf5-tools \
    && apt-get autoremove -y \
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

# Install CUDA from NVIDIA's official repository
# Using CUDA 12.6 (latest stable version)
# Note: We only install the toolkit, not drivers (drivers come from the host via --gpus flag)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y \
        cuda-toolkit-12-6 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

# Switch to iowarp user for conda and venv setup
USER iowarp
WORKDIR /home/iowarp

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
#   - Build tools: cmake, ninja, conda-build
#   - Core libraries: boost, hdf5, yaml-cpp, zeromq, cppzmq, cereal
#   - Testing: catch2
#   - Network: libcurl, openssl
#   - Compression: zlib
#   - Optional: poco (for Globus support), nlohmann_json
#   - GPU: CUDA installed from NVIDIA repos (see below)
RUN /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /home/iowarp/miniconda3/bin/conda install -y \
        conda-build \
        cmake \
        ninja \
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
        poco \
        nlohmann_json \
    && /home/iowarp/miniconda3/bin/conda clean -ya

# Create Python virtual environment in user's home directory
# Note: Users can choose between conda environments or venv
RUN python3 -m venv /home/iowarp/venv \
    && /home/iowarp/venv/bin/pip install --upgrade pip setuptools wheel \
    && /home/iowarp/venv/bin/pip install pyyaml nanobind

# Add conda activation and CUDA paths to bashrc
# Conda base environment is auto-activated to ensure all conda dependencies are available
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# CUDA environment variables' >> /home/iowarp/.bashrc \
    && echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> /home/iowarp/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# >>> conda initialize >>>' >> /home/iowarp/.bashrc \
    && echo '# Conda base environment is auto-activated with all dev dependencies' >> /home/iowarp/.bashrc \
    && echo '# This includes: boost, hdf5, yaml-cpp, zeromq, cereal, catch2, etc.' >> /home/iowarp/.bashrc \
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
