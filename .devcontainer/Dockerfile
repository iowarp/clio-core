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
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# CUDA environment variables for GPU execution' >> /home/iowarp/.bashrc \
    && echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> /home/iowarp/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/lib64/stubs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc \
    && echo 'export NVIDIA_VISIBLE_DEVICES=all' >> /home/iowarp/.bashrc \
    && echo 'export NVIDIA_DRIVER_CAPABILITIES=compute,utility' >> /home/iowarp/.bashrc

# Install Jarvis (IOWarp runtime deployment tool)
# Note: Must source conda in non-interactive shell since .bashrc isn't loaded during build
SHELL ["/bin/bash", "-c"]
RUN cd /home/iowarp \
    && git clone https://github.com/iowarp/runtime-deployment.git \
    && cd runtime-deployment \
    && source /home/iowarp/miniconda3/etc/profile.d/conda.sh \
    && conda activate base \
    && pip install -e . -r requirements.txt  \
    && jarvis init \
    && jarvis rg build


WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
