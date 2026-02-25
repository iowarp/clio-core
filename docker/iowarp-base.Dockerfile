# Install Ubuntu.
FROM ubuntu:24.04
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="0.0"
LABEL description="IoWarp spack docker image"

# Disable prompt during packages installation.
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y

# Install basic packages.
# NOTE: The following packages have been moved to conda and are commented out:
#   cmake, openssl, libssl-dev, zlib1g-dev, hdf5-tools
# They are still listed here in case we need to restore apt versions in the future.
RUN apt install -y \
    openssh-server \
    sudo git \
    gcc g++ gfortran make binutils gpg \
    tar zip xz-utils bzip2 \
    perl m4 libncurses5-dev libxml2-dev diffutils \
    pkg-config \
    python3 python3-pip python3-venv doxygen \
    lcov \
    build-essential ca-certificates \
    coreutils curl wget \
    lsb-release unzip liblz4-dev \
    bash jq gdbserver gdb gh nano vim dos2unix \
    clangd clang-format clang-tidy npm \
    redis-server redis-tools \
    gnupg \
    net-tools lsof iproute2 \
    && rm -rf /var/lib/apt/lists/*
# Commented apt packages now provided by conda:
#   cmake \
#   openssl libssl-dev \
#   zlib1g-dev \
#   hdf5-tools \

#------------------------------------------------------------
# User Configuration
#------------------------------------------------------------

# Create non-root user with sudo privileges
RUN useradd -m -s /bin/bash -G sudo iowarp && \
    echo "iowarp ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    passwd -d iowarp

# Switch to non-root user
USER iowarp
ENV USER="iowarp"
ENV HOME="/home/iowarp"
WORKDIR /home/iowarp

#------------------------------------------------------------
# Python Virtual Environment
#------------------------------------------------------------

# Create Python virtual environment in user's home directory
# Note: Users can choose between conda environments or venv
RUN python3 -m venv /home/iowarp/venv \
    && /home/iowarp/venv/bin/pip install --upgrade pip setuptools wheel \
    && /home/iowarp/venv/bin/pip install pyyaml nanobind

#------------------------------------------------------------
# Spack Configuration
#------------------------------------------------------------

# Setup basic environment.
ENV SPACK_DIR="${HOME}/spack"
ENV SPACK_VERSION="v1.1.0"

# Install Spack.
RUN git clone -b ${SPACK_VERSION} https://github.com/spack/spack ${SPACK_DIR} && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    spack external find

# Add GRC Spack repo.
RUN git clone https://github.com/grc-iit/grc-repo.git ${HOME}/grc-repo && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    spack repo add ${HOME}/grc-repo

# Add IOWarp Spack repo.
RUN git clone https://github.com/iowarp/iowarp-install.git ${HOME}/iowarp-install && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    spack repo add ${HOME}/iowarp-install/iowarp-spack

#------------------------------------------------------------
# SSH Configuration
#------------------------------------------------------------

# Configure SSH for iowarp user
RUN mkdir -p ~/.ssh && \
    echo "Host *" >> ~/.ssh/config && \
    echo "    StrictHostKeyChecking no" >> ~/.ssh/config && \
    chmod 600 ~/.ssh/config

# Enable passwordless SSH (requires root)
USER root
RUN sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config && \
    mkdir -p /run/sshd

# Switch back to iowarp user
USER iowarp

#------------------------------------------------------------
# Environment Configuration
#------------------------------------------------------------

# Add spack to bashrc
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# Spack environment' >> /home/iowarp/.bashrc \
    && echo 'source ${SPACK_DIR}/share/spack/setup-env.sh' >> /home/iowarp/.bashrc

WORKDIR /workspace

# Start SSH on container startup (using sudo since iowarp user has NOPASSWD)
CMD sudo service ssh start && /bin/bash
