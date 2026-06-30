# IOWarp CPU Development Container
# Inherits from deps-cpu which has all build dependencies
#
# This devcontainer adds development-specific tools like Claude Code.
#
FROM iowarp/deps-cpu:latest
USER root
RUN userdel -r ubuntu || true

# FUSE 3 runtime (fusermount3) + dev headers, needed to build the libfuse
# adapter (clio_cte_fuse) and run its mount integration tests. Installed here
# so a devcontainer rebuild picks it up immediately; it is also in the base
# image (docker/iowarp-base.Dockerfile) as the canonical default. The
# container runs --privileged, so /dev/fuse + SYS_ADMIN are available.
RUN apt-get update && apt-get install -y --no-install-recommends \
      fuse3 libfuse3-dev \
      libboost-context-dev libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# Build the Boost.Context stackful coroutine backend by default in the
# devcontainer so it is exercised during development (issue #620). Build tooling
# that honors this passes -DCLIO_CORE_ENABLE_BOOST_COROUTINES=ON.
ENV CLIO_CORE_ENABLE_BOOST_COROUTINES=ON

# The deps-cpu image already includes:
# - All apt + source-built dependencies (boost, hdf5, yaml-cpp, zeromq, cereal, etc.)
# - Compression libraries (zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2)
# - Lossy compression (ZFP, SZ3, FPZIP, LibPressio)
# - ADIOS2 with HDF5 and ZeroMQ support
# - Docker-in-Docker support
# - Jarvis (IOWarp runtime deployment tool)
# - MPI (OpenMPI)
# - libaio (for bdev ChiMod)

# Remap iowarp user UID/GID to match the host user.
# This avoids file permission issues with bind-mounted volumes.
# Override at build time: --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g)
ARG HOST_UID=1000
ARG HOST_GID=1000

USER root
# Ubuntu 24.04 ships a default "ubuntu" user at UID 1000 which blocks
# updateRemoteUserUID and manual UID remapping (see
# https://github.com/microsoft/vscode-remote-release/issues/10030).
# Move it out of the way so iowarp can claim the host user's UID.
RUN if id ubuntu >/dev/null 2>&1; then \
      usermod -u 59999 ubuntu && \
      groupmod -g 59999 ubuntu; \
    fi && \
    OLD_UID=$(id -u iowarp) && OLD_GID=$(id -g iowarp) && \
    NEW_UID="${HOST_UID:-1000}" && NEW_GID="${HOST_GID:-1000}" && \
    if [ "${NEW_UID}" != "${OLD_UID}" ] || [ "${NEW_GID}" != "${OLD_GID}" ]; then \
      groupmod -o -g "${NEW_GID}" iowarp && \
      usermod  -o -u "${NEW_UID}" -g "${NEW_GID}" iowarp && \
      chown -R "${NEW_UID}:${NEW_GID}" /home/iowarp; \
    fi

USER iowarp
WORKDIR /workspace

# Install Claude Code for AI-assisted development
RUN curl -fsSL https://claude.ai/install.sh | bash

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
