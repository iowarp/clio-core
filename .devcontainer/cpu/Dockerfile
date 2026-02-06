# IOWarp CPU Development Container
# Inherits from deps-cpu which has all build dependencies
#
# This devcontainer adds development-specific tools like Claude Code.
#
FROM iowarp/deps-cpu:latest

# The deps-cpu image already includes:
# - All conda dependencies (boost, hdf5, yaml-cpp, zeromq, cereal, etc.)
# - Compression libraries (zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2)
# - Lossy compression (ZFP, SZ3, FPZIP, LibPressio)
# - ADIOS2 with HDF5 and ZeroMQ support
# - Docker-in-Docker support
# - Jarvis (IOWarp runtime deployment tool)
# - MPI (OpenMPI)
# - libaio (for bdev ChiMod)

WORKDIR /workspace

# Install Claude Code for AI-assisted development
RUN curl -fsSL https://claude.ai/install.sh | bash

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
