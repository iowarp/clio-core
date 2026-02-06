# IOWarp NVIDIA GPU Development Container
# Inherits from deps-nvidia which has all build dependencies including CUDA
#
# This devcontainer adds development-specific tools like Claude Code.
#
FROM iowarp/deps-nvidia:latest

# The deps-nvidia image already includes (inherited from deps-cpu):
# - All conda dependencies (boost, hdf5, yaml-cpp, zeromq, cereal, etc.)
# - Compression libraries (zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2)
# - Lossy compression (ZFP, SZ3, FPZIP, LibPressio)
# - ADIOS2 with HDF5 and ZeroMQ support
# - Docker-in-Docker support
# - Jarvis (IOWarp runtime deployment tool)
# - MPI (OpenMPI)
# - libaio (for bdev ChiMod)
#
# Plus NVIDIA-specific:
# - CUDA 12.6 toolkit and runtime libraries
# - NVIDIA Container Toolkit
# - GPU environment variables (CUDA_HOME, NVIDIA_VISIBLE_DEVICES, etc.)

WORKDIR /workspace

# Install Claude Code for AI-assisted development
RUN curl -fsSL https://claude.ai/install.sh | bash

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
