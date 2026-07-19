# IOWarp ML Inference Development Container
# Inherits from deps-ml which has llama.cpp, vLLM, SGLang, and PyTorch pre-installed.
#
FROM iowarp/deps-ml:latest
USER root

# FUSE 3 runtime (fusermount3) + dev headers, needed to build the libfuse
# adapter (clio_cte_fuse) and run its mount integration tests. Installed here
# so a devcontainer rebuild picks it up immediately; it is also in the base
# image (docker/iowarp-base.Dockerfile) as the canonical default. The
# container runs --privileged, so /dev/fuse + SYS_ADMIN are available.
#
# deps-ml inherits deps-nvidia, which force-installs the HIP-NVCC `hip-dev`
# headers with `dpkg -i --force-depends` and deliberately never installs the
# AMD-flavored hip-runtime-amd / rocm-llvm / hsa-rocr-dev / hipcc, leaving apt
# in a broken-deps state (deps-nvidia.Dockerfile:196-199). apt refuses to run
# ANY install while those phantom deps are unmet, which aborts this layer with
# exit 100 (issue #670). Purge the unsatisfiable HIP metapackages and repair
# the dependency tree first so the fuse/boost install can proceed.
RUN set -eux; \
    dpkg -P --force-all hipcc hip-dev hip-runtime-amd rocm-llvm hsa-rocr-dev rocm-device-libs 2>/dev/null || true; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get --fix-broken install -y || true; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      fuse3 libfuse3-dev \
      libboost-context-dev libboost-dev; \
    rm -rf /var/lib/apt/lists/*

# Boost.Context stackful coroutine backend available + on by default (issue #620).
ENV CLIO_CORE_ENABLE_BOOST_COROUTINES=ON

# Remap iowarp user UID/GID to match the host user.
ARG HOST_UID=1000
ARG HOST_GID=1000

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
