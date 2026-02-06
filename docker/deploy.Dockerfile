# Deployment Dockerfile for IOWarp Core
# DEPRECATED: Use deploy-cpu.Dockerfile instead.
# This file is kept for backward compatibility.
#
# Inherits from the build container
FROM iowarp/build-cpu:latest

# Create empty runtime configuration file (if not inherited from base)
RUN sudo mkdir -p /etc/iowarp && \
    sudo touch /etc/iowarp/wrp_conf.yaml

# Set runtime configuration environment variable
ENV WRP_RUNTIME_CONF=/etc/iowarp/wrp_conf.yaml
