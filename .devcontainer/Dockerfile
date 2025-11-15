FROM iowarp/iowarp-deps:latest

# Switch to root for package installation
USER root

# Install Docker CLI and Docker-in-Docker dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    iptables \
    supervisor \
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

# Switch to iowarp user for venv creation
USER iowarp
WORKDIR /workspace

# Create Python virtual environment in user's home directory
RUN python3 -m venv /home/iowarp/venv \
    && /home/iowarp/venv/bin/pip install --upgrade pip setuptools wheel \
    && /home/iowarp/venv/bin/pip install pyyaml nanobind

# Add venv activation to bashrc
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# Activate Python virtual environment' >> /home/iowarp/.bashrc \
    && echo 'if [ -f /home/iowarp/venv/bin/activate ]; then' >> /home/iowarp/.bashrc \
    && echo '    source /home/iowarp/venv/bin/activate' >> /home/iowarp/.bashrc \
    && echo 'fi' >> /home/iowarp/.bashrc

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
