FROM iowarp/iowarp-deps:latest

COPY . /workspace

WORKDIR /workspace

RUN cd build && sudo make -j$(nproc) install
RUN sudo rm -rf /workspace

# Add iowarp-runtime to Spack configuration
RUN echo "  iowarp-runtime:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: iowarp-runtime@main" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr/local" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml
