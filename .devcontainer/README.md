# IOWarp Core DevContainer

This directory contains the DevContainer configuration for IOWarp Core development.

## Features

### Python Virtual Environment

The devcontainer automatically creates and activates a Python virtual environment at `/home/iowarp/venv`.

**Pre-installed packages:**
- `pip`, `setuptools`, `wheel` (latest versions)
- `pyyaml` (for configuration file parsing)
- `nanobind` (for Python bindings)

**Automatic activation:**
The virtual environment is automatically activated when:
- Opening a new terminal in VSCode
- Running the postCreateCommand
- Starting a new bash session (via `.bashrc`)

**Manual activation:**
If you need to manually activate the venv:
```bash
source /home/iowarp/venv/bin/activate
```

**Installing additional packages:**
```bash
# Activate venv (if not already active)
source /home/iowarp/venv/bin/activate

# Install packages
pip install <package-name>
```

**Building Python bindings:**
To enable Python bindings for IOWarp components:
```bash
# Configure with Python support
cmake --preset=debug -DWRP_CORE_ENABLE_PYTHON=ON

# Build
make -j8

# Install to venv (if WRP_CORE_INSTALL_WRAPPER_TO_VENV is ON)
make install
```

The Python modules will be installed to the virtual environment's site-packages directory and will be importable from Python scripts.

### VSCode Extensions

The following extensions are automatically installed:
- **C/C++ Development:**
  - C/C++ (ms-vscode.cpptools)
  - CMake Tools (ms-vscode.cmake-tools)
  - CMake (twxs.cmake)
  - C/C++ Debug (KylinIdeTeam.cppdebug)
  - clangd (llvm-vs-code-extensions.vscode-clangd)

- **Python Development:**
  - Python (ms-python.python)
  - Pylance (ms-python.vscode-pylance)

- **Container & DevOps:**
  - Docker (ms-azuretools.vscode-docker)

- **AI Assistant:**
  - Claude Code (anthropic.claude-code)

### Docker-in-Docker

Docker is available inside the container with the host's Docker socket mounted, allowing you to:
- Build and run containers from inside the devcontainer
- Use docker-compose
- Interact with the host's Docker daemon

### Environment Variables

- `IOWARP_CORE_ROOT`: Set to the workspace folder
- `VIRTUAL_ENV`: Points to `/home/iowarp/venv`
- `PATH`: Includes the venv's bin directory

## Python Configuration

The VSCode Python extension is configured with:
- **Default interpreter:** `/home/iowarp/venv/bin/python`
- **Auto-activate:** Terminal activation is enabled
- **Linting:** flake8 enabled (pylint disabled)
- **Formatting:** black (if installed)

## Rebuilding the Container

If you modify the Dockerfile, rebuild the container:

1. In VSCode: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"
2. Or manually: `docker-compose build` (if using docker-compose setup)

## Troubleshooting

**Virtual environment not activated:**
```bash
source /home/iowarp/venv/bin/activate
```

**Python packages not found:**
Ensure you're in the virtual environment and the package is installed:
```bash
which python  # Should show /home/iowarp/venv/bin/python
pip list      # Show installed packages
```

**Docker permission issues:**
The container should automatically fix Docker socket permissions. If issues persist:
```bash
sudo chmod 666 /var/run/docker.sock
```
