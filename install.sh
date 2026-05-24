#!/bin/bash
# install.sh - Install IOWarp Core using conda-build
# This script builds and installs IOWarp Core from source
# It will automatically install Miniconda if conda is not detected
#
# Usage:
#   ./install.sh                          # Build with default (release) preset
#   ./install.sh release                  # Build with release preset
#   ./install.sh release-fuse             # Build with FUSE adapter enabled
#   ./install.sh debug                    # Build with debug preset
#   ./install.sh conda                    # Build with conda-optimized preset
#   ./install.sh cuda                     # Build with CUDA preset
#   ./install.sh rocm                     # Build with ROCm preset
#   ./install.sh --only-deps [preset]     # Install ONLY iowarp-core's deps
#                                         # (build+host+run from the recipe),
#                                         # skip conda-build of iowarp-core itself.

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments: --only-deps flag + optional positional preset
ONLY_DEPS=false
PRESET=""
for arg in "$@"; do
    case "$arg" in
        --only-deps) ONLY_DEPS=true ;;
        --*)
            echo "Unknown flag: $arg" >&2
            echo "Usage: $0 [--only-deps] [preset]" >&2
            exit 1
            ;;
        *) PRESET="${PRESET:-$arg}" ;;
    esac
done
PRESET="${PRESET:-release}"

# Single source of truth for the build/target Python: the recipe's
# conda_build_config.yaml `python:` pin. We deliberately do NOT derive
# this from whatever `python3` is active — conda-forge's `python`
# metapackage moved its default to 3.14, and forcing `conda build
# --python=3.14` (overriding the recipe pin) crashes conda-build in
# get_upstream_pins/execute_download_actions with
# "IndexError: list index out of range". Pinning the env, the build,
# and the install to the recipe's value keeps all three consistent.
CBC_FILE="$SCRIPT_DIR/installers/conda/conda_build_config.yaml"
PYVER="$(grep -A2 '^python:' "$CBC_FILE" 2>/dev/null \
    | grep -oE '[0-9]+\.[0-9]+' | head -1)"
PYVER="${PYVER:-3.12}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================"
echo -e "IOWarp Core - Installation"
echo -e "======================================================================${NC}"
echo ""
echo -e "${BLUE}Preset: ${YELLOW}$PRESET${NC}"
echo ""

# Function to install Miniconda
install_miniconda() {
    echo -e "${YELLOW}Conda not detected. Installing Miniconda...${NC}"
    echo ""

    # Default Miniconda installation directory
    MINICONDA_DIR="$HOME/miniconda3"

    # Detect platform
    if [[ "$OSTYPE" == "linux"* ]]; then
        PLATFORM="Linux"
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$ARCH" == "aarch64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else
            echo -e "${RED}Error: Unsupported Linux architecture: $ARCH${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM="macOS"
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif [[ "$ARCH" == "arm64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            echo -e "${RED}Error: Unsupported macOS architecture: $ARCH${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: Unsupported operating system: $OSTYPE${NC}"
        exit 1
    fi

    echo -e "${BLUE}Detected platform: $PLATFORM ($ARCH)${NC}"
    echo -e "${BLUE}Installation directory: $MINICONDA_DIR${NC}"
    echo ""

    # Download Miniconda installer
    INSTALLER_SCRIPT="/tmp/miniconda_installer.sh"
    echo -e "${BLUE}Downloading Miniconda installer...${NC}"
    curl -L -o "$INSTALLER_SCRIPT" "$INSTALLER_URL"

    # Install Miniconda
    echo -e "${BLUE}Installing Miniconda...${NC}"
    bash "$INSTALLER_SCRIPT" -b -p "$MINICONDA_DIR"
    rm "$INSTALLER_SCRIPT"

    # Initialize conda for bash
    echo -e "${BLUE}Initializing conda for bash...${NC}"
    "$MINICONDA_DIR/bin/conda" init bash

    # Source conda to make it available in current shell
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"

    echo ""
    echo -e "${GREEN}Miniconda installed successfully!${NC}"
    echo ""
}

# Function to ensure conda is available
ensure_conda() {
    # Check if conda command is available
    if ! command -v conda &> /dev/null; then
        # Check if conda is installed but not in PATH
        if [ -f "$HOME/miniconda3/bin/conda" ]; then
            echo -e "${YELLOW}Conda found but not in PATH. Activating...${NC}"
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/bin/conda" ]; then
            echo -e "${YELLOW}Anaconda found but not in PATH. Activating...${NC}"
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        else
            # Install Miniconda
            install_miniconda
        fi
    else
        echo -e "${GREEN}Conda detected: $(conda --version)${NC}"
    fi
    echo ""
}

# Ensure conda is available
ensure_conda

# Accept Conda Terms of Service for Anaconda channels
echo -e "${BLUE}Accepting Conda Terms of Service...${NC}"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
echo -e "${GREEN}Conda ToS accepted${NC}"
echo ""

# Configure conda channels (add conda-forge if not already present)
echo -e "${BLUE}Configuring conda channels...${NC}"
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority flexible 2>/dev/null || true
echo -e "${GREEN}Conda channels configured${NC}"
echo ""

# Create and activate environment if not already in one
if [ -z "$CONDA_PREFIX" ]; then
    ENV_NAME="iowarp"
    echo -e "${BLUE}Creating conda environment: $ENV_NAME${NC}"

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Using existing environment.${NC}"
    else
        conda create -n "$ENV_NAME" -y "python=$PYVER"
        echo -e "${GREEN}Environment created (python=$PYVER)${NC}"
    fi

    echo -e "${BLUE}Activating environment: $ENV_NAME${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    echo ""
fi

echo -e "${GREEN}Active conda environment: $CONDA_PREFIX${NC}"
echo ""

# Check if conda-build is installed.
# conda-build registers its subcommand plugin against the *base* environment's
# conda CLI, so installing it into a non-base env leaves `conda build`
# unrecognized. Always install into base.
# conda-build is a *plugin* registered against the base env's conda CLI.
# When we invoke it from a non-base active env (e.g. `iowarp` created
# above), the active env's `conda` doesn't see the plugin. So always
# use base's conda binary explicitly. This was found in CI on a fresh
# Miniconda install: `conda install -n base conda-build` succeeds, but
# the subsequent `conda build` from iowarp's PATH errors out with
# "invalid choice: 'build'".
CONDA_BASE="$(conda info --base)"
CONDA_BIN="${CONDA_BASE}/bin/conda"

if ! "$CONDA_BIN" build --version &> /dev/null; then
    echo -e "${YELLOW}Installing conda-build into base environment...${NC}"
    "$CONDA_BIN" install -n base -y conda-build -c conda-forge
    echo ""
fi

if ! "$CONDA_BIN" build --version &> /dev/null; then
    echo -e "${RED}conda-build is still not available after install attempt.${NC}"
    echo -e "${YELLOW}Try manually:${NC} conda install -n base -y conda-build -c conda-forge"
    exit 1
fi

echo -e "${GREEN}conda-build detected: $("$CONDA_BIN" build --version)${NC}"
echo ""

# Initialize and update git submodules recursively (if in a git repository)
if [ -d ".git" ]; then
    echo -e "${BLUE}>>> Initializing git submodules...${NC}"
    git submodule update --init --recursive 2>/dev/null || {
        echo -e "${YELLOW}Some submodules failed to update (worktrees or optional repos). Continuing...${NC}"
    }
    echo ""
elif [ -d "context-transport-primitives" ] && [ "$(ls -A context-transport-primitives 2>/dev/null)" ]; then
    echo -e "${GREEN}>>> Submodules already present${NC}"
    echo ""
else
    echo -e "${RED}ERROR: Not a git repository and no submodule content found${NC}"
    echo "       Cannot proceed with build - missing dependencies"
    echo ""
    exit 1
fi

# Build the conda package
RECIPE_DIR="$SCRIPT_DIR/installers/conda"

OUTPUT_DIR="$SCRIPT_DIR/build/conda-output"
mkdir -p "$OUTPUT_DIR"

export IOWARP_PRESET="$PRESET"

# Extract PKG_VERSION from CMakeLists.txt's `project(iowarp-core VERSION X.Y.Z)`
# and export it so meta.yaml's `environ.get('PKG_VERSION', '1.0.0')` jinja
# resolves to the real version.  Without this, conda-build 26.x's jinja
# returns the literal string "None" for the unset env var (instead of the
# fallback default), and the package name ends up "iowarp-core-None-...",
# which then breaks at the _test_env solve step with
# `libmambapy.bindings.specs.ParseError: invalid version predicate in "None"`.
# Mirrors the same extraction step in .github/workflows/install-conda.yml.
if [ -z "${PKG_VERSION:-}" ]; then
    # Portable extraction (BSD sed on macOS lacks PCRE `\K`, so don't use `grep -oP`).
    PKG_VERSION="$(sed -nE 's/.*project\(iowarp-core VERSION ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' \
        "$SCRIPT_DIR/CMakeLists.txt" | head -1)"
    if [ -z "$PKG_VERSION" ]; then
        PKG_VERSION="1.0.0"
    fi
    export PKG_VERSION
fi
echo -e "${BLUE}Package version: $PKG_VERSION${NC}"

# Build/target Python comes from the recipe pin (computed above as
# $PYVER), NOT from the active interpreter. See the comment at the top.
echo -e "${BLUE}Target Python version: $PYVER (from conda_build_config.yaml)${NC}"

# --only-deps: render the recipe to resolve jinja, extract the union
# of build/host/run requirements, and conda-install them. Skip the
# conda-build of iowarp-core itself. Useful for dev iteration when
# you want to compile the tree manually (cmake --build) but let
# conda manage the C++/python dependencies.
if [ "$ONLY_DEPS" = true ]; then
    echo -e "${BLUE}>>> --only-deps: installing iowarp-core dependencies (no build)${NC}"
    echo ""
    # `mktemp --suffix=` is GNU-only; on BSD/macOS we get the name first
    # then rename to add the .yaml suffix (some tools key off the extension).
    RENDERED_BASE="$(mktemp -t iowarp-render.XXXXXX)"
    RENDERED="${RENDERED_BASE}.yaml"
    mv "$RENDERED_BASE" "$RENDERED"
    # -f writes the rendered YAML to FILE without the "Hash contents:"
    # / "meta.yaml:" prelude that `conda render` would otherwise emit
    # on stdout (which is not parseable as pure YAML).
    # No --python: the recipe's conda_build_config.yaml `python:` pin
    # drives the variant. Passing --python on the CLI overrides that
    # pin and trips conda-build's execute_download_actions IndexError.
    if ! "$CONDA_BIN" render "$RECIPE_DIR" \
            -c conda-forge \
            -f "$RENDERED" >/dev/null 2>&1; then
        echo -e "${RED}conda render failed${NC}"
        rm -f "$RENDERED"
        exit 1
    fi

    # Parse rendered meta.yaml with python+yaml (conda-build pulls in
    # pyyaml into base, so $CONDA_BIN's python has it).
    # NOTE: a heredoc inside `$(...)` triggers a parser bug in bash 3.2
    # (the version macOS ships at /bin/bash), so we write the script to
    # a temp file and run it instead. `conda render` resolves each dep
    # to "name version build" with the exact transitive build hash; for
    # a dev "install deps" flow we want the loosest practical pin so the
    # solver can fit them into the user's active env. Strip to just the
    # package name (drop version and build hash). Users who need strict
    # pinning should `conda build` the full package.
    DEPS_SCRIPT="$(mktemp -t iowarp-deps.XXXXXX)"
    cat >"$DEPS_SCRIPT" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
reqs = (data or {}).get("requirements", {}) or {}
seen_names = set()
ordered = []
for section in ("build", "host", "run"):
    for dep in reqs.get(section, []) or []:
        if not isinstance(dep, str):
            continue
        name = dep.split()[0]
        if name in seen_names:
            continue
        seen_names.add(name)
        ordered.append(name)
print(" ".join(ordered))
PY
    DEPS="$("$CONDA_BASE/bin/python" "$DEPS_SCRIPT" "$RENDERED")"
    rm -f "$DEPS_SCRIPT" "$RENDERED"

    if [ -z "$DEPS" ]; then
        echo -e "${RED}No dependencies extracted from recipe.${NC}"
        exit 1
    fi

    echo -e "${BLUE}Dependencies to install:${NC}"
    echo "  $DEPS"
    echo ""

    # shellcheck disable=SC2086  # intentional word-splitting of $DEPS
    if conda install -y -c conda-forge $DEPS; then
        echo ""
        echo -e "${GREEN}======================================================================"
        echo -e "Dependencies installed (iowarp-core itself was NOT built/installed)"
        echo -e "======================================================================${NC}"
        echo -e "${BLUE}Active env: ${CONDA_PREFIX}${NC}"
        exit 0
    else
        echo -e "${RED}conda install of dependencies failed.${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}>>> Building conda package with conda-build...${NC}"
echo -e "${YELLOW}This may take 10-30 minutes depending on your system${NC}"
echo ""

# No --python flag: the recipe's conda_build_config.yaml pins
# python (3.12). Passing --python on the CLI overrides that pin and
# crashes conda-build in execute_download_actions
# ("IndexError: list index out of range"), even when the value
# matches the pin. This mirrors the known-good install-conda.yml job.
if "$CONDA_BIN" build "$RECIPE_DIR" \
    --output-folder "$OUTPUT_DIR" \
    -c conda-forge \
    --no-anaconda-upload; then
    BUILD_SUCCESS=true
else
    BUILD_SUCCESS=false
fi

echo ""

if [ "$BUILD_SUCCESS" = true ]; then
    # Find the built package
    PACKAGE_PATH=$(find "$OUTPUT_DIR" -name "iowarp-core-*.tar.bz2" -o -name "iowarp-core-*.conda" | head -1)

    if [ -z "$PACKAGE_PATH" ]; then
        echo -e "${RED}Error: Could not find built package in $OUTPUT_DIR${NC}"
        exit 1
    fi

    echo -e "${GREEN}======================================================================"
    echo -e "Package built successfully!"
    echo -e "======================================================================${NC}"
    echo ""
    echo -e "${BLUE}Package location:${NC}"
    echo "  $PACKAGE_PATH"
    echo ""

    # Install using local output directory as a channel so that conda
    # resolves run dependencies (installing by file path skips dep resolution).
    echo -e "${BLUE}>>> Installing iowarp-core into current environment...${NC}"
    if conda install -y -c "$OUTPUT_DIR" -c conda-forge iowarp-core; then
        echo ""
        echo -e "${GREEN}======================================================================"
        echo -e "IOWarp Core installed successfully!"
        echo -e "======================================================================${NC}"
        echo ""
        echo -e "${BLUE}Installation prefix: $CONDA_PREFIX${NC}"
        echo ""
        echo -e "${BLUE}Verify installation:${NC}"
        echo "  conda list iowarp-core"
        echo ""
        echo -e "${YELLOW}NOTE: To use iowarp-core in a new terminal session, activate the environment:${NC}"
        echo "  conda activate $(basename $CONDA_PREFIX)"
        echo ""
    else
        echo ""
        echo -e "${RED}Installation failed.${NC}"
        echo ""
        echo -e "${YELLOW}You can try installing manually:${NC}"
        echo "  conda install \"$PACKAGE_PATH\""
        echo ""
        exit 1
    fi
else
    echo -e "${RED}======================================================================"
    echo -e "Build failed!"
    echo -e "======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo ""
    echo "1. Check that submodules are initialized:"
    echo "   git submodule update --init --recursive"
    echo ""
    echo "2. Verify conda-forge channel is configured:"
    echo "   conda config --show channels"
    echo ""
    echo "3. Try building with verbose output:"
    echo "   IOWARP_PRESET=$PRESET conda build $RECIPE_DIR -c conda-forge --no-anaconda-upload"
    echo ""
    exit 1
fi
