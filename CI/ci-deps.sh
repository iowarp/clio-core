#!/bin/bash
# CI/ci-deps.sh - Install IOWarp Core using conda-build
# This script builds and installs IOWarp Core from source
# It will automatically install Miniconda if conda is not detected
#
# Usage:
#   ./CI/ci-deps.sh                          # Build with default (release) preset
#   ./CI/ci-deps.sh release                  # Build with release preset
#   ./CI/ci-deps.sh release-fuse             # Build with FUSE adapter enabled
#   ./CI/ci-deps.sh debug                    # Build with debug preset
#   ./CI/ci-deps.sh conda                    # Build with conda-optimized preset
#   ./CI/ci-deps.sh cuda                     # Build with CUDA preset
#   ./CI/ci-deps.sh rocm                     # Build with ROCm preset
#   ./CI/ci-deps.sh --only-deps [preset]     # Install ONLY iowarp-core's deps
#                                         # (build+host+run from the recipe),
#                                         # skip conda-build of iowarp-core itself.

set -e  # Exit on error

# Get the repo root. This script lives in CI/, so the root is its parent dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

# Pinned Miniconda release with a Python 3.12 base. conda-build crashes on a
# Python >= 3.14 interpreter (execute_download_actions IndexError, same bug as
# the PYVER note above), and "Miniconda3-latest" now ships a 3.14 base — so
# fresh installs must pin, and a detected system conda must be checked.
MINICONDA_RELEASE="py312_26.5.3-2"
# Anaconda stopped shipping macOS-Intel installers after this release
# (Miniconda3-py312_26.5.3-2-MacOSX-x86_64.sh is a 404), so mac-Intel pins
# the last release that has one.
MINICONDA_RELEASE_MACOS_INTEL="py312_25.7.0-2"

# Function to install Miniconda
install_miniconda() {
    echo -e "${YELLOW}Installing pinned Miniconda ($MINICONDA_RELEASE)...${NC}"
    echo ""

    # Default Miniconda installation directory
    MINICONDA_DIR="$HOME/miniconda3"

    # Detect platform
    if [[ "$OSTYPE" == "linux"* ]]; then
        PLATFORM="Linux"
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_RELEASE-Linux-x86_64.sh"
        elif [[ "$ARCH" == "aarch64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_RELEASE-Linux-aarch64.sh"
        else
            echo -e "${RED}Error: Unsupported Linux architecture: $ARCH${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM="macOS"
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_RELEASE_MACOS_INTEL-MacOSX-x86_64.sh"
        elif [[ "$ARCH" == "arm64" ]]; then
            INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_RELEASE-MacOSX-arm64.sh"
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

    # Download Miniconda installer. -f: fail on HTTP errors instead of saving
    # an HTML error page that bash then chokes on (`syntax error near
    # unexpected token 'newline'`).
    INSTALLER_SCRIPT="/tmp/miniconda_installer.sh"
    echo -e "${BLUE}Downloading Miniconda installer...${NC}"
    if ! curl -fL -o "$INSTALLER_SCRIPT" "$INSTALLER_URL"; then
        echo -e "${RED}Error: failed to download $INSTALLER_URL${NC}"
        exit 1
    fi

    # Install Miniconda (-u: proceed even if the directory already exists,
    # e.g. when replacing an unusable py3.14-base install)
    echo -e "${BLUE}Installing Miniconda...${NC}"
    bash "$INSTALLER_SCRIPT" -b -u -p "$MINICONDA_DIR"
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

# conda-build runs in the BASE environment, and is broken on a Python >= 3.14
# base regardless of the recipe's target-python pin. Returns success when the
# detected conda's base interpreter is usable.
conda_base_usable() {
    local base
    base="$(conda info --base 2>/dev/null)" || return 1
    [ -x "$base/bin/python" ] || return 1
    "$base/bin/python" -c 'import sys; sys.exit(0 if sys.version_info < (3, 14) else 1)'
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
    # A detected conda with a py3.14 base (e.g. the GitHub runner image's
    # preinstalled /usr/share/miniconda since 2026-07-30) crashes conda-build;
    # replace it with the pinned py312 Miniconda and use that instead.
    if command -v conda &> /dev/null && ! conda_base_usable; then
        echo -e "${YELLOW}Detected conda has a Python >= 3.14 base, which breaks conda-build.${NC}"
        install_miniconda
    fi
    echo ""
}

# Ensure conda is available
ensure_conda

# Configure conda to use conda-forge ONLY, and never the Anaconda `defaults`
# channels (pkgs/main, pkgs/r).
#
# Those channels sit behind repo.anaconda.com, which now returns HTTP 403
# Forbidden for CI/commercial use unless the org has an Anaconda license:
#
#   CondaHTTPError: HTTP 403 Forbidden for url
#     <https://repo.anaconda.com/pkgs/main/noarch/repodata.json>
#   - The channel requires authentication.
#
# Previously this script did `conda tos accept ...` for pkgs/main and pkgs/r
# and then `conda config --add channels conda-forge`, which only PREPENDED
# conda-forge while leaving `defaults` in the channel list. conda still
# fetched pkgs/main repodata and intermittently 403'd — a flake that fails the
# dependency-install step before any build begins, and the create-retry loop
# below could not help because every retry hit the same forbidden channel.
# Accepting a ToS also does nothing when the runner IP is refused outright.
#
# `--remove-key channels` clears whatever the base install seeded (which
# includes defaults), then we add only conda-forge and pin strict priority so
# conda never falls back to defaults for a package conda-forge lacks.
echo -e "${BLUE}Configuring conda channels (conda-forge only, no defaults)...${NC}"
conda config --remove-key channels 2>/dev/null || true
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority strict 2>/dev/null || true
# Belt and braces: some conda versions still consult the implicit `defaults`
# unless it is explicitly disabled.
conda config --set default_channels "[]" 2>/dev/null || true
echo -e "${GREEN}Conda channels configured (conda-forge only)${NC}"
echo ""

# Create and activate environment if not already in one
if [ -z "$CONDA_PREFIX" ]; then
    ENV_NAME="iowarp"
    echo -e "${BLUE}Creating conda environment: $ENV_NAME${NC}"

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Using existing environment.${NC}"
    else
        # Retry up to 3 times: conda-forge CDN occasionally returns 403/502.
        _created=0
        for _attempt in 1 2 3; do
            if conda create -n "$ENV_NAME" -y -c conda-forge --override-channels "python=$PYVER"; then
                _created=1
                break
            fi
            echo -e "${YELLOW}conda create failed (attempt $_attempt/3), retrying in $((10 * _attempt))s...${NC}"
            conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
            sleep $((10 * _attempt))
        done
        [ "$_created" -eq 1 ] || { echo -e "${RED}conda create failed after 3 attempts.${NC}"; exit 1; }
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
    # Retry: conda's package-cache sqlite can transiently report
    # "database is locked" and conda-forge occasionally 502s during the
    # solve; a short back-off clears both (observed as an arm64 CI flake).
    for _attempt in 1 2 3; do
        "$CONDA_BIN" install -n base -y conda-build -c conda-forge && break
        echo -e "${YELLOW}conda-build install failed (attempt $_attempt/3), retrying in $((10 * _attempt))s...${NC}"
        sleep $((10 * _attempt))
    done
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
    # Retry: render performs a solve and can hit the same transient
    # "database is locked" / conda-forge 502 flakes as the installs above.
    _render_ok=0
    for _attempt in 1 2 3; do
        if "$CONDA_BIN" render "$RECIPE_DIR" \
                -c conda-forge \
                -f "$RENDERED" >/dev/null 2>&1; then
            _render_ok=1
            break
        fi
        echo -e "${YELLOW}conda render failed (attempt $_attempt/3), retrying in $((10 * _attempt))s...${NC}"
        sleep $((10 * _attempt))
    done
    if [ "$_render_ok" -ne 1 ]; then
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
    # Retry up to 3 times: conda-forge occasionally returns 502/503 on
    # transient CDN hiccups. A short back-off is enough to recover.
    _conda_ok=0
    for _attempt in 1 2 3; do
        if conda install -y -c conda-forge $DEPS; then
            _conda_ok=1
            break
        fi
        echo -e "${YELLOW}conda install failed (attempt $_attempt/3), retrying in $((10 * _attempt))s...${NC}"
        sleep $((10 * _attempt))
    done
    if [ "$_conda_ok" -eq 1 ]; then
        echo ""
        echo -e "${GREEN}======================================================================"
        echo -e "Dependencies installed (iowarp-core itself was NOT built/installed)"
        echo -e "======================================================================${NC}"
        echo -e "${BLUE}Active env: ${CONDA_PREFIX}${NC}"
        exit 0
    else
        echo -e "${RED}conda install of dependencies failed after 3 attempts.${NC}"
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
    _conda_ok=0
    for _attempt in 1 2 3; do
        if conda install -y -c "$OUTPUT_DIR" -c conda-forge iowarp-core; then
            _conda_ok=1
            break
        fi
        echo -e "${YELLOW}conda install failed (attempt $_attempt/3), retrying in $((10 * _attempt))s...${NC}"
        sleep $((10 * _attempt))
    done
    if [ "$_conda_ok" -eq 1 ]; then
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
        echo -e "${RED}Installation failed after 3 attempts.${NC}"
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
