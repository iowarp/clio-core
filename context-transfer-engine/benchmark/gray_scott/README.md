# IOWarp Gray-Scott Simulation

A production-ready implementation of the Gray-Scott reaction-diffusion simulation using ADIOS2 for I/O. This version is based on the ADIOS tutorial but follows modern CMake practices with proper RPATH configuration.

## Overview

The Gray-Scott simulation models a reaction-diffusion system that produces interesting pattern formation. This implementation uses ADIOS2 for scalable parallel I/O and includes analysis and visualization tools.

## Features

- **Parallel MPI simulation** with ADIOS2 I/O
- **RPATH-based linking** - no need for `LD_LIBRARY_PATH` configuration
- **Standard installation layout** - binaries in `bin/`, configs in `share/`
- **Analysis tools**: PDF calculation, isosurface extraction, blob detection, curvature computation
- **Optional VTK visualization** support

## Dependencies

- **CMake** >= 3.20
- **MPI** (OpenMPI or MPICH)
- **ADIOS2** >= 2.7
- **VTK** (optional, for visualization tools)

## Building

### Standard Build

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/path/to/install \
      ..
make -j$(nproc)
```

### Build with VTK Support

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/path/to/install \
      -DENABLE_VTK=ON \
      -DVTK_ROOT=/path/to/vtk \
      ..
make -j$(nproc)
```

### Build Options

- **`ENABLE_RPATH`** (default: ON) - Use RPATH for runtime library search
- **`USE_TIMERS`** (default: OFF) - Enable profiling timers
- **`ENABLE_VTK`** (default: OFF) - Build VTK visualization apps

### Installation

```bash
make install
```

This installs:
- **Binaries**: `${CMAKE_INSTALL_PREFIX}/bin/`
- **Config files**: `${CMAKE_INSTALL_PREFIX}/share/iowarp-gray-scott/`
- **Scripts**: `${CMAKE_INSTALL_PREFIX}/share/iowarp-gray-scott/scripts/`

## Executables

### Core Programs

- **`gray-scott`** - Main simulation executable
- **`pdf_calc`** - PDF (Probability Density Function) calculation

### VTK Analysis Tools (if built with ENABLE_VTK=ON)

- **`isosurface`** - Extract isosurfaces from simulation data
- **`find_blobs`** - Detect and analyze blob structures
- **`compute_curvature`** - Compute curvature of surfaces
- **`render_isosurface`** - Render isosurfaces to images

## Running the Simulation

### Basic MPI Run

**If ADIOS2 was built with conda dependencies (recommended):**
```bash
./scripts/run-with-conda.sh mpirun -n 4 ./install/bin/gray-scott settings.json
```

**If ADIOS2 was built with system libraries:**
```bash
mpirun -n 4 ./install/bin/gray-scott settings.json
```

Or from the build directory:
```bash
mpirun -n 4 ./build/bin/gray-scott settings.json
```

> **Note:** The wrapper script is needed when ADIOS2 depends on conda libraries (libcurl, OpenSSL) that conflict with system versions. See the [Troubleshooting](#openssl-version-mismatch-error) section for details.

### Simulation Parameters

Create a `settings.json` file with simulation parameters:

```json
{
  "L": 128,
  "Du": 0.2,
  "Dv": 0.1,
  "F": 0.02,
  "k": 0.048,
  "dt": 1.0,
  "plotgap": 10,
  "steps": 10000,
  "noise": 0.01,
  "output": "gs.bp",
  "checkpoint": false,
  "checkpoint_freq": 2000,
  "checkpoint_output": "ckpt.bp"
}
```

**Parameters:**
- **L**: Domain size (L x L x L)
- **Du, Dv**: Diffusion coefficients for U and V
- **F**: Feed rate
- **k**: Kill rate
- **dt**: Time step size
- **plotgap**: Output frequency (every N steps)
- **steps**: Total simulation steps
- **noise**: Initial noise level
- **output**: Output file path
- **checkpoint**: Enable checkpointing
- **checkpoint_freq**: Checkpoint frequency
- **checkpoint_output**: Checkpoint file path

### ADIOS2 Configuration

The `adios2.xml` file configures ADIOS2 I/O engines:

- **SimulationOutput**: For simulation output (default: FileStream)
- **PDFAnalysisOutput**: For PDF calculation output
- **IsosurfaceOutput**: For isosurface data
- **SimulationCheckpoint**: For checkpointing

Edit `adios2.xml` to change engine types, compression, or streaming parameters.

## Analysis Workflow

### 1. Run Simulation

```bash
mpirun -n 4 ./bin/gray-scott settings.json
```

### 2. Calculate PDF

```bash
mpirun -n 2 ./bin/pdf_calc gs.bp pdf.bp 100
```

### 3. Extract Isosurfaces (if VTK enabled)

```bash
./bin/isosurface gs.bp isosurface.bp 0.5
```

### 4. Find Blobs (if VTK enabled)

```bash
./bin/find_blobs isosurface.bp blobs.csv
```

## RPATH Configuration

This project uses **RPATH** for runtime library linking instead of requiring `LD_LIBRARY_PATH`:

- **Build RPATH**: Allows running executables from the build directory
- **Install RPATH**: Set to `${CMAKE_INSTALL_PREFIX}/lib` for installed binaries
- **Automatic dependency resolution**: All dependency paths are encoded in the binary

To disable RPATH (and use `LD_LIBRARY_PATH` instead):

```bash
cmake -DENABLE_RPATH=OFF ..
```

## Cleanup

Remove output files and build artifacts:

```bash
# Clean data files only
./scripts/cleanup.sh data

# Clean build artifacts only
./scripts/cleanup.sh code

# Clean everything
./scripts/cleanup.sh all
```

## Directory Structure

```
iowarp-gray-scott/
├── CMakeLists.txt          # Main build configuration
├── README.md               # This file
├── adios2.xml              # ADIOS2 engine configuration
├── simulation/             # Simulation source code
│   ├── main.cpp
│   ├── gray-scott.cpp
│   ├── settings.cpp
│   └── writer.cpp
├── analysis/               # Analysis tools
│   ├── pdf_calc.cpp
│   ├── isosurface.cpp
│   ├── find_blobs.cpp
│   └── curvature.cpp
├── plot/                   # Visualization tools
│   └── render_isosurface.cpp
└── scripts/                # Utility scripts
    └── cleanup.sh
```

## License

Based on the ADIOS2 Gray-Scott tutorial. See original tutorial for license information.

## References

- Original tutorial: https://github.com/ornladios/ADIOS2-Examples
- Gray-Scott model: https://doi.org/10.1126/science.261.5118.189
- ADIOS2 documentation: https://adios2.readthedocs.io/

## Troubleshooting

### OpenSSL Version Mismatch Error

**Symptom:**
```
./gray-scott: /usr/lib/x86_64-linux-gnu/libssl.so.3: version `OPENSSL_3.2.0' not found
(required by /home/iowarp/miniconda3/lib/./libcurl.so.4)
```

**Cause:** ADIOS2 depends on conda's libcurl, which requires OpenSSL 3.2.0, but your system has OpenSSL 3.0.x. While gray-scott has RPATH configured, RPATH only controls direct dependencies. When ADIOS2's transitive dependency (libcurl) needs libssl, and LD_LIBRARY_PATH contains system paths, the system OpenSSL is found first.

**Solution 1 (Recommended):** Use the wrapper script:
```bash
./scripts/run-with-conda.sh mpirun -n 4 ./install/bin/gray-scott settings.json
```

**Solution 2:** Set LD_LIBRARY_PATH manually before running:
```bash
export LD_LIBRARY_PATH="/home/iowarp/miniconda3/lib:$LD_LIBRARY_PATH"
mpirun -n 4 ./install/bin/gray-scott settings.json
```

**Solution 3:** Build ADIOS2 against system libraries instead of conda libraries.

### Library Not Found Errors

If you see library not found errors despite RPATH being enabled, ensure all dependencies were found during CMake configuration:

```bash
cmake .. 2>&1 | grep "Found"
```

### MPI Errors

Ensure you're using the same MPI implementation for building and running:

```bash
cmake -DMPI_C_COMPILER=$(which mpicc) \
      -DMPI_CXX_COMPILER=$(which mpicxx) \
      ..
```

### ADIOS2 Not Found

If ADIOS2 is installed in a non-standard location:

```bash
cmake -DADIOS2_ROOT=/path/to/adios2 ..
```

## Contributing

This is a production-ready implementation following modern CMake best practices. Contributions welcome!
