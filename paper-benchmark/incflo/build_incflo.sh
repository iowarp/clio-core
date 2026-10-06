#!/bin/bash
# incflo (AMReX-Fluids) 3D, CUDA sm_80, no MPI, EB on, double precision
# AMReX-Hydro from ~/src/amrex-hydro-incflo. (the plotfile converter writes float32). Superbuild with AMReX from ~/src/amrex-incflo.
set -ex
cd ~/src/incflo
cmake -S . -B build-clio -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAMREX_HOME=$HOME/src/amrex-incflo -DAMREX_HYDRO_HOME=$HOME/src/amrex-hydro-incflo -DINCFLO_DIM=3 \
  -DINCFLO_MPI=NO -DINCFLO_OMP=NO -DINCFLO_CUDA=YES -DINCFLO_EB=YES \
  -DAMReX_CUDA_ARCH=8.0 -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-clio -j48
find build-clio -maxdepth 2 -type f -executable -name 'incflo*' -ls
echo INCFLO_BUILD_OK
