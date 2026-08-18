/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file grayscott_sim.h
 * @brief Gray-Scott reaction-diffusion simulation, GPU-resident, in the two
 * variants this tree actually uses.
 *
 * Ported for issue #693 from NeuroPress's Gray-Scott workload -- the data
 * generator behind its `grayscott_benchmark`. Ported, not linked: the same
 * discretization, parameters and RNG, re-expressed in this project's style,
 * with ctp_neuropress_grayscott_parity running upstream's own kernels beside
 * these and diffing both fields bit-for-bit.
 *
 * TWO MODELS, deliberately kept distinct rather than reconciled:
 *
 *   kNeuroPress3D -- 3D L^3 grid, 6-point Laplacian normalized by 6,
 *     Du=0.05 Dv=0.1 F=0.04 k=0.06075 dt=0.2, centre 12^3 cube at
 *     U=0.25/V=0.33. This is the ADIOS2 reference model
 *     (external/iowarp-gray-scott) that NeuroPress ported to float, and the
 *     distribution its selection network was evaluated against.
 *
 *   kClio2D -- 2D nx*ny grid, 4-point Laplacian with NO normalization,
 *     Du=0.16 Dv=0.08 F=0.055 k=0.062 dt=1.0, centre 6x6 square at V=1.
 *     This is what adapter/kvhdf5/test/e2e/gray_scott_gpu_test.cu and
 *     gray_scott_threeway_bench.cu have always run.
 *
 * They are different reaction-diffusion configurations in different numbers
 * of dimensions, not two spellings of one model, so neither can stand in for
 * the other as a workload: a compression ratio measured on one says nothing
 * about the other. Both live here so the existing 2D harnesses and the new
 * NeuroPress workload share one verified simulation instead of each carrying
 * a private copy of the kernel.
 *
 * NO CUDA in this header, deliberately. clio_cte/compressor headers do not
 * compile under nvcc, so a Clio consumer of this simulation has to be a plain
 * host TU; the kernels live in grayscott_sim.cu and reach it only through
 * this interface. Same TU-separation rule as core_client.h/.cc.
 */

#ifndef CLIO_CTE_COMPRESSOR_GENERATOR_GRAYSCOTT_GRAYSCOTT_SIM_H_
#define CLIO_CTE_COMPRESSOR_GENERATOR_GRAYSCOTT_GRAYSCOTT_SIM_H_

#include <cstddef>

namespace clio::cte::compressor::grayscott {

/** @brief Which discretization, initial condition and parameter set to run. */
enum class Model {
  kNeuroPress3D,  /**< 3D, 6-point/6, 12^3 cube IC. NeuroPress + ADIOS2. */
  kClio2D,        /**< 2D, raw 4-point, 6x6 square IC. kvhdf5 e2e harnesses. */
};

/**
 * @brief Simulation parameters.
 *
 * Build these with NeuroPress3D() or Clio2D() rather than by hand -- the
 * coefficients are not interchangeable between models, and a 3D coefficient
 * set on a 2D stencil is a silently different simulation, not an error.
 */
struct SimSettings {
  Model model = Model::kNeuroPress3D;

  int nx = 128;   /**< Cells in x. */
  int ny = 128;   /**< Cells in y. */
  int nz = 128;   /**< Cells in z; 1 for kClio2D. */

  float Du = 0.05f;    /**< Diffusion coefficient for U. */
  float Dv = 0.1f;     /**< Diffusion coefficient for V. */
  float F = 0.04f;     /**< Feed rate. */
  float k = 0.06075f;  /**< Kill rate. */
  float dt = 0.2f;     /**< Forward-Euler step size. */
  float noise = 0.0f;  /**< Noise amplitude on U. kNeuroPress3D only. */

  int steps = 10000;   /**< Steps to advance in a full run. */
  int seed = 42;       /**< Seed for the init and per-step noise streams. */

  /**
   * @brief NeuroPress's / the ADIOS2 reference's defaults on an L^3 grid.
   * The (F, k) pair selects the pattern regime, and the regime is what moves
   * compressibility: {0.04, 0.06075} spots, {0.035, 0.065} stripes,
   * {0.014, 0.045} chaos (high entropy), {0.04, 0.065} sparse spots.
   */
  static SimSettings NeuroPress3D(int L);

  /** @brief The parameters this tree's existing 2D harnesses have always
   *  used, on an nx * ny grid. */
  static SimSettings Clio2D(int nx, int ny);
};

/**
 * @brief One simulation instance owning four device buffers (U, V and their
 * double-buffered scratch halves).
 *
 * The fields never leave the GPU: DeviceV() hands out a raw device pointer
 * intended to go straight into Clio's compressor or the HDF5 VOL. After a
 * run, DeviceU() is dead scratch and is what a benchmark can read decompressed
 * data back into, leaving V intact as the verification reference -- the
 * arrangement NeuroPress's harness uses. (Its 2D predecessors here snapshot U
 * and keep V; either field works, they are just different data.)
 */
class Simulation {
 public:
  explicit Simulation(const SimSettings &settings);
  ~Simulation();

  Simulation(const Simulation &) = delete;
  Simulation &operator=(const Simulation &) = delete;

  /** @brief True when all four device buffers were allocated. */
  bool Valid() const;

  /** @brief Apply the model's initial condition and reset the step counter.
   *  Built by a kernel, so nothing is staged through host memory. */
  bool Init();

  /** @brief Advance `steps` forward-Euler steps. Buffers are swapped, not
   *  copied, so this costs no extra bandwidth. */
  bool Run(int steps);

  /** @brief Current U field (scratch once a run has finished). */
  float *DeviceU() const;
  /** @brief Current V field. */
  float *DeviceV() const;

  /** @brief nx * ny * nz. */
  size_t NumElems() const;
  /** @brief NumElems() * sizeof(float). */
  size_t NumBytes() const;
  /** @brief Steps advanced since the last Init(). */
  int Step() const;

 private:
  struct Impl;
  Impl *impl_;
};

}  // namespace clio::cte::compressor::grayscott

#endif  // CLIO_CTE_COMPRESSOR_GENERATOR_GRAYSCOTT_GRAYSCOTT_SIM_H_
