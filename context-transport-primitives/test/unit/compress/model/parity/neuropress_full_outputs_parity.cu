/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */
/**
@file neuropress_full_outputs_parity.cu
@brief Differential test for model outputs 4-7 (rmse, max_error, mae, ssim).

The network has always produced eight outputs on both sides. Upstream inverts
all eight (nn_gpu.cu:207-216); Clio inverted only the first four, because those
are the only ones selection reads -- upstream says so itself with
NN_INFER_OUTPUTS = 4 (nn_weights.h:15). The remaining four are data-quality
predictions, reported by upstream through NNInferenceOutput and previously
absent from Clio entirely.

They are now ported (NeuroPressGpuInferBatchFull). This checks the port against
upstream's own numbers rather than against the formulas as re-read from its
source, because "I transcribed the expression correctly" and "the two produce
the same float" are different claims -- and output 7 is exactly where that
distinction bites: ssim is stored as -log(1-ssim) and inverts with
1 - exp(-max(0,x)), NOT with the expm1f that outputs 4-6 use. A port that
pattern-matched head 7 to its neighbours would look right and be wrong.

Outputs 0-3 are compared too, as a control: they were already known to agree,
so a failure there means the harness is feeding the two sides different inputs
and nothing else in the file can be believed.
*/

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

#if CLIO_HAVE_GRAYSCOTT
#include "grayscott_kernels.cuh"
namespace gs = clio::cte::compressor::grayscott;
#endif

#include "api/internal.hpp"
#include "nn/nn_weights.h"
#include "stats/auto_stats_gpu.h"

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

namespace gpucompress {
/* Upstream's byte->statistics pipeline, its own kernels. Declared here the way
   neuropress_dataset_parity.cu declares it. */
int runStatsOnlyPipeline(const void *d_input, size_t input_size,
                         cudaStream_t stream, double *out_entropy,
                         double *out_mad, double *out_deriv);
void freeStatsWorkspace();
}  // namespace gpucompress

namespace {

long g_checks = 0;
bool g_dumped = false;
/* How many comparisons were against a value that is not identically zero.
   Outputs 4-6 are clamped at 0, so a run in which they are always 0 would
   report "matches upstream" while never exercising the transform itself. */
long g_nonzero[8] = {0};
const char *kOutName[8] = {"comp_time", "decomp_time", "ratio", "psnr",
                           "rmse",      "max_error",   "mae",   "ssim"};
int g_failures = 0;

/* Relative tolerance. Both sides run the same weights through the same
   arithmetic, so the only expected difference is FMA contraction and the order
   the compiler picks -- parts per million, not parts per thousand. */
constexpr double kRelTol = 1e-5;

/* algo index 0-7 -> CTE base_id, the inverse of NeuroPressAlgoIdForBaseId.
   The predictor addresses codecs by base_id and upstream by index; getting
   this backwards would silently compare two different algorithms. */
const int kBaseIdForAlgo[8] = {13, 14, 17, 16, 15, 18, 23, 24};

long g_zero_pairs = 0;

/* Compare one output.
 *
 * A comparison where upstream's value is 0 is NOT counted as a substantive
 * check: "0 == 0" says nothing about the transform, and a tally padded with
 * them overstates the coverage. The headline number below is non-zero
 * comparisons only.
 *
 * Those cases are still VERIFIED, in a separate tally, and that is deliberate:
 * dropping them entirely would have hidden the bug this harness found on its
 * first run, where Clio produced rmse = -1.7e-4 exactly where upstream
 * produced 0 because the port had omitted upstream's fmaxf(0, ...) clamp.
 * The failure mode is "Clio is non-zero where upstream is zero", so the zero
 * cases are precisely where that bug lives -- they just should not be counted
 * as evidence that the arithmetic agrees. */
void CheckClose(double clio, double up, const char *what, int action) {
  if (up == 0.0) {
    ++g_zero_pairs;
    if (clio != 0.0) {
      ++g_failures;
      std::printf("  FAIL action %2d %-12s upstream=0 but clio=%.9g\n", action,
                  what, clio);
    }
    return;
  }
  ++g_checks;
  const double denom = std::fabs(up);
  const double rel = std::fabs(clio - up) / denom;
  if (!(rel <= kRelTol) || !std::isfinite(clio)) {
    ++g_failures;
    std::printf("  FAIL action %2d %-12s clio=%.9g upstream=%.9g rel=%.3e\n",
                action, what, clio, up, rel);
  }
}

/* Deterministic float32 chunks spanning regimes the model sees in practice:
   a smooth ramp, high-entropy noise, a sparse signal, and a constant buffer
   (the degenerate case where MAD and the derivative are exactly zero). */
enum class Regime { kSmooth, kRandom, kSparse, kConstant, kGrayScott };

const char *RegimeName(Regime r) {
  switch (r) {
    case Regime::kSmooth: return "smooth";
    case Regime::kRandom: return "random";
    case Regime::kSparse: return "sparse";
    case Regime::kConstant: return "constant";
    case Regime::kGrayScott: return "grayscott";
  }
  return "?";
}

std::vector<float> MakeChunk(size_t n_elems, Regime r) {
  std::vector<float> v(n_elems);
  unsigned seed = 987654321u;
  for (size_t i = 0; i < n_elems; ++i) {
    seed = seed * 1103515245u + 12345u;
    switch (r) {
      case Regime::kSmooth:   v[i] = static_cast<float>(i % 4096) * 0.001f; break;
      case Regime::kRandom:   v[i] = static_cast<float>(seed >> 8) * 1e-7f;  break;
      case Regime::kSparse:   v[i] = ((seed >> 16) % 32 == 0) ? 1.0f : 0.0f; break;
      case Regime::kConstant: v[i] = 0.5f;                                   break;
    }
  }
  return v;
}

#if CLIO_HAVE_GRAYSCOTT
/* Run the reaction-diffusion simulation and leave its V field on the device.
   Real output from a real solver: spatially correlated, with sharp fronts
   between smooth regions -- the structure the synthetic regimes lack, and the
   structure entropy and the second derivative are most sensitive to. */
bool MakeGrayScottField(float *d_out, size_t n_elems) {
  const int L = 128;                       /* 128^3 = 2,097,152 cells */
  const size_t cells = static_cast<size_t>(L) * L * L;
  if (n_elems > cells) return false;

  float *u = nullptr, *v = nullptr, *u2 = nullptr, *v2 = nullptr;
  bool ok = cudaMalloc(&u, cells * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&v, cells * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&u2, cells * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&v2, cells * sizeof(float)) == cudaSuccess;
  if (ok) {
    const int grid = static_cast<int>((cells + gs::kBlockSize - 1) / gs::kBlockSize);
    gs::GrayScottInitKernel<<<grid, gs::kBlockSize>>>(u, v, L, 0.01f, 1234ULL);
    /* Long enough for the reaction front to spread across the domain. At 200
       steps the pattern is still a small blob at the centre and most of the
       grid is untouched, which made every slice below 1 Mi element degenerate.
       Sampling U rather than V for the same reason: V is seeded to 0 outside a
       6x6 square, U is 1 everywhere plus noise, so U carries structure over the
       whole domain from the first step. */
    for (int i = 0; i < 1200; ++i) {
      gs::GrayScottStepKernel<<<grid, gs::kBlockSize>>>(
          u, v, u2, v2, L, 0.05f, 0.1f, 0.03f, 0.062f, 0.2f, 0.01f,
          1234ULL ^ (static_cast<unsigned long long>(i) << 20));
      float *tu = u; u = u2; u2 = tu;
      float *tv = v; v = v2; v2 = tv;
    }
    ok = cudaDeviceSynchronize() == cudaSuccess;
    if (ok) {
      /* Sample from the MIDDLE of the field, not the start. V is initialized
         to 0 everywhere except a small square at the grid centre, so a prefix
         of the buffer is a corner the pattern has not reached -- the first
         version of this took that prefix and measured a buffer of zeros, which
         both implementations agreed about perfectly and which proved nothing.
         The centre is where the reaction front actually is. */
      const size_t offset = (cells > n_elems) ? (cells - n_elems) / 2 : 0;
      ok = cudaMemcpy(d_out, u + offset, n_elems * sizeof(float),
                      cudaMemcpyDeviceToDevice) == cudaSuccess;
    }
  }
  cudaFree(u); cudaFree(v); cudaFree(u2); cudaFree(v2);
  return ok;
}
#endif

void CheckStat(double clio, double up, const char *what, const char *regime) {
  ++g_checks;
  const double denom = std::fmax(1e-12, std::fabs(up));
  const double rel = std::fabs(clio - up) / denom;
  /* Both sides reduce in double; they should agree far more tightly than the
     float32 network does. Same 1e-9 bound neuropress_dataset_parity uses. */
  if (!(rel <= 1e-9)) {
    ++g_failures;
    std::printf("  FAIL [%s] %-18s clio=%.17g upstream=%.17g rel=%.3e\n",
                regime, what, clio, up, rel);
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("No CUDA device -- skipping.\n");
    return 77;
  }

  cudaStreamCreate(&g_sgd_stream);
  cudaEventCreate(&g_sgd_done);
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const char *weights_dir = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  const std::string nnwt = std::string(weights_dir) + "/model.nnwt";
  if (!gpucompress::loadNNFromBinary(nnwt.c_str())) {
    std::printf("Could not load model.nnwt -- skipping.\n");
    return 77;
  }

  /* ---------------------------------------------------------------------
     For every data regime and error bound:
       (a) compute the three statistics from the SAME BYTES on both sides and
           compare them -- this is the half the previous version skipped, and
           without it "the models agree" only means they agree about numbers I
           typed in, not about data;
       (b) run both inferences from those statistics and compare all EIGHT
           outputs across all 32 actions.
     --------------------------------------------------------------------- */
  /* Chunk sizes. data_size is input 4 of the model, so size is not a scale
     factor here -- it moves the network's operating point. The set straddles
     the boundaries the two stats implementations could disagree on: well below
     one CUDA block, the 256 KiB byte-shuffle block, a non-power-of-two element
     count, and the 4 MiB chunk production actually writes. */
  const size_t kSizes[] = {1024, 65536, 262144, 300000, 1u << 20};
  const size_t kMaxElems = 1u << 20;
  const double kBounds[] = {0.0, 1e-5, 1e-3, 1e-1};

  float *d_chunk = nullptr;
  if (cudaMalloc(&d_chunk, kMaxElems * sizeof(float)) != cudaSuccess) {
    std::printf("Could not allocate chunk -- skipping.\n");
    return 77;
  }
  AutoStatsGPU *d_stats = nullptr;
  if (cudaMalloc(&d_stats, sizeof(AutoStatsGPU)) != cudaSuccess) {
    std::printf("Could not allocate stats -- skipping.\n");
    return 77;
  }

  CompContext ctx{};
  ctx.stream = stream;
  if (cudaMalloc(&ctx.d_fused_infer_output, sizeof(NNInferenceOutput)) !=
          cudaSuccess ||
      cudaMalloc(&ctx.d_fused_top_actions, 32 * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&ctx.d_fused_costs, 32 * sizeof(float)) != cudaSuccess) {
    std::printf("Could not allocate CompContext buffers -- skipping.\n");
    return 77;
  }

  namespace cm = ctp::compress::model;
  cm::NeuroPressNNPredictor predictor;
  if (!predictor.Load(weights_dir) || !predictor.IsReady()) {
    std::printf("Clio weight load failed -- skipping.\n");
    return 77;
  }

  for (Regime reg : {Regime::kSmooth, Regime::kRandom, Regime::kSparse,
                     Regime::kConstant, Regime::kGrayScott}) {
    if (reg == Regime::kGrayScott) {
#if CLIO_HAVE_GRAYSCOTT
      if (!MakeGrayScottField(d_chunk, kMaxElems)) {
        std::printf("[grayscott] simulation failed -- skipping regime\n");
        continue;
      }
#else
      continue;
#endif
    } else {
      const std::vector<float> host_chunk = MakeChunk(kMaxElems, reg);
      if (cudaMemcpy(d_chunk, host_chunk.data(), kMaxElems * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("chunk upload failed -- skipping.\n");
        return 77;
      }
    }

    for (size_t kElems : kSizes) {
    const size_t kBytes = kElems * sizeof(float);

    /* (a) statistics, from the same bytes, by each side's own kernels. */
    double n_ent = 0, n_mad = 0, n_der = 0;
    const int src = gpucompress::runStatsOnlyPipeline(d_chunk, kBytes, stream,
                                                      &n_ent, &n_mad, &n_der);
    cudaStreamSynchronize(stream);
    if (src != 0) {
      std::printf("  [%s n=%zu] upstream stats rc=%d -- skipping\n",
                  RegimeName(reg), kElems, src);
      continue;
    }
    double c_ent = 0, c_mad = 0, c_der = 0;
    ctp::ComputeCompressionFeatures(d_chunk, kElems, ctp::DataType::FLOAT32,
                                    &c_ent, &c_mad, &c_der);

    CheckStat(c_ent, n_ent, "entropy", RegimeName(reg));
    CheckStat(c_mad, n_mad, "mad", RegimeName(reg));
    CheckStat(c_der, n_der, "second_derivative", RegimeName(reg));
    std::printf("[%-9s n=%7zu] entropy %.6f  mad %.6g  deriv %.6g\n",
                RegimeName(reg), kElems, c_ent, c_mad, c_der);
    /* The constant regime is degenerate ON PURPOSE; every other regime must
       actually vary, or "the two agree" is a statement about zeros. */
    if (reg != Regime::kConstant) {
      ++g_checks;
      if (c_mad == 0.0 && c_der == 0.0) {
        ++g_failures;
        std::printf("  FAIL [%s n=%zu] degenerate data: mad and deriv both 0 "
                    "-- this regime is not exercising anything\n",
                    RegimeName(reg), kElems);
      }
    }

    /* Upstream's inference reads an AutoStatsGPU; build it from the stats it
       computed itself, so nothing is laundered through Clio's numbers. */
    AutoStatsGPU h{};
    h.entropy = n_ent;
    h.mad_normalized = n_mad;
    h.deriv_normalized = n_der;
    h.num_elements = kElems;
    if (cudaMemcpy(d_stats, &h, sizeof(h), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      continue;
    }

    for (double eb : kBounds) {
      /* (b) upstream: all 32 configs. */
      std::vector<NNDebugPerConfig> up_cfg(32);
      int chosen = -1;
      const int rc = gpucompress::runNNFusedInferenceCtx(
          d_stats, kBytes, eb, stream, &ctx, &chosen, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, up_cfg.data());
      cudaStreamSynchronize(stream);
      if (rc < 0) {
        std::printf("  [%s eb=%g] upstream inference failed (%d)\n",
                    RegimeName(reg), eb, rc);
        ++g_failures; ++g_checks;
        continue;
      }

      /* Clio: the same 32 actions, from ITS OWN stats. */
      std::vector<cm::CompressionFeatures> batch(32);
      for (int a = 0; a < 32; ++a) {
        const int quant = (a / 8) % 2;
        const int shuf = (a / 16) % 2;
        cm::CompressionFeatures &f = batch[a];
        f.chunk_size_bytes = static_cast<double>(kBytes);
        f.shannon_entropy = c_ent;
        f.mad = c_mad;
        f.second_derivative_mean = c_der;
        f.data_type_float = 1.0;
        f.quantize = static_cast<double>(quant);
        f.byte_shuffle = static_cast<double>(shuf);
        f.error_bound = quant ? eb : 0.0;
        f.library_config_id =
            static_cast<double>(kBaseIdForAlgo[a % 8] * 10 + 2);
        f.config_balanced = 1.0;
      }

      cm::NeuroPressNNPredictor::FullOutputs full;
      if (!predictor.PredictBatchFull(batch, &full)) {
        std::printf("  [%s eb=%g] PredictBatchFull failed\n", RegimeName(reg),
                    eb);
        ++g_failures; ++g_checks;
        continue;
      }

      /* Show the actual numbers for one representative point, so the result
         is legible as a comparison rather than only as a pass/fail count.
         up_cfg[] is filled by upstream's own inference kernel; full.* by
         Clio's. Nothing here is re-derived from a formula. */
      if (kElems == (1u << 20) && eb == 1e-3 && !g_dumped) {
        g_dumped = true;
        std::printf("\n  %-6s %-12s %18s %18s\n", "action", "output",
                    "clio", "upstream(NeuroPress)");
        for (int a : {0, 4, 12, 20}) {
          std::printf("  %-6d %-12s %18.9g %18.9g\n", a, "ratio",
                      full.ratio[a], up_cfg[a].ratio);
          std::printf("  %-6d %-12s %18.9g %18.9g\n", a, "psnr",
                      full.psnr_db[a], up_cfg[a].psnr);
          std::printf("  %-6d %-12s %18.9g %18.9g\n", a, "rmse",
                      full.rmse[a], up_cfg[a].rmse);
          std::printf("  %-6d %-12s %18.9g %18.9g\n", a, "mae",
                      full.mae[a], up_cfg[a].mae);
          std::printf("  %-6d %-12s %18.9g %18.9g\n", a, "ssim",
                      full.ssim[a], up_cfg[a].ssim);
        }
        std::printf("\n");
      }

      for (int a = 0; a < 32; ++a) {
        if (up_cfg[a].comp_time != 0.0f) ++g_nonzero[0];
        if (up_cfg[a].decomp_time != 0.0f) ++g_nonzero[1];
        if (up_cfg[a].ratio != 0.0f) ++g_nonzero[2];
        if (up_cfg[a].psnr != 0.0f) ++g_nonzero[3];
        if (up_cfg[a].rmse != 0.0f) ++g_nonzero[4];
        if (up_cfg[a].max_error != 0.0f) ++g_nonzero[5];
        if (up_cfg[a].mae != 0.0f) ++g_nonzero[6];
        if (up_cfg[a].ssim != 0.0f) ++g_nonzero[7];
        CheckClose(full.ratio[a], up_cfg[a].ratio, "ratio", a);
        CheckClose(full.comp_time_ms[a], up_cfg[a].comp_time, "comp_time", a);
        CheckClose(full.decomp_time_ms[a], up_cfg[a].decomp_time, "decomp_time", a);
        CheckClose(full.psnr_db[a], up_cfg[a].psnr, "psnr", a);
        CheckClose(full.rmse[a], up_cfg[a].rmse, "rmse", a);
        CheckClose(full.max_error[a], up_cfg[a].max_error, "max_error", a);
        CheckClose(full.mae[a], up_cfg[a].mae, "mae", a);
        CheckClose(full.ssim[a], up_cfg[a].ssim, "ssim", a);
      }
    }
    }  // size sweep
  }

  /* ---------------------------------------------------------------------
     Phase 2: ten million random float32 values, chunked, with the entropy
     deliberately swept across chunks.

     The regimes above are four hand-written shapes. This is the opposite: a
     large body of random data whose *entropy* is the thing being varied, so
     the two implementations are compared across the whole range the model's
     input 5 can take rather than at a handful of points. Each chunk draws its
     values from a different number of distinct levels -- 2 up to 2^24 -- which
     moves the byte histogram from nearly degenerate to nearly uniform.

     Entropy here is computed over BYTES in 256 bins on both sides, so it
     saturates near 8; the sweep is designed to walk most of that interval.
     --------------------------------------------------------------------- */
  {
    const size_t kTotal = 10u * 1000u * 1000u;   /* ten million values */
    const size_t kChunk = 262144;                /* 1 MiB per chunk */
    const size_t kChunks = kTotal / kChunk;      /* 38 */

    std::vector<float> host(kTotal);
    unsigned seed = 0xC0FFEEu;
    for (size_t c = 0; c < kChunks; ++c) {
      /* Distinct levels for this chunk: 2, 4, 8, ... 2^24, cycling. Fewer
         levels means a peakier byte histogram and lower entropy. */
      const unsigned bits = 1u + static_cast<unsigned>(c % 24u);
      const double levels = static_cast<double>(1u << bits);
      for (size_t i = 0; i < kChunk; ++i) {
        seed = seed * 1103515245u + 12345u;
        const double q = static_cast<double>((seed >> 8) % (1u << bits));
        host[c * kChunk + i] = static_cast<float>(q / levels);
      }
    }

    float *d_big = nullptr;
    if (cudaMalloc(&d_big, kChunk * sizeof(float)) != cudaSuccess) {
      std::printf("Phase 2: allocation failed -- skipping.\n");
    } else {
      std::printf("\n=== Phase 2: %zu random values in %zu chunks, entropy swept ===\n",
                  kTotal, kChunks);
      double ent_min = 1e30, ent_max = -1e30;

      for (size_t c = 0; c < kChunks; ++c) {
        if (cudaMemcpy(d_big, host.data() + c * kChunk,
                       kChunk * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          break;
        }
        const size_t bytes = kChunk * sizeof(float);

        double n_ent = 0, n_mad = 0, n_der = 0;
        if (gpucompress::runStatsOnlyPipeline(d_big, bytes, stream, &n_ent,
                                              &n_mad, &n_der) != 0) {
          ++g_failures; ++g_checks;
          continue;
        }
        cudaStreamSynchronize(stream);

        double c_ent = 0, c_mad = 0, c_der = 0;
        ctp::ComputeCompressionFeatures(d_big, kChunk, ctp::DataType::FLOAT32,
                                        &c_ent, &c_mad, &c_der);

        char tag[32];
        std::snprintf(tag, sizeof(tag), "rand[%zu]", c);
        CheckStat(c_ent, n_ent, "entropy", tag);
        CheckStat(c_mad, n_mad, "mad", tag);
        CheckStat(c_der, n_der, "second_derivative", tag);
        ent_min = std::fmin(ent_min, c_ent);
        ent_max = std::fmax(ent_max, c_ent);

        AutoStatsGPU hh{};
        hh.entropy = n_ent;
        hh.mad_normalized = n_mad;
        hh.deriv_normalized = n_der;
        hh.num_elements = kChunk;
        if (cudaMemcpy(d_stats, &hh, sizeof(hh), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
          continue;
        }

        const double eb = 1e-3;
        std::vector<NNDebugPerConfig> up2(32);
        int chosen2 = -1;
        if (gpucompress::runNNFusedInferenceCtx(
                d_stats, bytes, eb, stream, &ctx, &chosen2, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, up2.data()) < 0) {
          ++g_failures; ++g_checks;
          continue;
        }
        cudaStreamSynchronize(stream);

        std::vector<cm::CompressionFeatures> b2(32);
        for (int a = 0; a < 32; ++a) {
          const int quant = (a / 8) % 2;
          const int shuf = (a / 16) % 2;
          cm::CompressionFeatures &f = b2[a];
          f.chunk_size_bytes = static_cast<double>(bytes);
          f.shannon_entropy = c_ent;
          f.mad = c_mad;
          f.second_derivative_mean = c_der;
          f.data_type_float = 1.0;
          f.quantize = static_cast<double>(quant);
          f.byte_shuffle = static_cast<double>(shuf);
          f.error_bound = quant ? eb : 0.0;
          f.library_config_id =
              static_cast<double>(kBaseIdForAlgo[a % 8] * 10 + 2);
          f.config_balanced = 1.0;
        }
        cm::NeuroPressNNPredictor::FullOutputs f2;
        if (!predictor.PredictBatchFull(b2, &f2)) {
          ++g_failures; ++g_checks;
          continue;
        }
        for (int a = 0; a < 32; ++a) {
          if (up2[a].rmse != 0.0f) ++g_nonzero[4];
          if (up2[a].max_error != 0.0f) ++g_nonzero[5];
          if (up2[a].mae != 0.0f) ++g_nonzero[6];
          if (up2[a].ssim != 0.0f) ++g_nonzero[7];
          if (up2[a].comp_time != 0.0f) ++g_nonzero[0];
          if (up2[a].decomp_time != 0.0f) ++g_nonzero[1];
          if (up2[a].ratio != 0.0f) ++g_nonzero[2];
          if (up2[a].psnr != 0.0f) ++g_nonzero[3];
          CheckClose(f2.ratio[a], up2[a].ratio, "ratio", a);
          CheckClose(f2.comp_time_ms[a], up2[a].comp_time, "comp_time", a);
          CheckClose(f2.decomp_time_ms[a], up2[a].decomp_time, "decomp_time", a);
          CheckClose(f2.psnr_db[a], up2[a].psnr, "psnr", a);
          CheckClose(f2.rmse[a], up2[a].rmse, "rmse", a);
          CheckClose(f2.max_error[a], up2[a].max_error, "max_error", a);
          CheckClose(f2.mae[a], up2[a].mae, "mae", a);
          CheckClose(f2.ssim[a], up2[a].ssim, "ssim", a);
        }
      }

      std::printf("  entropy spanned %.4f .. %.4f over %zu chunks\n", ent_min,
                  ent_max, kChunks);
      /* The sweep must actually sweep: if every chunk lands on the same
         entropy, this phase is 38 copies of one comparison. */
      ++g_checks;
      if (!(ent_max - ent_min > 1.0)) {
        ++g_failures;
        std::printf("  FAIL entropy range %.4f is too narrow to be a sweep\n",
                    ent_max - ent_min);
      }
      cudaFree(d_big);
    }
  }

  /* ---------------------------------------------------------------------
     Phase 3: the DEVICE-STATS entry point.

     Phases 1-2 went through PredictBatchFull, which takes a host-built input
     matrix. That proves the transforms are right; it does not prove they are
     reachable the way upstream's are. Upstream's device-resident call
     (runNNFusedInferenceCtx) reads AutoStatsGPU from device memory and returns
     all eight outputs. Clio's equivalent used to return four.

     This drives Clio's NeuroPressGpuInferBatchDeviceStats -- same device
     stats, nothing staged through the host -- and checks all eight against
     upstream's per-config output.
     --------------------------------------------------------------------- */
  {
    std::printf("\n=== Phase 3: device-stats entry point, all 8 outputs ===\n");
    const std::vector<float> host_chunk = MakeChunk(1u << 20, Regime::kRandom);
    if (cudaMemcpy(d_chunk, host_chunk.data(), (1u << 20) * sizeof(float),
                   cudaMemcpyHostToDevice) == cudaSuccess) {
      const size_t bytes = (1u << 20) * sizeof(float);
      double n_ent = 0, n_mad = 0, n_der = 0;
      if (gpucompress::runStatsOnlyPipeline(d_chunk, bytes, stream, &n_ent,
                                            &n_mad, &n_der) == 0) {
        cudaStreamSynchronize(stream);
        AutoStatsGPU hh{};
        hh.entropy = n_ent;
        hh.mad_normalized = n_mad;
        hh.deriv_normalized = n_der;
        hh.num_elements = 1u << 20;
        cudaMemcpy(d_stats, &hh, sizeof(hh), cudaMemcpyHostToDevice);

        const double eb = 1e-3;
        std::vector<NNDebugPerConfig> up3(32);
        int chosen3 = -1;
        if (gpucompress::runNNFusedInferenceCtx(
                d_stats, bytes, eb, stream, &ctx, &chosen3, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, up3.data()) >= 0) {
          cudaStreamSynchronize(stream);

          /* Clio, device-stats path, through the PUBLIC API: the chunk's
             statistics stay on the device and all eight come back, which is
             what upstream's runNNFusedInferenceCtx does. */
          std::vector<cm::CompressionFeatures> b3(32);
          for (int a = 0; a < 32; ++a) {
            const int quant = (a / 8) % 2;
            const int shuf = (a / 16) % 2;
            cm::CompressionFeatures &f = b3[a];
            f.chunk_size_bytes = static_cast<double>(bytes);
            f.data_type_float = 1.0;
            f.quantize = static_cast<double>(quant);
            f.byte_shuffle = static_cast<double>(shuf);
            f.error_bound = quant ? eb : 0.0;
            f.library_config_id =
                static_cast<double>(kBaseIdForAlgo[a % 8] * 10 + 2);
            f.config_balanced = 1.0;
            /* Inputs 5-7 are read from device_stats, not from here. */
          }
          /* Clio's device-stats handle, NOT upstream's AutoStatsGPU. The two
             structs have different layouts; passing upstream's here made the
             kernel read the wrong fields and produce a capped ratio of 100
             against upstream's 0.77. The API documents the requirement
             ("device pointer from ComputeDeviceStatsResident") and the mistake
             was mine, not the port's -- but it is exactly the kind of thing a
             harness should surface rather than paper over, so the correct call
             is spelled out here. */
          const void *clio_stats = ctp::ComputeDeviceStatsResident(
              d_chunk, 1u << 20, ctp::DataType::FLOAT32,
              ctp::DeviceStatsStream());
          cm::NeuroPressNNPredictor::FullOutputs f3;
          const auto preds =
              (clio_stats == nullptr)
                  ? std::vector<cm::CompressionPrediction>{}
                  : predictor.PredictBatchDeviceStats(
                        clio_stats, b3, ctp::DeviceStatsStream(),
                        /*weights=*/nullptr, /*out_order=*/nullptr,
                        /*min_psnr=*/0.0, /*out_scores=*/nullptr, &f3);
          ++g_checks;
          if (preds.empty() || f3.rmse.size() != 32) {
            ++g_failures;
            std::printf("  FAIL device-stats call returned nothing\n");
          } else {
            for (int a = 0; a < 32; ++a) {
              CheckClose(f3.rmse[a], up3[a].rmse, "ds-rmse", a);
              CheckClose(f3.max_error[a], up3[a].max_error, "ds-max_error", a);
              CheckClose(f3.mae[a], up3[a].mae, "ds-mae", a);
              CheckClose(f3.ssim[a], up3[a].ssim, "ds-ssim", a);
              CheckClose(f3.ratio[a], up3[a].ratio, "ds-ratio", a);
              CheckClose(f3.psnr_db[a], up3[a].psnr, "ds-psnr", a);
            }
            std::printf("  device-stats path returned all 8 and matched\n");
          }
        }
      }
    }
  }

  gpucompress::freeStatsWorkspace();

  std::printf("\n  SUBSTANTIVE comparisons per output (upstream value non-zero;\n"
              "  zero-valued pairs are verified separately, not counted here):\n");
  for (int i = 0; i < 8; ++i) {
    std::printf("    %-12s %6ld\n", kOutName[i], g_nonzero[i]);
    ++g_checks;
    if (g_nonzero[i] == 0) {
      ++g_failures;
      std::printf("  FAIL output '%s' was zero in every comparison -- "
                  "agreement there is vacuous\n", kOutName[i]);
    }
  }

  std::printf("\n  %ld zero-valued pairs verified separately (upstream=0 =>\n"
              "  clio must also be 0; this is where a missing clamp shows up)\n",
              g_zero_pairs);
  std::printf("\n===== %ld substantive checks, %d failures =====\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
