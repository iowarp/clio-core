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
 * @file neuropress_v2_kernels.cu
 * @brief The GPU side of NeuroPress v2: inputs from device statistics, the
 * network, and the ranking of every setting, in one single-block kernel.
 */

#include "neuropress_v2_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF

#include <cmath>

#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

namespace ctp::compress::model::v2 {

namespace {

constexpr int kThreads = 256;  ///< one block; >= kMaxWidth

/**
 * Thread 0: the four standardised inputs from the chunk's statistics, with
 * the training definitions (see neuropress_v2_predictor.h).
 */
__device__ void BuildInputs(const float *p, const NetDesc &d,
                            const ctp::DeviceFeatureStats *st,
                            double chunk_bytes, float *x_raw, float *x_std) {
  double range = st->value_max - st->value_min;
  if (!(range > 0.0)) range = 1.0;
  x_raw[0] = static_cast<float>(log2(chunk_bytes));
  x_raw[1] = static_cast<float>(st->entropy);
  x_raw[2] = static_cast<float>(log10(st->mad / range + 1e-12));
  x_raw[3] = static_cast<float>(log10(st->second_derivative / range + 1e-12));
  for (int i = 0; i < 4; ++i) {
    x_std[i] = (x_raw[i] - p[d.xm_off + i]) / p[d.xs_off + i];
  }
}

/**
 * All layers, block-wide: activations ping-pong between two shared buffers.
 * @return the buffer holding the (still standardised) outputs
 */
__device__ const float *RunLayers(const float *p, const NetDesc &d,
                                  float (*buf)[kMaxWidth]) {
  int in = 0;
  for (int l = 0; l < d.n_layers; ++l) {
    const int n_in = d.dims[l];
    const int n_out = d.dims[l + 1];
    const float *w = p + d.w_off[l];
    const float *b = p + d.b_off[l];
    for (int o = threadIdx.x; o < n_out; o += blockDim.x) {
      float s = b[o];
      for (int i = 0; i < n_in; ++i) s += w[o * n_in + i] * buf[in][i];
      buf[1 - in][o] = (l + 1 < d.n_layers) ? fmaxf(0.0f, s) : s;
    }
    __syncthreads();
    in = 1 - in;
  }
  return buf[in];
}

/** Single block: inputs, network, per-setting outcome and cost, ordering. */
__global__ void RankKernel(const float *p, NetDesc d,
                           const unsigned char *available,
                           const ctp::DeviceFeatureStats *st, RankArgs a,
                           RankOut *out) {
  __shared__ float buf[2][kMaxWidth];
  __shared__ double cost[kMaxSettings];
  if (threadIdx.x == 0) BuildInputs(p, d, st, a.chunk_bytes, out->x, buf[0]);
  __syncthreads();
  const float *y = RunLayers(p, d, buf);
  for (int s = threadIdx.x; s < d.n_settings; s += blockDim.x) {
    float v[3];
    for (int k = 0; k < 3; ++k) {
      const int o = 3 * s + k;
      v[k] = expf(y[o] * p[d.ys_off + o] + p[d.ym_off + o]);
    }
    out->ct[s] = v[0];
    out->dt[s] = v[1];
    out->ratio[s] = v[2];
    double c = a.w_ct * v[0] + a.w_dt * v[1] +
               a.w_io * a.chunk_bytes / (static_cast<double>(v[2]) *
                                         a.bw_bytes_per_ms);
    if (!available[s] || !isfinite(c) || !(v[2] > 0.0f)) c = CUDART_INF;
    cost[s] = c;
    out->cost[s] = c;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Insertion sort, cheapest first; ties keep the lower index first.
    for (int s = 0; s < d.n_settings; ++s) {
      int j = s;
      while (j > 0 && cost[out->order[j - 1]] > cost[s]) {
        out->order[j] = out->order[j - 1];
        --j;
      }
      out->order[j] = s;
    }
  }
}

/**
 * Thread 0: scale the raw inputs; then every hidden layer, block-wide.
 * @return the buffer holding the last hidden layer's activations
 */
__device__ const float *RunHidden(const float *p, const NetDesc &d,
                                  const float *x_raw, float (*buf)[kMaxWidth]) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < d.dims[0]; ++i) {
      buf[0][i] = (x_raw[i] - p[d.xm_off + i]) / p[d.xs_off + i];
    }
  }
  __syncthreads();
  int in = 0;
  for (int l = 0; l + 1 < d.n_layers; ++l) {
    const int n_in = d.dims[l];
    const int n_out = d.dims[l + 1];
    const float *w = p + d.w_off[l];
    const float *b = p + d.b_off[l];
    for (int o = threadIdx.x; o < n_out; o += blockDim.x) {
      float s = b[o];
      for (int i = 0; i < n_in; ++i) s += w[o * n_in + i] * buf[in][i];
      buf[1 - in][o] = fmaxf(0.0f, s);
    }
    __syncthreads();
    in = 1 - in;
  }
  return buf[in];
}

/**
 * Normalised LMS on the labelled output rows of one setting: thread 0 forms
 * each row's error against the label, then the block subtracts
 * lr * err / (1 + |h|^2) * h from the row and the bias.
 */
__global__ void TrainKernel(float *p, NetDesc d, TrainArgs a,
                            double *abs_err) {
  __shared__ float buf[2][kMaxWidth];
  __shared__ double step[3];
  const float *h = RunHidden(p, d, a.x, buf);
  const int last = d.n_layers - 1;
  const int n_in = d.dims[last];
  float *w = p + d.w_off[last];
  float *b = p + d.b_off[last];
  if (threadIdx.x == 0) {
    double h2 = 1.0, err_sum = 0.0;
    int n = 0;
    for (int i = 0; i < n_in; ++i) h2 += static_cast<double>(h[i]) * h[i];
    for (int k = 0; k < 3; ++k) {
      step[k] = 0.0;
      if (!(a.label[k] > 0.0) || !isfinite(a.label[k])) continue;
      const int o = 3 * a.setting + k;
      double pred = b[o];
      for (int i = 0; i < n_in; ++i) pred += static_cast<double>(w[o * n_in + i]) * h[i];
      const double sd = p[d.ys_off + o];
      const double target = (log(a.label[k]) - p[d.ym_off + o]) / sd;
      const double err = pred - target;
      err_sum += fabs(err * sd);
      ++n;
      step[k] = a.lr * err / h2;
      b[o] -= static_cast<float>(step[k]);
    }
    if (abs_err != nullptr) *abs_err = n > 0 ? err_sum / n : 0.0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n_in; i += blockDim.x) {
    for (int k = 0; k < 3; ++k) {
      if (step[k] != 0.0) {
        w[(3 * a.setting + k) * n_in + i] -= static_cast<float>(step[k] * h[i]);
      }
    }
  }
}

/** One element of a NeuroPressV2Dtype code as float. */
__device__ float ElementToFloat(const void *in, size_t i, int dtype) {
  switch (dtype) {
    case 2: return static_cast<float>(static_cast<const double *>(in)[i]);
    case 3: return __half2float(static_cast<const __half *>(in)[i]);
    case 4: return __bfloat162float(static_cast<const __nv_bfloat16 *>(in)[i]);
    case 5: return static_cast<float>(static_cast<const int8_t *>(in)[i]);
    case 6: return static_cast<float>(static_cast<const uint8_t *>(in)[i]);
    case 7: return static_cast<float>(static_cast<const int16_t *>(in)[i]);
    case 8: return static_cast<float>(static_cast<const uint16_t *>(in)[i]);
    case 9: return static_cast<float>(static_cast<const int32_t *>(in)[i]);
    case 10: return static_cast<float>(static_cast<const uint32_t *>(in)[i]);
    case 11: return static_cast<float>(static_cast<const long long *>(in)[i]);
    case 12:
      return static_cast<float>(static_cast<const unsigned long long *>(in)[i]);
    default: return static_cast<const float *>(in)[i];
  }
}

/** Grid-stride conversion of n elements to float32. */
__global__ void ConvertKernel(const void *in, float *out, size_t n,
                              int dtype) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    out[i] = ElementToFloat(in, i, dtype);
  }
}

/** @return this thread's device copy of a RankOut (allocated once). */
RankOut *ThreadOut() {
  static thread_local RankOut *d_out = nullptr;
  if (d_out == nullptr && cudaMalloc(&d_out, sizeof(RankOut)) != cudaSuccess) {
    d_out = nullptr;
  }
  return d_out;
}

}  // namespace

bool Upload(const void *host, size_t bytes, void **dev) {
  *dev = nullptr;
  if (cudaMalloc(dev, bytes) != cudaSuccess) return false;
  if (cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*dev);
    *dev = nullptr;
    return false;
  }
  return true;
}

bool CopyToDevice(void *dev, size_t offset_bytes, const void *host,
                  size_t bytes) {
  return cudaMemcpy(static_cast<char *>(dev) + offset_bytes, host, bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

float *ConvertToFloat32(const void *in, size_t n, int dtype, void *stream,
                        double *convert_ms) {
  static thread_local float *scratch = nullptr;
  static thread_local size_t cap = 0;
  if (n > cap) {
    if (scratch != nullptr) cudaFree(scratch);
    scratch = nullptr;
    cap = 0;
    if (cudaMalloc(&scratch, n * sizeof(float)) != cudaSuccess) return nullptr;
    cap = n;
  }
  static thread_local cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (e0 == nullptr && (cudaEventCreate(&e0) != cudaSuccess ||
                        cudaEventCreate(&e1) != cudaSuccess)) {
    return nullptr;
  }
  auto s = static_cast<cudaStream_t>(stream);
  cudaEventRecord(e0, s);
  ConvertKernel<<<1024, 256, 0, s>>>(in, scratch, n, dtype);
  cudaEventRecord(e1, s);
  if (cudaGetLastError() != cudaSuccess || cudaEventSynchronize(e1) != cudaSuccess) {
    return nullptr;
  }
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);
  *convert_ms = static_cast<double>(ms);
  return scratch;
}

bool TrainOnDevice(float *d_params, const NetDesc &desc, const TrainArgs &args,
                   void *stream, void *done, double *d_abs_err, void *t_start,
                   void *t_stop) {
  if (desc.n_layers > kMaxLayers || args.setting < 0 ||
      args.setting >= desc.n_settings) {
    return false;
  }
  auto s = static_cast<cudaStream_t>(stream);
  if (t_start != nullptr) cudaEventRecord(static_cast<cudaEvent_t>(t_start), s);
  TrainKernel<<<1, kThreads, 0, s>>>(d_params, desc, args, d_abs_err);
  if (cudaGetLastError() != cudaSuccess) return false;
  if (t_stop != nullptr) cudaEventRecord(static_cast<cudaEvent_t>(t_stop), s);
  return done == nullptr ||
         cudaEventRecord(static_cast<cudaEvent_t>(done), s) == cudaSuccess;
}

void *CreateTimingEvent() {
  cudaEvent_t e = nullptr;
  if (cudaEventCreate(&e) != cudaSuccess) return nullptr;
  return e;
}

void DestroyEvent(void *event) {
  if (event != nullptr) cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

bool RecordEvent(void *event, void *stream) {
  return event != nullptr &&
         cudaEventRecord(static_cast<cudaEvent_t>(event),
                         static_cast<cudaStream_t>(stream)) == cudaSuccess;
}

bool EventDone(void *event) {
  if (event == nullptr) return false;
  const cudaError_t rc = cudaEventQuery(static_cast<cudaEvent_t>(event));
  if (rc == cudaErrorNotReady) return false;
  return rc == cudaSuccess;
}

bool EventSync(void *event) {
  return event != nullptr &&
         cudaEventSynchronize(static_cast<cudaEvent_t>(event)) == cudaSuccess;
}

double EventElapsedMs(void *start, void *stop) {
  float ms = 0.0f;
  if (start == nullptr || stop == nullptr ||
      cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start),
                           static_cast<cudaEvent_t>(stop)) != cudaSuccess) {
    cudaGetLastError();
    return -1.0;
  }
  return static_cast<double>(ms);
}

bool CreateStreamAndEvent(void **stream, void **event) {
  cudaStream_t s = nullptr;
  cudaEvent_t e = nullptr;
  if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
    return false;
  }
  if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
    cudaStreamDestroy(s);
    return false;
  }
  *stream = s;
  *event = e;
  return true;
}

void DestroyStreamAndEvent(void *stream, void *event) {
  if (event != nullptr) cudaEventDestroy(static_cast<cudaEvent_t>(event));
  if (stream != nullptr) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

bool StreamWaitEvent(void *stream, void *event) {
  return event == nullptr ||
         cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                             static_cast<cudaEvent_t>(event), 0) == cudaSuccess;
}

bool CopyToHost(void *host, const void *dev, size_t bytes, void *stream) {
  auto s = static_cast<cudaStream_t>(stream);
  return cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s) ==
             cudaSuccess &&
         cudaStreamSynchronize(s) == cudaSuccess;
}

void FreeDevice(void *dev) {
  if (dev != nullptr) cudaFree(dev);
}

bool RankLaunch(const float *d_params, const NetDesc &desc,
                const unsigned char *d_available, const void *device_stats,
                const RankArgs &args, void *stream, RankOut *host_out,
                void *done) {
  if (desc.n_settings > kMaxSettings || desc.n_layers > kMaxLayers) {
    return false;
  }
  for (int l = 0; l <= desc.n_layers; ++l) {
    if (desc.dims[l] > kMaxWidth) return false;
  }
  // The thread's device result buffer is reused by the next ranking on this
  // thread, which is enqueued on the same stream after this copy.
  RankOut *d_out = ThreadOut();
  if (d_out == nullptr || device_stats == nullptr || host_out == nullptr) {
    return false;
  }
  auto s = static_cast<cudaStream_t>(stream);
  RankKernel<<<1, kThreads, 0, s>>>(
      d_params, desc, d_available,
      static_cast<const ctp::DeviceFeatureStats *>(device_stats), args, d_out);
  if (cudaGetLastError() != cudaSuccess) return false;
  if (cudaMemcpyAsync(host_out, d_out, sizeof(RankOut), cudaMemcpyDeviceToHost,
                      s) != cudaSuccess) {
    return false;
  }
  return RecordEvent(done, stream);
}

void *AllocPinned(size_t bytes) {
  void *p = nullptr;
  if (cudaMallocHost(&p, bytes) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  return p;
}

bool RankOnDevice(const float *d_params, const NetDesc &desc,
                  const unsigned char *d_available, const void *device_stats,
                  const RankArgs &args, void *stream, RankOut *out) {
  if (desc.n_settings > kMaxSettings || desc.n_layers > kMaxLayers) {
    return false;
  }
  for (int l = 0; l <= desc.n_layers; ++l) {
    if (desc.dims[l] > kMaxWidth) return false;
  }
  RankOut *d_out = ThreadOut();
  if (d_out == nullptr || device_stats == nullptr) return false;
  auto s = static_cast<cudaStream_t>(stream);
  RankKernel<<<1, kThreads, 0, s>>>(
      d_params, desc, d_available,
      static_cast<const ctp::DeviceFeatureStats *>(device_stats), args, d_out);
  if (cudaGetLastError() != cudaSuccess) return false;
  if (cudaMemcpyAsync(out, d_out, sizeof(RankOut), cudaMemcpyDeviceToHost,
                      s) != cudaSuccess) {
    return false;
  }
  return cudaStreamSynchronize(s) == cudaSuccess;
}

}  // namespace ctp::compress::model::v2
