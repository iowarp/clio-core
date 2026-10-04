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
 * @file gpucodecs_capi.cu
 * @brief C interface over gpu_codecs.cuh for the producer-consumer pipeline
 * driver (pipeline.py, via ctypes). Device pointers in, device pointers out;
 * every call synchronizes the device first (the caller's tensors may be on
 * another stream) and returns only after the codec's work is done.
 *
 * Timing matches corpus_sweep.cu: CUDA events around the codec's own call,
 * with any pre-shuffle inside the compress timing and the un-shuffle inside
 * the decompress timing.
 */
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include "gpu_codecs.cuh"

using gpu_codecs::Codec;

namespace {

/** @brief A built codec plus the events used to time it. */
struct Handle {
  std::unique_ptr<Codec> codec;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
};

/** @return ms between the handle's two events (already synchronized). */
float Elapsed(Handle *h) {
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, h->e0, h->e1);
  return ms;
}

}  // namespace

extern "C" {

/**
 * Build a codec from a setting string ("zstd shuffle=byte elem=8", ...).
 * @return opaque handle, or nullptr when the spec is unknown or invalid
 *         (the reason is printed to stderr)
 */
void *gc_create(const char *spec) {
  auto *h = new Handle();
  try {
    h->codec = gpu_codecs::MakeCodec(gpu_codecs::ParseSpec(spec));
    if (!h->codec) throw std::invalid_argument(std::string("unknown codec: ") + spec);
    cudaEventCreate(&h->e0);
    cudaEventCreate(&h->e1);
    return h;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "gc_create(%s): %s\n", spec, e.what());
    delete h;
    return nullptr;
  }
}

/** Free a handle from gc_create. */
void gc_destroy(void *p) {
  auto *h = static_cast<Handle *>(p);
  if (!h) return;
  if (h->e0) cudaEventDestroy(h->e0);
  if (h->e1) cudaEventDestroy(h->e1);
  delete h;
}

/** @return worst-case compressed bytes for n input bytes (0 = not accepted). */
size_t gc_bound(void *p, size_t n) {
  auto *h = static_cast<Handle *>(p);
  return h->codec->Accepts(n) ? h->codec->Bound(n) : 0;
}

/**
 * Compress n bytes at device pointer in into device buffer out (cap bytes).
 * @param ms out: GPU time of the shuffle (if any) + the codec's compress call
 * @return compressed bytes, 0 on failure or when the codec cannot take n
 */
size_t gc_compress(void *p, const void *in, size_t n, void *out, size_t cap,
                   float *ms) {
  auto *h = static_cast<Handle *>(p);
  Codec *c = h->codec.get();
  if (!c->Accepts(n)) return 0;
  try {
    cudaDeviceSynchronize();
    c->PrepareCompress(n);
    cudaStream_t s = c->stream();
    cudaStreamSynchronize(s);
    cudaEventRecord(h->e0, s);
    const uint8_t *src = c->Preprocess(static_cast<const uint8_t *>(in), n);
    c->Compress(src, n, static_cast<uint8_t *>(out), cap);
    cudaEventRecord(h->e1, s);
    cudaEventSynchronize(h->e1);
    const size_t bytes = c->CompressedBytes(static_cast<const uint8_t *>(out));
    if (ms) *ms = Elapsed(h);
    return (bytes == 0 || bytes > cap) ? 0 : bytes;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "gc_compress: %s\n", e.what());
    cudaGetLastError();
    return 0;
  }
}

/**
 * Decompress comp bytes at device pointer in into n bytes at device pointer
 * out.
 * @param ms out: GPU time of the codec's decompress call + the un-shuffle
 * @return 1 on success, 0 on failure
 */
int gc_decompress(void *p, const void *in, size_t comp, void *out, size_t n,
                  float *ms) {
  auto *h = static_cast<Handle *>(p);
  Codec *c = h->codec.get();
  try {
    cudaDeviceSynchronize();
    c->PrepareCompress(n);  // sizes scratch (shuffle decode buffer), untimed
    uint8_t *o = static_cast<uint8_t *>(out);
    uint8_t *dst = c->DecodeTarget(o);
    c->PrepareDecompress(static_cast<const uint8_t *>(in), comp);
    cudaStream_t s = c->stream();
    cudaEventRecord(h->e0, s);
    c->Decompress(static_cast<const uint8_t *>(in), comp, dst, n);
    c->Postprocess(o, n);
    cudaEventRecord(h->e1, s);
    cudaEventSynchronize(h->e1);
    if (ms) *ms = Elapsed(h);
    return c->DecompressOk() ? 1 : 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "gc_decompress: %s\n", e.what());
    cudaGetLastError();
    return 0;
  }
}

}  // extern "C"
