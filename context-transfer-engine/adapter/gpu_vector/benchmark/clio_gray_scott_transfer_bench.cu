// clio_gray_scott_transfer_bench.cu
// ---------------------------------------------------------------------------
// Proof-of-concept microbenchmark for the tier-aware compressed GPU vector.
//
// Workload: Gray-Scott reaction-diffusion. Each simulation step produces an
// output snapshot of (nblocks * per_block_bytes) that must be moved GPU->host
// (modeling per-step simulation checkpoint I/O). We compare four GPU->host
// transfer strategies:
//
//   1. raw          : step -> wait -> D2H copy -> relaunch (serial baseline)
//   2. async        : double-buffered; D2H of step N overlaps compute of N+1
//   3. compressed   : compress the snapshot on a stream, then D2H the (smaller)
//                     compressed bytes. Host-orchestrated (NO CDP).
//   4. async+comp   : triple-buffered; compute || compress || D2H overlapped
//
// COMPRESSOR NOTE
// ---------------
// The "compressor" here is a host-launched, error-bounded quantizer kernel used
// as a STAND-IN so the compressed pipeline runs and is measurable on any GPU,
// including Blackwell sm_120 where real cuSZp is unreliable (PTX-JIT clobbering;
// see clio_ctp/compress/cuszp.h). On Delta (sm_80/sm_90), define USE_CUSZP to
// link real cuSZp via its host API (cuSZp_compress on a stream) in the exact
// same pipeline slot. The transfer mechanics (case 1-4) are identical either way.
//
// Build (local, sm_120):
//   nvcc -O3 -arch=sm_120 -o gs_bench clio_gray_scott_transfer_bench.cu
// Build (Delta, real cuSZp, e.g. sm_80):
//   nvcc -O3 -arch=sm_80 -DUSE_CUSZP -I<cuszp_inc> clio_gray_scott_transfer_bench.cu \
//        -L<cuszp_lib> -lcuSZp -o gs_bench
//
// Run:
//   ./gs_bench [nblocks] [threads_per_block] [per_block_bytes] [nsteps] [reps]
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(_e));                                   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Gray-Scott reaction-diffusion (2D, 5-point Laplacian).
// Field is laid out as [nblocks rows] x [elems_per_block cols]; each CUDA block
// owns one row, threads stride over the columns. Boundaries are clamped.
//   u2 = u + (Du*lap(u) - u*v^2 + F*(1-u)) * dt
//   v2 = v + (Dv*lap(v) + u*v^2 - (F+k)*v) * dt
// ---------------------------------------------------------------------------
__global__ void GrayScottStepKernel(const float* __restrict__ u,
                                     const float* __restrict__ v,
                                     float* __restrict__ u2,
                                     float* __restrict__ v2, int rows, int cols,
                                     float Du, float Dv, float F, float k,
                                     float dt) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int idx = row * cols + col;
    const int up = (row > 0 ? row - 1 : row) * cols + col;
    const int dn = (row < rows - 1 ? row + 1 : row) * cols + col;
    const int lf = row * cols + (col > 0 ? col - 1 : col);
    const int rt = row * cols + (col < cols - 1 ? col + 1 : col);

    const float uc = u[idx], vc = v[idx];
    const float lap_u = u[up] + u[dn] + u[lf] + u[rt] - 4.0f * uc;
    const float lap_v = v[up] + v[dn] + v[lf] + v[rt] - 4.0f * vc;
    const float uvv = uc * vc * vc;
    u2[idx] = uc + (Du * lap_u - uvv + F * (1.0f - uc)) * dt;
    v2[idx] = vc + (Dv * lap_v + uvv - (F + k) * vc) * dt;
  }
}

// ---------------------------------------------------------------------------
// STAND-IN compressor: per-block absolute-error-bounded quantization of floats
// into int8 (4x reduction). One CUDA block compresses one logical block. Real,
// runs anywhere; replaced by cuSZp host API on Delta (USE_CUSZP). Each block:
//   range scan -> quantize each float to int8 around the block's min, store an
//   8-byte header (min, scale) + cols int8 codes.
// Layout of compressed block b: [float min][float scale][cols * int8].
// ---------------------------------------------------------------------------
__global__ void QuantCompressKernel(const float* __restrict__ in,
                                     uint8_t* __restrict__ out, int rows,
                                     int cols, int comp_block_bytes) {
  extern __shared__ float sdata[];  // blockDim.x for reduction
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* src = in + row * cols;

  // Block-wide min/max via shared-memory reduction.
  float lmin = 3.4e38f, lmax = -3.4e38f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float x = src[c];
    lmin = x < lmin ? x : lmin;
    lmax = x > lmax ? x : lmax;
  }
  sdata[threadIdx.x] = lmin;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] = fminf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    __syncthreads();
  }
  const float bmin = sdata[0];
  __syncthreads();
  sdata[threadIdx.x] = lmax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    __syncthreads();
  }
  const float bmax = sdata[0];

  const float range = (bmax - bmin);
  const float scale = range > 0.0f ? range / 255.0f : 1.0f;
  uint8_t* dst = out + (size_t)row * comp_block_bytes;
  if (threadIdx.x == 0) {
    // Byte-wise header store (device-safe, no alignment assumption).
    const uint8_t* pmin = reinterpret_cast<const uint8_t*>(&bmin);
    const uint8_t* psc = reinterpret_cast<const uint8_t*>(&scale);
    for (int i = 0; i < 4; ++i) {
      dst[i] = pmin[i];
      dst[i + 4] = psc[i];
    }
  }
  __syncthreads();
  uint8_t* codes = dst + 2 * sizeof(float);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    int q = (int)((src[c] - bmin) / scale + 0.5f);
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    codes[c] = (uint8_t)q;
  }
}

// ---------------------------------------------------------------------------
struct Opts {
  int nblocks = 4096;
  int threads = 128;
  int per_block_bytes = 4096;  // bytes each block emits per step
  int nsteps = 200;
  int reps = 3;
};

struct Result {
  double ms_total = 0;     // best-of-reps wall time for the whole run
  double bytes_moved = 0;  // total bytes actually transferred to host
  double comp_ratio = 1.0;
};

static const float kDu = 0.16f, kDv = 0.08f, kF = 0.060f, kK = 0.062f,
                   kDt = 1.0f;

// Compress the snapshot in `d_v` (rows*cols floats) into `d_comp`. Host-side
// launch on `stream` (no CDP). Returns compressed size in bytes.
static size_t CompressOnStream(const float* d_v, uint8_t* d_comp, int rows,
                               int cols, int comp_block_bytes,
                               cudaStream_t stream) {
#ifdef USE_CUSZP
  // Real cuSZp host API on Delta. cmp_size filled by cuSZp; eb=1e-3 absolute.
  // (Linked when -DUSE_CUSZP; header/lib supplied via -I/-L.)
  size_t cmp_size = 0;
  uint3 dims = {0, 0, 0};
  cuSZp_compress((float*)d_v, d_comp, (size_t)rows * cols, &cmp_size, 1e-3f,
                 CUSZP_DIM_1D, dims, CUSZP_TYPE_FLOAT, CUSZP_MODE_OUTLIER,
                 stream);
  return cmp_size;
#else
  const size_t shmem = (size_t)/*threads*/ 0;  // set by caller via launch below
  (void)shmem;
  // Stand-in quantizer: deterministic 4x reduction (1 byte/float + 8B header).
  QuantCompressKernel<<<rows, /*threads*/ 256, 256 * sizeof(float), stream>>>(
      d_v, d_comp, rows, cols, comp_block_bytes);
  return (size_t)rows * comp_block_bytes;
#endif
}

// ===========================================================================
// Case 1: RAW — step, sync, D2H, relaunch. Fully serial.
// ===========================================================================
static Result RunRaw(const Opts& o, int rows, int cols, size_t snap_bytes) {
  float *u, *v, *u2, *v2;
  CUDA_CHECK(cudaMalloc(&u, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v, snap_bytes));
  CUDA_CHECK(cudaMalloc(&u2, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v2, snap_bytes));
  CUDA_CHECK(cudaMemset(u, 0, snap_bytes));
  CUDA_CHECK(cudaMemset(v, 0, snap_bytes));
  uint8_t* h_snap;
  CUDA_CHECK(cudaMallocHost(&h_snap, snap_bytes));

  Result best;
  best.ms_total = 1e30;
  for (int r = 0; r < o.reps; ++r) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int s = 0; s < o.nsteps; ++s) {
      GrayScottStepKernel<<<rows, o.threads>>>(u, v, u2, v2, rows, cols, kDu,
                                               kDv, kF, kK, kDt);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_snap, v2, snap_bytes, cudaMemcpyDeviceToHost));
      std::swap(u, u2);
      std::swap(v, v2);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (ms < best.ms_total) best.ms_total = ms;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
  }
  best.bytes_moved = (double)snap_bytes * o.nsteps;
  cudaFree(u); cudaFree(v); cudaFree(u2); cudaFree(v2); cudaFreeHost(h_snap);
  return best;
}

// ===========================================================================
// Case 2: ASYNC — double-buffered snapshot; D2H of step N overlaps compute N+1.
// Two device snapshot buffers + two pinned host buffers, compute & copy streams.
// ===========================================================================
static Result RunAsync(const Opts& o, int rows, int cols, size_t snap_bytes) {
  float *u, *v, *u2, *v2;
  CUDA_CHECK(cudaMalloc(&u, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v, snap_bytes));
  CUDA_CHECK(cudaMalloc(&u2, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v2, snap_bytes));
  CUDA_CHECK(cudaMemset(u, 0, snap_bytes));
  CUDA_CHECK(cudaMemset(v, 0, snap_bytes));
  // Two pinned host buffers (one per double-buffer slot). No device staging:
  // the GS field is already double-buffered (u/v/u2/v2), so we D2H straight
  // from the step's output field v2, gated by events.
  uint8_t* h_snap[2];
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaMallocHost(&h_snap[i], snap_bytes));
  }
  cudaStream_t comp, copy;
  CUDA_CHECK(cudaStreamCreateWithFlags(&comp, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
  cudaEvent_t gs_done[2], copy_done[2];
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaEventCreateWithFlags(&gs_done[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copy_done[i], copy));  // initially "free"
  }

  Result best;
  best.ms_total = 1e30;
  for (int r = 0; r < o.reps; ++r) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int s = 0; s < o.nsteps; ++s) {
      const int b = s & 1;
      // Don't overwrite a buffer whose D2H (two steps ago) is still in flight.
      CUDA_CHECK(cudaStreamWaitEvent(comp, copy_done[b], 0));
      GrayScottStepKernel<<<rows, o.threads, 0, comp>>>(
          u, v, u2, v2, rows, cols, kDu, kDv, kF, kK, kDt);
      CUDA_CHECK(cudaEventRecord(gs_done[b], comp));
      // Copy stream: wait for this step's GS, then async D2H STRAIGHT from the
      // output field v2 to pinned host. No device-to-device staging copy --
      // the field is double-buffered, so v2 is not overwritten until step b+2,
      // which is gated by copy_done[b] above.
      CUDA_CHECK(cudaStreamWaitEvent(copy, gs_done[b], 0));
      CUDA_CHECK(cudaMemcpyAsync(h_snap[b], v2, snap_bytes,
                                 cudaMemcpyDeviceToHost, copy));
      CUDA_CHECK(cudaEventRecord(copy_done[b], copy));
      std::swap(u, u2);
      std::swap(v, v2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (ms < best.ms_total) best.ms_total = ms;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
  }
  best.bytes_moved = (double)snap_bytes * o.nsteps;
  for (int i = 0; i < 2; ++i) {
    cudaFreeHost(h_snap[i]);
    cudaEventDestroy(gs_done[i]); cudaEventDestroy(copy_done[i]);
  }
  cudaStreamDestroy(comp); cudaStreamDestroy(copy);
  cudaFree(u); cudaFree(v); cudaFree(u2); cudaFree(v2);
  return best;
}

// ===========================================================================
// Case 3: COMPRESSED — step, compress snapshot on a stream, D2H compressed bytes.
// Serial dependency (compute -> compress -> copy) but moves far fewer bytes.
// ===========================================================================
static Result RunCompressed(const Opts& o, int rows, int cols,
                            size_t snap_bytes, int comp_block_bytes) {
  const size_t comp_bytes = (size_t)rows * comp_block_bytes;
  float *u, *v, *u2, *v2;
  CUDA_CHECK(cudaMalloc(&u, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v, snap_bytes));
  CUDA_CHECK(cudaMalloc(&u2, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v2, snap_bytes));
  CUDA_CHECK(cudaMemset(u, 0, snap_bytes));
  CUDA_CHECK(cudaMemset(v, 0, snap_bytes));
  uint8_t *d_comp, *h_comp;
  CUDA_CHECK(cudaMalloc(&d_comp, comp_bytes));
  CUDA_CHECK(cudaMallocHost(&h_comp, comp_bytes));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  Result best;
  best.ms_total = 1e30;
  size_t last_csize = comp_bytes;
  for (int r = 0; r < o.reps; ++r) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int s = 0; s < o.nsteps; ++s) {
      GrayScottStepKernel<<<rows, o.threads, 0, stream>>>(
          u, v, u2, v2, rows, cols, kDu, kDv, kF, kK, kDt);
      last_csize =
          CompressOnStream(v2, d_comp, rows, cols, comp_block_bytes, stream);
      CUDA_CHECK(cudaMemcpyAsync(h_comp, d_comp, last_csize,
                                 cudaMemcpyDeviceToHost, stream));
      std::swap(u, u2);
      std::swap(v, v2);
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (ms < best.ms_total) best.ms_total = ms;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
  }
  best.bytes_moved = (double)last_csize * o.nsteps;
  best.comp_ratio = (double)snap_bytes / (double)last_csize;
  cudaFree(d_comp); cudaFreeHost(h_comp); cudaStreamDestroy(stream);
  cudaFree(u); cudaFree(v); cudaFree(u2); cudaFree(v2);
  return best;
}

// ===========================================================================
// Case 4: ASYNC + COMPRESSED — triple-buffered; compute || compress || D2H.
// ===========================================================================
static Result RunAsyncCompressed(const Opts& o, int rows, int cols,
                                 size_t snap_bytes, int comp_block_bytes) {
  const size_t comp_bytes = (size_t)rows * comp_block_bytes;
  float *u, *v, *u2, *v2;
  CUDA_CHECK(cudaMalloc(&u, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v, snap_bytes));
  CUDA_CHECK(cudaMalloc(&u2, snap_bytes));
  CUDA_CHECK(cudaMalloc(&v2, snap_bytes));
  CUDA_CHECK(cudaMemset(u, 0, snap_bytes));
  CUDA_CHECK(cudaMemset(v, 0, snap_bytes));
  uint8_t *d_comp[2], *h_comp[2];
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp[i], comp_bytes));
    CUDA_CHECK(cudaMallocHost(&h_comp[i], comp_bytes));
  }
  cudaStream_t comp, copy;
  CUDA_CHECK(cudaStreamCreateWithFlags(&comp, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
  cudaEvent_t comp_ready[2], copy_done[2];
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaEventCreateWithFlags(&comp_ready[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copy_done[i], copy));
  }

  Result best;
  best.ms_total = 1e30;
  size_t last_csize = comp_bytes;
  for (int r = 0; r < o.reps; ++r) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int s = 0; s < o.nsteps; ++s) {
      const int b = s & 1;
      CUDA_CHECK(cudaStreamWaitEvent(comp, copy_done[b], 0));
      GrayScottStepKernel<<<rows, o.threads, 0, comp>>>(
          u, v, u2, v2, rows, cols, kDu, kDv, kF, kK, kDt);
      last_csize =
          CompressOnStream(v2, d_comp[b], rows, cols, comp_block_bytes, comp);
      CUDA_CHECK(cudaEventRecord(comp_ready[b], comp));
      CUDA_CHECK(cudaStreamWaitEvent(copy, comp_ready[b], 0));
      CUDA_CHECK(cudaMemcpyAsync(h_comp[b], d_comp[b], last_csize,
                                 cudaMemcpyDeviceToHost, copy));
      CUDA_CHECK(cudaEventRecord(copy_done[b], copy));
      std::swap(u, u2);
      std::swap(v, v2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (ms < best.ms_total) best.ms_total = ms;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
  }
  best.bytes_moved = (double)last_csize * o.nsteps;
  best.comp_ratio = (double)snap_bytes / (double)last_csize;
  for (int i = 0; i < 2; ++i) {
    cudaFree(d_comp[i]); cudaFreeHost(h_comp[i]);
    cudaEventDestroy(comp_ready[i]); cudaEventDestroy(copy_done[i]);
  }
  cudaStreamDestroy(comp); cudaStreamDestroy(copy);
  cudaFree(u); cudaFree(v); cudaFree(u2); cudaFree(v2);
  return best;
}

static void PrintRow(const char* name, const Opts& o, const Result& r) {
  const double gbps = (r.bytes_moved / 1e9) / (r.ms_total / 1e3);
  const double us_step = (r.ms_total * 1e3) / o.nsteps;
  std::printf("%-16s  %10.3f  %10.2f  %12.3f  %9.2fx  %12.3f\n", name,
              r.ms_total, us_step, gbps, r.comp_ratio, r.bytes_moved / 1e6);
}

int main(int argc, char** argv) {
  Opts o;
  if (argc > 1) o.nblocks = std::atoi(argv[1]);
  if (argc > 2) o.threads = std::atoi(argv[2]);
  if (argc > 3) o.per_block_bytes = std::atoi(argv[3]);
  if (argc > 4) o.nsteps = std::atoi(argv[4]);
  if (argc > 5) o.reps = std::atoi(argv[5]);

  // per_block_bytes must be a whole number of floats.
  if (o.per_block_bytes % (int)sizeof(float) != 0) {
    std::fprintf(stderr, "per_block_bytes must be a multiple of 4\n");
    return 1;
  }
  const int rows = o.nblocks;
  const int cols = o.per_block_bytes / (int)sizeof(float);
  const size_t snap_bytes = (size_t)rows * cols * sizeof(float);
  const int comp_block_bytes = 2 * (int)sizeof(float) + cols;  // header + int8s

  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("# GPU: %s (sm_%d%d), %.1f GB\n", prop.name, prop.major,
              prop.minor, prop.totalGlobalMem / 1e9);
#ifdef USE_CUSZP
  std::printf("# compressor: cuSZp (real, host API)\n");
#else
  std::printf("# compressor: stand-in int8 quantizer (cuSZp swaps in on Delta)\n");
#endif
  std::printf("# params: nblocks=%d threads=%d per_block_bytes=%d nsteps=%d reps=%d\n",
              o.nblocks, o.threads, o.per_block_bytes, o.nsteps, o.reps);
  std::printf("# snapshot/step = %.2f MB\n", snap_bytes / 1e6);
  std::printf("%-16s  %10s  %10s  %12s  %9s  %12s\n", "case", "ms_total",
              "us/step", "GB/s(eff)", "ratio", "MB_moved");

  PrintRow("raw", o, RunRaw(o, rows, cols, snap_bytes));
  PrintRow("async", o, RunAsync(o, rows, cols, snap_bytes));
  PrintRow("compressed", o,
           RunCompressed(o, rows, cols, snap_bytes, comp_block_bytes));
  PrintRow("async+comp", o,
           RunAsyncCompressed(o, rows, cols, snap_bytes, comp_block_bytes));
  return 0;
}
