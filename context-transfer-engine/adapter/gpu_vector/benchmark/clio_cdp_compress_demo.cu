// clio_cdp_compress_demo.cu
// ---------------------------------------------------------------------------
// PART 2 of the CDP proof: demonstrates that a custom, NON-cooperative
// compressor kernel CAN be launched from inside the Gray-Scott kernel via
// CUDA Dynamic Parallelism (CDP) -- answering the original PoC question
// ("can we compress from within a kernel?") with: YES, just not with cuSZp.
//
// Each Gray-Scott block, after computing its row, performs a __threadfence()
// and (thread 0) launches a child quantizer kernel over ITS OWN row. Because a
// block only compresses the row its own threads wrote (fenced before launch),
// there is no cross-block visibility hazard and no grid-wide cooperation -- the
// exact property cuSZp's fused kernel lacks.
//
// Requires separable compilation: -rdc=true (+ cudadevrt).
//   nvcc -O3 -rdc=true -gencode arch=compute_90,code=compute_90 \
//        -o cdp_demo clio_cdp_compress_demo.cu -lcudadevrt
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(_e));                                \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

__device__ unsigned long long g_child_launches = 0;

// CHILD kernel: compress one row of `cols` floats -> [float min][float scale][int8 codes].
__global__ void QuantChildKernel(const float* __restrict__ row,
                                 uint8_t* __restrict__ cblk, int cols) {
  extern __shared__ float s[];
  const int t = threadIdx.x, n = blockDim.x;
  float lmin = 3.4e38f, lmax = -3.4e38f;
  for (int c = t; c < cols; c += n) {
    const float x = row[c];
    lmin = x < lmin ? x : lmin;
    lmax = x > lmax ? x : lmax;
  }
  s[t] = lmin; __syncthreads();
  for (int d = n / 2; d > 0; d >>= 1) {
    if (t < d) s[t] = fminf(s[t], s[t + d]);
    __syncthreads();
  }
  const float bmin = s[0]; __syncthreads();
  s[t] = lmax; __syncthreads();
  for (int d = n / 2; d > 0; d >>= 1) {
    if (t < d) s[t] = fmaxf(s[t], s[t + d]);
    __syncthreads();
  }
  const float scale = (s[0] - bmin) > 0.0f ? (s[0] - bmin) / 255.0f : 1.0f;
  if (t == 0) {
    const uint8_t* pm = reinterpret_cast<const uint8_t*>(&bmin);
    const uint8_t* ps = reinterpret_cast<const uint8_t*>(&scale);
    for (int i = 0; i < 4; ++i) { cblk[i] = pm[i]; cblk[i + 4] = ps[i]; }
    atomicAdd(&g_child_launches, 1ULL);
  }
  __syncthreads();
  uint8_t* codes = cblk + 8;
  for (int c = t; c < cols; c += n) {
    int q = (int)((row[c] - bmin) / scale + 0.5f);
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    codes[c] = (uint8_t)q;
  }
}

// PARENT kernel: one Gray-Scott step for block's row, then CDP-launch the child
// compressor over that row.
__global__ void GsThenCdpCompressKernel(const float* __restrict__ u,
                                        const float* __restrict__ v,
                                        float* __restrict__ v2,
                                        uint8_t* __restrict__ comp, int rows,
                                        int cols, int cbb, float Du, float Dv,
                                        float F, float k, float dt) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int idx = row * cols + col;
    const int up = (row > 0 ? row - 1 : row) * cols + col;
    const int dn = (row < rows - 1 ? row + 1 : row) * cols + col;
    const int lf = row * cols + (col > 0 ? col - 1 : col);
    const int rt = row * cols + (col < cols - 1 ? col + 1 : col);
    const float uc = u[idx], vc = v[idx];
    const float lap_v = v[up] + v[dn] + v[lf] + v[rt] - 4.0f * vc;
    const float uvv = uc * vc * vc;
    v2[idx] = vc + (Dv * lap_v + uvv - (F + k) * vc) * dt;
  }
  __syncthreads();
  __threadfence();  // make this block's writes to its row visible to the child
  if (threadIdx.x == 0) {
    // >>> CDP: a kernel launching the compressor kernel from device code. <<<
    QuantChildKernel<<<1, 128, 128 * sizeof(float)>>>(
        v2 + (size_t)row * cols, comp + (size_t)row * cbb, cols);
  }
}

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 256;
  const int cols = argc > 2 ? std::atoi(argv[2]) : 1024;
  const int cbb = 8 + cols;  // compressed bytes per row
  const size_t n = (size_t)rows * cols;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("# GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("# rows=%d cols=%d  (parent grid: %d blocks, each CDP-launches a child)\n",
              rows, cols, rows);

  std::vector<float> h_u(n), h_v(n);
  for (size_t i = 0; i < n; ++i) {
    h_u[i] = 1.0f - 0.5f * std::sin(0.01f * i);
    h_v[i] = 0.25f + 0.2f * std::cos(0.013f * i);
  }
  float *u, *v, *v2;
  uint8_t* comp;
  CUDA_CHECK(cudaMalloc(&u, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&v, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&v2, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&comp, (size_t)rows * cbb));
  CUDA_CHECK(cudaMemcpy(u, h_u.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v, h_v.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const unsigned long long zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_child_launches, &zero, sizeof(zero)));

  GsThenCdpCompressKernel<<<rows, 128>>>(u, v, v2, comp, rows, cols, cbb, 0.16f,
                                         0.08f, 0.060f, 0.062f, 1.0f);
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::fprintf(stderr, "parent launch error: %s\n",
                 cudaGetErrorString(launch_err));
    return 1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long child_count = 0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&child_count, g_child_launches,
                                  sizeof(child_count)));

  // Verify the round-trip: decompress on host and check the error bound.
  std::vector<uint8_t> h_comp((size_t)rows * cbb);
  std::vector<float> h_v2(n);
  CUDA_CHECK(cudaMemcpy(h_comp.data(), comp, (size_t)rows * cbb,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_v2.data(), v2, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double max_err = 0;
  for (int r = 0; r < rows; ++r) {
    const uint8_t* blk = h_comp.data() + (size_t)r * cbb;
    float bmin, scale;
    std::memcpy(&bmin, blk, 4);
    std::memcpy(&scale, blk + 4, 4);
    const uint8_t* codes = blk + 8;
    for (int c = 0; c < cols; ++c) {
      const float recon = bmin + codes[c] * scale;
      const double e = std::fabs(recon - h_v2[(size_t)r * cols + c]);
      if (e > max_err) max_err = e;
    }
  }

  const double ratio = (double)(n * sizeof(float)) / ((double)rows * cbb);
  std::printf("CDP child-launches executed : %llu  (expected %d)\n", child_count,
              rows);
  std::printf("compression ratio           : %.2fx\n", ratio);
  std::printf("max abs reconstruction error: %.6f\n", max_err);
  std::printf("RESULT: %s\n",
              (child_count == (unsigned long long)rows)
                  ? "CDP compress-from-within-kernel WORKS"
                  : "FAILED (child did not run)");
  return child_count == (unsigned long long)rows ? 0 : 1;
}
