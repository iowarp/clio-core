/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * MLP training, MPI edition: MODEL-parallel, rank <-> the paged bench's
 * block. A rank owns h-rows of W1 and o-rows of W2 in plain device memory;
 * per step it computes its a1 rows (allgathered), its z2/d2 rows
 * (allgathered), its o-partial of the backprop sum (allgathered and
 * combined IN RANK ORDER, which preserves the paged bench's page-blocked
 * float ordering), and updates its own shards in place. One thread per
 * output element with fixed-order sums throughout, so the per-step LOSS and
 * the final WEIGHT DIGEST must be BIT-EQUAL to the in-process dense
 * reference at any rank count. Links nothing from clio.
 *
 * Run recipe: mpirun -n 4 clio_lbann_mpi_bench --hidden 4096 --steps 5
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define LB_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                        \
  } while (0)

using u32 = unsigned int;
using u64 = unsigned long long;

__host__ __device__ inline u64 Lcg(u64 s) {
  return s * 6364136223846793005ull + 1442695040888963407ull;
}
__host__ __device__ inline float Sym01(u64 s) {
  return (static_cast<float>((s >> 40) & 0xFFFFFF) / 8388608.0f) - 1.0f;
}

/** Weight/batch seeding and every training loop below are the paged
 *  bench's, verbatim: one thread per output element, fixed-order sums, no
 *  atomics -- so ANY decomposition of the elements produces bit-identical
 *  floats, and the gates can demand equality. */
__global__ void SeedW(float *w, u64 n) {
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    w[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + i)) * 0.05f;
  }
}

/** a1[h,b] over h in [h0,h1): w1 points at row h0 of W1, b1 at bias h0. */
__global__ void Fwd1(const float *w1, const float *b1, u64 h0, u64 h1, u64 I,
                     u64 B, const float *x, float *a1) {
  const u64 nout = (h1 - h0) * B;
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < nout;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 h = h0 + t / B;
    const u64 b = t % B;
    float acc = b1[h - h0];
    for (u64 i = 0; i < I; ++i) {
      acc += w1[(h - h0) * I + i] * x[b * I + i];
    }
    a1[h * B + b] = acc > 0.0f ? acc : 0.0f;
  }
}

__global__ void Fwd2(const float *w2, const float *b2, u64 o0, u64 o1, u64 H,
                     u64 O, u64 B, const float *a1, const float *y, float *d2,
                     double *loss_parts) {
  const u64 nout = (o1 - o0) * B;
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < nout;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 o = o0 + t / B;
    const u64 b = t % B;
    float acc = b2[o - o0];
    for (u64 h = 0; h < H; ++h) {
      acc += w2[(o - o0) * H + h] * a1[h * B + b];
    }
    const float diff = acc - y[b * O + o];
    d2[o * B + b] = 2.0f * diff / static_cast<float>(B * O);
    loss_parts[o * B + b] =
        static_cast<double>(diff) * static_cast<double>(diff);
  }
}

/** This rank's o-rows' PARTIAL of d1, page-blocked in o exactly like the
 *  paged bwd1 so the float order matches when partials are combined in
 *  ascending rank (= ascending o) order. */
__global__ void Bwd1Partial(const float *w2, u64 o0, u64 o1, u64 H, u64 B,
                            const float *d2, float *d1_part, u64 rpp) {
  const u64 nout = H * B;
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < nout;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 h = t / B;
    const u64 b = t % B;
    float acc = 0.0f;
    for (u64 op = o0; op < o1; op += rpp) {
      const u64 oend = (op + rpp < o1) ? op + rpp : o1;
      float blk = 0.0f;
      for (u64 o = op; o < oend; ++o) {
        blk += w2[(o - o0) * H + h] * d2[o * B + b];
      }
      acc += blk;
    }
    d1_part[h * B + b] = acc;
  }
}

/** Combine rank partials IN RANK ORDER (deterministic), apply relu mask. */
__global__ void Bwd1Combine(const float *parts, u64 nranks, u64 H, u64 B,
                            const float *a1, float *d1) {
  const u64 nout = H * B;
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < nout;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    float acc = 0.0f;
    for (u64 r = 0; r < nranks; ++r) acc += parts[r * nout + t];
    d1[t] = (a1[t] <= 0.0f) ? 0.0f : acc;
  }
}

__global__ void Upd2(float *w2, float *b2, u64 o0, u64 o1, u64 H, u64 B,
                     const float *a1, const float *d2, float lr) {
  const u64 nw = (o1 - o0) * H;
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < nw;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 o = o0 + t / H;
    const u64 h = t % H;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[o * B + b] * a1[h * B + b];
    w2[(o - o0) * H + h] -= lr * g;
  }
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < (o1 - o0);
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 o = o0 + t;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[o * B + b];
    b2[o - o0] -= lr * g;
  }
}

__global__ void Upd1(float *w1, float *b1, u64 h0, u64 h1, u64 I, u64 B,
                     const float *x, const float *d1, float lr) {
  const u64 nw = (h1 - h0) * I;
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < nw;
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 h = h0 + t / I;
    const u64 i = t % I;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[h * B + b] * x[b * I + i];
    w1[(h - h0) * I + i] -= lr * g;
  }
  for (u64 t = blockIdx.x * blockDim.x + threadIdx.x; t < (h1 - h0);
       t += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 h = h0 + t;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[h * B + b];
    b1[h - h0] -= lr * g;
  }
}

__global__ void Digest(const float *w, u64 gbase, u64 n,
                       unsigned long long *out) {
  unsigned long long acc = 0;
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    acc += static_cast<unsigned long long>(__float_as_uint(w[i])) *
           (2ull * (gbase + i) + 1ull);
  }
  atomicAdd(out, acc);
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  {
    int ndev = 0;
    LB_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    LB_CUDA_CHECK(cudaSetDevice(ndev > 0 ? rank % ndev : 0));
  }
  u32 blocks = 8, threads = 256;
  u64 I = 256, H = 4096, O = 64, B = 64, steps = 5, rpp = 16;
  float lr = 0.01f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--in") I = next();
    else if (a == "--hidden") H = next();
    else if (a == "--out") O = next();
    else if (a == "--batch") B = next();
    else if (a == "--steps") steps = next();
    else if (a == "--rpp") rpp = next();
    else if (a == "--lr" && i + 1 < argc) lr = std::strtof(argv[++i], nullptr);
  }
  if (H % nranks != 0 || O % nranks != 0) {
    if (rank == 0) std::fprintf(stderr, "LBANN MPI: H and O must divide the "
                                "rank count\n");
    MPI_Finalize();
    return 2;
  }
  const u64 hper = H / nranks, oper = O / nranks;
  const u64 h0 = rank * hper, h1 = h0 + hper;
  const u64 o0 = rank * oper, o1 = o0 + oper;

  if (rank == 0) {
    std::printf("MLP training, MPI edition: %llu -> %llu -> %llu, batch=%llu"
                ", steps=%llu, %d ranks (model-parallel)\n",
                (unsigned long long)I, (unsigned long long)H,
                (unsigned long long)O, (unsigned long long)B,
                (unsigned long long)steps, nranks);
  }

  // Batch and targets, deterministic and replicated.
  std::vector<float> hxv(B * I), hyv(B * O);
  for (u64 i = 0; i < B * I; ++i) hxv[i] = Sym01(Lcg(0xA02BDBF7BB3C0A7ull + i));
  for (u64 i = 0; i < B * O; ++i) hyv[i] = Sym01(Lcg(0x6C62272E07BB0142ull + i));
  float *d_x, *d_y;
  LB_CUDA_CHECK(cudaMalloc(&d_x, B * I * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_y, B * O * sizeof(float)));
  LB_CUDA_CHECK(cudaMemcpy(d_x, hxv.data(), B * I * sizeof(float),
                           cudaMemcpyHostToDevice));
  LB_CUDA_CHECK(cudaMemcpy(d_y, hyv.data(), B * O * sizeof(float),
                           cudaMemcpyHostToDevice));

  // Shards. The GLOBAL seeding index keeps every substrate's initial
  // weights identical: W1 rows live at [h*I], b1 at w1_n + h (padded
  // layout matching the paged bench's page-aligned offsets is NOT needed
  // here -- the seed function is indexed by LOGICAL element, so the seeds
  // must use the same logical ids as the dense reference below).
  const u64 w1_n = H * I, w2_n = O * H;
  float *d_w1, *d_b1, *d_w2, *d_b2;
  LB_CUDA_CHECK(cudaMalloc(&d_w1, hper * I * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_b1, hper * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_w2, oper * H * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_b2, oper * sizeof(float)));
  {
    // Seed on host straight from the logical-id generator; shard layouts
    // differ from the dense buffer but values must not.
    std::vector<float> t(hper * I);
    for (u64 h = h0; h < h1; ++h) {
      for (u64 i = 0; i < I; ++i) {
        t[(h - h0) * I + i] =
            Sym01(Lcg(0xB5297A4D3F84D5B5ull + h * I + i)) * 0.05f;
      }
    }
    LB_CUDA_CHECK(cudaMemcpy(d_w1, t.data(), t.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    std::vector<float> tb(hper);
    for (u64 h = h0; h < h1; ++h) {
      tb[h - h0] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + h)) * 0.05f;
    }
    LB_CUDA_CHECK(cudaMemcpy(d_b1, tb.data(), tb.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    std::vector<float> t2(oper * H);
    for (u64 o = o0; o < o1; ++o) {
      for (u64 h = 0; h < H; ++h) {
        t2[(o - o0) * H + h] =
            Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + H + o * H + h)) * 0.05f;
      }
    }
    LB_CUDA_CHECK(cudaMemcpy(d_w2, t2.data(), t2.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    std::vector<float> tb2(oper);
    for (u64 o = o0; o < o1; ++o) {
      tb2[o - o0] =
          Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + H + w2_n + o)) * 0.05f;
    }
    LB_CUDA_CHECK(cudaMemcpy(d_b2, tb2.data(), tb2.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
  }

  float *d_a1, *d_d1, *d_d2, *d_d1p;
  double *d_lp;
  LB_CUDA_CHECK(cudaMalloc(&d_a1, H * B * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_d1, H * B * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_d2, O * B * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_d1p, nranks * H * B * sizeof(float)));
  LB_CUDA_CHECK(cudaMalloc(&d_lp, O * B * sizeof(double)));

  std::vector<float> h_gath(H * B);
  std::vector<float> h_gath_o(O * B);
  std::vector<double> h_lp(O * B);
  std::vector<float> h_d1p(static_cast<size_t>(nranks) * H * B);
  std::vector<double> loss(steps);

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    // fwd1 own rows -> allgather a1.
    Fwd1<<<blocks, threads>>>(d_w1, d_b1, h0, h1, I, B, d_x, d_a1);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    LB_CUDA_CHECK(cudaMemcpy(h_gath.data() + h0 * B, d_a1 + h0 * B,
                             hper * B * sizeof(float),
                             cudaMemcpyDeviceToHost));
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_gath.data(),
                  static_cast<int>(hper * B), MPI_FLOAT, MPI_COMM_WORLD);
    LB_CUDA_CHECK(cudaMemcpy(d_a1, h_gath.data(), H * B * sizeof(float),
                             cudaMemcpyHostToDevice));
    // fwd2 own rows -> allgather d2 and the loss parts.
    Fwd2<<<blocks, threads>>>(d_w2, d_b2, o0, o1, H, O, B, d_a1, d_y, d_d2,
                              d_lp);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    LB_CUDA_CHECK(cudaMemcpy(h_gath_o.data() + o0 * B, d_d2 + o0 * B,
                             oper * B * sizeof(float),
                             cudaMemcpyDeviceToHost));
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_gath_o.data(),
                  static_cast<int>(oper * B), MPI_FLOAT, MPI_COMM_WORLD);
    LB_CUDA_CHECK(cudaMemcpy(d_d2, h_gath_o.data(), O * B * sizeof(float),
                             cudaMemcpyHostToDevice));
    std::vector<double> lp_own(oper * B);
    LB_CUDA_CHECK(cudaMemcpy(lp_own.data(),
                             d_lp + o0 * B, oper * B * sizeof(double),
                             cudaMemcpyDeviceToHost));
    std::memcpy(h_lp.data() + o0 * B, lp_own.data(),
                oper * B * sizeof(double));
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_lp.data(),
                  static_cast<int>(oper * B), MPI_DOUBLE, MPI_COMM_WORLD);
    double l = 0.0;
    for (u64 i = 0; i < O * B; ++i) l += h_lp[i];
    loss[s] = l / static_cast<double>(B * O);
    // bwd1: own o-partial -> allgather -> combine IN RANK ORDER.
    Bwd1Partial<<<blocks, threads>>>(d_w2, o0, o1, H, B, d_d2,
                                     d_d1p + rank * H * B, rpp);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    LB_CUDA_CHECK(cudaMemcpy(h_d1p.data() + rank * H * B,
                             d_d1p + rank * H * B, H * B * sizeof(float),
                             cudaMemcpyDeviceToHost));
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_d1p.data(),
                  static_cast<int>(H * B), MPI_FLOAT, MPI_COMM_WORLD);
    LB_CUDA_CHECK(cudaMemcpy(d_d1p, h_d1p.data(),
                             h_d1p.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    Bwd1Combine<<<blocks, threads>>>(d_d1p, nranks, H, B, d_a1, d_d1);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    // updates on own shards (d2/a1/d1 are full everywhere).
    Upd2<<<blocks, threads>>>(d_w2, d_b2, o0, o1, H, B, d_a1, d_d2, lr);
    Upd1<<<blocks, threads>>>(d_w1, d_b1, h0, h1, I, B, d_x, d_d1, lr);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
  }
  const double ms = NowMs() - t0;

  // Weight digest over the LOGICAL ids (order-independent integer sum).
  unsigned long long *d_dg;
  LB_CUDA_CHECK(cudaMalloc(&d_dg, sizeof(unsigned long long)));
  LB_CUDA_CHECK(cudaMemset(d_dg, 0, sizeof(unsigned long long)));
  Digest<<<64, 256>>>(d_w1, h0 * I, hper * I, d_dg);
  Digest<<<64, 256>>>(d_b1, w1_n + h0, hper, d_dg);
  Digest<<<64, 256>>>(d_w2, w1_n + H + o0 * H, oper * H, d_dg);
  Digest<<<64, 256>>>(d_b2, w1_n + H + w2_n + o0, oper, d_dg);
  LB_CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long dg_loc = 0, dg = 0;
  LB_CUDA_CHECK(cudaMemcpy(&dg_loc, d_dg, sizeof(dg_loc),
                           cudaMemcpyDeviceToHost));
  MPI_Allreduce(&dg_loc, &dg, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  int rc = 0;
  if (rank == 0) {
    // Dense in-process reference: the same kernels run single-shard.
    float *r_w1, *r_b1, *r_w2, *r_b2;
    LB_CUDA_CHECK(cudaMalloc(&r_w1, w1_n * sizeof(float)));
    LB_CUDA_CHECK(cudaMalloc(&r_b1, H * sizeof(float)));
    LB_CUDA_CHECK(cudaMalloc(&r_w2, w2_n * sizeof(float)));
    LB_CUDA_CHECK(cudaMalloc(&r_b2, O * sizeof(float)));
    SeedW<<<64, 256>>>(r_w1, w1_n);
    {
      std::vector<float> t(H);
      for (u64 h = 0; h < H; ++h)
        t[h] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + h)) * 0.05f;
      LB_CUDA_CHECK(cudaMemcpy(r_b1, t.data(), H * sizeof(float),
                               cudaMemcpyHostToDevice));
      std::vector<float> t2(w2_n);
      for (u64 i = 0; i < w2_n; ++i)
        t2[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + H + i)) * 0.05f;
      LB_CUDA_CHECK(cudaMemcpy(r_w2, t2.data(), w2_n * sizeof(float),
                               cudaMemcpyHostToDevice));
      std::vector<float> t3(O);
      for (u64 o = 0; o < O; ++o)
        t3[o] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + w1_n + H + w2_n + o)) *
                0.05f;
      LB_CUDA_CHECK(cudaMemcpy(r_b2, t3.data(), O * sizeof(float),
                               cudaMemcpyHostToDevice));
    }
    std::vector<double> loss_ref(steps);
    for (u64 s = 0; s < steps; ++s) {
      Fwd1<<<blocks, threads>>>(r_w1, r_b1, 0, H, I, B, d_x, d_a1);
      LB_CUDA_CHECK(cudaDeviceSynchronize());
      Fwd2<<<blocks, threads>>>(r_w2, r_b2, 0, O, H, O, B, d_a1, d_y, d_d2,
                                d_lp);
      LB_CUDA_CHECK(cudaDeviceSynchronize());
      LB_CUDA_CHECK(cudaMemcpy(h_lp.data(), d_lp, O * B * sizeof(double),
                               cudaMemcpyDeviceToHost));
      double l = 0.0;
      for (u64 i = 0; i < O * B; ++i) l += h_lp[i];
      loss_ref[s] = l / static_cast<double>(B * O);
      // SAME ASSOCIATION AS THE DISTRIBUTED PATH: the reference computes
      // one partial per rank-range and combines them in rank order --
      // (blk0+blk1)+(blk2+blk3) is not ((blk0+blk1)+blk2)+blk3 in floats,
      // and that one reassociation was the whole 1e-10-per-step drift.
      for (int r = 0; r < nranks; ++r) {
        Bwd1Partial<<<blocks, threads>>>(r_w2 + static_cast<u64>(r) * oper * H,
                                         static_cast<u64>(r) * oper,
                                         static_cast<u64>(r + 1) * oper, H, B,
                                         d_d2, d_d1p + static_cast<u64>(r) *
                                         H * B, rpp);
      }
      LB_CUDA_CHECK(cudaDeviceSynchronize());
      Bwd1Combine<<<blocks, threads>>>(d_d1p, nranks, H, B, d_a1, d_d1);
      Upd2<<<blocks, threads>>>(r_w2, r_b2, 0, O, H, B, d_a1, d_d2, lr);
      Upd1<<<blocks, threads>>>(r_w1, r_b1, 0, H, I, B, d_x, d_d1, lr);
      LB_CUDA_CHECK(cudaDeviceSynchronize());
    }
    LB_CUDA_CHECK(cudaMemset(d_dg, 0, sizeof(unsigned long long)));
    Digest<<<64, 256>>>(r_w1, 0, w1_n, d_dg);
    Digest<<<64, 256>>>(r_b1, w1_n, H, d_dg);
    Digest<<<64, 256>>>(r_w2, w1_n + H, w2_n, d_dg);
    Digest<<<64, 256>>>(r_b2, w1_n + H + w2_n, O, d_dg);
    LB_CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long dg_ref = 0;
    LB_CUDA_CHECK(cudaMemcpy(&dg_ref, d_dg, sizeof(dg_ref),
                             cudaMemcpyDeviceToHost));

    std::printf("  %llu steps in %.1f ms\n", (unsigned long long)steps, ms);
    bool ok = true;
    for (u64 s = 0; s < steps; ++s) {
      if (loss[s] != loss_ref[s]) {
        std::printf("  LOSS GATE: step %llu %.17g != %.17g\n",
                    (unsigned long long)s, loss[s], loss_ref[s]);
        ok = false;
      }
    }
    if (ok) {
      std::printf("  LOSS GATE: PASS (all steps bit-equal; %.6f -> %.6f)\n",
                  loss_ref[0], loss_ref[steps - 1]);
    } else {
      rc = 1;
    }
    if (dg != dg_ref) {
      std::printf("  WEIGHT GATE: FAIL (digest %llu != %llu)\n", dg, dg_ref);
      rc = 1;
    } else {
      std::printf("  WEIGHT GATE: PASS (bit-equal to dense reference)\n");
    }
    std::printf("%s\n", rc == 0 ? "LBANN MPI: ALL GATES PASS"
                                : "LBANN MPI: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return rc;
}
