# Task: an out-of-core k-means benchmark on the CLIO GPU vector

Implement a GPU k-means (Lloyd's algorithm) benchmark whose point set is too
large for the GPU cache you are given. The point set must be stored in a
CLIO paged GPU vector (`gv::Vector<float>`, in
`context-transfer-engine/adapter/gpu_vector`), accessed from inside your GPU
kernels through its device-side API. Centroids, sums and counts are small and
may live in plain device memory.

Deliver a correct, working benchmark.

## The workload (must be reproduced exactly)

- `P = data_mb * 1048576 / (4 * dims)` points of `dims` floats each, stored
  point-major: element `i` of the vector is coordinate `i % dims` of point
  `i / dims`.
- Element `i` has the value `PointVal(i, dims, k)`:

```cpp
float PointVal(unsigned long long idx, unsigned dims, unsigned k) {
  const unsigned long long point = idx / dims;
  const unsigned long long dim = idx % dims;
  const unsigned long long cluster = point % k;
  const float centre = static_cast<float>(cluster) * 8.0f;
  const unsigned long long h =
      (point * 6364136223846793005ull + dim * 1442695040888963407ull);
  const float jitter =
      static_cast<float>(static_cast<unsigned>(h >> 40)) * (2.0f / 16777216.0f) - 1.0f;
  return centre + jitter;
}
```

- The data must be produced on the GPU into the vector (not on the host) and
  must survive being evicted from the GPU cache: later iterations must read the
  same values back.
- Initial centroids: the first `k` points (centroid `c` = point `c`).
- One iteration: assign every point to its nearest centroid by squared
  Euclidean distance over `dims` (ties go to the lowest centroid index); then
  each centroid becomes the mean of its assigned points (a centroid with no
  points keeps its position).
- Run `iters` iterations. Time only the iterations (not setup, runtime start,
  or data generation).

## Flags

| flag | default | meaning |
|---|---|---|
| `--data-mb N` | 256 | size of the point set in MB (MiB) |
| `--dims N` | 32 | coordinates per point |
| `--clusters N` | 16 | `k` |
| `--iters N` | 4 | iterations timed |
| `--cache-mb N` | 512 | the most GPU memory your vector's page cache may use, in MB. When it is smaller than `--data-mb`, the data must not all be resident on the GPU at once. |

The grader uses `dims=32`, `clusters=16`, `data-mb` from 128 to 512, and
`cache-mb` from 32 to 256.

The vector's backing storage (the storage targets your runtime
configuration gives the transfer engine) must be host memory only
(`bdev_type: ram`), sized to hold the whole data set: no GPU-memory storage
tier. The only GPU memory holding point data is the vector's page cache,
bounded by `--cache-mb`.

## RESULT keys

`RESULT counts=<n0>,<n1>,...,<n(k-1)> csum=<sum of all final centroid coordinates, %.6f> ms_per_iter=<mean ms per timed iteration> faults=<n> evicts=<n>`

`counts` are the per-cluster point counts of the final assignment step.
