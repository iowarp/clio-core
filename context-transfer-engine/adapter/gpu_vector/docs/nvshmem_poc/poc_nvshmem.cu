#include <cstdio>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void PutToPeer(int *dst, int myid, int peer) {
  nvshmem_int_p(dst, myid, peer);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  cudaSetDevice(0);
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;
  int expect = (mype + npes - 1) % npes;
  int *dst = (int *)nvshmem_malloc(sizeof(int));
  int init = -1;
  cudaMemcpy(dst, &init, sizeof(int), cudaMemcpyHostToDevice);
  nvshmem_barrier_all();
  PutToPeer<<<1, 1>>>(dst, mype, peer);
  cudaDeviceSynchronize();
  nvshmem_barrier_all();
  int got = -2;
  cudaMemcpy(&got, dst, sizeof(int), cudaMemcpyDeviceToHost);
  std::printf("[NVSHMEM] PE %d/%d on GPU0: got %d from peer (expected %d)  %s\n",
              mype, npes, got, expect, got == expect ? "OK" : "FAIL");
  nvshmem_free(dst);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
