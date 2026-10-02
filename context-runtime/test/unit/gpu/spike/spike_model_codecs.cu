// Sweep nvcomp 5.x GPU codecs over weight samples: ratio + throughput.
#include <nvcomp/nvcompManager.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/zstd.hpp>
#include <nvcomp/deflate.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/cascaded.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

#define CK(x) do { auto e=(x); if (e!=cudaSuccess) { \
  printf("cuda err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

using namespace nvcomp;
static const size_t kChunk = 64 * 1024;

int main(int argc, char **argv) {
  cudaStream_t st; CK(cudaStreamCreate(&st));
  for (int fi = 1; fi < argc; ++fi) {
    std::ifstream f(argv[fi], std::ios::binary | std::ios::ate);
    size_t n = (size_t) f.tellg(); f.seekg(0);
    std::vector<char> host(n); f.read(host.data(), n);
    uint8_t *d_in = nullptr, *d_out = nullptr;
    CK(cudaMalloc(&d_in, n)); CK(cudaMalloc(&d_out, n));
    CK(cudaMemcpy(d_in, host.data(), n, cudaMemcpyHostToDevice));
    const char *base = strrchr(argv[fi], '/'); base = base ? base+1 : argv[fi];

    struct Entry { const char *name; std::shared_ptr<nvcompManagerBase> mgr; };
    std::vector<Entry> mgrs;
    mgrs.push_back({"nv-lz4     ", std::make_shared<LZ4Manager>(kChunk, nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts, st)});
    mgrs.push_back({"nv-snappy  ", std::make_shared<SnappyManager>(kChunk, nvcompBatchedSnappyCompressDefaultOpts, nvcompBatchedSnappyDecompressDefaultOpts, st)});
    mgrs.push_back({"nv-zstd    ", std::make_shared<ZstdManager>(kChunk, nvcompBatchedZstdCompressDefaultOpts, nvcompBatchedZstdDecompressDefaultOpts, st)});
    mgrs.push_back({"nv-deflate ", std::make_shared<DeflateManager>(kChunk, nvcompBatchedDeflateCompressDefaultOpts, nvcompBatchedDeflateDecompressDefaultOpts, st)});
    mgrs.push_back({"nv-gdeflate", std::make_shared<GdeflateManager>(kChunk, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, st)});
    mgrs.push_back({"nv-ans     ", std::make_shared<ANSManager>(kChunk, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, st)});
    mgrs.push_back({"nv-bitcomp ", std::make_shared<BitcompManager>(kChunk, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, st)});
    mgrs.push_back({"nv-cascaded", std::make_shared<CascadedManager>(kChunk, nvcompBatchedCascadedCompressDefaultOpts, nvcompBatchedCascadedDecompressDefaultOpts, st)});

    for (auto &e : mgrs) {
      try {
        CompressionConfig cc = e.mgr->configure_compression(n);
        uint8_t *d_comp = nullptr;
        CK(cudaMalloc(&d_comp, cc.max_compressed_buffer_size));
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        e.mgr->compress(d_in, d_comp, cc);      // warm
        CK(cudaStreamSynchronize(st));
        cudaEventRecord(a, st);
        for (int r = 0; r < 3; ++r) e.mgr->compress(d_in, d_comp, cc);
        cudaEventRecord(b, st);
        CK(cudaStreamSynchronize(st));
        float cms = 0; cudaEventElapsedTime(&cms, a, b); cms /= 3;
        size_t csz = e.mgr->get_compressed_output_size(d_comp);
        DecompressionConfig dc = e.mgr->configure_decompression(d_comp);
        CK(cudaMemset(d_out, 0, n));
        e.mgr->decompress(d_out, d_comp, dc);   // warm
        CK(cudaStreamSynchronize(st));
        cudaEventRecord(a, st);
        for (int r = 0; r < 3; ++r) e.mgr->decompress(d_out, d_comp, dc);
        cudaEventRecord(b, st);
        CK(cudaStreamSynchronize(st));
        float dms = 0; cudaEventElapsedTime(&dms, a, b); dms /= 3;
        std::vector<char> back(n);
        CK(cudaMemcpy(back.data(), d_out, n, cudaMemcpyDeviceToHost));
        bool ok = memcmp(back.data(), host.data(), n) == 0;
        printf("%s %-14s ratio=%.4f comp=%7.1fMB/s decomp=%7.1fMB/s %s\n",
               e.name, base, (double) csz / n, n / (cms * 1e3),
               n / (dms * 1e3), ok ? "OK" : "MISMATCH");
        cudaFree(d_comp);
        cudaEventDestroy(a); cudaEventDestroy(b);
      } catch (const std::exception &ex) {
        printf("%s %-14s FAILED: %s\n", e.name, base, ex.what());
      }
      fflush(stdout);
    }
    cudaFree(d_in); cudaFree(d_out);
  }
  return 0;
}
