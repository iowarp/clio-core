// Cross-decompression check: can native NeuroPress decode the payload Clio
// produced for the same chunk, and does it reconstruct the source exactly?
#include <gpucompress.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

static std::vector<char> Slurp(const char *p) {
  std::FILE *f = std::fopen(p, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(1); }
  std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  std::vector<char> v(n);
  if (std::fread(v.data(), 1, n, f) != (size_t)n) { std::exit(1); }
  std::fclose(f);
  return v;
}

int main(int argc, char **argv) {
  const char *weights = argv[1];
  const char *native_full = argv[2];   // 64B header + native payload
  const char *clio_payload = argv[3];  // Clio payload, no header
  const char *data = argv[4];
  long chunk = std::atol(argv[5]);
  const size_t CH = 4ull << 20;

  if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) return 1;

  std::vector<char> nf = Slurp(native_full);
  std::vector<char> cp = Slurp(clio_payload);
  std::vector<char> src = Slurp(data);
  const char *srcchunk = src.data() + chunk * CH;
  std::printf("native framed : %zu bytes (header 64 + payload %zu)\n",
              nf.size(), nf.size() - 64);
  std::printf("clio payload  : %zu bytes\n", cp.size());
  std::printf("payload sizes %s\n",
              (nf.size() - 64 == cp.size()) ? "MATCH" : "DIFFER");

  // Graft Clio's payload behind native's header. The header carries the
  // algorithm, sizes and preprocessing flags -- if it also carried a
  // payload checksum this would fail loudly rather than silently pass.
  std::vector<char> grafted(64 + cp.size());
  std::memcpy(grafted.data(), nf.data(), 64);
  std::memcpy(grafted.data() + 64, cp.data(), cp.size());

  void *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, grafted.size() > nf.size() ? grafted.size() : nf.size());
  cudaMalloc(&d_out, CH);

  struct Case { const char *name; std::vector<char> *buf; };
  Case cases[] = {{"native payload (control)", &nf},
                  {"clio payload via native header", &grafted}};

  int rc_all = 0;
  for (auto &c : cases) {
    cudaMemcpy(d_in, c.buf->data(), c.buf->size(), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, CH);
    size_t out = CH;
    gpucompress_error_t rc =
        gpucompress_decompress_gpu(d_in, c.buf->size(), d_out, &out, nullptr);
    if (rc != GPUCOMPRESS_SUCCESS) {
      std::printf("  %-32s DECOMPRESS FAILED rc=%d\n", c.name, (int)rc);
      rc_all = 1;
      continue;
    }
    std::vector<char> host(CH);
    cudaMemcpy(host.data(), d_out, CH, cudaMemcpyDeviceToHost);
    size_t bad = 0;
    for (size_t i = 0; i < CH; ++i) if (host[i] != srcchunk[i]) ++bad;
    std::printf("  %-32s out=%zu bytes, %zu of %zu bytes differ from source -> %s\n",
                c.name, out, bad, CH, bad == 0 ? "EXACT" : "MISMATCH");
    if (bad) rc_all = 1;
  }
  gpucompress_cleanup();
  return rc_all;
}
