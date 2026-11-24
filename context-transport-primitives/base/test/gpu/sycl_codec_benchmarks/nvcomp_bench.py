#!/usr/bin/env python3
# nvcomp LZ4 on the A100 -- FAIR harness: prebuilt compression/decompression
# configs (the default path re-runs configure_decompression + a stream sync on
# every call, which unfairly penalizes nvcomp and makes throughput fall with
# size) and reused output buffers. Reports host-to-host and GPU-resident.
import sys, time, numpy as np
from nvidia import nvcomp
ALGO = "LZ4"

def bench(path):
    data = np.fromfile(path, dtype=np.uint8)
    n = data.nbytes; gb = n / 1e9
    reps = max(8, min(30, int(4e9 // n)))
    codec = nvcomp.Codec(algorithm=ALGO)
    h_in = nvcomp.as_array(data); d_in = h_in.cuda()

    ccfg = codec.compression_config(n)                       # no sync
    comp = codec.encode(d_in, compression_config=ccfg); comp.cpu()
    comp_bytes = comp.buffer_size; ratio = n / comp_bytes
    dcfg = codec.decompression_config(comp)                  # ONE sync, outside timing
    dec = codec.decode(comp, decompression_config=dcfg); dec.cpu()
    h_comp = comp.cpu()

    def timed(fn):
        r = fn(); r.cpu()                                    # warm + sync
        t0 = time.perf_counter()
        for _ in range(reps): r = fn()
        r.cpu()                                              # sync once (stream-ordered)
        return (time.perf_counter() - t0) / reps

    # GPU-resident (kernel): input already on device, sync-free configs, reuse out
    c_gpu = timed(lambda: codec.encode(d_in, out=comp, compression_config=ccfg))
    d_gpu = timed(lambda: codec.decode(comp, out=dec, decompression_config=dcfg))
    # host-to-host (fair vs SYCL codec: H2D + kernel + D2H via .cpu())
    c_h2h = timed(lambda: codec.encode(h_in, compression_config=ccfg))
    d_h2h = timed(lambda: codec.decode(h_comp, decompression_config=dcfg))

    print(f"[nvcomp-{ALGO.lower()}] {path.split('/')[-1]}  size={n/1048576:.0f}MB  reps={reps}")
    print(f"  ratio                 = {ratio:.3f}x")
    print(f"  compress   host2host  = {gb/c_h2h:.2f} GB/s")
    print(f"  decompress host2host  = {gb/d_h2h:.2f} GB/s")
    print(f"  compress   GPU-only   = {gb/c_gpu:.2f} GB/s")
    print(f"  decompress GPU-only   = {gb/d_gpu:.2f} GB/s")

if __name__ == "__main__":
    if sys.argv[1] in ("LZ4","Snappy","ANS","GDeflate","Deflate","Zstd","Bitcomp"):
        ALGO = sys.argv[1]; files = sys.argv[2:]
    else:
        files = sys.argv[1:]
    for p in files:
        try: bench(p)
        except Exception as e:
            import traceback; traceback.print_exc()
