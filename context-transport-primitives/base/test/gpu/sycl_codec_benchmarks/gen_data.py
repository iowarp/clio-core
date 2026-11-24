#!/usr/bin/env python3
import sys, numpy as np
def base_block(nbytes, redundancy, seed):
    rng = np.random.default_rng(seed)
    words = [b"the ", b"quick ", b"brown ", b"fox ", b"GPU ", b"compress ",
             b"sycl ", b"lz4 ", b"data ", b"block "]
    out = bytearray()
    while len(out) < nbytes:
        if rng.random() < redundancy: out += words[rng.integers(0,10)]
        else: out += bytes([int(rng.integers(0,256))])
    return bytes(out[:nbytes])
if __name__ == "__main__":
    path, mb, red, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    target = mb*1024*1024
    blk = base_block(min(target, 16*1024*1024), red, seed)   # <=16MB base
    reps = (target + len(blk) - 1)//len(blk)
    data = (blk*reps)[:target]
    with open(path,"wb") as f: f.write(data)
    print(f"wrote {path}: {len(data)} bytes ({mb}MB) red={red}")
