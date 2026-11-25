"""Stand-in data for the WfCommons phases (third application), in each workflow's real file format.

    python3 wf_data.py [out_dir]        (once; ~1 min; written next to the Gray-Scott campaign)

The phases' sizes, counts and reuse come from WfCommons execution traces (WfInstances, Pegasus on
Chameleon; see WF_TRACES in workload.py). The traces record file sizes, not contents, so the bytes
are synthesized in the formats those workflows really write, with realistic statistics:
    fits.bin   Montage reprojected images: big-endian float32 tiles, sky background + noise +
               point sources (stars), as in 2MASS FITS data
    fit.txt    Montage mDiffFit output: per-pair plane-fit parameter records (text)
    fastq.txt  Epigenomics reads: FASTQ records (100 bp, Illumina 1.5 quality strings)
    map.txt    Epigenomics alignments (MAQ-style map text: read, chromosome, position, strand, ...)
    vcf.txt    1000Genome chromosome VCF: one variant per line, genotypes of 2,504 samples,
               mostly 0|0 (rare alleles), as in the 1000 Genomes phase 3 release
"""

import os
import sys

import numpy as np

MB = 1 << 20
DEFAULT_DIR = os.environ.get("GS_CAMPAIGN", "/ssc-bench/gs_campaign")


def fits(rng, n_tiles=48, side=512):
    out = []
    yy, xx = np.mgrid[0:side, 0:side]
    for _ in range(n_tiles):
        img = rng.normal(400.0, 12.0, (side, side)).astype(np.float32)      # sky + read noise
        for _ in range(300):                                                  # stars, Gaussian PSF
            cx, cy, flux, sig = rng.uniform(0, side), rng.uniform(0, side), rng.lognormal(7, 1.2), rng.uniform(1.0, 2.2)
            x0, x1, y0, y1 = int(max(cx - 8, 0)), int(min(cx + 8, side)), int(max(cy - 8, 0)), int(min(cy + 8, side))
            sub = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2
            img[y0:y1, x0:x1] += (flux * np.exp(-sub / (2 * sig * sig))).astype(np.float32)
        out.append(img.astype(">f4").tobytes())
    return b"".join(out)


def fit_txt(rng, n=200000):
    lines = []
    for i in range(n):
        a, b, c = rng.normal(0, 1e-3), rng.normal(0, 1e-3), rng.normal(0, 5)
        lines.append(f"plus=p2mass-atlas-{i:06d}.fits minus=p2mass-atlas-{i + 1:06d}.fits "
                     f"a={a:.6e} b={b:.6e} c={c:.6e} crpix1={rng.uniform(0, 900):.2f} "
                     f"crpix2={rng.uniform(0, 900):.2f} xmin={rng.integers(-500, 0)} xmax={rng.integers(0, 500)} "
                     f"ymin={rng.integers(-500, 0)} ymax={rng.integers(0, 500)} xcenter={rng.uniform(-5, 5):.4f} "
                     f"ycenter={rng.uniform(-5, 5):.4f} npixel={rng.integers(10000, 400000)} "
                     f"rms={abs(rng.normal(3, 1)):.6f} boxx=0 boxy=0 boxwidth=0 boxheight=0 boxang=0\n")
    return "".join(lines).encode()


QUAL = np.frombuffer(b"BCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefgh", dtype=np.uint8)   # Illumina 1.5 (Phred+64)


def fastq(rng, n=600000, bp=100):
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    seq = bases[rng.integers(0, 4, (n, bp))]
    # quality decays along the read with noise (what real Illumina runs show)
    q = np.clip(38 - np.arange(bp) / 6 + rng.normal(0, 3, (n, bp)), 0, len(QUAL) - 1).astype(int)
    qual = QUAL[q]
    out = []
    for i in range(n):
        h = f"@HWI-EAS{rng.integers(100, 999)}:4:{rng.integers(1, 120)}:{rng.integers(0, 2000)}:{rng.integers(0, 2000)}#0/1"
        out.append(f"{h}\n{seq[i].tobytes().decode()}\n+\n{qual[i].tobytes().decode()}\n")
    return "".join(out).encode()


def map_txt(rng, n=600000, bp=100):
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    pos = np.sort(rng.integers(9_400_000, 48_100_000, n))       # chr21, sorted by position
    out = []
    for i in range(n):
        s = bases[rng.integers(0, 4, bp)].tobytes().decode()
        q = QUAL[np.clip(38 - np.arange(bp) / 6 + rng.normal(0, 3, bp), 0, len(QUAL) - 1).astype(int)].tobytes().decode()
        out.append(f"HWI-EAS{rng.integers(100, 999)}:4:{rng.integers(1, 120)}:{rng.integers(0, 2000)}:"
                   f"{rng.integers(0, 2000)}\tchr21\t{pos[i]}\t{'+-'[rng.integers(0, 2)]}\t0\t0\t{rng.integers(0, 60)}\t"
                   f"{rng.integers(0, 60)}\t{rng.integers(0, 60)}\t{rng.integers(0, 3)}\t{rng.integers(0, 4)}\t"
                   f"0\t0\t{bp}\t{s}\t{q}\n")
    return "".join(out).encode()


def vcf(rng, n=12000, samples=2504):
    out = ["##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
           + "\t".join(f"HG{i:05d}" for i in range(samples)) + "\n"]
    pos = np.sort(rng.integers(9_411_000, 48_119_000, n))
    gts = np.array(["0|0", "0|1", "1|0", "1|1"])
    for i in range(n):
        af = min(rng.beta(0.3, 8.0), 0.95)                   # mostly rare variants
        a = rng.random((samples, 2)) < af
        g = gts[a[:, 0] * 2 + a[:, 1]]
        ref, alt = "ACGT"[rng.integers(0, 4)], "ACGT"[rng.integers(0, 4)]
        out.append(f"21\t{pos[i]}\trs{rng.integers(1, 10**9)}\t{ref}\t{alt}\t100\tPASS\t"
                   f"AC={a.sum()};AF={af:.6f};AN={2 * samples};NS={samples};DP={rng.integers(10000, 30000)}\tGT\t"
                   + "\t".join(g) + "\n")
    return "".join(out).encode()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(2026)
    for name, fn in (("fits.bin", fits), ("fit.txt", fit_txt), ("fastq.txt", fastq), ("map.txt", map_txt),
                     ("vcf.txt", vcf)):
        data = fn(rng)
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
        print(f"{name:10s} {len(data) / MB:7.1f} MB", flush=True)


if __name__ == "__main__":
    main()
