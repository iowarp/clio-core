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
    cycles.txt Cycles crop-model season output: one row per simulated day (date, weather,
               crop growth, soil water and nitrogen columns), as the Cycles agroecosystem model writes
    sac.bin    Seismology deconvolved receiver functions: SAC files (632-byte header + float32
               samples of a band-limited, decaying waveform), as sG1IterDecon writes
Files that already exist are kept (the earlier ones are not regenerated).
"""

import os
import zlib
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


def cycles_txt(rng, days=365 * 40):
    out = ["YEAR  DOY  CROP           STAGE     THERMAL_TIME  CUM_BIOMASS  AG_BIOMASS  ROOT_BIOMASS  "
           "FRAC_INTERCEPT  TOTAL_N  AG_N  ROOT_N  N_STRESS  WATER_STRESS  POTENTIAL_TR  ACTUAL_TR  "
           "SOIL_EVAP  PRECIP  TMAX  TMIN  SOLAR_RAD\n"]
    bio = 0.0
    for d in range(days):
        year, doy = 1980 + d // 365, d % 365 + 1
        season = 0.5 + 0.5 * np.sin((doy - 100) / 365 * 2 * np.pi)
        growing = 120 <= doy <= 280
        bio = bio + max(0.0, rng.normal(0.08, 0.02)) * season if growing else 0.0
        out.append(f"{year}  {doy:3d}  {'CornRM.110' if growing else 'N/A':13s}  "
                   f"{'Vegetative' if growing and doy < 200 else 'Reproductive' if growing else 'N/A':9s}  "
                   f"{rng.uniform(0, 1800) if growing else 0:12.1f}  {bio:11.3f}  {0.8 * bio:10.3f}  "
                   f"{0.2 * bio:12.3f}  {min(1.0, bio / 8):14.3f}  {bio * 0.012:7.4f}  {bio * 0.009:5.4f}  "
                   f"{bio * 0.003:6.4f}  {rng.uniform(0, 0.3) if growing else 0:8.3f}  "
                   f"{rng.uniform(0, 0.4) if growing else 0:12.3f}  {6 * season:12.3f}  {5 * season:9.3f}  "
                   f"{rng.uniform(0, 2):9.3f}  {max(0, rng.normal(2.5, 6)):6.2f}  {12 + 18 * season + rng.normal(0, 3):5.1f}  "
                   f"{0 + 15 * season + rng.normal(0, 3):5.1f}  {8 + 20 * season + rng.normal(0, 2):10.2f}\n")
    return "".join(out).encode()


def sac_bin(rng, n_files=2000, npts=1000):
    out = []
    t = np.arange(npts) / 20.0                                   # 20 Hz receiver function
    for _ in range(n_files):
        hdr = np.zeros(158, dtype=np.float32).tobytes()           # 632-byte SAC header (70 floats,
        hdr = hdr[:632]                                           #  40 ints, 192 chars), zero-filled
        w = np.zeros(npts, dtype=np.float32)
        for _ in range(rng.integers(3, 7)):                       # Ps conversions + multiples
            t0, a, f = rng.uniform(1, 40), rng.normal(0, 0.3), rng.uniform(0.5, 2.0)
            w += (a * np.exp(-((t - t0) ** 2) * f * 4) * np.cos(2 * np.pi * f * (t - t0))).astype(np.float32)
        w += rng.normal(0, 0.01, npts).astype(np.float32)
        out.append(hdr + w.tobytes())
    return b"".join(out)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(2026)
    for name, fn in (("fits.bin", fits), ("fit.txt", fit_txt), ("fastq.txt", fastq), ("map.txt", map_txt),
                     ("vcf.txt", vcf), ("cycles.txt", cycles_txt), ("sac.bin", sac_bin)):
        if os.path.exists(os.path.join(out, name)):
            print(f"{name:10s} kept", flush=True)
            continue
        data = fn(np.random.default_rng(zlib.crc32(name.encode())) if name in ("cycles.txt", "sac.bin") else rng)
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
        print(f"{name:10s} {len(data) / MB:7.1f} MB", flush=True)


if __name__ == "__main__":
    main()
