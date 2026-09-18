#!/usr/bin/env python3
"""Per-chunk distribution class, HCompress's only data-dependent input.

    classify_chunks.py --explore E.csv --raw DIR              --out dist.csv
    classify_chunks.py --explore E.csv --fields DIR [--ext .f32] --out dist.csv
    classify_chunks.py --explore E.csv --label-from-frame     --out dist.csv

HCompress (Devarajan et al., IPDPS 2020, Sec. IV-D) deduces a data
distribution CLASS -- Normal, Gamma, Exponential or Uniform -- "by
sub-sampling the buffer", and feeds that class, not any per-chunk statistic,
to its cost predictor. This script reproduces that step for chunks a Clio
exploration run already measured, so the baseline can be scored on exactly the
chunks our own model was scored on.

The classifier is a line-by-line port of the repo's own
clio::cte::compressor::DistributionClassifier (models/distribution_classifier.h):
moments -> 64-bin histogram -> chi-squared-like fit against each family ->
moment-based tie-break. Porting rather than inventing one matters: the class
boundaries are a modelling choice, and a second set of them would make the
baseline's accuracy an artefact of this script.

Two things the port keeps that a summary would lose:

  * CONSTANT is a fifth outcome, returned when the variance underflows. It is
    NOT one of HCompress's four classes; it is reported as-is rather than
    folded into `uniform`, and the seeding step decides what to do with it.
  * The sub-sample is STRIDED over the whole chunk, not a prefix. A prefix of a
    field dump is one corner of the simulation domain, whose moments can differ
    from the chunk's by more than the classes differ from each other.

Where the bytes come from, per workload:

  --raw DIR      the in-situ drivers' own `--raw` / CLIO_VPIC_RAW_DIR output,
                 one file per blob named after the blob with '/' -> '_'. This
                 is the ONLY way to see an in-situ workload's buffers: they are
                 never written to disk otherwise.
  --fields DIR   the replay datasets. A blob is named <frame>/<stem>/chunk_<i>,
                 so its bytes are at DIR/<frame>/<stem><ext> + i*chunk_bytes --
                 the same derivation the replay driver's SourceOfBlob() does.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

#: Elements sub-sampled per chunk. 65,536 float32 keeps every moment estimate
#: far inside sampling noise of the classes' separation while reading one
#: strided pass per chunk.
SAMPLE_ELEMS = 65536
#: DistributionClassifier::Classify's own default.
NUM_BINS = 64


# ---------------------------------------------------------------------------
# Port of DistributionClassifier (distribution_classifier.h)
# ---------------------------------------------------------------------------
def _moments(x: np.ndarray):
    """ComputeMoments: mean, variance, skewness, EXCESS kurtosis."""
    n = x.size
    mean = float(x.mean())
    d = x - mean
    d2 = d * d
    m2 = float(d2.mean())
    m3 = float((d2 * d).mean())
    m4 = float((d2 * d2).mean())
    skew = kurt = 0.0
    if m2 > 1e-10:
        sigma = math.sqrt(m2)
        skew = m3 / (sigma ** 3)
        kurt = (m4 / (m2 * m2)) - 3.0
    return mean, m2, skew, kurt, n


def _histogram(x: np.ndarray, num_bins: int):
    """BuildHistogram: equal-width bins over [min,max], last bin inclusive."""
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / num_bins
    idx = ((x - lo) / width).astype(np.int64)
    np.clip(idx, 0, num_bins - 1, out=idx)
    h = np.bincount(idx, minlength=num_bins).astype(np.float64)
    total = h.sum()
    if total > 0:
        h /= total
    return h, lo, hi


def _uniform_score(h: np.ndarray) -> float:
    n = h.size
    expected = 1.0 / n
    return float(((h - expected) ** 2 / expected).sum())


def _normal_score(h, mean, sigma, lo, hi) -> float:
    n = h.size
    width = (hi - lo) / n
    centres = lo + (np.arange(n) + 0.5) * width
    z = (centres - mean) / sigma
    exp = np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi)) * width
    exp = np.maximum(exp, 1e-10)
    return float(((h - exp) ** 2 / exp).sum())


def _gamma_score(h, mean, var, lo, hi) -> float:
    rng = hi - lo
    if lo < 0 and abs(lo) > 0.1 * rng:
        return 1e10
    if mean <= 0 or var <= 0:
        return 1e10
    k = (mean * mean) / var
    theta = var / mean
    if k < 0.1 or k > 100:
        return 1e10
    n = h.size
    width = rng / n
    centres = lo + (np.arange(n) + 0.5) * width
    x = centres - lo                      # shift so the minimum is 0
    log_gamma_k = ((k - 0.5) * math.log(k) - k + 0.5 * math.log(2 * math.pi)
                   if k > 10 else math.lgamma(k))
    exp = np.zeros(n)
    pos = x > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pdf = ((k - 1) * np.log(np.where(pos, x, 1.0)) - x / theta
                   - k * math.log(theta) - log_gamma_k)
    exp[pos] = np.exp(log_pdf[pos]) * width
    exp = np.maximum(exp, 1e-10)
    # A bin skipped by `continue` in the C++ contributes nothing at all.
    contrib = (h - exp) ** 2 / exp
    contrib[~pos] = 0.0
    return float(contrib.sum())


def _exponential_score(h, mean, lo, hi) -> float:
    rng = hi - lo
    if lo < 0 and abs(lo) > 0.1 * rng:
        return 1e10
    if mean <= 0:
        return 1e10
    lam = 1.0 / mean
    n = h.size
    width = rng / n
    centres = lo + (np.arange(n) + 0.5) * width
    x = centres - lo
    exp = np.maximum(lam * np.exp(-lam * x) * width, 1e-10)
    return float(((h - exp) ** 2 / exp).sum())


def _by_moments(skew: float, kurt: float) -> int:
    """ClassifyByMoments: 0 uniform, 1 normal, 2 gamma, 3 exponential."""
    if skew > 1.5:
        return 3
    if skew > 0.4:
        return 2
    if kurt < -0.5:
        return 0
    return 1


NAMES = ["uniform", "normal", "gamma", "exponential"]


def classify(x: np.ndarray) -> dict:
    """DistributionClassifier<float>::Classify, including its tie-break."""
    out = {"n_sampled": int(x.size), "dist_class": "unknown", "confidence": 0.0,
           "mean": 0.0, "variance": 0.0, "skewness": 0.0, "kurtosis": 0.0,
           "uniform_score": float("nan"), "normal_score": float("nan"),
           "gamma_score": float("nan"), "exponential_score": float("nan")}
    if x.size < 10:
        return out
    x = x.astype(np.float64)
    mean, var, skew, kurt, _ = _moments(x)
    out.update(mean=mean, variance=var, skewness=skew, kurtosis=kurt)
    if var < 1e-10:
        out.update(dist_class="constant", confidence=1.0)
        return out

    h, lo, hi = _histogram(x, NUM_BINS)
    scores = [
        _uniform_score(h),
        _normal_score(h, mean, math.sqrt(var), lo, hi),
        _gamma_score(h, mean, var, lo, hi),
        _exponential_score(h, mean, lo, hi),
    ]
    out.update(uniform_score=scores[0], normal_score=scores[1],
               gamma_score=scores[2], exponential_score=scores[3])

    best = int(np.argmin(scores))
    min_score = scores[best]
    hint = _by_moments(skew, kurt)
    conf = 0.0
    if hint == best:
        conf = 0.9
    else:
        conf = 0.6
        if scores[hint] < min_score * 2.0:
            best = hint
    # Confidence from score separation overrides the agreement value above,
    # exactly as SelectBestDistribution does.
    second = min((s for i, s in enumerate(scores) if i != best), default=1e10)
    if min_score > 0 and second < 1e9:
        conf = min(0.95, 0.5 + 0.25 * math.log(second / min_score))
    out.update(dist_class=NAMES[best], confidence=conf)
    return out


# ---------------------------------------------------------------------------
# Getting a chunk's bytes
# ---------------------------------------------------------------------------
def _subsample(path: str, offset: int, nbytes: int, itemsize: int,
               dtype) -> np.ndarray:
    """A STRIDED sub-sample of one chunk, SAMPLE_ELEMS elements at most."""
    count = nbytes // itemsize
    if count <= 0:
        return np.empty(0, dtype=dtype)
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(count * itemsize)
    a = np.frombuffer(buf, dtype=dtype, count=len(buf) // itemsize)
    if a.size > SAMPLE_ELEMS:
        a = a[:: max(1, a.size // SAMPLE_ELEMS)][:SAMPLE_ELEMS]
    return a


def _source_of_blob(blob: str, fields_dir: str, ext: str, chunk: int):
    """SourceOfBlob() from neuropress_field_replay.cc, verbatim in Python."""
    c = blob.rfind("/chunk_")
    if c < 0:
        return None
    slash = blob.rfind("/", 0, c)
    if slash < 0:
        return None
    frame = blob[:slash]
    stem = blob[slash + 1:c]
    idx = int(blob[c + 7:].split("/")[0])
    return os.path.join(fields_dir, frame, stem + ext), idx * chunk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explore", required=True, help="explore.csv naming the chunks")
    ap.add_argument("--raw", help="directory of per-blob raw dumps (in-situ workloads)")
    ap.add_argument("--fields", help="replay field directory")
    ap.add_argument("--ext", default=".f32", help="field file extension (default .f32)")
    ap.add_argument("--f64", action="store_true", help="fields are float64")
    ap.add_argument("--label-from-frame", action="store_true",
                    help="synthetic data: the class is in the blob's frame name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (a.raw or a.fields or a.label_from_frame):
        ap.error("one of --raw / --fields / --label-from-frame is required")

    e = pd.read_csv(a.explore, usecols=["blob", "chunk_bytes", "role"])
    chunks = (e[e["role"] == "primary"][["blob", "chunk_bytes"]]
              .drop_duplicates("blob"))
    dtype = np.float64 if a.f64 else np.float32
    itemsize = 8 if a.f64 else 4

    rows, missing = [], 0
    for blob, nbytes in zip(chunks["blob"], chunks["chunk_bytes"]):
        nbytes = int(nbytes)
        if a.label_from_frame:
            # <class>_<something>/<stem>/chunk_<i>: generated data carries its
            # own ground-truth class, so the synthetic block never depends on
            # the classifier agreeing with the generator.
            rows.append({"blob": blob, "dist_class": blob.split("/")[0].split("_")[0],
                         "confidence": 1.0, "n_sampled": 0, "mean": float("nan"),
                         "variance": float("nan"), "skewness": float("nan"),
                         "kurtosis": float("nan"), "uniform_score": float("nan"),
                         "normal_score": float("nan"), "gamma_score": float("nan"),
                         "exponential_score": float("nan")})
            continue
        if a.raw:
            path = os.path.join(a.raw, blob.replace("/", "_") + ".bin")
            offset = 0
        else:
            src = _source_of_blob(blob, a.fields, a.ext, nbytes)
            if src is None:
                missing += 1
                continue
            path, offset = src
        if not os.path.exists(path):
            missing += 1
            continue
        x = _subsample(path, offset, nbytes, itemsize, dtype)
        r = classify(x)
        r["blob"] = blob
        rows.append(r)

    if not rows:
        print(f"no chunk bytes found (missing {missing}); nothing written",
              file=sys.stderr)
        return 1
    cols = ["blob", "dist_class", "confidence", "n_sampled", "mean", "variance",
            "skewness", "kurtosis", "uniform_score", "normal_score",
            "gamma_score", "exponential_score"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(a.out, index=False, float_format="%.9g")
    mix = df["dist_class"].value_counts().to_dict()
    print(f"{len(df)} chunk(s) classified"
          + (f", {missing} without bytes" if missing else "")
          + f" -> {a.out}\n  class mix: {mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
