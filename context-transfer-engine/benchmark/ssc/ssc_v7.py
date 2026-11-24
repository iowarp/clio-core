"""SSC v7 candidates (developed offline on the practice schedules, seeds 1-3, with ablate_ssc.py).

v3 with the ChiMod descriptions corrected to what this node measures (the v3 text said lz4 "adds
little CPU time"; the grid shows either codec several times slower than uncompressed here, and the
cache helping only compressed or SSD-placed data):

  v7c  corrected cache and compressor descriptions (deployment profile) + current placement on the
       cache question
  v7   v7c + compression question evidence: sampled ratio, whether the stream fits uncompressed
       (stated as its consequence), and how many reads pay decompression. No codec is named.
"""

import ssc_live
from ssc_live import TIER_TEXT

_V3_COMPOSABILITY = ssc_live.SSCBuilder.composability

CACHE_TEXT = ("- cache: sits in front of core and keeps uncompressed copies of recently written or read "
              "blobs in the DRAM tier. It speeds up repeat reads of data that is compressed (no "
              "decompression on a hit) or placed on the SSD tier; for uncompressed data in the DRAM tier "
              "it only adds work to every write and read.\n")
CODEC_TEXT = ("- compressor: compresses blobs on the way in and decompresses on the way out. On this node "
              "(measured at deployment) either codec makes writes and reads several times slower than "
              "uncompressed; it pays off only when the data would not fit otherwise. Deployed codecs:\n"
              "  - lz4: the faster codec, modest ratio.\n"
              "  - zstd: the slower codec, higher ratio.\n")


def composability():
    text = _V3_COMPOSABILITY()
    text = text[:text.index("- cache:")] + CACHE_TEXT + text[text.index("- compressor:"):]
    return text[:text.index("- compressor:")] + CODEC_TEXT + text[text.index("- indexer:"):]


def add_evidence(qs, builder, obs, compression):
    w, r = builder.history[-1][1], builder.history[-1][3]
    qs["cache"]["instructions"] += ("\n- placement of this stream now: "
                                    f"{TIER_TEXT[builder.current['placement']]}")
    if compression:
        need = w["bytes_put"] / (1 << 30)
        free = sum(t["remaining_gb"] for t in obs["tiers"].values())
        if need <= 0.8 * free:
            fit = (f"this stream fits uncompressed in the free space ({need:.2f} GB per window, {free:.1f} GB "
                   f"free), so compressing it would only add CPU time to every write and read")
        else:
            fit = (f"this stream does not fit uncompressed in the free space ({need:.2f} GB per window, "
                   f"{free:.1f} GB free); at the sampled ratio it "
                   f"{'fits' if need / max(r, 1.0) <= 0.8 * free else 'still does not fit'} compressed")
        reads = (f"reads per window: {w['n_get']} blobs, and every read of a compressed blob is "
                 f"decompressed first" if w["n_get"] else "this stream is written and not read back")
        qs["compression"]["instructions"] += ("\nEvidence for this question:\n" + "\n".join(
            f"- {e}" for e in (f"sampled compressibility of this stream: {r:.1f}x", fit, reads)))
    return qs
