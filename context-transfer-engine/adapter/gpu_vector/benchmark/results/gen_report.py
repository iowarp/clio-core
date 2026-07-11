#!/usr/bin/env python3
"""Colleague-facing report: Gray-Scott + CLIO GPU compression + the compressed
GPU vector. Regenerates GS_CLIO_compression_report.pdf."""
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "GS_CLIO_compression_report.pdf")

BLUE = colors.HexColor("#2a78d6")
YELLOW = colors.HexColor("#eda100")
GREEN = colors.HexColor("#008300")
INK = colors.HexColor("#1a1a1a")
GREY = colors.HexColor("#5a5a5a")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=19, textColor=INK,
                    spaceAfter=4, leading=23)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10.5, textColor=GREY,
                     spaceAfter=10, leading=14)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, textColor=BLUE,
                    spaceBefore=7, spaceAfter=3, leading=15)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.7, textColor=INK,
                      leading=13.0, spaceAfter=4)
SMALL = ParagraphStyle("SMALL", parent=ss["Normal"], fontSize=8.3,
                       textColor=GREY, leading=11)
BANNER = ParagraphStyle("BANNER", parent=ss["Normal"], fontSize=10,
                        textColor=colors.white, leading=14)


def banner(text, fill):
    t = Table([[Paragraph(text, BANNER)]], colWidths=[6.9 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


KVLABEL = ParagraphStyle("KVLABEL", parent=ss["Normal"], fontSize=8.8,
                         textColor=BLUE, fontName="Helvetica-Bold", leading=11)
KVVAL = ParagraphStyle("KVVAL", parent=ss["Normal"], fontSize=8.8,
                       textColor=INK, leading=12)


def kv_table(rows, col0=1.5):
    cells = [[Paragraph(a, KVLABEL), Paragraph(b, KVVAL)] for a, b in rows]
    t = Table(cells, colWidths=[col0 * inch, (6.9 - col0) * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e2e2e2")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def data_table(header, rows, widths):
    data = [header] + rows
    t = Table(data, colWidths=[w * inch for w in widths])
    st = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.6),
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dcdcdc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f8fd")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(st))
    return t


def fig(name, w=6.5):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return Spacer(1, 2)
    from PIL import Image as PILImage
    iw, ih = PILImage.open(p).size
    return Image(p, width=w * inch, height=w * inch * ih / iw)


story = []
story.append(Paragraph("Tier-Aware Compressed GPU Vector for CLIO / IOWarp", H1))
story.append(Paragraph("GPU-native error-bounded compression on the Gray-Scott "
                       "workload &mdash; Delta A100 &middot; cuSZp &middot; "
                       "CLIO embedded runtime", SUB))
story.append(banner("STATUS &mdash; WORKING. The compressed GPU vector "
                    "round-trips write&rarr;evict&rarr;compress(HBM)&rarr;store"
                    "&rarr;decompress&rarr;device-read, and stores a dataset "
                    "2&times; the GPU-memory budget entirely on the GPU "
                    "(16&times; capacity). Verified on A100.", GREEN))
story.append(Spacer(1, 7))

story.append(Paragraph("What this is", H2))
story.append(Paragraph(
    "A <b>vector-like abstraction whose pages live across HBM &rarr; DRAM &rarr; "
    "NVMe &rarr; PFS</b> with automatic <b>GPU</b> compression. Data produced by "
    "a GPU kernel (here, a Gray-Scott reaction-diffusion simulation) is stored "
    "compressed <b>without ever leaving the GPU to compress</b>: pages are "
    "compressed in-place in HBM by cuSZp (an error-bounded lossy float codec) "
    "and only the small compressed result is written down the tier stack through "
    "CLIO's real runtime. The compressor is a CLIO chimod placed in front of the "
    "CTE core; the <code>Vector&lt;T&gt;</code> routes its page evictions and "
    "faults through it transparently.", BODY))

story.append(Paragraph("Compressed GPU vector &mdash; end-to-end result", H2))
story.append(Paragraph(
    "A <code>Vector&lt;float&gt;</code> (256&nbsp;KiB pages) is written on-device; "
    "the async cache manager <b>evicts every dirty page through the compressor</b>, "
    "which compresses it in HBM and stores the compressed blob in the CTE core. "
    "Reading back, <code>FaultAllSync()</code> decompresses every page straight "
    "into its HBM slot, then a device read kernel reads them. Verified on A100 "
    "(cuSZp, abs error bound 1e-3):", BODY))
story.append(data_table(
    ["Stage", "Pages", "Per page", "Result"],
    [["Evict → compress (HBM)", "4 / 4", "262144 B → ~16.6 KB", "~16:1"],
     ["Store (compressed, in core)", "4 / 4", "compressed blob", "ok"],
     ["FaultAllSync → decompress", "4 / 4", "→ HBM slot", "ok"],
     ["Device read kernel", "262144 elems", "max_abs_err 1.0e-3", "== eb"]],
    [2.5, 1.2, 1.9, 1.3]))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "max_abs_err = 1.0e-3 (equals cuSZp's error bound), mean 4.9e-4 &mdash; the "
    "loss is exactly the requested bound, i.e. correct. Test: "
    "<code>cte_gpu_vector_compress_cuda</code> (PASS). No regression in the "
    "existing <code>cte_gpu_vector_cuda</code>.", SMALL))

story.append(Paragraph("How it is wired (no DRAM copy to compress)", H2))
story.append(kv_table([
    ["Zero-copy HBM", "Page data is passed as a device pointer; cuSZp compresses "
     "it in place using its own temp HBM buffer. Nothing is staged to DRAM to "
     "compress &mdash; only the compressed bytes leave the device."],
    ["Compressor handlers", "Raw core PutBlob / GetBlob arriving at the compressor "
     "entrypoint are transparently compressed / decompressed and forwarded to the "
     "core (per-page name <code>&lt;tag&gt;_b&lt;blk&gt;_pi&lt;page&gt;</code>)."],
    ["Vector routing", "New <code>storage_pool_id</code> ctor arg: page traffic "
     "goes to the compressor pool; the tag stays on the core, which the compressor "
     "forwards to. A plain vector becomes compressed with one argument."],
    ["Library pin", "<code>CLIO_CTE_COMPRESS_LIB=cuszp</code> selects the GPU "
     "codec; the factory (<code>WireIdForName</code>/<code>MakeCuszp</code>) "
     "registers it. lz4/zstd (CPU) also available."],
]))

story.append(Paragraph("Capacity win &mdash; a dataset larger than GPU memory, "
                       "stored on the GPU", H2))
story.append(Paragraph(
    "The headline benefit: because compression shrinks cold pages, a dataset "
    "<b>larger than the GPU-memory budget fits entirely on the GPU</b>. The "
    "compressor's storage tier is a <b>kHbm bdev (device memory)</b>, so "
    "compressed cold pages live in HBM, not host DRAM. A Gray-Scott field of "
    "logical size <b>2&times; the HBM budget</b> was streamed through the "
    "compressed vector; measured HBM footprint (kHbm-tier used bytes):", BODY))
story.append(data_table(
    ["Quantity", "Value", "Fits on GPU?"],
    [["Logical dataset", "256 MiB (2× budget)", "—"],
     ["HBM budget", "128 MiB", "—"],
     ["HBM used (compressed 15.9×)", "16 MiB", "yes — 111 MiB to spare"],
     ["Uncompressed would need", "256 MiB", "no — exceeds budget"]],
    [2.6, 2.1, 2.2]))
story.append(Spacer(1, 2))
story.append(fig("fig_capacity.png", 4.3))
story.append(Paragraph(
    "So the same GPU holds ~16&times; more logical data. Chunk-0 read-back "
    "(via FaultAllSync) matched within the error bound (max_abs_err 1.0e-3). "
    "Test: <code>cte_gpu_vector_capacity_cuda</code> (PASS). Streaming is "
    "chunked + host-paced because paging larger-than-HBM <i>on-device</i> "
    "deadlocks under GPU compression (see limitation below).", SMALL))

story.append(Paragraph("Head-to-head vs. the traditional checkpoint path", H2))
story.append(Paragraph(
    "Same canonical Gray-Scott loop (iterate steps, checkpoint the field every "
    "N steps) checkpointed two ways and timed on the same evolving snapshots: "
    "the <b>traditional path</b> (cudaMemcpy device&rarr;host, then write the "
    "full uncompressed field to a Lustre/PFS file &mdash; what an HDF5 checkpoint "
    "does) vs. the <b>compressed GPU vector</b> (compress in HBM, blob stays on "
    "the GPU). A100, 4096&times;4096 field (64 MiB), 200 steps, checkpoint every "
    "25:", BODY))
story.append(data_table(
    ["Checkpoint path", "Latency", "Eff. BW", "Footprint", "Where"],
    [["Traditional (D2H + PFS file)", "96.7 ms", "0.65 GiB/s", "512 MiB", "disk"],
     ["Compressed GPU vector", "14.6 ms", "4.27 GiB/s", "68 MiB", "HBM"],
     ["Advantage", "6.6× faster", "6.6×", "7.5× smaller", "on-GPU"]],
    [2.5, 1.15, 1.05, 1.15, 0.85]))
story.append(Spacer(1, 2))
story.append(fig("fig_checkpoint_compare.png", 5.8))
story.append(Paragraph(
    "The traditional path is dominated by the PFS write (single-stream Lustre "
    "~0.68 GiB/s); the compressed path is compute-bound (cuSZp) but never leaves "
    "the GPU and writes 7.5× fewer bytes. Bench: "
    "<code>clio_gs_checkpoint_bench</code>.", SMALL))

story.append(Paragraph("What we reused vs. changed", H2))
story.append(Paragraph(
    "The <code>Vector&lt;T&gt;</code> API was <b>already implemented</b> &mdash; we "
    "reused it and made two small, backward-compatible <b>additions</b>; nothing "
    "was rewritten.", BODY))
story.append(kv_table([
    ["Reused as-is", "The whole <code>Vector&lt;T&gt;</code>: ctor, "
     "<code>Device()</code>, <code>FlushAllSync()</code>, the device-side "
     "<code>write_range</code>/<code>read_range</code>, the HBM/DRAM paging + "
     "cache-manager machinery, legacy/async modes, cold-miss fault, and the "
     "existing passing test (left untouched)."],
    ["Added #1 &mdash; one ctor arg", "<code>storage_pool_id</code> (defaults to "
     "the core = old behavior). Pointed at the compressor pool, it routes page "
     "evictions/faults through compression &mdash; this is what turns a plain "
     "vector into a compressed one."],
    ["Added #2 &mdash; one method", "<code>FaultAllSync()</code> (+ a small "
     "resident-marking kernel) to materialize compressed pages back into HBM for "
     "device-side reads."],
    ["Outside the vector", "Compressor PutBlob/GetBlob handlers, task "
     "serialization cases, and a new test file &mdash; not modifications to the "
     "Vector."],
]))

story.append(Paragraph("Transfer micro-benchmark (context)", H2))
story.append(Paragraph(
    "Before the vector, a 4-case Gray-Scott transfer benchmark characterized the "
    "codec on the same data (raw / async / compressed / async+compressed). "
    "Directly-measured compression ratio and the compress-vs-raw slowdown:", BODY))
story.append(fig("fig_compression_ratio.png", 2.35))
story.append(fig("fig_slowdown.png", 2.35))

story.append(Paragraph("Device-side reads: the fault deadlock, and the fix", H2))
story.append(banner("The transparent on-ACCESS device fault is undeadlockable on "
                    "one GPU (kernel co-scheduling). The fix is the #700 "
                    "Transaction API: host-orchestrated WINDOWED PREFETCH.",
                    GREEN))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "An on-device page fault spin-waits on the GPU for its GetBlob, but the "
    "compressor services that fault by launching cuSZp's decompress kernel on the "
    "<b>same</b> GPU &mdash; and a spin-waiting kernel and the decompress kernel "
    "do not co-schedule, so they deadlock. Two fixes were tried and both failed: "
    "(1) a dedicated non-blocking stream + <code>cudaMallocAsync</code> in the "
    "wrapper and a cuSZp source patch; (2) a preallocated pinned staging pool + "
    "pre-warmed mempool. Both remove every device-synchronizing CUDA call, yet the "
    "fault still hangs &mdash; proving the residual cause is kernel "
    "<b>co-scheduling</b>, not device-sync. (Uncompressed faults are fine: "
    "serviced by a CPU memcpy, no GPU kernel. Eviction is fine too: flush kernels "
    "exit before the compressor runs.)", BODY))
story.append(Paragraph(
    "<b>The working answer</b> is the issue-#700 Transaction API &mdash; "
    "host-orchestrated <b>windowed prefetch</b>. Pages are pulled into HBM "
    "<i>ahead</i> of use while the GPU is idle, so the device kernel only ever "
    "reads resident pages. <code>Transaction&lt;T&gt;</code> + "
    "<code>SequentialTransaction</code> / <code>PseudoRandomTransaction</code> "
    "(with <code>Vector::PrefetchWindowSync</code>) sweep a dataset far larger than "
    "the HBM cache one window at a time. Verified A100: an 8 MiB dataset swept in "
    "1 MiB windows (8&times; the resident footprint) through both a Sequential and "
    "a PseudoRandom transaction &mdash; 2,097,152 elems each, max_abs_err 1.0e-3 "
    "== eb, no on-device fault. Remaining perf work: pipeline the next window's "
    "prefetch on a copy stream while the kernel reads the current one.", BODY))
story.append(Spacer(1, 8))
story.append(HRFlowable(width="100%", thickness=0.6,
                        color=colors.HexColor("#dcdcdc")))
story.append(Paragraph(
    "Environment: A100-SXM4-40GB &middot; CUDA 12.6 (iowarp/deps-nvidia, "
    "Apptainer) &middot; cuSZp GPU codec &middot; CLIO embedded runtime "
    "(compressor pool &rarr; CTE core &rarr; HBM/RAM bdev). Key commits: c069d50 "
    "(transparent compress), c59eb5c (compressed vector), ce1cc9a (FaultAllSync), "
    "bcda63a (capacity), 6077f9f (checkpoint bench), 89c4735 (Transaction API).",
    SMALL))

doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                        title="Compressed GPU Vector for CLIO/IOWarp")
doc.build(story)
print("wrote", OUT)
