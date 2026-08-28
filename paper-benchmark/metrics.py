#!/usr/bin/env python3
"""Aggregate one paper-benchmark cell into metrics.json, and normalise its
per-chunk rows into results.csv.

    ./metrics.py <cell-dir> [<cell-dir> ...]
    ./metrics.py --table warpx/*/            # one line per cell, for eyeballing

A cell is <workload>/<config>/, as run_paper.sh writes it. Everything here is
derived from files that already exist -- blobs.csv, selection.csv, config.json
-- so it can be re-run at any time without touching the GPU, and re-running it
cannot change what was measured.

BLOB NAMES CARRY THE COORDINATES, AND NOT IN ONE FORMAT. Three of the four
workloads run in situ and name a blob for the step it came from; Nyx runs
through replay and names it for the DUMP INDEX, because the replay driver only
ever sees files:

    warpx   Bx/step_0/chunk_0                              step 0
    vpic    cbx/step_00040/chunk_0                         step 40
    lammps  position/step_0/chunk_0                        step 0
    nyx     plt00000/fab0000_comp00_density/chunk_0        dump 0 -> step 0

So Nyx's step has to be reconstructed as dump_index * interval, which is why
config.json's "interval" is read here rather than assumed. A cell whose names
do not parse is reported rather than silently given step 0.

DECOMPRESSION TIME IS NOT MEASURED FOR EVERY CHUNK. On WarpX only 53 of 240
rows carried a decompress_ms in the smoke run -- the codec is timed where the
runtime had reason to run it, not on every write. Decompression throughput is
therefore computed over the rows that HAVE it, and n_decompress_measured is
reported next to it so the denominator is never in doubt.

THROUGHPUT IS BYTES-IN OVER TIME, both directions. For compression that is the
uncompressed input over the compress kernel time; for decompression it is the
same uncompressed size over the decompress time, because that is the volume
the codec had to produce. Reporting compressed bytes per second instead would
make a better ratio look like worse throughput.
"""
import argparse, csv, json, os, re, statistics as st, sys

# <field>/step_<N>/chunk_<c>  -- the three in-situ workloads
IN_SITU = re.compile(r"^(?P<field>[^/]+)/step_(?P<step>\d+)/chunk_(?P<chunk>\d+)$")
# plt<NNNNN>/fab<NNNN>_comp<NN>_<field>/chunk_<c>  -- Nyx replay
REPLAY = re.compile(r"^plt(?P<dump>\d+)/fab\d+_comp\d+_(?P<field>[^/]+)/chunk_(?P<chunk>\d+)$")


def parse_blob(name, interval):
    """-> (field, step, chunk) or None. step is None when it cannot be known."""
    m = IN_SITU.match(name)
    if m:
        return m["field"], int(m["step"]), int(m["chunk"])
    m = REPLAY.match(name)
    if m:
        # The replay driver sees files, not timesteps: the dump index is what
        # the name carries, so the step is only recoverable with the interval
        # the run was launched at.
        dump = int(m["dump"])
        return m["field"], (dump * interval if interval else None), int(m["chunk"])
    return None


def normalise_selection_key(k):
    """WarpX selection keys are dataset paths; blobs.csv uses field/step."""
    m = re.match(r"^/data/(\d+)/fields/([A-Za-z]+)/([xyz])/chunk_(\d+)$", k)
    if m:
        return f"{m[2]}{m[3]}/step_{m[1]}/chunk_{m[4]}"
    m = re.match(r"^/data/(\d+)/fields/([A-Za-z]+)/chunk_(\d+)$", k)
    if m:
        return f"{m[2]}/step_{m[1]}/chunk_{m[3]}"
    return k


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default



def prediction_error(cell, cfg):
    """MAPE of the model's predictions against what the sweep MEASURED.

    Upstream computes exactly this (diagnostics_store.hpp:337-343) and does not
    merely report it: mape_cost drives a three-band exploration policy
    (:347-348). Clio has neither, so this is the missing half of the
    comparison -- the benchmark could say WHICH candidate the model got wrong
    but never BY HOW MUCH.

    THE SOURCE IS explore.csv, NOT selection.csv, and that is the whole
    correctness argument. A prediction and a measurement may only be compared
    when they describe the SAME ACTION. selection.csv's primary row carries the
    model's prediction for the PRIMARY action while its adopted row carries the
    measurement for the ADOPTED one -- different actions whenever exploration
    overrode, which is 20-89% of chunks. Pairing those would report the model's
    error against a codec it never predicted. explore.csv has both halves on
    one row, per candidate.

    mape_of is upstream's, including its degenerate branch: it returns 0 --
    "perfect prediction" -- whenever the MEASURED value is <= 1e-6. That is a
    silent zero, so `*_degenerate` counts how often it fired. A mape of 0 with
    a high degenerate count is not accuracy.
    """
    ep = os.path.join(cell, "explore.csv")
    if not os.path.exists(ep):
        return {}
    rows = list(csv.DictReader(open(ep)))
    if not rows:
        return {}

    def mape_of(pred, real):
        return abs(pred - real) / real if real > 1e-6 else None  # None = degenerate

    acc = {k: [] for k in ("ratio", "ct_ms", "dt_ms", "cost")}
    degen = {k: 0 for k in acc}

    # The cost model's own weights, so predicted and measured cost are formed
    # the same way the selector formed them. balance = (1,1,1); ratio zeroes
    # the two latency terms. Times are floored at 1 ms and the ratio capped at
    # 100 BEFORE the cost -- on megabyte chunks those clamps dominate, so
    # omitting them would not be a rounding difference.
    model = (cfg.get("cost_model") or "").lower()
    w_ct = w_dt = 0.0 if model == "ratio" else 1.0
    w_io = 0.0 if model == "speed" else 1.0
    bw = num(cfg.get("bw_bytes_per_ms"), 5e6) or 5e6

    def cost_of(nbytes, ratio, ct, dt):
        if ratio <= 0:
            return None
        return (w_ct * max(ct, 1.0) + w_dt * max(dt, 1.0)
                + w_io * nbytes / (min(ratio, 100.0) * bw))

    for r in rows:
        nbytes = num(r.get("chunk_bytes"))
        pairs = [("ratio", num(r.get("pred_ratio")), num(r.get("ratio"))),
                 ("ct_ms", num(r.get("pred_ct_ms")), num(r.get("ct_ms")))]
        # dt is measured only under CLIO_NEUROPRESS_EXPLORE_MEASURE_DT; -1
        # means not measured, which is not the same as a measured zero.
        mdt = num(r.get("dt_ms"), -1.0)
        if mdt >= 0:
            pairs.append(("dt_ms", num(r.get("pred_dt_ms")), mdt))
        for key, pred, real in pairs:
            v = mape_of(pred, real)
            if v is None: degen[key] += 1
            else: acc[key].append(v)
        pc = cost_of(nbytes, num(r.get("pred_ratio")), num(r.get("pred_ct_ms")),
                     num(r.get("pred_dt_ms")))
        rc = cost_of(nbytes, num(r.get("ratio")), num(r.get("ct_ms")),
                     mdt if mdt >= 0 else num(r.get("pred_dt_ms")))
        if pc is not None and rc is not None:
            v = mape_of(pc, rc)
            if v is None: degen["cost"] += 1
            else: acc["cost"].append(v)

    out = {"mape_candidates": len(rows)}
    for k, vals in acc.items():
        # A FRACTION, not a percentage: 0.0 is perfect, 1.0 is 100% off.
        out["mape_" + k] = round(st.mean(vals), 6) if vals else None
        out["mape_" + k + "_median"] = round(st.median(vals), 6) if vals else None
        out["mape_" + k + "_n"] = len(vals)
        out["mape_" + k + "_degenerate"] = degen[k]
    return out


def summarise(cell):
    cfg_path = os.path.join(cell, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    interval = int(cfg.get("interval") or 0)

    blobs_path = os.path.join(cell, "blobs.csv")
    if not os.path.exists(blobs_path):
        return None
    rows = list(csv.DictReader(open(blobs_path)))
    if not rows:
        return None

    # The model's own view of each chunk, when the run recorded one. Keyed by
    # blob name so a missing selection.csv simply leaves the columns empty.
    # TWO MAPS, BECAUSE A CHUNK NOW APPEARS TWICE. 75442318 made the selection
    # log emit a second row at the adopt site: `role=primary` is what the model
    # ranked first, `role=adopted` is what the sweep actually kept, and the LAST
    # row for a blob wins. So:
    #   sel          last row -> the ADOPTED action. Right for the feature
    #                columns, since quantize/shuffle must describe what really
    #                ran.
    #   sel_primary  the `role=primary` row -> the MODEL'S OWN PICK, which is
    #                the only thing "was the model overridden?" can be asked of.
    # Keying both identically matters: reading the override count off `sel`
    # alone compares the adopted row with blobs.csv, and those agree BY
    # CONSTRUCTION, so the count was structurally 0 on every workload that
    # emits adopted rows. Measured on nyx/explore-balance at K=3: 0 reported
    # where 45 of 66 chunks (68.2%) had genuinely been overridden.
    # Logs written before that commit have no `role` column; there the first
    # row seen is the primary, which is what setdefault gives.
    sel, sel_primary = {}, {}
    has_adopted = False          # does this log predate 75442318?
    sel_path = os.path.join(cell, "selection.csv")
    if os.path.exists(sel_path):
        for s in csv.DictReader(open(sel_path)):
            if s.get("role") == "adopted":
                has_adopted = True
            k = s.get("blob", "")
            sel[k] = s
            if s.get("role", "primary") == "primary":
                sel_primary.setdefault(k, s)
            # THE TWO LOGS DO NOT AGREE ON BLOB KEYS. WarpX's selection log
            # names a chunk by its HDF5 dataset path while blobs.csv names it
            # by field and step:
            #     /data/0/fields/B/x/chunk_0   vs   Bx/step_0/chunk_0
            # A plain join therefore matches NOTHING for WarpX and leaves every
            # model-feature column silently empty -- 240 of 240 rows, with no
            # error. Normalise so the features actually land.
            nk = normalise_selection_key(k)
            if nk != k:
                sel.setdefault(nk, s)
                if s.get("role", "primary") == "primary":
                    sel_primary.setdefault(nk, s)

    out_rows, unparsed = [], 0
    for r in rows:
        name = r["blob"]
        p = parse_blob(name, interval)
        if p is None:
            unparsed += 1
            field, step, chunk = "", None, None
        else:
            field, step, chunk = p
        b, stored = num(r["bytes"]), num(r["stored"])
        ct, dt = num(r.get("compress_ms")), num(r.get("decompress_ms"), -1.0)
        # Prefer the runtime's own key when the workload carries it (WarpX
        # writes runtime_blob for exactly this reason); fall back to the
        # normalised name for everything else.
        s = sel.get(r.get("runtime_blob") or "", {}) or sel.get(name, {})
        sp = (sel_primary.get(r.get("runtime_blob") or "", {})
              or sel_primary.get(name, {}))
        out_rows.append({
            "blob": name, "field": field,
            "step": "" if step is None else step,
            "chunk": "" if chunk is None else chunk,
            "bytes": int(b), "stored": int(stored),
            "ratio": round(b / stored, 6) if stored else "",
            "codec": r.get("codec", ""),
            "compress_ms": ct, "decompress_ms": dt if dt >= 0 else "",
            # Uncompressed volume over kernel time, both directions. See the
            # module docstring for why it is bytes-IN on the decompress side.
            "compress_MBps": round(b / 1e6 / (ct / 1e3), 3) if ct > 0 else "",
            "decompress_MBps": round(b / 1e6 / (dt / 1e3), 3) if dt > 0 else "",
            # What the MODEL ranked first, beside what was stored. Kept per
            # chunk so anyone can recompute the override rate from results.csv
            # instead of trusting the aggregate.
            "model_codec": sp.get("lib_name", ""),
            "adopted_codec": s.get("lib_name", ""),
            # THE ACTION, NOT JUST THE CODEC. NeuroPress's action is the tuple
            # (library, shuffle, quantize, preset) -- viz_actions.py draws all
            # four for exactly this reason -- so a sweep that keeps the library
            # and changes the stride HAS overridden the model. Measured on
            # nyx/explore-balance: 36 chunks were stored by exploration but
            # only 33 changed library, so comparing lib_name alone lost 3.
            "model_action": "|".join((sp.get("lib_name", ""),
                                      sp.get("quantize", ""),
                                      sp.get("shuffle", ""),
                                      sp.get("preset", ""))) if sp else "",
            "adopted_action": "|".join((s.get("lib_name", ""),
                                        s.get("quantize", ""),
                                        s.get("shuffle", ""),
                                        s.get("preset", ""))) if s else "",
            # Ground truth from the runtime rather than inferred: the adopted
            # row is emitted ONLY in compressor_runtime.cc's
            # `stored_by_exploration = true` branch, so its presence IS the
            # statement that the sweep superseded the primary. Carried so the
            # comparison below can be cross-checked against it.
            "explored_adopted": 1 if s.get("role") == "adopted" else 0,
            "entropy": s.get("entropy", ""), "mad": s.get("mad", ""),
            "quantize": s.get("quantize", ""), "shuffle": s.get("shuffle", ""),
            "rc": r.get("rc", ""),
        })

    with open(os.path.join(cell, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    tot_in = sum(r["bytes"] for r in out_rows)
    tot_st = sum(r["stored"] for r in out_rows)
    ratios = [r["ratio"] for r in out_rows if r["ratio"] != ""]
    ct = [r["compress_ms"] for r in out_rows if r["compress_ms"] > 0]
    dt = [r["decompress_ms"] for r in out_rows if r["decompress_ms"] != ""]
    steps = sorted({r["step"] for r in out_rows if r["step"] != ""})
    chunks = sorted({r["chunk"] for r in out_rows if r["chunk"] != ""})
    sizes = sorted({r["bytes"] for r in out_rows})

    codecs = {}
    for r in out_rows:
        c = codecs.setdefault(r["codec"] or "?", {"chunks": 0, "bytes": 0, "stored": 0})
        c["chunks"] += 1; c["bytes"] += r["bytes"]; c["stored"] += r["stored"]
    for c in codecs.values():
        c["ratio"] = round(c["bytes"] / c["stored"], 4) if c["stored"] else None

    wall = num(cfg.get("wall_s"))
    m = {
        "workload": cfg.get("workload"), "config": cfg.get("config"),
        "cost_model": cfg.get("cost_model"), "mode": cfg.get("mode"),
        "error_bound": cfg.get("error_bound"), "explore_k": cfg.get("explore_k"),
        "route": cfg.get("route"), "rc": cfg.get("rc"),
        "verification": cfg.get("verification"),
        "verify_result": cfg.get("verify_result"),

        "total_bytes_in": tot_in,
        "total_bytes_stored": tot_st,
        "overall_ratio": round(tot_in / tot_st, 4) if tot_st else None,
        "chunks": len(out_rows),
        "chunk_bytes": sizes[0] if len(sizes) == 1 else sizes[:1] + ["..."] + sizes[-1:],
        "fields": sorted({r["field"] for r in out_rows if r["field"]}),
        "timesteps": steps, "n_timesteps": len(steps),
        "chunks_per_field_frame": len(chunks),
        "unparsed_blob_names": unparsed,
        # HOW OFTEN THE SWEEP BEAT THE MODEL: the `role=primary` row against
        # the `role=adopted` one, BOTH FROM selection.csv. Comparing the
        # primary against blobs.csv instead would miscount every chunk stored
        # raw -- lib==0 is "Compression not beneficial" and reads as
        # `raw(not-beneficial)` in the codec column (4d8024c8), which is the
        # chunk being DECLINED, not the model being overridden.
        # Rows that did not join contribute nothing rather than counting as a
        # disagreement; selection_features_joined below is what exposes those.
        # A CELL WRITTEN BEFORE 75442318 HAS NO ADOPTED ROW to compare against
        # -- its single row per blob IS the primary -- so there the only
        # available comparison is the primary against what the tier holds, and
        # that is what the counter meant at the time. Raw-stored chunks are
        # still excluded: "Compression not beneficial" is the chunk being
        # declined, not the sweep overriding anything. Without this branch every
        # archived cell would silently read 0.
        "selection_overridden_by_explore": sum(
            1 for r in out_rows
            if r["model_action"] and (
                (r["adopted_action"] and r["model_action"] != r["adopted_action"])
                if has_adopted else
                (not str(r["codec"]).startswith("raw")
                 and r["model_codec"] != r["codec"]))),
        # The same fact counted a second way, from the runtime's own emission
        # site rather than by comparing fields. These two MUST agree; if they
        # do not, the action tuple is missing something the sweep changed.
        "selection_explored_adopted": sum(r["explored_adopted"] for r in out_rows),
        # How many of the overrides are visible as a codec change alone. Lower
        # than the above by exactly the chunks where only the stride, preset or
        # quantize bit moved.
        "selection_codec_changed": sum(
            1 for r in out_rows
            if r["model_codec"] and r["adopted_codec"]
            and r["model_codec"] != r["adopted_codec"]),
        # The cross-log guard the old counter used to be, kept because it is
        # still worth checking -- it just is not the override rate. The adopted
        # row must name what landed on the tier, so anything but 0 means the
        # log and the tier disagree about a chunk. Raw-stored chunks are
        # excluded: nothing ran on them to name.
        "selection_stored_mismatch": sum(
            1 for r in out_rows
            if r["adopted_codec"] and not str(r["codec"]).startswith("raw")
            and r["adopted_codec"] != r["codec"]),
        "selection_features_joined": sum(1 for r in out_rows if r["entropy"] != ""),

        "ratio_mean": round(st.mean(ratios), 4) if ratios else None,
        "ratio_median": round(st.median(ratios), 4) if ratios else None,
        "ratio_min": round(min(ratios), 4) if ratios else None,
        "ratio_max": round(max(ratios), 4) if ratios else None,

        "compress_ms_total": round(sum(ct), 3) if ct else 0.0,
        "compress_ms_mean": round(st.mean(ct), 4) if ct else None,
        "compress_ms_median": round(st.median(ct), 4) if ct else None,
        # Aggregate, not the mean of per-chunk rates: total volume over total
        # kernel time is what a reader can multiply back out.
        "compress_MBps": round(tot_in / 1e6 / (sum(ct) / 1e3), 2) if sum(ct) > 0 else None,

        "n_decompress_measured": len(dt),
        "decompress_ms_total": round(sum(dt), 3) if dt else 0.0,
        "decompress_ms_mean": round(st.mean(dt), 4) if dt else None,
        "decompress_ms_median": round(st.median(dt), 4) if dt else None,
        "decompress_MBps": (
            round(sum(r["bytes"] for r in out_rows if r["decompress_ms"] != "")
                  / 1e6 / (sum(dt) / 1e3), 2) if sum(dt) > 0 else None),

        "codecs": dict(sorted(codecs.items(), key=lambda kv: -kv[1]["chunks"])),
        "wall_s": wall,
        # Everything the run did, over everything it took: the number to quote
        # when comparing configurations end to end rather than kernel to kernel.
        "end_to_end_MBps": round(tot_in / 1e6 / wall, 2) if wall > 0 else None,
    }
    m.update(prediction_error(cell, cfg))
    json.dump(m, open(os.path.join(cell, "metrics.json"), "w"), indent=1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--table", action="store_true", help="one line per cell")
    a = ap.parse_args()

    got = []
    for c in a.cells:
        c = c.rstrip("/")
        m = summarise(c)
        if m is None:
            print(f"skip (no blobs.csv): {c}", file=sys.stderr); continue
        got.append((c, m))

    if a.table:
        print(f"{'cell':<34} {'chunks':>7} {'in MB':>9} {'ratio':>7} "
              f"{'cMB/s':>9} {'dMB/s':>9} {'wall':>6} {'verify':>10}")
        for c, m in got:
            print(f"{'/'.join(c.split('/')[-2:]):<34} {m['chunks']:>7} "
                  f"{m['total_bytes_in']/1e6:>9.1f} {m['overall_ratio'] or 0:>7.2f} "
                  f"{m['compress_MBps'] or 0:>9.1f} {m['decompress_MBps'] or 0:>9.1f} "
                  f"{m['wall_s']:>6.0f} {str(m['verify_result']):>10}")
    else:
        for c, m in got:
            print(f"{c}: {m['chunks']} chunks, {m['total_bytes_in']/1e6:.1f} MB in, "
                  f"ratio {m['overall_ratio']}, {m['compress_MBps']} MB/s compress")


if __name__ == "__main__":
    main()
