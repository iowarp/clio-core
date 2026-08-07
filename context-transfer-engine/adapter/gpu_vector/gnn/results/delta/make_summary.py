#!/usr/bin/env python3
"""Build the cross-run summary table from the CSVs written by the GNN tests.

  dataset | method | peak GPU | store ratio | final train+val acc | epoch time | bit-exact

Training rows come from gnn_train_results.csv (one row per run: it contains BOTH
the in-core and the Eternia arm, so each row expands into up to two table rows).
Forward-capacity rows come from gnn_cap_results.csv.
"""
import argparse
import csv
import os


def human_mib(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{v/1024:.1f} GiB" if v >= 1024 else f"{v:.0f} MiB"


def pct(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if v < 0 else f"{100*v:.2f}%"


def secs(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if v < 0 else (f"{v:.2f} s" if v < 60 else f"{v/60:.1f} min")


def dsname(path):
    b = os.path.basename(str(path).rstrip("/"))
    return b or str(path)


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    # Default to the CSVs committed alongside this script rather than an absolute
    # home path, so the tree works wherever it is checked out.
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--train-csv", default=os.path.join(here, "csv", "gnn_train_results.csv"))
    ap.add_argument("--cap-csv", default=os.path.join(here, "csv", "gnn_cap_results.csv"))
    ap.add_argument("--out", default=os.path.join(here, "csv", "SUMMARY.md"))
    args = ap.parse_args()

    rows = []
    for r in load(args.train_csv):
        ds = dsname(r.get("data", "?"))
        lib = r.get("lib", "?")
        preset = r.get("preset", "")
        amib = r.get("A_mib", "-")
        # Several runs share a (dataset, lib, preset) triple and differ only in epoch
        # count (e.g. the 3-epoch probe vs the 30-epoch headline), so carry the epoch
        # count into the label -- otherwise the rows are indistinguishable.
        ep = r.get("epochs", "")
        tag = f", {ep} ep" if ep else ""
        # --- in-core arm ---
        if r.get("base_status") == "OK":
            rows.append([ds, f"in-core (resident{tag})", human_mib(amib), "1.00x (uncompressed)",
                         f"{pct(r.get('base_final_acc'))} / {pct(r.get('base_final_vacc'))}",
                         secs(r.get("base_epoch_s")), "baseline"])
        else:
            rows.append([ds, f"in-core (resident{tag})", f"OOM (needs {human_mib(amib)})", "-", "OOM", "OOM",
                         "n/a (OOM)"])
        # --- Eternia arm ---
        try:
            ratio = f"{float(r.get('ratio', 0)):.3f}x"
        except (TypeError, ValueError):
            ratio = "-"
        rows.append([ds, f"Eternia-{lib}" + (f" ({preset}{tag})" if preset else f" ({tag.strip(', ')})"),
                     human_mib(r.get("peak_gpu_mib")), ratio,
                     f"{pct(r.get('eternia_final_acc'))} / {pct(r.get('eternia_final_vacc'))}",
                     secs(r.get("eternia_epoch_s")), r.get("bit_exact", "-")])

    hdr = ["dataset", "method", "peak GPU", "store ratio",
           "final train / val acc", "epoch time", "bit-exact?"]
    lines = ["# GNN on the Eternia compressed feature store - summary", "",
             "## Training (test_gpu_vector_gnn_train)", ""]
    if rows:
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(c) for c in r) + " |")
    else:
        lines.append("_no training rows_")

    cap = load(args.cap_csv)
    lines += ["", "## Forward capacity (test_gpu_vector_gnn_capacity)", ""]
    if cap:
        keys = list(cap[0].keys())
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in cap:
            lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    else:
        lines.append("_no capacity rows_")

    out = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"[summary] wrote {args.out}")


if __name__ == "__main__":
    main()
