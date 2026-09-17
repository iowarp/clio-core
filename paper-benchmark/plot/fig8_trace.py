#!/usr/bin/env python3
"""Figure 8 trace from a Clio-NeuroPress exploration log.

  fig8_trace.py <explore.csv> --out trace.csv [--bw 5e6] [--chunks N]

Input: explore.csv from a run that measures all 32 configurations per chunk.
Output, one row per (chunk, configuration): chunk_id, config_id, chosen,
real_cost and predicted_cost at MIN_TIME_MS (both sides floored alike),
real_cost_raw and predicted_cost_raw (no floor), runtime_ranked_cost, and the
components (ct_ms, dt_ms, ratio, pred_ct_ms, pred_dt_ms, pred_ratio).

Prints checks: configurations per chunk, one pick per chunk, measured dt, and
penalty rows (never measured; dropped).
"""
import argparse, sys
import pandas as pd

RATIO_CAP = 100.0
# Reporting floor, upstream's trace mape_cost (diagnostics_store.hpp:318).
# Upstream's regret uses its unfloored real_cost (H5VLgpucompress.cu:2734),
# which is real_cost_raw here.
MIN_TIME_MS = 5.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("explore_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bw", type=float, default=5e6, help="bytes per ms in the cost (default 5e6)")
    ap.add_argument("--chunks", type=int, default=0, help="keep only the first N chunks (0 = all)")
    a = ap.parse_args()

    e = pd.read_csv(a.explore_csv, usecols=["seq", "blob", "chunk_bytes", "role", "lib_name",
                                            "quantize", "shuffle", "pred_ratio", "pred_ct_ms",
                                            "pred_dt_ms", "ratio", "ct_ms", "dt_ms", "cost",
                                            "adopted"])
    picks = e[e["role"] == "primary"].sort_values("seq")
    if a.chunks:
        picks = picks.head(a.chunks)
    order = {b: i for i, b in enumerate(picks["blob"])}
    e = e[e["blob"].isin(order)].copy()

    penalty = (e["ct_ms"] <= 0) & (e["dt_ms"] < 0) & (e["cost"] >= 1e5)
    e["config_id"] = e["lib_name"] + "|q" + e["quantize"].astype(str) + "|s" + e["shuffle"].astype(str)
    per_chunk = e.groupby("blob").agg(rows=("config_id", "size"), configs=("config_id", "nunique"),
                                      picks=("role", lambda s: int((s == "primary").sum())))
    m = e[~penalty]
    print(f"chunks {len(order)} | configurations per chunk {per_chunk.configs.min()}..{per_chunk.configs.max()} "
          f"(rows {per_chunk.rows.min()}..{per_chunk.rows.max()}) | picks per chunk "
          f"{sorted(per_chunk.picks.unique().tolist())}")
    print(f"penalty rows dropped {int(penalty.sum())} (picks {int((penalty & (e.role == 'primary')).sum())}) | "
          f"decompression measured on {100 * (m.dt_ms >= 0).mean():.1f}% of rows | "
          f"alternative adopted on {int(e[(e.role != 'primary') & (e.adopted == 1)].blob.nunique())} chunk(s)")

    # An unmeasured decompression (dt < 0) is scored at the compression time, as
    # the runtime's own cost model does; below the floor that substitution shows.
    dt_meas = m["dt_ms"].where(m["dt_ms"] > 0, m["ct_ms"])
    io = m["chunk_bytes"] / (a.bw * m["ratio"].clip(lower=0.1, upper=RATIO_CAP))
    io_pred = m["chunk_bytes"] / (a.bw * m["pred_ratio"].clip(lower=0.1, upper=RATIO_CAP))
    m = m.assign(
        chunk_id=m["blob"].map(order),
        chosen=(m["role"] == "primary").astype(int),
        predicted_cost=m["pred_ct_ms"].clip(lower=MIN_TIME_MS) + m["pred_dt_ms"].clip(lower=MIN_TIME_MS)
        + io_pred,
        # Recomputed, not the runtime's `cost` column, so both sides share a floor.
        real_cost=m["ct_ms"].clip(lower=MIN_TIME_MS) + dt_meas.clip(lower=MIN_TIME_MS) + io,
        real_cost_raw=m["ct_ms"].clip(lower=0.0) + dt_meas.clip(lower=0.0) + io,
        predicted_cost_raw=m["pred_ct_ms"].clip(lower=0.0) + m["pred_dt_ms"].clip(lower=0.0) + io_pred,
    ).sort_values(["chunk_id", "chosen"], ascending=[True, False])
    # The six components as well, so plot_fig8.py can re-score the cost MAPE at
    # any reporting floor without a rerun (--mape-floor).
    cols = ["chunk_id", "config_id", "real_cost", "chosen", "predicted_cost",
            "real_cost_raw", "predicted_cost_raw", "chunk_bytes", "cost",
            "ct_ms", "dt_ms", "ratio", "pred_ct_ms", "pred_dt_ms", "pred_ratio"]
    m[cols].rename(columns={"cost": "runtime_ranked_cost"}).to_csv(
        a.out, index=False, float_format="%.9g")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
