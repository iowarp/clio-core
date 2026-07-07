#!/usr/bin/env python3
"""
Q-table trainer for compression-metric prediction.

Reads a unified benchmark CSV, discretizes the 11-feature space into bins,
and aggregates labels per state. Emits qtable.csv and binning.csv in the exact
format that the C++ QTablePredictor::Load() expects.

CSV schema and feature order same as train_xgboost.py.
Labels: compression_ratio, psnr_db, compression_time_ms.
Filter: success == 1.
"""

import argparse
import sys
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Train Q-table compression predictor")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out-dir", required=True, help="Output directory for models")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins per feature")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed", file=sys.stderr)
        return 1

    # Read CSV
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    # Check required columns
    required_cols = [
        "chunk_size_bytes",
        "shannon_entropy",
        "mad",
        "second_derivative_mean",
        "library_config_id",
        "config_fast",
        "config_balanced",
        "config_best",
        "data_type_char",
        "data_type_float",
        "compression_ratio",
        "psnr_db",
        "compression_time_ms",
        "success",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing CSV columns: {missing}", file=sys.stderr)
        return 1

    if df.empty:
        print("Error: CSV is empty", file=sys.stderr)
        return 1

    # Filter success == 1
    df = df[df["success"] == 1].copy()
    if df.empty:
        print("Error: No rows with success=1", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} samples")

    # Build 11-feature matrix with feature names in order
    X = pd.DataFrame()
    X["chunk_size_bytes"] = df["chunk_size_bytes"]
    X["target_cpu_util"] = 50.0  # Synthesized constant
    X["shannon_entropy"] = df["shannon_entropy"]
    X["mad"] = df["mad"]
    X["second_derivative_mean"] = df["second_derivative_mean"]
    X["library_config_id"] = df["library_config_id"]
    X["config_fast"] = df["config_fast"]
    X["config_balanced"] = df["config_balanced"]
    X["config_best"] = df["config_best"]
    X["data_type_char"] = df["data_type_char"]
    X["data_type_float"] = df["data_type_float"]

    feature_names = list(X.columns)

    # Build bin edges using percentiles (for continuous features)
    # Discrete features (library_config_id, config_*) are handled separately
    bin_edges = [[] for _ in range(11)]

    # For continuous features (indices 0, 2, 3, 4): use percentile-based bins
    continuous_indices = [0, 2, 3, 4]
    for idx in continuous_indices:
        values = sorted(X.iloc[:, idx].unique())
        edges = []
        for i in range(1, args.bins):
            percentile = i / args.bins
            pos = int(percentile * (len(values) - 1))
            pos = min(pos, len(values) - 1)
            edge_val = values[pos]
            if not edges or edges[-1] != edge_val:
                edges.append(edge_val)
        bin_edges[idx] = edges

    print(f"Built bin edges for {args.bins} bins")

    # Discretize features into state vectors
    qtable = defaultdict(lambda: {"ratio": [], "psnr": [], "time": []})

    for idx, row in df.iterrows():
        state = [0] * 11

        # Discrete features: library_config_id, config_*, data_type_*
        state[5] = int(X.iloc[idx]["library_config_id"])  # library_config_id

        # config_* are one-hot encoded
        if X.iloc[idx]["config_fast"] > 0.5:
            state[6] = 1  # config_fast
        elif X.iloc[idx]["config_best"] > 0.5:
            state[8] = 1  # config_best
        elif X.iloc[idx]["config_balanced"] > 0.5:
            state[7] = 1  # config_balanced

        # data_type_* are one-hot encoded
        if X.iloc[idx]["data_type_char"] > 0.5:
            state[9] = 1  # data_type_char
        elif X.iloc[idx]["data_type_float"] > 0.5:
            state[10] = 1  # data_type_float

        # Continuous features: chunk_size_bytes, shannon_entropy, mad, second_derivative_mean
        for i, cont_idx in enumerate(continuous_indices):
            value = X.iloc[idx, cont_idx]
            edges = bin_edges[cont_idx]
            if edges:
                # Binary search to find bin index
                bin_idx = 0
                for edge in edges:
                    if value >= edge:
                        bin_idx += 1
                bin_idx = min(bin_idx, args.bins - 1)
            else:
                bin_idx = 0
            state[cont_idx] = bin_idx

        state_key = tuple(state)
        qtable[state_key]["ratio"].append(df.iloc[idx]["compression_ratio"])
        qtable[state_key]["psnr"].append(df.iloc[idx]["psnr_db"])
        qtable[state_key]["time"].append(df.iloc[idx]["compression_time_ms"])

    # Compute global averages
    all_ratios = []
    all_psnrs = []
    all_times = []
    for labels in qtable.values():
        all_ratios.extend(labels["ratio"])
        all_psnrs.extend(labels["psnr"])
        all_times.extend(labels["time"])

    global_avg_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0
    global_avg_psnr = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0.0
    global_avg_time = sum(all_times) / len(all_times) if all_times else 0.0

    print(
        f"Q-table: {len(qtable)} states, "
        f"global avg ratio={global_avg_ratio:.4f}, "
        f"psnr={global_avg_psnr:.4f}, "
        f"time={global_avg_time:.4f}"
    )

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Write qtable.csv
    qtable_path = os.path.join(args.out_dir, "qtable.csv")
    with open(qtable_path, "w") as f:
        # Config section
        f.write("# Q-Table Configuration\n")
        f.write(f"n_bins={args.bins}\n")
        f.write("use_nearest_neighbor=0\n")
        f.write("nn_k=3\n")

        # Global average
        f.write("global_avg_ratio,global_avg_psnr,global_avg_time,global_avg_count\n")
        f.write(f"{global_avg_ratio},{global_avg_psnr},{global_avg_time},{len(df)}\n")

        # Q-table header
        f.write("# Q-Table States\n")
        f.write(
            "bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10,"
            "ratio,psnr,time,count\n"
        )

        # Write states (sorted for consistency)
        for state in sorted(qtable.keys()):
            labels = qtable[state]
            avg_ratio = sum(labels["ratio"]) / len(labels["ratio"])
            avg_psnr = sum(labels["psnr"]) / len(labels["psnr"])
            avg_time = sum(labels["time"]) / len(labels["time"])
            count = len(labels["ratio"])

            state_str = ",".join(str(b) for b in state)
            f.write(f"{state_str},{avg_ratio},{avg_psnr},{avg_time},{count}\n")

    print(f"Saved Q-table to {qtable_path}")

    # Write binning.csv
    binning_path = os.path.join(args.out_dir, "binning.csv")
    with open(binning_path, "w") as f:
        f.write("# Binning Edges (feature 0 to 10)\n")
        for edges in bin_edges:
            if edges:
                f.write(",".join(str(e) for e in edges) + "\n")
            else:
                f.write("\n")

    print(f"Saved binning edges to {binning_path}")
    print("Q-table training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
