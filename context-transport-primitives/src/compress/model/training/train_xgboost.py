#!/usr/bin/env python3
"""
XGBoost multi-output regressor trainer for compression-metric prediction.

Reads a unified benchmark CSV and trains 3 independent XGBoost regressors
(compression_ratio, psnr_db, compression_time_ms) using the 11-feature vector.
Emits one JSON model file per output to be loaded by the C++ XGBoostPredictor.

CSV schema:
  library, preset, distribution, data_type, chunk_size_bytes, shannon_entropy,
  mad, second_derivative_mean, library_config_id, config_fast, config_balanced,
  config_best, data_type_char, data_type_float, original_bytes, compressed_bytes,
  compression_ratio, compression_time_ms, decompression_time_ms, psnr_db, success

The 11 model features (in order):
  1. chunk_size_bytes
  2. target_cpu_util (synthesized as 50.0)
  3. shannon_entropy
  4. mad
  5. second_derivative_mean
  6. library_config_id
  7. config_fast
  8. config_balanced
  9. config_best
  10. data_type_char
  11. data_type_float

Labels:
  - compression_ratio
  - psnr_db
  - compression_time_ms

Filter: success == 1
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost multi-output compression-metric predictor"
    )
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out-dir", required=True, help="Output directory for models")
    parser.add_argument("--cv", action="store_true", help="Enable cross-validation")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed", file=sys.stderr)
        return 1

    try:
        import xgboost as xgb
    except ImportError:
        print("Error: xgboost not installed", file=sys.stderr)
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

    # Build 11-feature matrix
    features = []
    feature_names = [
        "chunk_size_bytes",
        "target_cpu_util",
        "shannon_entropy",
        "mad",
        "second_derivative_mean",
        "library_config_id",
        "config_fast",
        "config_balanced",
        "config_best",
        "data_type_char",
        "data_type_float",
    ]

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

    # Labels
    y_ratio = df["compression_ratio"]
    y_psnr = df["psnr_db"]
    y_time = df["compression_time_ms"]

    print(f"Loaded {len(df)} samples")
    print(f"Feature matrix shape: {X.shape}")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Train ratio model
    print("Training compression_ratio model...")
    ratio_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    ratio_model.fit(X, y_ratio)
    ratio_path = os.path.join(args.out_dir, "compression_ratio_model.json")
    ratio_model.get_booster().save_model(ratio_path)
    print(f"  Saved to {ratio_path}")

    # Train psnr model
    print("Training psnr_db model...")
    psnr_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    psnr_model.fit(X, y_psnr)
    psnr_path = os.path.join(args.out_dir, "psnr_model.json")
    psnr_model.get_booster().save_model(psnr_path)
    print(f"  Saved to {psnr_path}")

    # Train time model
    print("Training compression_time_ms model...")
    time_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    time_model.fit(X, y_time)
    time_path = os.path.join(args.out_dir, "compression_time_model.json")
    time_model.get_booster().save_model(time_path)
    print(f"  Saved to {time_path}")

    print("XGBoost training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
