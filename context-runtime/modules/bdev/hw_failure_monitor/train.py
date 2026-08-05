import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import weibull_min
import joblib

# Backblaze HDD SMART attributes (_raw suffix, based on Backblaze failure research).
# These are the most predictive attributes for HDD failure identified across
# millions of drives in the Backblaze fleet.
SMART_HDD = [
    'smart_5_raw',   # Reallocated Sectors Count     — strongest single predictor
    'smart_187_raw', # Reported Uncorrectable Errors
    'smart_188_raw', # Command Timeout
    'smart_197_raw', # Current Pending Sector Count
    'smart_198_raw', # Uncorrectable Sector Count
    'smart_1_raw',   # Read Error Rate
    'smart_7_raw',   # Seek Error Rate
    'smart_9_raw',   # Power On Hours               — drive age proxy
    'smart_10_raw',  # Spin Retry Count
    'smart_192_raw', # Power-Off Retract Count
    'smart_193_raw', # Load/Unload Cycle Count
    'smart_194_raw', # Temperature
]
SMART_SSD = [
    'smart_5',   # Reallocated Sectors Count
    'smart_9',   # Power On Hours                   — drive age proxy
    'smart_173', # Wear Leveling Count
    'smart_174', # Unexpected Power Loss Count
    'smart_177', # Wear Range Delta
    'smart_183', # SATA Downshift Error Count
    'smart_187', # Reported Uncorrectable Errors
    'smart_194', # Temperature
    'smart_230', # Media Wearout Indicator
    'smart_241', # Total LBAs Written
    'smart_242', # Total LBAs Read
]

# Canonical names after stripping _raw (used as model feature names)
SMART_HDD_CANONICAL = [
    'smart_5', 'smart_187', 'smart_188', 'smart_197', 'smart_198',
    'smart_1', 'smart_7', 'smart_9', 'smart_10', 'smart_192', 'smart_193', 'smart_194',
]

def load_failure_dates(data_dir):
    """
    Load SSD failure label file and return a dict mapping disk_id -> failure_date.

    Args:
        data_dir: Directory containing ssd_failure_label.csv.

    Returns:
        Dict of {str(disk_id): datetime.date} or empty dict if file not found.
    """
    failure_file = os.path.join(data_dir, 'ssd_failure_label.csv')
    if not os.path.exists(failure_file):
        return {}
    fail_df = pd.read_csv(failure_file, usecols=['disk_id', 'failure_time'])
    fail_df['failure_date'] = pd.to_datetime(fail_df['failure_time'], errors='coerce').dt.date
    fail_df = fail_df.dropna(subset=['failure_date'])
    return dict(zip(fail_df['disk_id'].astype(str), fail_df['failure_date']))


def load_files(filepaths, drive_type, failure_dates=None, label_window_days=30):
    """
    Load a list of CSV files into a single DataFrame with memory-efficient dtypes.

    For SSD (Alibaba) data, applies a precise 30-day failure label window instead
    of tagging the entire drive history as failure=1.

    Args:
        filepaths:         Absolute paths to CSV files to load.
        drive_type:        'hdd' or 'ssd'.
        failure_dates:     Dict of {disk_id: failure_date} for SSD labeling.
        label_window_days: Only label the final N days before failure as failure=1.

    Returns:
        Concatenated DataFrame with standardised column names.
    """
    import datetime
    features = SMART_HDD if drive_type == 'hdd' else SMART_SSD
    keep_cols = ['serial_number', 'date', 'failure'] + features
    dfs = []

    for filepath in filepaths:
        sample_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
        is_alibaba = 'disk_id' in sample_cols

        if is_alibaba:
            alibaba_features = ['r_' + f.split('_')[1] for f in features if f.startswith('smart_')]
            usecols = ['disk_id', 'ds'] + [c for c in alibaba_features if c in sample_cols]
            if 'failure' in sample_cols:
                usecols.append('failure')
        else:
            usecols = [c for c in keep_cols if c in sample_cols]

        dtype_dict = {}
        if 'failure' in usecols:
            dtype_dict['failure'] = 'int8'
        for c in usecols:
            if c.startswith('smart_') or (is_alibaba and c.startswith('r_')):
                dtype_dict[c] = 'float32'

        df = pd.read_csv(filepath, usecols=usecols, dtype=dtype_dict)
        print(f"  Loaded {os.path.basename(filepath)}: {len(df):,} rows")

        if is_alibaba:
            rename_map = {'disk_id': 'serial_number', 'ds': 'date'}
            for c in usecols:
                if c.startswith('r_'):
                    rename_map[c] = f"smart_{c.split('_')[1]}"
            df = df.rename(columns=rename_map)

            # Fix #2: Only label the final `label_window_days` before actual failure as 1.
            # Previously the entire drive history was marked failure=1, causing label leakage.
            df['failure'] = 0
            if failure_dates:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                def label_row(row):
                    """Return 1 if this row falls within the failure window."""
                    fd = failure_dates.get(str(row['serial_number']))
                    if fd is None:
                        return 0
                    row_date = row['date'].date() if pd.notna(row['date']) else None
                    if row_date is None:
                        return 0
                    delta = (fd - row_date).days
                    return 1 if 0 <= delta <= label_window_days else 0
                df['failure'] = df.apply(label_row, axis=1).astype('int8')

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    full_df = pd.concat(dfs, ignore_index=True)
    if 'date' in full_df.columns:
        full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce')
    return full_df

def create_features(df, drive_type, rolling_days=7):
    """
    Create both static and rolling delta features.
    """
    print("Creating features...")
    features = SMART_HDD if drive_type == 'hdd' else SMART_SSD
    canonical = SMART_HDD_CANONICAL if drive_type == 'hdd' else SMART_SSD
    
    # For Backblaze HDD data: rename smart_X_raw -> smart_X so columns are canonical
    if drive_type == 'hdd':
        rename_raw = {raw: canon for raw, canon in zip(SMART_HDD, SMART_HDD_CANONICAL)
                      if raw in df.columns}
        df = df.rename(columns=rename_raw)
    
    # Check if features exist, if not, fill with 0 for script robustness
    for f in canonical:
        if f not in df.columns:
            df[f] = 0.0
            
    if 'failure' not in df.columns:
        df['failure'] = 0 # Default if label is missing
        
    df = df.sort_values(by=['serial_number', 'date'])
    
    # Static features use canonical names
    static_features = canonical.copy()
    
    # Create rolling delta features using canonical names
    rolling_features = canonical.copy()
    for col in canonical:
        delta_col = f'{col}_delta'
        df[delta_col] = df.groupby('serial_number')[col].diff(periods=rolling_days)
        rolling_features.append(delta_col)
        
    # Drop rows with NaN in rolling features to train the rolling model cleanly
    rolling_df = df.dropna(subset=[f'{col}_delta' for col in canonical])
    
    return df, static_features, rolling_df, rolling_features

def fit_weibull_params(df, drive_type):
    """
    Fit Weibull survival parameters from actual failure data in the training set.

    Uses the maximum observed Power-On Hours (smart_9) per drive as the
    time-to-failure proxy. Drives that never failed are excluded (only true
    failure events are used to fit the distribution).

    Falls back to empirically validated defaults if fewer than 10 failures
    are observed (not enough data to fit reliably).

    Args:
        df:         Full training DataFrame with 'failure' column and 'smart_9'.
        drive_type: 'hdd' or 'ssd'.

    Returns:
        Dict with 'k' (shape) and 'lam' (scale in days).
    """
    DEFAULTS = {
        'hdd': {'k': 2.1, 'lam': 1730.0},
        'ssd': {'k': 1.8, 'lam': 2400.0},
    }

    if 'smart_9' not in df.columns or 'failure' not in df.columns:
        print("  Weibull: smart_9 not available, using defaults.")
        return DEFAULTS[drive_type]

    # Get max smart_9 (power-on hours) for drives that actually failed
    failed = df[df['failure'] == 1]
    if len(failed) == 0:
        print("  Weibull: no failure events found, using defaults.")
        return DEFAULTS[drive_type]

    failure_hours = (
        failed.groupby('serial_number')['smart_9']
        .max()
        .dropna()
    )
    failure_hours = failure_hours[failure_hours > 0]

    if len(failure_hours) < 10:
        print(f"  Weibull: only {len(failure_hours)} failure events — too few to fit, using defaults.")
        return DEFAULTS[drive_type]

    # Convert hours to days and fit Weibull
    failure_days = failure_hours / 24.0
    shape, _, scale = weibull_min.fit(failure_days, floc=0)
    result = {'k': round(float(shape), 4), 'lam': round(float(scale), 2)}
    print(f"  Weibull fit from {len(failure_hours)} failures: k={result['k']}, lam={result['lam']} days "
          f"(mean life ≈ {result['lam'] * 0.8873:.0f} days)")
    return result


def train_model(X, y, init_model=None):
    """
    Train (or continue training) a high-quality LightGBM gradient-boosted classifier.

    When init_model is provided, adds more trees on top of the existing booster
    enabling memory-safe incremental training across chunked datasets.

    Args:
        X:          Feature matrix (DataFrame).
        y:          Binary failure labels (Series).
        init_model: Optional existing LGBMClassifier to continue training from.

    Returns:
        Fitted LGBMClassifier.
    """
    model = LGBMClassifier(
        n_estimators=200,         # Per-chunk tree budget (×N_chunks = total trees)
        learning_rate=0.01,       # Low rate — precise convergence across chunks
        num_leaves=63,            # Richer trees (default is 31)
        max_depth=8,              # Prevent extreme depth overfitting
        min_child_samples=50,     # Need 50+ samples to form a leaf
        subsample=0.8,            # Use 80% of rows per tree (row bagging)
        subsample_freq=1,
        colsample_bytree=0.8,     # Use 80% of features per tree
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=0.1,           # L2 regularization
        class_weight='balanced',  # Critical for rare-failure imbalance
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    if init_model is not None:
        model.fit(X, y, init_model=init_model.booster_)
    else:
        model.fit(X, y)
    return model

def get_all_csvs(data_dir):
    """
    Recursively find all CSV files in data_dir, excluding failure label files.

    Args:
        data_dir: Root directory to search.

    Returns:
        Sorted list of absolute CSV file paths.
    """
    all_csvs = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    all_csvs = [f for f in all_csvs if 'failure_label' not in os.path.basename(f)]
    return sorted(all_csvs)


def main():
    parser = argparse.ArgumentParser(description="Train Drive Failure Models")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--drive-type', type=str, choices=['ssd', 'hdd'], required=True)
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--limit-files', type=int, default=None,
                        help="Cap total CSV files (memory safety). Default: use all.")
    parser.add_argument('--chunk-size', type=int, default=15,
                        help="Files per training chunk for incremental learning (default: 15).")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover files
    all_csvs = get_all_csvs(args.data_dir)
    if not all_csvs:
        print("Error: No CSV files found.")
        return
    if args.limit_files:
        all_csvs = all_csvs[:args.limit_files]
    print(f"Found {len(all_csvs)} CSV files. Training in chunks of {args.chunk_size}.")

    # Fix #2: Pre-load SSD failure dates for precise 30-day window labeling
    failure_dates = load_failure_dates(args.data_dir) if args.drive_type == 'ssd' else {}
    if failure_dates:
        print(f"Loaded failure dates for {len(failure_dates)} drives (30-day label window active).")

    # Chunked incremental training — only the static model.
    # The rolling model is always a copy of the static model.
    # Rolling predictions are enriched at inference time using live SQLite history.
    chunks = [all_csvs[i:i + args.chunk_size] for i in range(0, len(all_csvs), args.chunk_size)]
    static_model = None
    all_dfs = []  # Small samples collected for Weibull fitting

    for chunk_idx, chunk_files in enumerate(chunks):
        print(f"\n--- Chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_files)} files) ---")
        df = load_files(chunk_files, args.drive_type, failure_dates=failure_dates)
        if df.empty:
            continue

        # Collect sample for Weibull fitting — rename smart_9_raw -> smart_9 here
        # since create_features() does this rename and we sample before that step.
        sample = df.sample(min(len(df), 200_000), random_state=42).copy()
        if 'smart_9_raw' in sample.columns and 'smart_9' not in sample.columns:
            sample = sample.rename(columns={'smart_9_raw': 'smart_9'})
        keep = ['serial_number', 'failure']
        if 'smart_9' in sample.columns:
            keep.append('smart_9')
        all_dfs.append(sample[keep])

        static_df, static_cols, _, _ = create_features(df, args.drive_type)

        print("  Training static model...")
        X_s = static_df[static_cols].fillna(0)
        y_s = static_df['failure']
        static_model = train_model(X_s, y_s, init_model=static_model)

        del df, static_df  # Free RAM immediately after each chunk

    if static_model is None:
        print("Error: no data could be loaded.")
        return

    # Save static model
    static_path = os.path.join(args.output_dir, f'{args.drive_type}_static_model.pkl')
    joblib.dump(static_model, static_path)
    total_trees = static_model.booster_.num_trees()
    print(f"\nStatic model saved: {static_path} ({total_trees} total trees)")

    # Rolling model = exact copy of static model.
    # After 7 days of live drive readings the server computes delta features
    # from SQLite history and feeds them to this same model for richer predictions.
    rolling_path = os.path.join(args.output_dir, f'{args.drive_type}_rolling_model.pkl')
    joblib.dump(static_model, rolling_path)
    print(f"Rolling model saved: {rolling_path} (copy of static — enriched by server after 7 days of live data)")

    # Fit Weibull parameters from actual failure data in the training set
    print("\nFitting Weibull survival parameters from training data...")
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    weibull_params = fit_weibull_params(combined, args.drive_type)
    weibull_path = os.path.join(args.output_dir, f'{args.drive_type}_weibull_params.json')
    with open(weibull_path, 'w') as f:
        json.dump(weibull_params, f, indent=2)
    print(f"Weibull params saved: {weibull_path} -> {weibull_params}")

    print(f"\nAll models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
