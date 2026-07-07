# Compression-Metric Model Training

Unified Python training scripts for compression-metric prediction models in the Clio compression transport subsystem.

## CSV Input Schema

All trainers consume a single unified benchmark CSV with the following columns:

```
library, preset, distribution, data_type, chunk_size_bytes, shannon_entropy,
mad, second_derivative_mean, library_config_id, config_fast, config_balanced,
config_best, data_type_char, data_type_float, quantize, byte_shuffle,
error_bound, original_bytes, compressed_bytes, compression_ratio,
compression_time_ms, decompression_time_ms, psnr_db, success
```

### Filtering
- Only rows with `success == 1` are used for training.
- Empty or malformed CSV raises an error.

### Generated CSV and Preprocessor Sweep
The unified benchmark CSV is produced by the `benchmark_compress_metrics_exec` (defined in `/workspace/test/unit/compress/model`). Run it to generate the training data:

```bash
# Basic run (no preprocessor variants)
benchmark_compress_metrics_exec --rows 50 --chunk-bytes 1048576 --out metrics.csv

# With preprocessor sweep (also emits byte-shuffle variants)
benchmark_compress_metrics_exec --rows 50 --chunk-bytes 1048576 --preprocessors --out metrics.csv
```

The `--preprocessors` flag causes the benchmark to emit additional rows for each baseline configuration with `byte_shuffle=1` set, allowing the models (especially NeuroPress NN) to learn the effect of preprocessing.

## The 11 Model Features

All trainers use the same 11-feature vector (in this order):

1. **chunk_size_bytes** — Size of the data chunk in bytes
2. **target_cpu_util** — Target CPU utilization (0–100%). **Synthesized as constant 50.0** in training (not in CSV)
3. **shannon_entropy** — Shannon entropy in bits/byte (0–8)
4. **mad** — Mean absolute deviation
5. **second_derivative_mean** — Mean second derivative (curvature)
6. **library_config_id** — Encodes library + preset; library_id = config_id // 10, preset_id = config_id % 10
7. **config_fast** — One-hot: 1 if FAST preset
8. **config_balanced** — One-hot: 1 if BALANCED preset
9. **config_best** — One-hot: 1 if BEST preset
10. **data_type_char** — One-hot: 1 if char/int data
11. **data_type_float** — One-hot: 1 if float data

## Training Labels

All trainers predict three outputs (multi-output regression):

- **compression_ratio** — Compressed / original bytes (>1 means smaller)
- **psnr_db** — Peak signal-to-noise ratio in dB (0 for lossless)
- **compression_time_ms** — Time to compress in milliseconds

## Model Artifacts and C++ Loaders

Each trainer emits artifact files that correspond to a specific C++ predictor class:

### 1. XGBoost (`train_xgboost.py`)

**Output files:**
- `compression_ratio_model.json` — JSON model for compression-ratio prediction
- `psnr_model.json` — JSON model for PSNR prediction
- `compression_time_model.json` — JSON model for compression-time prediction

**C++ Loader:** `ctp::compress::model::XGBoostPredictor` (requires XGBoost C API; optional)

**Command:**
```bash
python3 train_xgboost.py --csv metrics.csv --out-dir ./xgboost_models
```

**Parameters:**
- `--csv` (required) — Path to the input CSV
- `--out-dir` (required) — Output directory for the three JSON files
- `--cv` (optional) — Enable cross-validation (reserved for future use)

**Python dependencies:** `pandas`, `xgboost`

### 2. Q-Table (`train_qtable.py`)

**Output files:**
- `qtable.csv` — Discretized states and aggregated label averages
- `binning.csv` — Bin edges for continuous features

**C++ Loader:** `ctp::compress::model::QTablePredictor` (pure C++, no external deps)

**Command:**
```bash
python3 train_qtable.py --csv metrics.csv --out-dir ./qtable_models --bins 5
```

**Parameters:**
- `--csv` (required) — Path to the input CSV
- `--out-dir` (required) — Output directory for `qtable.csv` and `binning.csv`
- `--bins` (optional, default=5) — Number of bins per continuous feature (quantization resolution)

**Python dependencies:** `pandas`

**Format details:**

`qtable.csv` structure:
```
# Q-Table Configuration
n_bins=5
use_nearest_neighbor=0
nn_k=3
global_avg_ratio,global_avg_psnr,global_avg_time,global_avg_count
<ratio>,<psnr>,<time>,<count>
# Q-Table States
bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10,ratio,psnr,time,count
<state_bins>...,<avg_ratio>,<avg_psnr>,<avg_time>,<sample_count>
...
```

`binning.csv` structure:
```
# Binning Edges (feature 0 to 10)
<edges for feature 0 (chunk_size_bytes)>
<edges for feature 1 (target_cpu_util)>
<edges for feature 2 (shannon_entropy)>
<edges for feature 3 (mad)>
<edges for feature 4 (second_derivative_mean)>
<empty for discrete features 5-10>
...
```

### 3. NeuroPress Dense NN (`train_neuropress_nn.py`)

**Output file:**
- `model.nnwt` — Binary neural-network weights file (little-endian)

**C++ Loader:** `ctp::compress::model::NeuroPressNNPredictor`

**Command:**
```bash
python3 train_neuropress_nn.py --csv metrics.csv --out-dir ./neuropress_models \
  --epochs 100 --lr 0.001
```

**Parameters:**
- `--csv` (required) — Path to the input CSV
- `--out-dir` (required) — Output directory for `model.nnwt`
- `--epochs` (optional, default=100) — Number of training epochs
- `--lr` (optional, default=0.001) — Learning rate

**Python dependencies:** `pandas`, `numpy`, optionally `torch` (PyTorch)

**Architecture:**
- Inputs: 8-dimensional feature vector (see below)
- Hidden layers: 4 × 64 units with ReLU activation
- Output: 8-dimensional prediction vector
- Activation: ReLU hidden, linear output

**Input feature mapping** (8-dim):
1. `algo_id` = (library_config_id // 10) % 8
2. `quant` = quantize (from CSV, default 0 if missing)
3. `shuffle` = byte_shuffle (from CSV, default 0 if missing)
4. `error_bound` = error_bound (from CSV, default 0 if missing)
5. `data_size` = chunk_size_bytes
6. `entropy` = shannon_entropy
7. `mad`
8. `second_derivative_mean`

**Output labels** (8-dim):
1. `compression_time_ms`
2. `decompression_time_ms` (synthesized as mean compression_time_ms if unavailable)
3. `compression_ratio`
4. `psnr_db`
5–8. Padding zeros

**Binary format** (`model.nnwt`):
```
[Header: 24 bytes]
  uint32 magic = 0x4E4E5754 ('NNWT')
  uint32 version = 2
  uint32 num_layers = 5
  uint32 input_dim = 8
  uint32 hidden_dim = 64
  uint32 output_dim = 8

[Normalization: 32 floats]
  float x_means[8]
  float x_stds[8]
  float y_means[8]
  float y_stds[8]

[Layer weights and biases]
  For each of 5 layers (in order):
    float weights[fan_out][fan_in]  (row-major flattened)
    float biases[fan_out]

[Feature bounds: 16 floats]
  float x_mins[8]
  float x_maxs[8]
```

All floats are IEEE 754 single precision (4 bytes each). The file is written in little-endian byte order.

After training, the script verifies:
- File size matches the expected layout
- Magic number and version are correct

**Training modes:**
- **PyTorch (preferred):** Uses `torch.nn` with Adam optimizer if available
- **NumPy fallback:** Simple SGD implementation if PyTorch is not installed

### Preprocessor Column Usage
- **NeuroPress NN (`train_neuropress_nn.py`)**: Reads `quantize`, `byte_shuffle`, `error_bound` columns from CSV if present (defaults to 0 if missing for backward compatibility)
- **XGBoost (`train_xgboost.py`)**: Uses the 11-feature `ToVector()` only; preprocessor columns are ignored
- **Q-Table (`train_qtable.py`)**: Uses the 11-feature `ToVector()` only; preprocessor columns are ignored

## Python Dependencies

**Required for all scripts:**
- `pandas` — CSV I/O and data manipulation

**Required by specific trainers:**
- `train_xgboost.py`: `xgboost`
- `train_qtable.py`: (none beyond pandas)
- `train_neuropress_nn.py`: `numpy` (required), `torch` (optional, fallback to numpy SGD)

**Installation:**
```bash
pip install pandas xgboost numpy torch
```

Or selectively:
```bash
pip install pandas xgboost numpy  # For XGBoost and Q-table
pip install torch  # Optional, for faster NeuroPress training
```

## Usage Examples

### Full pipeline (all three models)

```bash
# Generate metrics CSV
benchmark_compress_metrics_exec --output-csv metrics.csv

# Train XGBoost
python3 train_xgboost.py --csv metrics.csv --out-dir ./models/xgboost

# Train Q-table
python3 train_qtable.py --csv metrics.csv --out-dir ./models/qtable --bins 5

# Train NeuroPress NN
python3 train_neuropress_nn.py --csv metrics.csv --out-dir ./models/neuropress \
  --epochs 100 --lr 0.001
```

### Loading in C++

See the C++ predictor classes in `/workspace/context-transport-primitives/include/clio_ctp/compress/model/`:
- `XGBoostPredictor` — Load from `--out-dir` containing JSON files
- `QTablePredictor` — Load from `--out-dir` containing CSV files
- `NeuroPressNNPredictor` — Load from `--out-dir` containing `model.nnwt`

All implement the unified `CompressionPredictor` interface.

## Implementation Notes

- **Feature standardization:** All inputs and outputs are standardized (mean=0, std=1) before training. Statistics are stored with each model for inference-time normalization.
- **Robustness:** Scripts validate required columns, check for empty CSVs, and warn on missing dependencies.
- **Determinism:** XGBoost and NeuroPress trainers use fixed random seeds for reproducibility.
- **Error handling:** All scripts exit with status 1 on error, 0 on success.

## Testing

### Compilation check
```bash
python3 -m py_compile train_xgboost.py
python3 -m py_compile train_qtable.py
python3 -m py_compile train_neuropress_nn.py
```

### Help text
```bash
python3 train_xgboost.py --help
python3 train_qtable.py --help
python3 train_neuropress_nn.py --help
```

### Minimal test (if dependencies are available)
```bash
# Create a small dummy CSV
python3 -c "
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'chunk_size_bytes': [1024, 2048, 4096],
    'shannon_entropy': [5.0, 6.0, 7.0],
    'mad': [10.0, 20.0, 30.0],
    'second_derivative_mean': [0.1, 0.2, 0.3],
    'library_config_id': [11, 22, 33],
    'config_fast': [1, 0, 0],
    'config_balanced': [0, 1, 0],
    'config_best': [0, 0, 1],
    'data_type_char': [1, 0, 0],
    'data_type_float': [0, 1, 1],
    'compression_ratio': [2.0, 2.5, 3.0],
    'psnr_db': [50.0, 55.0, 60.0],
    'compression_time_ms': [10.0, 15.0, 20.0],
    'success': [1, 1, 1],
})
df.to_csv('test.csv', index=False)
"

# Run trainers
python3 train_qtable.py --csv test.csv --out-dir ./test_qtable
python3 train_neuropress_nn.py --csv test.csv --out-dir ./test_neuropress --epochs 10
```
