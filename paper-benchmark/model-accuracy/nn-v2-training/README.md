# NeuroPress v2 model: training data and records

Everything used to train the NeuroPress v2 predictor
(`context-transport-primitives/src/compress/model/weights/v2/model_v2.{nnwt,json,pt}`).
The model is trained only on this synthetic corpus, never on benchmark
workload data.

## Contents

| File | What it is |
|---|---|
| `corpus_v2.csv.xz` | **The training CSV** (441,000 rows = 9,800 files x 45 lossless settings): measured compress / decompress time and ratio per file and setting, plus each file's statistics. `train_nn_v2.py` and `nn_v2_cv.py` read it by default (pandas reads `.xz` directly). |
| `sweep/raw_full_ok.csv.xz`, `sweep/raw_patch.csv` | Raw rows of the corpus sweep (`corpus_sweep`, 3 timed round trips per row, every one bit-exact); the patch re-measures one row that failed once. |
| `sweep/files.txt`, `sweep/settings_full.txt` | The 9,800 files and the 45 settings of the sweep. |
| `sweep/sweep_run.log` | The sweep's log (A100-PCIE-40GB, SM clock 1410 MHz). |
| `timing/` | The earlier timing probe (subset of files and settings), raw rows, lists and log. |
| `synthetic-9800.sha256.xz` | sha256 of every corpus file (the corpus itself is 11 GB and is regenerated, see below). |
| `synthetic-9800-stats.csv.xz` | Per-file statistics of the corpus, used by `eval_nn_v2_real.py`. |
| `logs/gen_synthetic.log`, `logs/cv_run.log`, `logs/train_v2.log` | Logs of the corpus generation, the 5-fold cross-validation and the final training. |

The cross-validation results that set the number of epochs are in
`../../codec-sweep/results/nn-v2/cv_training.csv` (with `cv_accuracy.csv`,
`cv_selection.csv`).

Checksums of the uncompressed CSVs:

```
b54bd5ef73f2ad8381acd31c823591770bbb91fbad9e6c51bc5d03fef2d46573  corpus_v2.csv
6421afc9eb429a8b4241d849713422baded81463adcb64029d944abc32c1ce72  raw_full_ok.csv
```

## Pipeline (all scripts are in this repository)

1. **Corpus.** `../generate_synthetic_corpus.py --output-dir DIR` writes the
   9,800 float32 files (10.2 GiB). The seeds are fixed: a regeneration on
   2026-10-06 matched every sha256 in `synthetic-9800.sha256.xz`:

   ```bash
   python3 ../generate_synthetic_corpus.py --output-dir DIR --workers 32
   (cd DIR && xz -dc .../synthetic-9800.sha256.xz | sha256sum -c --quiet -)
   ```
2. **Sweep.** `../../codec-sweep/corpus_sweep.cu` through
   `../../codec-sweep/run_corpus_sweep.sh` (`FILES=sweep/files.txt
   SETTINGS=sweep/settings_full.txt CORPUS=DIR OUT=...`) measures every file
   with every setting.
3. **CSV.** `../../codec-sweep/build_corpus_csv.py --corpus DIR --raw
   raw_full_ok.csv raw_patch.csv --files sweep/files.txt --settings
   sweep/settings_full.txt --out corpus_v2.csv` adds the file statistics; no
   value is rounded, floored or clamped.
4. **Cross-validation.** `../nn_v2_cv.py` (5 folds by file group) gives the
   accuracy and the best epoch per fold.
5. **Training.** `../train_nn_v2.py` trains model B (4 features -> 64 x 4
   ReLU -> 3 outputs per setting: log compress ms, log decompress ms, log
   ratio) on all files for the median best epoch of the folds (1540), and
   writes `model_v2.nnwt`, `.json`, `.pt`. Result (`logs/train_v2.log`): CV
   MAPE on unseen files 7.99% compress time, 8.13% decompress time, 10.09%
   ratio.
