#!/usr/bin/env bash
# Collect everything needed to rebuild Figure 8 panel (d) from CSV alone.
#
#   ./collect_fig8_full.sh [CAMPAIGN_TAG] [DEST]
#
# Files are HARD-LINKED (campaign output and destination share one filesystem),
# so the bundle costs no extra space and the originals are never touched.
set -euo pipefail

TAG=${1:-full2k-09181841}
ROOT=/projects/bekn/imuradli/np-hcompress
DEST=${2:-$ROOT/fig8_full}
OUT=$ROOT/out-$TAG
RUNS=$ROOT/runs/$TAG
REPO=/u/imuradli/clio-core

[ -d "$OUT" ]  || { echo "no campaign output at $OUT" >&2; exit 2; }
[ -d "$RUNS" ] || { echo "no run dir at $RUNS" >&2; exit 2; }

rm -rf "$DEST"
mkdir -p "$DEST"/{scored,inputs,logs,raw,figures,scripts}

link() { cp -l "$1" "$2" 2>/dev/null || cp "$1" "$2"; }

# --- scored results: the CSV the plot is drawn from, plus the accuracy tables
for f in fig8_models.csv accuracy_long.csv accuracy_long_unclamped.csv \
         accuracy_table.md accuracy_table.tex accuracy_table_unclamped.md \
         accuracy_table_unclamped.tex correctness.csv correctness_unclamped.csv \
         coverage.csv coverage_unclamped.csv; do
  [ -f "$OUT/$f" ] && link "$OUT/$f" "$DEST/scored/$f"
done

# --- scorer inputs: per-workload measurements + the shared seed corpus
for f in "$OUT"/inputs/seed.csv "$OUT"/inputs/seed_largest.csv \
         "$OUT"/inputs/hcompress_ccp_seed.json; do
  [ -f "$f" ] && link "$f" "$DEST/inputs/$(basename "$f")"
done
for d in "$OUT"/inputs/*/; do
  w=$(basename "$d"); mkdir -p "$DEST/inputs/$w"
  for f in eval.csv rows.csv hcompress.csv; do
    [ -f "$d$f" ] && link "$d$f" "$DEST/inputs/$w/$f"
  done
done

# --- logs: scoring, table build, per-workload console/runtime, SLURM stdout
for f in "$OUT"/fig8_models.log "$OUT"/make_table.log; do
  [ -f "$f" ] && link "$f" "$DEST/logs/$(basename "$f")"
done
for d in "$RUNS"/*/; do
  w=$(basename "$d"); mkdir -p "$DEST/logs/$w"
  [ -f "$d/console.log" ] && link "$d/console.log" "$DEST/logs/$w/console.log"
  for f in stdout.log runtime.log meta.json compose.yaml; do
    [ -f "$d/$w/$f" ] && link "$d/$w/$f" "$DEST/logs/$w/$f"
  done
done
for f in "$ROOT"/jobs/acc-*.out "$ROOT"/jobs/fig8dump-*.out; do
  [ -f "$f" ] && link "$f" "$DEST/logs/$(basename "$f")"
done

# --- raw selector traces, one level down from the scorer
for d in "$RUNS"/*/; do
  w=$(basename "$d"); mkdir -p "$DEST/raw/$w"
  for f in selection.csv selection.csv.payload selection.csv.quality \
           explore.csv blobs.csv dist.csv; do
    [ -f "$d/$w/$f" ] && link "$d/$w/$f" "$DEST/raw/$w/$f"
  done
done

# --- the exact code that produced all of it
for f in prepare_inputs.py accuracy_table.py fig8_model_chunks.py \
         hcompress_ccp_eval.cc run_campaign.sh make_table.sh \
         collect_fig8_full.sh; do
  [ -f "$REPO/paper-benchmark/model-accuracy/$f" ] && \
    cp "$REPO/paper-benchmark/model-accuracy/$f" "$DEST/scripts/$f"
done
cp "$REPO/paper-benchmark/plot/plot_fig8_models.py" "$DEST/scripts/" 2>/dev/null || true
for f in "$ROOT"/jobs/acc.sbatch "$ROOT"/jobs/fig8_dump.sbatch; do
  [ -f "$f" ] && cp "$f" "$DEST/scripts/$(basename "$f")"
done

# --- the figure itself
for f in "$REPO"/paper-benchmark/figures/fig8/full/fig8d_models.png; do
  [ -f "$f" ] && link "$f" "$DEST/figures/$(basename "$f")"
done

# --- provenance
{
  echo "campaign      : $TAG"
  echo "collected     : $(date -Is)"
  echo "host          : $(hostname)"
  echo "clio-core git : $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch        : $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty files   :"
  git -C "$REPO" status --porcelain 2>/dev/null | sed 's/^/  /'
} > "$DEST/PROVENANCE.txt"

( cd "$DEST" && find . -type f ! -name MANIFEST.txt -printf '%10s  %p\n' | sort -k2 ) > "$DEST/MANIFEST.txt"
echo "bundle -> $DEST  ($(du -sh --apparent-size "$DEST" | cut -f1) apparent, $(find "$DEST" -type f | wc -l) files)"
