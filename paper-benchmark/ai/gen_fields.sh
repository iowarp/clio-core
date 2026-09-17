#!/usr/bin/env bash
# Train the AI workload once and dump its checkpoints as float32 files, which
# every evaluation then replays (nyx/run_config.sh --fields).
#
#   ./gen_fields.sh --out DIR [--cache DIR] [--epochs N]
#                   [--checkpoint-epochs LIST] [--max-batches N]
#
# Runs upstream NeuroPress's exporter, scripts/train_and_export_checkpoints.py
# from $NEUROPRESS_DIR, unmodified: ViT-B/16 fine-tuned on CIFAR-10, batch 64,
# 20 batches per epoch, no validation. Each checkpoint is four 327 MiB tensors
# (weights, adam_m, adam_v, gradients), moved to epochNN/<tensor>.f32 for the
# replay driver. Default: a checkpoint at epoch 1 and every 5th to 50, 11 in
# all (1,804 chunks at 8 MiB); ~9 s per epoch on an A100. --cache holds the torchvision weights and
# CIFAR-10; fetch them on a login node.
#
# Environment: NEUROPRESS_DIR, PYTHON, GEN_TIMEOUT (s; 0 = no limit).
set -euo pipefail

NEUROPRESS_DIR=${NEUROPRESS_DIR:-$HOME/NeuroPress}
PYTHON=${PYTHON:-python3}
OUT="" CACHE="" EPOCHS=50 CKPTS="" MAXB=20 WORKERS=8

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --cache) CACHE=$2; shift 2 ;;
    --epochs) EPOCHS=$2; shift 2 ;;
    --checkpoint-epochs) CKPTS=$2; shift 2 ;;
    --max-batches) MAXB=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "--out DIR is required" >&2; exit 2; }
CACHE=${CACHE:-$(dirname "$OUT")/cache}
# 1, then every 5th epoch, as upstream's package
[ -n "$CKPTS" ] || CKPTS=$(echo 1 $(seq 5 5 "$EPOCHS") | tr ' ' ',')
SCRIPT=$NEUROPRESS_DIR/scripts/train_and_export_checkpoints.py
[ -f "$SCRIPT" ] || { echo "no upstream exporter at $SCRIPT" >&2; exit 1; }
[ -d "$CACHE/cifar10/cifar-10-batches-py" ] || [ -f "$CACHE/cifar10/cifar-10-python.tar.gz" ] || {
  echo "no CIFAR-10 under $CACHE/cifar10 -- fetch it on a login node first" >&2; exit 1; }

mkdir -p "$OUT"
echo "== ViT-B/16 on CIFAR-10: $EPOCHS epochs x $MAXB batches, checkpoints $CKPTS -> $OUT"
rc=0
TORCH_HOME=$CACHE/torch timeout -k 10 "${GEN_TIMEOUT:-0}" \
  "$PYTHON" "$SCRIPT" --model vit_b_16 --dataset cifar10 \
  --epochs "$EPOCHS" --checkpoint-epochs "$CKPTS" --batch-size 64 \
  --max-batches-per-epoch "$MAXB" --no-validate --num-workers "$WORKERS" \
  --outdir "$OUT" --data-root "$CACHE" || rc=$?

# Lay out whatever was exported, even after a timeout; drop a partial checkpoint.
shopt -s nullglob
for f in "$OUT"/epoch[0-9][0-9]_*.f32; do
  b=$(basename "$f" .f32)
  mkdir -p "$OUT/${b%%_*}"
  mv "$f" "$OUT/${b%%_*}/${b#*_}.f32"
done
for d in "$OUT"/epoch[0-9][0-9]; do
  n=$(find "$d" -name '*.f32' | wc -l)
  sizes=$(find "$d" -name '*.f32' -printf '%s\n' | sort -u | wc -l)
  [ "$n" -eq 4 ] && [ "$sizes" -eq 1 ] || {
    echo "   $(basename "$d"): $n of 4 tensors, $sizes size(s) -- incomplete, removed"; rm -rf "$d"; }
done

N=$(find "$OUT" -name '*.f32' | wc -l)
DONE=$(cd "$OUT" && ls -d epoch[0-9][0-9] 2>/dev/null | sed 's/epoch0*//' | paste -sd,)
cat > "$OUT/gen.json" <<JSON
{"workload":"ai-vit_b_16-cifar10","epochs":$EPOCHS,"checkpoint_epochs":"$CKPTS",
 "max_batches_per_epoch":$MAXB,"batch_size":64,"lr":1e-4,"weight_decay":0.01,
 "checkpoints":$(( N / 4 )),"exported_epochs":"$DONE","files":$N,"precision":"float32","route":"replay",
 "generator":"NeuroPress scripts/train_and_export_checkpoints.py @ $(git -C "$NEUROPRESS_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)",
 "exit":$rc}
JSON
echo "   $(( N / 4 )) checkpoints, $N tensor files, $(du -sh "$OUT" | cut -f1)"
exit $rc
