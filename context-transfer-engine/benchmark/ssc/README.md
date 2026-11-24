# SSC: model-driven composition of the CTE chain

A controller that chooses the CTE module chain (placement, cache, indexer, compressor) for each
workload window, using a fast language model and a **Storage System Context (SSC)** built from
measurements on the node. Python only; it drives a running CLIO runtime through `clio_run` and the
`clio_cte_core_ext` Python extension.

## Files

| File | Role |
|---|---|
| `workload.py` | Gray-Scott-derived workload phases (output, analysis, products, log, archive, field re-read, events) |
| `compose.py`, `lifecycle.py` | Render the compose YAML for one composition; start/stop the runtime |
| `measure.py`, `run_grid.py` | Measure every phase under every composition (the node's grid) |
| `ssc_v8.py` | SSC builders: v9, **v10 (frozen)**, v11; the no-model lookup baselines |
| `ssc_live.py`, `bench_shared.py` | Earlier SSC (v2/v3) and the question text shared with SSC-Bench |
| `e2e.py`, `step.py`, `controller.py` | Live loop: run a window, build the SSC, ask the model, recompose |
| `ablate_ssc.py` | Offline replay: score any controller against a measured grid |
| `jev_bridge.py`, `s1_bridge.py` | File-queue bridges to the model server (Ollama; `<model>+think` turns thinking on) |
| `run_grid_v2.sh`, `run_test_v10.sh`, `run_short_qwen.sh`, `replay_v2.sh` | The runs behind the current results |

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `CLIO_BIN` | `/build/bin` | CLIO build `bin/` (`clio_run`, the Python extension) |
| `SSC_ROOT` | `/tmp/ssc_live` | Runtime config and the SSD-tier file |
| `SSC_DRAM_GB`, `SSC_DISK_GB` | `1`, `2` | Tier sizes |
| `SSC_DISK_PERSISTENCE` | `long_term` | SSD tier persistence (cache copies then stay in DRAM) |
| `SSC_CACHE_OWNER_COPIES` | `true` | Single-node read cache (`owner_copies`, needs the cache fix on this branch) |
| `SSC_QUEUE_ROOT` | `/ssc-bench` | Parent of the model request queues (`live_queue_s1_<model>`) |

## Typical runs

```bash
# 1. measure the node (about 1 h per node size)
SSC_DRAM_GB=1 SSC_DISK_GB=2 python3 run_grid.py grid_1g.jsonl --repeat 3
# 2. offline: frozen SSC v10 on the test schedules (model bridge must be running)
SSC_DRAM_GB=1 SSC_DISK_GB=2 python3 ablate_ssc.py out.jsonl --seeds 101-110 --grid grid_1g.jsonl \
    --variants "const,rulesh,none@qwen3:4b,v10@qwen3:4b"
python3 ablate_ssc.py --summary out.jsonl
# 3. live: one schedule, qwen3-4B with the SSC, thinking off
python3 e2e.py s1:qwen3:4b run.jsonl --schedule seed:201 --ssc v10 --rules --grid grid_1g.jsonl
# model side (where Ollama runs): python3 s1_bridge.py $SSC_QUEUE_ROOT/live_queue_s1_qwen3_4b qwen3:4b
```
