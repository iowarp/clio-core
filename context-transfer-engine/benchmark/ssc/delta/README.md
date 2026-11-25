# Running the SSC benchmark on NCSA Delta

Everything runs on one A100x4 GPU node (`--exclusive`, so the storage measurements see no
neighbours). CLIO and the workload use cores 0-31; each GPU runs its own Ollama server, pinned to
8 of cores 32-63. CLIO's tiers and the model request queues live on the node-local NVMe (`/tmp`).

## One-time setup (login node)

```bash
B=/u/$USER/clio-core-ssc/ssc_work            # workspace inside the checkout (git-excluded)
mkdir -p $B/{logs,results,pylib,ollama,models}
SIF=/u/$USER/containers/deps-nvidia.sif        # the clio-core deps image (gcc 13, python 3.12, nanobind)
# 1. CLIO, CPU build with the Python extension and the compressor/cache/indexer modules (~25 min)
sbatch $B/build.sbatch                          # runs build_clio.sh inside $SIF on a CPU node
# 2. python extra for the campaign generator
apptainer exec $SIF python3 -m pip install --target $B/pylib zstandard
# 3. Gray-Scott campaign data (~1 min)
PYTHONPATH=$B/pylib apptainer exec -B /work,/u $SIF python3 gs_campaign.py $B/gs_campaign
# 4. Ollama (single static tarball) and the models; compute nodes need no internet afterwards
curl -L https://github.com/ollama/ollama/releases/download/v0.40.17/ollama-linux-amd64.tar.zst | tar --zstd -x -C $B/ollama
OLLAMA_MODELS=$B/models OLLAMA_HOST=127.0.0.1:11555 $B/ollama/bin/ollama serve &   # stop it afterwards
for m in qwen3:4b llama3.1:8b qwen3:1.7b qwen3:8b qwen3:14b; do OLLAMA_HOST=127.0.0.1:11555 $B/ollama/bin/ollama pull $m; done
```

The CMake options used for the build are in `build_clio.sh` (CTE, compression, Python, tests ON;
CUDA OFF; the model runs in Ollama, not in CLIO).

## Jobs

| Script | What | Node-hours |
|---|---|---|
| `smoke.sbatch` | CLIO + both bridges on 3 windows (interactive partition) | 0.2 |
| `campaign1.sbatch` | grids at 1 / 0.5 / 4 GB DRAM (10 phases, 3 repeats) + all offline replays | ~14 |
| `campaign2.sbatch` | live end-to-end runs (resumable), more System 2 offline schedules | ~20 |

`env.sh` holds every path and port (`SSC_PORT`, `OLLAMA_URL` per GPU) so several runs can share a
node without colliding. Results land in `ssc_work/results/{grid,offline,live}` (in the checkout); figures and the
summary table come from `python3 ../paper_plots.py $SSC_BASE/results <figdir>` on the host.
