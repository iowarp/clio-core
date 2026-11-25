# Source on the host inside a Slurm job on Delta (GPU node). Sets paths, ports and helpers.
#   SSC_BASE: shared workspace (campaign data, Ollama binary/models, python extras, results)
#   node-local state (CLIO tiers, model queues) lives under /tmp, unique per job
: "${SSC_BASE:=/u/rpawar/clio-core-ssc/ssc_work}"
: "${SIF:=/u/rpawar/containers/deps-nvidia.sif}"
: "${REPO:=/u/rpawar/clio-core-ssc}"
export SSC_BASE SIF REPO
export HERE=$REPO/context-transfer-engine/benchmark/ssc
export CLIO_BIN=$REPO/build/bin
export GS_CAMPAIGN=$SSC_BASE/gs_campaign
export PYTHONPATH=$SSC_BASE/pylib
JOB=${SLURM_JOB_ID:-$$}
export SSC_TMP=/tmp/ssc_$JOB
export SSC_ROOT=$SSC_TMP/root
export SSC_QUEUE_ROOT=$SSC_TMP/q
export JEV_QUEUE=$SSC_QUEUE_ROOT/live_queue
export SSC_PORT=$((9100 + JOB % 700))
export SSC_VIZ=false
export RES=$SSC_BASE/results
mkdir -p $SSC_ROOT $SSC_QUEUE_ROOT $RES

export PATH=$HERE/delta:$PATH      # ctr: run inside the container

# start_ollama <gpu> <port> <cpu list>: one server per GPU, models stay loaded
start_ollama() {
  CUDA_VISIBLE_DEVICES=$1 OLLAMA_HOST=127.0.0.1:$2 OLLAMA_MODELS=$SSC_BASE/models OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_MAX_LOADED_MODELS=6 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=16384 \
    taskset -c "$3" "$SSC_BASE/ollama/bin/ollama" serve > "$LOGDIR/ollama_g$1.log" 2>&1 &
  for _ in $(seq 120); do curl -s "http://127.0.0.1:$2/api/version" > /dev/null && return 0; sleep 1; done
  echo "ollama on gpu $1 did not start"; return 1
}

# start_bridge <model[+think]> <port> <queue root> <cpu list>: load the model, then serve its queue
start_bridge() {
  local m=${1%+think} q=$3/live_queue_s1_$(echo "$1" | tr ':.' '__')
  mkdir -p "$q"
  curl -s "http://127.0.0.1:$2/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":-1,\"options\":{\"num_ctx\":16384}}" > /dev/null
  OLLAMA_URL=http://127.0.0.1:$2 taskset -c "$4" apptainer exec -B /work,/u,/tmp "$SIF" \
    python3 "$HERE/s1_bridge.py" "$q" "$1" > "$LOGDIR/bridge_$(basename "$q")_$2.log" 2>&1 &
}
