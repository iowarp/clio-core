# clio::cte::dtschedule — design

The dtschedule chimod is the *policy* layer of DTSchedule. It decides, per
PutBlob, (1) the codec and preset, (2) the node that compresses, (3) the
node that holds the consumer-readable copy, and (4) the storage tier, then
dispatches the work to existing mechanisms: the `ctp::compress` factory
for codecs, the CTE core for storage, and the core's cache-replica slot
for consumer-side copies. It also records every decision in a trace so the
paper's figures come from measurements.

## 1. Position in the interposition chain

```
client / FUSE / POSIX / HDF5 / ADIOS2
  -> dtschedule (566.0)     THIS MODULE: runs on the SUBMITTER's node, decides,
                            compresses here or ships raw to the consumer
  -> cache (563.0)          node-local raw copies, replica-0 lookup
  -> indexer (564.0)
  -> replication (561.0)    durable copies (unchanged)
  -> core (512.0)           storage, hash-owned metadata
```

dtschedule sits at the TOP of the chain, not between indexer and
replication as first planned. Reason (verified in the tree): the cache's
authoritative write goes down with `PoolQuery::Dynamic()`, and every
interposer below it inherits the core's hash routing, so a module below the
cache executes PutBlob on the blob's hash OWNER after the raw bytes have
already crossed the network. Only a module above the cache can decide
anything about the producer→consumer transfer.

**Phase 4a: Chain Position Override**
dtschedule overrides `ScheduleTask` to execute Dynamic queries on data verbs
(PutBlob, GetBlob, GetBlobSize, MultiPutBlob) on the submitter's node
(`PoolQuery::Local()`), not on the hash owner. This allows dtschedule to make
local decisions about compression and placement. All other tasks (dtschedule
verbs and core non-data methods) use the default CoreInterposer routing, which
delegates to the core's hash-based routing. Everything dtschedule forwards
goes to the cache pool (via `ForwardToCore`), so the cache's local raw copy
and the hash-routed authoritative write happen as before. The cache runs
replica-0 traffic locally via its own ScheduleTask override; the two overrides
compose: dtschedule makes decisions at the producer/reader's node, and the
cache keeps the node-local copy.

dtschedule reuses the compressor's blob header (`CompressionHeader`, magic
"CTEC") so blobs written by either module can be read by either. The
compressor chimod is not in the chain when dtschedule is.

## 2. Facts about the core that the design is built on

These were verified against the tree (see `EVAL_PLAN.md` §0 and the
routing notes):

- A blob's metadata lives on its **hash owner** O:
  `BlobHash(tag, blob) % num_containers`, one container per node. A
  `Dynamic` GetBlob from anywhere goes there. Writing with
  `PoolQuery::Physical(N)` moves the metadata to N and hides the blob from
  other nodes. So the authoritative copy always goes to O, never to a node
  dtschedule picks.
- **Tier** is chosen by the DPE from `blob_score`: targets with
  `target_score_ <= blob_score` are preferred, hottest first, spilling to
  colder tiers when full. Give every device an explicit YAML `score`
  (ram 1.0, nvme 0.7, ssd 0.4, nfs 0.1) and dtschedule selects a tier by
  passing that score.
- A **readable copy on node N** is a core replica in the `kCacheReplica`
  slot (`REPLICA_CACHE`, `transform_flags_ = 0`, score -1), written to the
  core pool with `Physical(N)` and registered at the owner with
  `RegisterReplicaContainer(tag, blob, N, Dynamic, version)`. The cache
  chimod on N serves it on the next read (`replica_ = kCacheReplica` hit
  before the owner fetch); a client bound to the top pool also reaches it
  through the zero-IPC path. The owner's next write invalidates every
  registered copy. Cache copies must hold RAW bytes: the cache's own size
  and hit paths bypass dtschedule and the zero-IPC reader refuses
  transformed replicas.
- The registration is version-gated and replica-slot writes bump
  `last_modified_`, so an asynchronous replication sweep can reject a
  late registration. Preferred fix: register through the put's own
  `origin_node_` (`REPLICA_VERIFY_COMPLETE`) when the slot is free; else a
  one-line core change so replica writes do not bump the content version.
- Node CPU load: `ctp::SystemInfo::GetCpuTimes()` deltas; worker load from
  `WorkerStats`. dtschedule implements its own `PollNodeLoad` verb (the
  admin pool's `system_stats` monitor is the msgpack alternative).
- No prefetch-on-open exists in the tree. "Prefetch" in this design means
  "push the consumer copy at write time" (scenario 2/3 below), which is
  what the paper's +w actually needs.
- Replica-slot puts and already-compressed puts must pass through
  dtschedule untouched, and PutBlob must restore the task's buffer after
  forwarding (callers above build regions from it).

## 3. Scenarios (Algorithm 2 mapped onto the mechanisms)

Let P = producer node (where the PutBlob is submitted and dtschedule runs),
C = chosen consumer node, O = hash owner of the blob (may equal P or C or be
a third node). The cache keeps a raw copy at P in every scenario.

| scenario | compress at | authoritative copy | consumer copy | cost model (write + first read) |
|---|---|---|---|---|
| S1 | P | O (compressed) | none; C fetches from O and dtschedule at C decompresses | T_c(P)·m_P + net(z, P→O) + net(z, O→C) + T_d(C) |
| S2 | P | O (compressed) | raw copy pushed to C at write time | T_c(P)·m_P + net(z, P→O) + net(raw, P→C) |
| S3 | C | O (compressed, written from C) | raw copy written locally at C | net(raw, P→C) + T_c(C)·m_C + net(z, C→O) |

z = predicted compressed size, net(b, A→B) = 0 when A = B else b / BW_net,
m_X = load multiplier for node X (§5). Storage time on the chosen tier is
added to every scenario as z / BW_tier (×m for the node that writes). The
minimum-cost scenario wins; `force_scenario` overrides for the ablation.
With `workflow_aware: none` there is no C, so only S1 is possible and the
decision reduces to Algorithm 1 (codec + tier).

S2 pushes a raw copy because cache replicas must be raw (§2); its benefit
is that C's reads are local and need no decompression. S3 ships raw bytes
once (P→C) and moves compression to C; it wins when P is loaded or C has
spare CPU, which is the paper's load-balancing case. In S3 dtschedule at P
sends a `CompressAt` task with the bulk payload to dtschedule at C
(`Physical(C)`); C writes its raw cache copy, compresses, puts the
compressed bytes at O through the indexer pool (skipping the cache, with
the compressed flag pre-set so dtschedule's own PutBlob passes it through),
registers itself as the copy holder, and returns the owner's result.

## 4. Codec ranking (Algorithm 1)

Candidates = QoS `compression_preference` if given, else the built-in list
(zstd{fast,balanced,best}, lz4, zlib, bzip2, snappy, brotli, lzma, blosc2;
plus sz3/zfp/fpzip at low/medium/high presets when the blob matches the
`lossy_allowlist` and `max_error > 0`). A lossy preset is dropped when its
bound exceeds `max_error`. For each remaining (lib, preset) the CCM
predicts compress time, decompress time and ratio from
(lib, preset, dtype, size bin, entropy bin, MAD bin, 2nd-derivative bin)
and the cost is computed per scenario as in §3. `objective: ratio` ranks
by predicted size with the time cost as a tie-breaker; `objective:
performance` ranks by total time.
Every ranking also scores the raw alternative (`lib=raw` in the `.cand`
file: cost = m x size / bw, no compression, no decompression); under the
performance objective a codec is chosen only when it beats it, so on a
fabric faster than the codecs (40 GbE + RAM tier) dtschedule stores raw
and only a slow link (1 GbE, NFS, disk) makes compression pay. The store
bandwidth `bw` the ranker sees is the tier's when this node owns the blob
and min(tier, network) when the owner is remote.

CCM back-ends (knob `ccm`):

- `qtable`: the trained table shipped at `models/qtable_v1.json` with the
  online running-mean update (state = feature bins, action = (lib,
  preset), value = running mean of observed time and ratio, update =
  incremental mean; "exploration" = `resample_chance` running all
  candidates when the error exceeds `resample_error`).
  Resampling runs on a background thread (single flight, at most 256 KiB
  of the chunk, timings scaled to the chunk size) so the put path never
  waits on a slow codec; the online running mean caps the effective
  training count at 16 so a model trained elsewhere converges quickly.
- `ema`: per-(lib, preset) exponential moving average of observed cost, no
  data features (the reviewer's moving-average baseline).
- `fixed:<lib>[:<preset>]`: no model; always this codec.
- `oracle`: run every candidate on the chunk and use the measured values
  (for regret and the prediction-error study only).
- `ratio_noise_sigma`: multiplicative log-normal noise on predicted ratio.

## 5. Load awareness

Each dtschedule container samples its own node every `load_period_ms`
(cpu %, worker load, queued tasks) and polls the nodes in its
consumer set with `PollNodeLoad`. The ranker uses
`m_X = clamp(cpu_X / max(cpu_other, 1), 1/load_cap, load_cap)` on the
compress and store terms of node X. `load_aware: off` fixes m = 1.

## 6. Workflow awareness

- `none`: no consumer; S1 only.
- `consumer`: live tracking, as in the paper. The first GetBlob of a tag
  from node N registers N as the tag's consumer (recorded at the hash
  owner of the tag via a dtschedule `RegisterConsumer` verb so every node
  sees the same set); later writes to that tag consider C = N.
- `dag`: a workflow spec is loaded at Create (`dag_path`): the WfCommons
  WfFormat JSON plus the task→node map our translator emits. From it
  dtschedule derives, per file name, the set of consumer nodes before the
  first byte is written. Policies on top:
  - `dag.colocate_fanin: on` — when one downstream task consumes outputs
    of several producers, C for all of them is that task's node.
  - `dag.replicate_fanout_min: R` — when a file has consumers on ≥ R
    distinct nodes, the consumer copy is pushed to every one of them (up
    to `dag.replicate_max`), otherwise only to the first.
  The spec is per workflow; the QoS block may also be given per stage
  (`stages: [{match: <regex>, ...}]`), which answers "per workflow or per
  stage".

## 7. Configuration (compose YAML)

```yaml
- mod_name: clio_cte_dtschedule
  pool_name: clio_cte_dtschedule
  pool_query: local
  pool_id: "566.0"
  next_pool_id: "563.0"             # the cache (top of the old chain)
  compressed_next_pool_id: "564.0"  # where COMPRESSED puts go (the cache's
                                    # own next pool): cache copies must stay raw
  core_pool_id: "512.0"             # core pool used for consumer copies
  qos:
    objective: performance          # performance | ratio
    max_error: 1e-3                 # relative; 0 = lossless only
    lossy_allowlist: ['.*\.(fits|nc|h5|bp|sac|dat)$']
    compression_preference: []      # empty = all built-in candidates
    resample_error: 0.4
    resample_chance: 1.0
    stages:                         # optional per-stage overrides
      - match: '.*/stage2/.*'
        max_error: 0
  ccm: qtable                       # qtable | ema | fixed:zstd:balanced | oracle
  qtable_model_path: ""             # default: models/qtable_v1.json
  ratio_noise_sigma: 0.0
  load_aware: true
  load_period_ms: 1000
  load_cap: 4.0
  workflow_aware: consumer          # none | consumer | dag
  dag_path: ""
  dag:
    colocate_fanin: true
    replicate_fanout_min: 4
    replicate_max: 8
  force_scenario: auto              # auto | 1 | 2 | 3
  decision_order: joint             # joint | codec_first | tier_first
  tiers:                            # name -> score, must match device scores
    ram: 1.0
    nvme: 0.7
    ssd: 0.4
    nfs: 0.1
  tier_bw_mbps:                     # tier name -> MB/s, for the cost model
    ram: 20000
    nvme: 3000
    ssd: 500
    nfs: 1000
  net_bw_gbps: 25
  load_peers: []                    # extra node ids to poll for load (tests)
  trace_path: ""                    # per-container CSV, "" = off
  min_compress_bytes: 4096
```

## 8. Methods (`clio_mod.yaml`)

Core verbs intercepted (core ids kept so a core client works unchanged):
kCreate 0, kDestroy 1, kMonitor 9, kPutBlob 15, kGetBlob 16,
kGetBlobSize 23, kMultiPutBlob 48. Own verbs: kPollNodeLoad 110,
kRegisterConsumer 111, kCompressAt 112 (raw bytes shipped to node C for
scenario 3), kGetDecisionStats 113 (counters for tests and the trace
flush), kSetKnobs 114 (change ablation knobs without restarting, used by
the load-response timeline experiment).

## 9. Trace record (one CSV row per decision, 28 columns)

`ts_ms, node, tag, blob, size, dtype, entropy, mad, d2, producer_cpu,
consumer_node, consumer_cpu, owner_node, n_candidates, chosen_lib,
chosen_preset, chosen_scenario, chosen_tier, pred_ctime_ms, pred_dtime_ms,
pred_ratio, obs_ctime_ms, obs_ratio, store_ms, forced, knobs_hash,
obs_dtime_ms, select_ms`

- Every put that went through selection gets a decision row; a blob stored
  raw has `chosen_lib = raw` and `obs_ratio = 1`. `select_ms` is the wall
  time of SelectCodec. Ratios are original/compressed (≥ 1).
- GetBlob appends a decompress row for the same (tag, blob) with the
  decision fields empty and `obs_dtime_ms` set.
- `forced` carries `stage:<i>` when a per-stage QoS override matched and
  `lm:<value>` for the load multiplier used, semicolon-separated.
- With `trace_candidates: true` a second file `<trace_path>.cand.<node>.csv`
  holds one row per candidate per decision: `ts_ms, node, tag, blob, lib,
  preset, pred_ctime_ms, pred_dtime_ms, pred_ratio, cost_ms, reason` with
  reason ∈ {ok, qos_lossy_not_allowed, qos_error_bound, qos_preference,
  unavailable, fixed, skip_ratio}; `cost_ms` is set only for `ok`.

## 10. Implementation phases

1. Skeleton: module files (copied from the replication interposer pattern,
   hand-maintained autogen), config parsing, tier table from core
   ListTargets, trace writer, fixed-codec PutBlob/GetBlob round trip with
   the shared header. Unit test: put/get through the chain with
   `ccm: fixed:zstd`.
2. Algorithm 1: data stats, Q-table load + predict + online update, EMA and
   oracle back-ends, candidate filter, QoS and stage matching.
3. Load: PollNodeLoad verb, self-sampling, consumer polling, load
   multiplier.
4. Scenarios: consumer registration, S2 replica push, S3 CompressAt,
   cost model, force_scenario, decision_order.
5. DAG: spec loader, colocate/replicate policies.
6. Jarvis package `clio_dtschedule` and pipelines.
