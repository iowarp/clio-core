# Data organizer reference -- where things are and what bites

Paths are relative to the repo root. Line numbers drift; names do not.

## The interfaces

| what | where |
|---|---|
| `DataOrganizer` (abstract), `DataOrganizerFactory::Get`, `OrganizerBlobStat` | `context-transfer-engine/core/include/clio_cte/core/data_organizer/data_organizer.h` |
| Factory registration (`"frecency"` -> `FrecencyDataOrganizer`) | `context-transfer-engine/core/src/data_organizer/data_organizer.cc` |
| Reference organizer: frecency = recency `2^(-age/half_life)` blended with saturating frequency | `.../data_organizer/frecency_organizer.{h,cc}` |
| Periodic driver `Runtime::DynamicReorganize` (Method::kDynamicReorganize) -- just delegates to `organizer_->Reorganize` | `core/src/core_runtime.cc` |
| Stats collection, partitioned by replica: `Runtime::CollectOrganizerBlobStats` | `core/src/core_runtime.cc` |
| Rescore-and-move without a task: `Runtime::ReorganizeBlobInternal(tag, name, score, rc)` | `core/include/clio_cte/core/core_runtime.h` |
| Phase hint: `Runtime::organizer_hint_`, `Runtime::OrganizerHint()`, `Runtime::ReorganizeHint` (Method::kReorganizeHint = 54) | `core_runtime.h/.cc`, `core_tasks.h` (`ReorganizeHintTask`), `core_client.h` (`AsyncReorganizeHint`, `ReorganizeHint`) |
| Config: `OrganizerConfig{name_, organizer_tasks_, period_ms_}`; YAML keys `organizer`, `organizer_tasks`, `organizer_period_ms` | `core/include/clio_cte/core/core_config.h`, `core/src/core_config.cc` |
| Device prefetch vocabulary `PrefetchPolicy{hot_ 1.0, warm_ 0.2, cold_ 0.0, lookahead_, epsilon_, stall gate}` | `context-transfer-engine/adapter/gpu_vector/include/clio_cte/gpu_vector/prefetch.h` |
| Page score constant `kVectorBlobScore = 0.5f` and why | `.../gpu_vector/page.h` |
| Existing tests: factory + frecency score + end-to-end rescoring; the hint API | `context-transfer-engine/test/unit/test_data_organizer.cc`, `test_reorganize_hint.cc` |
| Paged benchmarks (each header states its access pattern) | `context-transfer-engine/adapter/gpu_vector/benchmark/{kmeans,weights,gmx,grayscott,lbann,lammps_md,gnn}/` |

`OrganizerBlobStat` gives you: `tag_id_`, `blob_name_`, `score_` (current),
`last_modified_`, `last_read_` (steady-clock ns; may be 0 briefly after
creation -- treat 0 as "not yet stamped", see the frecency sentinel note),
`access_count_` (puts+gets), `size_`.

## Score direction -- read this twice

`MaxBwDpe` splits tiers on `target_score <= blob_score` and ranks the
preferred group **descending**. So:

- **higher blob score = faster tier**;
- a tier scored above the blob's score is *not eligible* for that blob;
- every gpu_vector page is written at `kVectorBlobScore = 0.5`, so with the
  usual tier config (HBM 1.0, host RAM 0.2, NVMe 0.0) pages **never reach HBM
  on their own** -- the bench's `TIER SPLIT ... nothing landed in the fastest
  tier` line is this rule, not a bug. Something (prefetch hint, organizer) has
  to promote them to >= 1.0.

The GNN trainer's put path uses blob score 0.5 with the *opposite* intent;
copying its tier numbers into another workload silently inverts the policy.
`prefetch.h` describes the symptom: "a prefetcher that demotes everything it
means to promote while reporting perfect health."

`ReorganizeBlobInternal` clamps to [0, 1] and is a successful no-op when the
delta is below `score_difference_threshold` (config, default 0.05). Do not
issue rescores you know are below it; each one still costs a task.

## The economics that decide whether a policy can pay

Measured in `prefetch.h` on 256 KB pages:

- a migration = one page read + one page write ~ **271 us**;
- the most it can save = slow-tier fault minus fast-tier fault ~ **46 us**;
- a page-fault round trip is ~110 us of which ~6 us is the copy
  (`clio_mod.yaml`, `kPodMultiGetBlob` note) -- batching, not placement, is
  the lever for that part.

So a promotion has to be *hit* several times in the fast tier to break even,
and promoting a whole window pays full price on every page while only the
ones that were genuinely on a slow tier can repay it. `prefetch.h`'s answer is
to gate on **stalls** (a block parked on a transfer is proof the page was on
the wrong tier). An organizer working from stats has the same problem in a
different costume: rescoring everything every period is a tax; rescoring what
the hint or the stats say is about to change regime is a policy.

## Running an organizer under test

Fixture pattern (from `test_data_organizer.cc`): write a YAML with two tiers
(`ram::` fast at 1.0 with a small `capacity_limit`, a `file` or second `ram::`
slow tier at 0.2), `organizer: <name>`, `organizer_tasks: 2` (exercises
replica partitioning), a short `organizer_period_ms`; `Setenv("CLIO_SERVER_CONF")`;
`CLIO_INIT(kClient, true)` then `CLIO_CTE_CLIENT_INIT()`; `REQUIRE` both.
Put some blobs, wait a few periods, observe tier changes via
`GetBlobInfo`/`GetBlobScore`. Use `simple_test.h`; register with
`add_test(NAME cte_<x> ...)` and a `cte` label.

Bench-level A/B: any paged bench takes `--hbm-mb`, `--nvme-mb`, `--nvme-path`,
`--slots`/`--cap`; make the fast tier smaller than the working set so eviction
is real, then compare `faults=`, `evicts=` and `ms=` with `organizer: none`
against yours. The distributed harness's rule applies single-node too: a run
that did not evict is a vacuous placement test.

## Phase hint wiring, end to end

Host side, at a phase boundary:

```cpp
enum : clio::run::i32 { kPhaseBuild = 1, kPhaseSweep = 2, kPhaseCheckpoint = 3 };
CLIO_CTE_CLIENT->ReorganizeHint(kPhaseSweep);        // broadcast, returns rc
```

Organizer side, once per round:

```cpp
clio::run::TaskResume MyOrganizer::Reorganize(Runtime *server, clio::run::u32 replica) {
  const clio::run::i32 phase = server->OrganizerHint();   // 0 = no hint yet
  std::vector<OrganizerBlobStat> stats;
  server->CollectOrganizerBlobStats(replica, stats);
  for (const auto &s : stats) {
    const float target = ScoreFor(s, phase);             // your policy
    if (std::fabs(target - s.score_) < kEpsilon) continue;
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(server->ReorganizeBlobInternal(s.tag_id_, s.blob_name_, target, rc));
  }
  CLIO_CO_RETURN;
}
```

The hint is per container and persists until the next call; the core never
interprets it. Write the enum down where the organizer lives.

## Gotchas collected from the tree

- Task structs are ABI across the client/runtime boundary: after touching one,
  rebuild with no `--target` (AGENTS.md).
- The CTE `clio_mod.yaml` has drifted from `autogen/core_methods.h`; the
  header is the source of truth and `repo refresh` deletes methods. Edit both
  by hand when adding a method.
- `organizer_tasks` replicas partition blobs by name hash; a policy that needs
  a global view (e.g. "top-k hottest") must not assume it sees every blob.
- Lock paths must not log; the organizer coroutine runs on a worker thread.

## Worked example: Gray-Scott (what "measure it either way" looks like)

`GrayScottDataOrganizer` (`organizer: "grayscott"`) demotes write-once
checkpoint tags (`gv_gs_ck*`) to the slowest tier and, given
`ReorganizeHint(step)`, demotes the region pair about to be overwritten. Full
table in its header. The result:

| regime | organizer none | grayscott | app `--ckpt-drain` |
|---|---|---|---|
| HBM 1.0 (live pages can't reach HBM) | 1798 ms | 1781 (-1.0%, noise) | +6.3% |
| HBM 0.5 (live pages compete with ckpts) | 1752 ms | 1831 (**+4.5%**) | +11.6% |
| no checkpoints (idle cost) | 1157 ms | 1153 (-0.3%) | -- |

Three lessons, each of which cost a round of measurement:

1. **Stats can be blind.** A first version also demoted "written but never
   read" blobs. The paged fault path (`PodMultiGetBlob`) never stamps
   `last_read_`, so every live page looked never-read and the whole dataset was
   rescored every round: +9.9% with nothing to demote. Check what the hot path
   actually records before building a policy on a timestamp.
2. **A demotion is a migration.** Rescoring a dead page moves its bytes down a
   tier on the critical path. If the freed fast slot is refilled by the live
   sweep anyway (it was: HBM 64/64 in every run), you paid the write for
   nothing. Dead data wants to be DROPPED, and the organizer interface has no
   drop verb -- that is the real finding.
3. **Freeing a tier only helps if live data can use it.** With `kVectorBlobScore
   = 0.5` and HBM at 1.0, no live page ever reaches HBM, so any organizer that
   only frees HBM is optimising a tier the workload does not touch.

Keep the organizer as the reference implementation and the test as a guard on
correctness and idle cost; run Gray-Scott with `organizer: "none"`.

## Worked example 2: the whole benchmark suite, and when promotion pays

Four organizers were built and measured against `frecency` and `none` on a
file-backed slow tier (3 reps each, RTX 4070 Laptop):

| benchmark | none | frecency | cyclic | hotset | scatter |
|---|---|---|---|---|---|
| kmeans (cyclic sweep)  | 1581 ms | -1.0% | **-3.4%** | -3.2% | -2.7% |
| lbann (fwd/bwd/update) | 29.2 ms/step | +0.1% | -2.1% | -2.9% | **-3.7%** |
| gmx (scatter->gather)  | 45.0 ms | -0.2% | +13.8% | +13.6% | **+23.3%** |
| grayscott (stencil)    | 1798 ms | -- | -- | -- | grayscott organizer: +4.5% in its own regime |

`weights` could not be measured: its seed refuses to converge in every tier
configuration this harness builds ("SEED DID NOT CONVERGE ... Refusing to
report a measurement"). `lammps_md` stage 1 is resident by design and asserts
zero faults, so there is nothing for an organizer to place.

**The cost rule, which predicts the sign before you write any code.** Filling
the fast tier costs one migration per page: `(tier_bytes / page_bytes) x
~271 us`. Compare it with the workload's runtime:

- kmeans: 64 MB / 256 KB = 230 migrations ~= 62 ms against 1581 ms -> **4%**,
  and the pinned 22% of the dataset repays it. Wins.
- gmx: 48 MB / 128 KB = 384 migrations ~= 104 ms against **45 ms** -- the tier
  costs more to fill than the benchmark takes to run. Cannot win at any hit
  rate. Loses by 14-23%.

**The regime rule.** Off a file tier the organizers win 2-4%; with the slow
tier in host RAM every arm lands inside +-1% noise. A fault is ~110 us of
round trip against ~6 us of data movement, so placement only touches ~5% of
a fault's cost, and 5% of a fault is not worth a migration unless the tier
underneath is genuinely slow.

**Frecency never moves anything, on any of them** (-1.0% to +0.1%). Two
reasons worth knowing before reaching for it: its scores top out near 0.85 and
so never clear a fast tier scored 1.0; and on a cyclic sweep recency is an
anti-signal, because the most recently touched page is the one furthest from
its next use.

**Which policy won where** justifies writing per-workload organizers rather
than one generic one: `cyclic` (stable prefix) on kmeans' pure sweep,
`scatter` (promote at the read-back turn) on lbann's phase structure, and
nothing at all on gmx or grayscott.
