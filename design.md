# safe-bdev: Erasure-Coded Block Device — Design & Algorithm

Issue: [#543](https://github.com/iowarp/clio-core/issues/543)
Branch: `543-safe-bdev-erasure-coding`

This document specifies the exact algorithms `safe-bdev` implements. It is split
into:

- **Part A — Erasure-coding core** (GF(2⁸), Reed–Solomon, the member array).
  *Implemented and unit-tested* (`modules/safe-bdev/.../ec/`, 8 passing tests).
- **Part B — Runtime data plane** (mapping the core onto real, possibly remote
  member bdevs; async fan-out; dirty-row log; recovery). *Implemented and
  daemon-tested.*
- **Part C — RAID-0-data + dedicated-parity layout** (the layout the runtime
  data plane actually implements: a single fixed `k`/`m`, RAID-0 striping of
  data, dedicated parity members, a real reclaimable allocator). Declustered
  placement + variable-width stripes are set aside as a future direction
  (see C.5).

Notation: `k` = number of data shards in a stripe, `m` = number of parity shards,
`M` = target maximum tolerated failures (`m` grows toward `M`). All shard
arithmetic is over GF(2⁸); `⊕` is XOR (field addition), `·` is field
multiplication.

---

## Part A — Erasure-coding core

### A.1 Field: GF(2⁸)

Elements are bytes. Addition is XOR. Multiplication is carry-less polynomial
multiplication modulo the primitive polynomial `0x11D`
(x⁸+x⁴+x³+x²+1), implemented with log/exp tables built once:

```
x = 1
for i in 0..254:  exp[i] = x;  log[x] = i;  x <<= 1;  if x & 0x100: x ^= 0x11D
exp[255..511] = exp[0..256]              # doubled to avoid a modulo
mul(a,b) = (a==0||b==0) ? 0 : exp[log[a] + log[b]]
inv(a)   = exp[255 - log[a]]             # a != 0
```

Inner loop (used by both encode and decode):
`region_mul_add(dst, src, c, len): for i: dst[i] ⊕= mul(c, src[i])`.

### A.2 Code: systematic Reed–Solomon with a Cauchy generator

The full generator matrix `G` is `(k+m) × k`:

```
G = [ I_k  ]      <- top k rows: identity  (systematic: data stored verbatim)
    [ C    ]      <- bottom m rows: Cauchy
```

Cauchy entry for **parity row r** (0-based) and **data column c**:

```
C[r][c] = inv( x_r ⊕ y_c ),  with  x_r = k + r,  y_c = c
```

`x_r` and `y_c` are drawn from disjoint ranges, so `x_r ⊕ y_c ≠ 0`, and every
`k×k` submatrix of `G` is invertible (the MDS / "any k of n" property). Requires
`k + M ≤ 256`.

**Why Cauchy and not Vandermonde:** the two properties safe-bdev needs both fall
out of the row structure:

1. **Independence of parity rows.** Parity shard `P_r = Σ_c C[r][c]·D_c` depends
   only on the data, never on other parity. So raising tolerance `m → m+1`
   computes exactly one new shard and touches nothing else.
2. **MDS at every intermediate `m`.** Any `k` of the `k+m` shards reconstruct,
   for every `m ≤ M`.

### A.3 Encode

Single parity shard (the incremental primitive):

```
EncodeParityShard(r, data[0..k-1], len) -> out[len]:
    out = 0
    for c in 0..k-1:  region_mul_add(out, data[c], C[r][c], len)
```

Full encode = `EncodeParityShard(r, …)` for `r in 0..m-1`.

### A.4 Decode / reconstruct (any k survivors)

Given **any k** surviving shards with global indices `g_0..g_{k-1}` (data shards
are `0..k-1`, parity shards are `k..k+m-1`):

```
DecodeData(survivor_index[0..k-1], survivor_shard[0..k-1], len) -> data[0..k-1]:
    # A = the k rows of G corresponding to the survivors
    for i in 0..k-1:
        g = survivor_index[i]
        if g < k:  A[i] = e_g            # identity row (data survivor)
        else:      A[i] = C[g-k]         # Cauchy row (parity survivor)
    Ainv = InvertMatrix(A, k)            # Gauss–Jordan over GF(2⁸); fails if singular
    for i in 0..k-1:                     # data_i = Σ_j Ainv[i][j] · survivor_j
        data[i] = 0
        for j in 0..k-1:  region_mul_add(data[i], survivor_shard[j], Ainv[i][j], len)
```

This single routine handles every loss pattern (data and/or parity, up to `m`
losses): recover the data, then re-encode any missing parity via A.3.

`InvertMatrix` is standard Gauss–Jordan on `[A | I]` using `inv()` to normalize
pivots and `region`-style row ops to eliminate; returns false on a zero pivot
column (singular ⇒ too few independent survivors).

### A.5 Member array (`EcArray`)

A fixed set of `k` data members + `0..M` parity members over an abstract
`MemberStore` (RAM in tests, a bdev pool in the runtime). Capacity is sliced into
fixed-size **stripes** of `shard_len` bytes per member; stripe `s` occupies
`[s·shard_len, (s+1)·shard_len)` on every member. Each member has a role
(`data`/`parity`), a state (`active`/`faulty`/`removed`), and an index (data
column `c`, or parity row `r`). Global shard index = `c` for data, `k+r` for
parity.

**WriteStripe(s, data[0..k-1]):** write each data shard to its data member at
offset `s·shard_len`; for every active parity row `r`, `EncodeParityShard(r,
data, …)` and write it.

**ReadStripeData(s):** if all data members active, read them directly (no
decode). Otherwise reconstruct (A.4) using the active members as survivors.

**AddParityDrive(store)** — *the central "add a drive for resilience" op:*

```
if parity_level == M: return CAPPED
r = parity_level
append parity member (role=parity, index=r, store)
for each stripe s:
    data = ReadStripeData(s)              # reads existing data members
    parity = EncodeParityShard(r, data, shard_len)
    store.write(s·shard_len, parity)      # ONLY the new row is written
parity_level += 1
```

Existing data and existing parity rows are never read-modified-written — this is
the incremental upgrade, proved by the `reed_solomon_incremental_parity` test.

**RemoveDrive(member, was_faulty):** `was_faulty ⇒` mark `faulty` (kept as a
recovery candidate, excluded from I/O); else mark `removed`. Neither migrates
data.

**RecoverMember(member, new_store):**

```
for each stripe s:
    data = ReconstructStripeData(s, exclude=member)   # decode from OTHER survivors
    if member.role == data:    shard = data[member.index]
    else:                      shard = EncodeParityShard(member.index, data, …)
    new_store.write(s·shard_len, shard)
member.store = new_store;  member.state = active
```

Works identically for a lost data drive (take the reconstructed column) or a lost
parity drive (re-encode its row). Verified byte-for-byte by
`ec_array_recover_{data,parity}_drive`.

---

## Part B — Runtime data plane (mapping onto member bdevs)

`safe-bdev` is a Chimaera `Container` presenting the **bdev task interface**
(`AllocateBlocks`/`FreeBlocks`/`Write`/`Read`/`GetStats`) so callers (e.g. CTE)
treat it as an ordinary bdev. Internally each member is a real bdev pool reached
through a `bdev::Client` + `PoolQuery` (local or remote).

### B.1 State

```
struct Member { PoolId pool_id; string pool_name; u32 node_id; Role; State; }
members_[]            # ordered; data columns first, then parity rows
member_clients_[]     # bdev::Client per member (index-aligned)
max_failures_  (M)
parity_level_  (current m)
shard_len_            # per-member stripe block size
generations_[]        # see Part C
dirty_stripes_        # queue of stripe ids needing/updating parity
```

`Create` resolves each member `pool_name → PoolId` via the pool manager and
constructs its `bdev::Client`. (Today's scaffold stores members but leaves
PoolId resolution + fan-out as `TODO(#543)`; the algorithms below replace the
single-member passthrough.)

### B.2 Address mapping

`safe-bdev` exposes a linear logical capacity = `(Σ data columns) · shard_len ·
num_stripes`. A logical byte offset `L` maps to `(stripe s, column c, byte b)`:

```
per_stripe_data = k · shard_len
s = L / per_stripe_data
within = L mod per_stripe_data
c = within / shard_len
b = within mod shard_len
```

A `bdev::Write`/`Read` covering `[L, L+n)` decomposes into per-(s,c) shard
sub-ranges. Member shard offset = `s·shard_len`.

### B.3 Write (inline path — low latency)

```
Write(blocks, data, len):
    split into stripe-aligned shard writes
    for each touched stripe s:
        parallel for c in data columns of s:
            buf = IPC.AllocateBuffer(shard_len); copy data slice
            co_await member_clients_[c].AsyncWrite(pq_c, shard_block, buf, shard_len)
        mark_dirty(s)                 # parity deferred to BuildParity (B.6)
    co_await all
```

Parity is **not** computed on the write path (avoids the read-modify-write and
CPU encode cost); the stripe is queued dirty. This realizes "async encoding, no
heavy write penalty." A synchronous-parity mode (compute parity inline) is a
config option for workloads that want full redundancy on write completion.

### B.4 Read (with reconstruct-on-read)

```
Read(blocks, out, len):
    for each touched stripe s:
        if all needed data members active:
            parallel co_await AsyncRead each data shard directly
        else:
            # degraded: gather k survivors (data + parity) for s
            survivors = first k active members
            parallel co_await AsyncRead survivor shards
            data = DecodeData(survivor_index, survivor_shard, shard_len)   # A.4, CPU
            copy requested columns out
```

Common case (no failure) is a plain parallel read of data shards — zero decode,
because the code is systematic.

### B.5 Management ops (delegate to Part A over the bdev-backed members)

- **AddBdev(pool_name, node_id):** resolve PoolId, append a member + client.
  - If it is added as **parity** and `parity_level_ < M`: run `AddParityDrive`
    (B.6 incremental build over all stripes) to raise tolerance by one.
  - If added as **data**: it begins a new generation (Part C) for *future*
    stripes; existing stripes are untouched (their new column is implicitly zero,
    so their parity stays valid).
- **RemoveBdev(target_pool_id, was_faulty):**
  - `was_faulty`: mark the member faulty. If a spare member is available, kick
    off `RecoverBdev` onto it; otherwise leave the array degraded (tolerance
    drops by one until recovered). **No migration.**
  - `!was_faulty`: unlink the member (drop from `members_`/clients). **No
    migration, no recovery.**
- **RecoverBdev(old_bdev_id, new pool_name/node):** resolve/allocate the new
  member, then run `RecoverMember` (A.5) streaming stripe-by-stripe:
  `reconstruct (read survivors + DecodeData) → AsyncWrite to the new member`.
  Throttled and yielding so it shares workers with foreground I/O.

### B.6 BuildParity (periodic background task)

Registered like `admin`'s periodic tasks (`SetPeriod` + `TASK_PERIODIC`). Each
pass drains up to `max_batch` dirty stripes:

```
BuildParity(max_batch):
    for s in dirty_stripes_.take(max_batch):
        data = ReadStripeData(s)                 # async reads of data shards
        for r in 0..parity_level_-1:
            parity = EncodeParityShard(r, data, shard_len)   # CPU
            co_await member_clients_[parity_member(r)].AsyncWrite(s·shard_len, parity)
        unmark_dirty(s)
```

When tolerance is raised (a new parity row `r`) or members change, the affected
stripes are marked dirty and this task converges them to the current
`parity_level_` — no foreground stall.

---

## Part C — RAID-0-data + dedicated-parity layout

The runtime data plane implements a **single fixed `k`/`m`** layout: the data is
**RAID-0 striped** across `k` dedicated DATA members, and `m` dedicated PARITY
members carry per-row Reed–Solomon parity. There is no rotation and there are no
generations — `members_[0..k-1]` are DATA (data column = index), and
`members_[k..k+m-1]` are PARITY (parity row `j`, global RS shard `k+j`). One
`ReedSolomon(k, M)` code serves the whole array; raising `m` toward `M` only
appends a parity member and re-derives its independent parity row.

### C.1 Constants & sizing

`kChunkLen = 65536` is the RAID-0 stripe unit (one data shard / one parity
chunk). `kSuperblockSize = 65536` is reserved at absolute offset 0 of every
member (Part B.5 superblock). At `Create`:

```
avail              = min_member_remaining - kSuperblockSize
usable_per_member  = floor(avail / kChunkLen) * kChunkLen
num_rows           = usable_per_member / kChunkLen
logical_capacity   = k * num_rows * kChunkLen
```

Every member's usable region starts at `kSuperblockSize`; the chunk for row `r`
lives at absolute member offset `kSuperblockSize + r·kChunkLen`.

### C.2 RAID-0 address mapping

A logical byte offset `L` maps to a data member + physical offset:

```
chunk    = L / kChunkLen ;   within = L mod kChunkLen
data_col = chunk mod k   ;   row    = chunk / k
phys     = kSuperblockSize + row·kChunkLen + within   (on DATA member data_col)
```

So data member `d` holds logical chunks `d, d+k, d+2k, …` stacked at successive
`kChunkLen` rows — a textbook RAID-0 stripe across the `k` data members.

### C.3 Per-row parity over FULL chunks

Parity member `j` stores, at member offset `kSuperblockSize + row·kChunkLen`, the
RS parity shard `j` computed over the `k` data chunks of that row, where each
data chunk is the **full `kChunkLen` chunk** read from data member `i` at that
same offset (`i in 0..k-1`). Parity is computed over whole chunks even when a
write only partially filled one — this is exactly what makes arbitrary,
non-chunk-aligned and partial writes correct: a partial write touches a chunk,
marks its row dirty, and `BuildParity` later re-reads the full (now-updated)
chunks and recomputes parity. Reconstruction (A.4) of a down data member's full
chunk gathers `k` survivors among the row's data + parity chunks and decodes;
the read path then copies the requested within-chunk slice.

Parity is **deferred off the write path** (Part B.6): writes record dirty rows;
a single periodic `BuildParity` drains them. A degraded read or recovery of a
data member refuses a row whose parity is not yet built (it would be
unprotected).

### C.4 Real reclaimable allocator

`AllocateBlocks` / `FreeBlocks` use a real free-list + heap allocator over
`[0, logical_capacity)` (the same `GlobalBlockMap` + `Heap` the `bdev` module
uses): allocate from the free list first, else carve from the bump heap; free
returns the block to the free list classified by size. Freed space is genuinely
reclaimed (a free-then-alloc of the same size reuses the region), and block
sizes are uneven exactly like `bdev`. `GetStats` reports
`logical_capacity − live_allocated`.

### C.5 Set aside: declustering & variable-width

The earlier declustered-parity + variable-width-generation design (rotated shard
placement over epochs) is **set aside**. Its standalone, daemon-free artifact
(`ec/declustered.h` + its unit tests) remains in the tree and passing, but the
runtime no longer uses it. The dedicated-parity layout trades declustering's
spread rebuild-read load for simplicity and correct partial writes: parity I/O
concentrates on the `m` parity members, so they are a **write/rebuild hotspot**
(every row's parity update and every rebuild reads/writes them). Capacity-aware
and load-aware placement (and reintroducing declustering / variable width on top
of a real allocator) remain **TODO #543**.

---

## Implementation status & order

| Component | Status |
|---|---|
| GF(2⁸), RS encode/decode, incremental parity, `EcArray`, recovery | **Done + tested** (`ec/`, 10 passing unit tests) |
| Module scaffold, tasks, client, autogen, build wiring | **Done** |
| RAID-0 data mapping + per-row parity (Part C) | **Done + tested** (daemon roundtrip/striping/partial tests) |
| Runtime async fan-out (B.3/B.4), PoolId resolution, buffers | **Done + tested** (daemon test) |
| Dirty-row log + `BuildParity` periodic task (B.6) | **Done + tested** |
| `RecoverBdev` over member bdevs (data decode / parity re-encode, B.5) | **Done + tested** |
| Real reclaimable `AllocateBlocks`/`FreeBlocks` (free-list + heap, C.4) | **Done + tested** (daemon reclaim test) |
| Member superblocks (fresh / re-attach / foreign-refuse) | **Done + tested** |
| Partial / arbitrary-size writes | **Done + tested** (parity over full chunks) |
| Declustered placement + variable-width generations | **Set aside** (daemon-free `ec/declustered.h` kept + unit-tested) |
| Daemon-level end-to-end recovery test | **Done** (`cr_all_safe_bdev_tests`) |

Remaining refinements (TODO #543): `AddBdev` as DATA (`k` reshape — currently
refused in RAID-0 mode), capacity/load-aware placement, the dedicated-parity
write/rebuild hotspot, concurrent write-during-`BuildParity` re-dirty handling,
and multi-node remote member routing.

### Testing strategy

- **Core (daemon-free):** correctness of all field/RS/recovery math against
  in-memory stores — fast, deterministic, already passing.
- **Runtime (daemon):** create member bdev pools + a safe-bdev over them, write
  data, fault a member (`RemoveBdev(was_faulty)`), read-reconstruct, `RecoverBdev`
  onto a fresh pool, and assert byte equality through the full async path.

The daemon test (`cr_all_safe_bdev_tests`) covers roundtrip across striped data,
RAID-0 striping (members verified directly), allocator reclaim, degraded read +
recover, partial-chunk writes, and the superblock reattach / foreign-refuse
flows; the EC core/declustered math is covered by `cr_safe_bdev_ec_tests`.
