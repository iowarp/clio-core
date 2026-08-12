
## PageRank page-access prediction vs simple caching (papers100M)

N=111059956 nodes, F=128 (row=512 B), directed E=1,615,685,872, reverse-PR damping 0.85/50 it. GraphSAGE fanout (15, 10), batch 1024. Each MISS = one zstd decompress + page refetch.

**Node-level skew:** the top **10%** of nodes by reverse-PageRank absorb **34.7%** of all feature reads in a full-graph sweep (vs 10% for a random 10%). That skew is the whole opportunity.

### Hit rate @ 10% cache budget, by page granularity and trace

| page granularity | trace | PR-pin | PR-REORDER+pin | degree-pin | first-C | random-C | LRU |
|---|---|---|---|---|---|---|---|
| 1 row (512B, 111059956 pages) | minibatch | **48.2** | **48.2** | 34.6 | 7.5 | 10.0 | n/a |
| 1 row (512B, 111059956 pages) | full-sweep | **34.8** | **34.8** | 40.6 | 8.4 | 10.0 | n/a |
| 8 rows (4KiB, 13882495 pages) | minibatch | **28.7** | **48.1** | 16.4 | 7.5 | 10.0 | 15.8 |
| 8 rows (4KiB, 13882495 pages) | full-sweep | **19.3** | **34.8** | 21.0 | 8.4 | 10.0 | 17.5 |
| 64 rows (32KiB, 1735312 pages) | minibatch | **23.5** | **47.2** | 12.2 | 7.5 | 10.0 | 13.2 |
| 64 rows (32KiB, 1735312 pages) | full-sweep | **14.8** | **34.8** | 16.2 | 8.4 | 10.0 | 14.4 |
| 512 rows (256KiB, 216914 pages) | minibatch | **21.4** | **42.1** | 12.0 | 7.7 | 10.0 | 10.4 |
| 512 rows (256KiB, 216914 pages) | full-sweep | **13.9** | **34.8** | 16.0 | 8.4 | 10.0 | 13.6 |

### Full budget sweep @ 1 row/page (111059956 pages)

_minibatch_

| policy | 1% | 2% | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| PageRank-pin | 12.7 | 19.6 | 33.3 | 48.2 | 67.1 | 94.0 |
| PageRank-REORDER+pin | 12.7 | 19.6 | 33.3 | 48.2 | 67.1 | 94.0 |
| degree-pin | 6.2 | 10.5 | 20.5 | 34.6 | 57.5 | 90.0 |
| first-C | 0.7 | 1.4 | 3.4 | 7.5 | 17.9 | 48.6 |
| random-C | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
| LRU | n/a | n/a | n/a | n/a | n/a | n/a |
_full-sweep_

| policy | 1% | 2% | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| PageRank-pin | 7.3 | 11.9 | 22.1 | 34.8 | 53.9 | 90.9 |
| PageRank-REORDER+pin | 7.3 | 11.9 | 22.1 | 34.8 | 53.9 | 90.9 |
| degree-pin | 7.9 | 13.2 | 25.2 | 40.6 | 63.0 | 93.1 |
| first-C | 0.9 | 1.9 | 4.2 | 8.4 | 21.0 | 61.6 |
| random-C | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
| LRU | n/a | n/a | n/a | n/a | n/a | n/a |

### Full budget sweep @ 512 row/page (216914 pages)

_minibatch_

| policy | 1% | 2% | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| PageRank-pin | 3.2 | 5.8 | 12.4 | 21.4 | 35.2 | 67.5 |
| PageRank-REORDER+pin | 8.0 | 13.9 | 26.9 | 42.1 | 62.5 | 93.0 |
| degree-pin | 1.2 | 2.5 | 6.1 | 12.0 | 22.3 | 60.2 |
| first-C | 0.7 | 1.5 | 3.5 | 7.7 | 18.3 | 49.4 |
| random-C | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
| LRU | 0.0 | 0.0 | 0.0 | 10.4 | 23.2 | 56.8 |
_full-sweep_

| policy | 1% | 2% | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| PageRank-pin | 1.8 | 3.3 | 7.5 | 13.9 | 25.8 | 62.8 |
| PageRank-REORDER+pin | 7.3 | 11.9 | 22.1 | 34.8 | 53.9 | 90.9 |
| degree-pin | 1.7 | 3.3 | 8.3 | 16.0 | 29.1 | 67.4 |
| first-C | 0.9 | 1.9 | 4.2 | 8.4 | 21.0 | 61.6 |
| random-C | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.1 |
| LRU | 1.4 | 2.8 | 6.9 | 13.6 | 26.7 | 63.0 |

### Headline (node granularity, full-sweep, 10% budget)

- PageRank-pin hit rate **34.8%** vs random **10.0%**.
- Over one epoch (30,484,640 feature reads): PageRank serves 19,891,001 misses vs random 27,436,683 -> **27.5% fewer zstd decompressions / refetches**. Each avoided miss saves one 512-B feature decompress+copy.

