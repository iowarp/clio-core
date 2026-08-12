# GNN on the Eternia compressed feature store - summary

## Training (test_gpu_vector_gnn_train)

| dataset | method | peak GPU | store ratio | final train / val acc | epoch time | bit-exact? |
|---|---|---|---|---|---|---|
| arxiv | in-core (resident, 30 ep) | 82 MiB | 1.00x (uncompressed) | 18.90% / 18.88% | 0.15 s | baseline |
| arxiv | Eternia-zstd (balanced, 30 ep) | 1 MiB | 1.067x | 18.90% / 18.88% | 0.31 s | BIT-EXACT |
| papers100M | in-core (resident, 3 ep) | OOM (needs 52.9 GiB) | - | OOM | OOM | n/a (OOM) |
| papers100M | Eternia-zstd (balanced, 3 ep) | 64 MiB | 1.079x | 2.09% / 1.99% | 4.7 min | n/a(OOM) |
| papers100M | in-core (resident, 30 ep) | OOM (needs 52.9 GiB) | - | OOM | OOM | n/a (OOM) |
| papers100M | Eternia-zstd (balanced, 30 ep) | 64 MiB | 1.079x | 6.26% / 6.23% | 4.6 min | n/a(OOM) |
| papers100M | in-core (resident, 30 ep) | OOM (needs 52.9 GiB) | - | OOM | OOM | n/a (OOM) |
| papers100M | Eternia-cuszp (balanced, 30 ep) | 64 MiB | 3.126x | 6.26% / 6.23% | 4.5 min | n/a(OOM) |
| papers100M | in-core (resident, 30 ep) | OOM (needs 52.9 GiB) | - | OOM | OOM | n/a (OOM) |
| papers100M | Eternia-cuszp (best, 30 ep) | 64 MiB | 2.340x | 6.26% / 6.23% | 4.5 min | n/a(OOM) |
| arxiv | in-core (resident, 30 ep) | 96 MiB | 1.00x (uncompressed) | 18.82% / 18.87% | 0.14 s | baseline |
| arxiv | Eternia-zstd (balanced, 30 ep) | 32 MiB | 1.078x | 18.82% / 18.87% | 0.53 s | BIT-EXACT |
| arxiv | in-core (resident, 30 ep) | 84 MiB | 1.00x (uncompressed) | 58.59% / 58.86% | 0.12 s | baseline |
| arxiv | Eternia-zstd (balanced, 30 ep) | 2 MiB | 1.076x | 58.59% / 58.86% | 0.38 s | BIT-EXACT |
| arxiv | in-core (resident, 5 ep) | 84 MiB | 1.00x (uncompressed) | 34.91% / 34.86% | 0.12 s | baseline |
| arxiv | Eternia-zstd (balanced, 5 ep) | 2 MiB | 1.076x | 34.91% / 34.86% | 0.38 s | BIT-EXACT |
| arxiv | in-core (resident, 30 ep) | 88 MiB | 1.00x (uncompressed) | 45.15% / 45.19% | 0.11 s | baseline |
| arxiv | Eternia-zstd (balanced, 30 ep) | 8 MiB | 1.077x | 45.15% / 45.19% | 0.39 s | BIT-EXACT |
| papers100M | in-core (resident, 10 ep) | OOM (needs 52.9 GiB) | - | OOM | OOM | n/a (OOM) |
| papers100M | Eternia-zstd (balanced, 10 ep) | 64 MiB | 1.079x | 57.86% / 57.89% | 6.0 min | n/a(OOM) |
| papers100M | in-core (resident, 30 ep) | OOM (needs 52.9 GiB) | - | OOM | OOM | n/a (OOM) |
| papers100M | Eternia-zstd (balanced, 30 ep) | 64 MiB | 1.079x | 61.13% / 61.18% | 6.0 min | n/a(OOM) |

## Forward capacity (test_gpu_vector_gnn_capacity)

| features_mib | nodes | page_rows | window_pages | traditional_status | traditional_s | eternia_store_s | eternia_readout_s | eternia_stored_mib | eternia_ratio | hbm_used_mib | dram_used_mib | eternia_peak_gpu_mib | bit_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 54208 | 111017984 | 65536 | 2 | OOM | -1.000 | 63.674 | 221.544 | 17573 | 3.085 | 0 | 17573 | 64 | n/a(OOM) |
| 960 | 1966080 | 65536 | 2 | OK | 0.254 | 2.962 | 4.044 | 890 | 1.078 | 0 | 890 | 64 | BIT-EXACT |
| 3904 | 7995392 | 65536 | 2 | OK | 0.995 | 11.952 | 15.992 | 3621 | 1.078 | 0 | 3621 | 64 | BIT-EXACT |
| 9792 | 20054016 | 65536 | 2 | OK | 2.494 | 21.462 | 39.799 | 9081 | 1.078 | 0 | 9081 | 64 | BIT-EXACT |
| 29312 | 60030976 | 65536 | 2 | OK | 7.463 | 71.646 | 118.876 | 27180 | 1.078 | 0 | 27180 | 64 | BIT-EXACT |
| 54208 | 111017984 | 65536 | 2 | OOM | -1.000 | 141.716 | 219.810 | 50262 | 1.078 | 0 | 50262 | 64 | n/a(OOM) |
