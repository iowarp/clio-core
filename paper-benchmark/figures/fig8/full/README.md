# Figure 8 — full campaign

`fig8a_regret.png`, `fig8b_cost_mape.png`, `fig8c_per_chunk.png`, `fig8_preview.png`
come from `fig8_trace.py` (5 ms reporting floor).

`fig8d_models.png` compares five cost models on the same chunks and comes from
campaign `full2k-09181841` (2,000-chunk replay per workload, 18 Sep 2026).

Everything needed to regenerate panel (d) from CSV — the scored per-chunk CSV,
the per-workload measurements, every log, the raw selector traces and frozen
copies of the whole script chain — is bundled outside the repo, because data and
CSVs do not belong in git:

    /projects/bekn/imuradli/np-hcompress/fig8_full

Rebuild the figure from that bundle with:

    python3 paper-benchmark/plot/plot_fig8_models.py \
        --csv /projects/bekn/imuradli/np-hcompress/fig8_full/scored/fig8_models.csv \
        --out paper-benchmark/figures/fig8/full \
        --name fig8d_models --bin 25 --cap 100

The bundle's own `README.md` documents the two upstream steps (scoring and
measurement), the cost model, and the known caveats. Regenerate the bundle after
any campaign with `paper-benchmark/model-accuracy/collect_fig8_full.sh <tag>`.
