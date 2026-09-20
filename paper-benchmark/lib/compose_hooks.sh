# Optional compose settings, sourced by every <workload>/common.sh. Unset, the
# compose file is unchanged.
#   BENCH_TIER1_TYPE / _MB / _PERSIST  first tier (default file, workload size, temporary)
#   BENCH_TIER1_PATH   where tier 1's file lives (default: <results>/cte_tier.dat).
#                      Set it to put the bytes on a chosen DEVICE while the run's
#                      CSVs and logs stay in the results dir. Inert for a `ram`
#                      tier, whose path is only a shared-memory name.
#   BENCH_TIER2_PATH / _MB / _SCORE    add a long_term file tier (score 0.3)
#   BENCH_FLUSH_MS     flush_data_period_ms; 0 disables it (unset: core default 10 s)
#   BENCH_NP_LR        neuropress_learning_rate
#   BENCH_NP_MAPE      neuropress_mape_threshold

# bench_tier2_yaml <tier_mb> -- the second storage entry, or nothing.
bench_tier2_yaml() {
  [ -n "${BENCH_TIER2_PATH:-}" ] || return 0
  printf '\n      - path: "%s"\n        bdev_type: "file"\n        capacity_limit: "%sMB"\n        score: %s\n        persistence_level: "long_term"' \
    "$BENCH_TIER2_PATH" "${BENCH_TIER2_MB:-$1}" "${BENCH_TIER2_SCORE:-0.3}"
}

# bench_flush_yaml -- the flush_data_period_ms line, or nothing.
bench_flush_yaml() {
  [ -n "${BENCH_FLUSH_MS:-}" ] || return 0
  printf '\n      flush_data_period_ms: %s' "$BENCH_FLUSH_MS"
}

# bench_learning_yaml -- the online-learning rate lines, or nothing. Follows the
# neuropress_exploration_threshold line.
bench_learning_yaml() {
  [ -n "${BENCH_NP_LR:-}" ] && printf '\n    neuropress_learning_rate: %s' "$BENCH_NP_LR"
  [ -n "${BENCH_NP_MAPE:-}" ] && printf '\n    neuropress_mape_threshold: %s' "$BENCH_NP_MAPE"
  return 0
}
