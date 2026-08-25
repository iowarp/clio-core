# NeuroPress NN predictor — adversarial accuracy review (2026-08-25)

Branch `neuropress-693-continued`, A100-PCIE-40GB, CUDA 12.6, real nvcomp codecs,
4 MiB float32 chunks, every candidate action compressed and decompressed for real.

Deployment mode under review: **in-memory adaptation from fresh model weights** —
every process loads the shipped `model.nnwt` (`compressor_runtime.cc:376`), adapts
its live weights via phase-1/phase-2 SGD for the process lifetime, and never persists
them (`compressor_runtime.cc:1465-1471`; `Save()` is never called).

| file | author | role |
|---|---|---|
| `01_findings.md` | investigator agent | how the prediction path works (Section A), 17 claims F1–F17 with measured evidence (B), ranked improvements (C), reproduction appendix (D) |
| `02_rebuttal.md` | adversarial agent | independent re-derivation of every claim with its own scorer and a held-out corpus; per-claim verdicts, attacks, missed items M1–M12, learning-loop battery L0–L6 / M1 / N1 |
| `03_debate_round1.md` | investigator | concede / contest / partial for every attack; 8 points left in dispute (D1–D8) |
| `04_debate_round2.md` | adversary | settles D1–D8 with idle-GPU, common-ground-truth runs; agreed gaps and suggestions |
| `05_agreed_findings.md` | moderator | **start here** — the converged, concise list of gaps and bottlenecks and the ranked suggestions |

Reproduction: the measurement harnesses (`np_eval.cu`, `np_sgd.cu`,
`np_ood_upstream.cu`, `np_adv.cu`) live in the gitignored
`context-transport-primitives/test/unit/compress/model/parity/` directory by repo
convention and are not committed; the reports' Section D / "Reproduction" sections give
the exact env knobs and commands. Paths under `/tmp/claude-.../scratchpad/npreview/` refer
to the session scratchpad that held the CSVs and analysis scripts.
