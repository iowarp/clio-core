# Agora / SSC paper draft (ICS)

- `main.tex`, `refs.bib`: the draft (ACM `acmart`, sigconf). Red `[...]` = `\pending{}` numbers still being measured.
- `tables/*.tex`, `figures/*.pdf`: **generated** from the Delta results, never edit by hand:

      cd context-transfer-engine/benchmark/ssc
      python3 paper_tables.py ../../../ssc_work/results ../../../paper

- `REVIEW.md`: self-review as an ICS reviewer and the experiments chosen to answer it.
- Compile: `latexmk -pdf main.tex` (or Overleaf). Delta has no TeX installation; compile elsewhere.
