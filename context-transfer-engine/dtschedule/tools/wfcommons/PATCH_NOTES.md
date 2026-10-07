# Changes relative to stock wfcommons 1.2

Base: the `wfcommons-1.2` sdist from PyPI (`bin/wfbench`,
`wfcommons/wfbench/bench.py`, `wfcommons/wfbench/translator/bash.py`,
`wfcommons/wfbench/translator/abstract_translator.py`). Line numbers below
refer to those stock files. Nothing in the installed package is modified;
every change lives in this directory and is applied by subclassing or by
shipping a patched copy of the `wfbench` script.

## `wfbench_dt` (patched copy of `bin/wfbench`, 499 stock lines)

| stock line(s) | change |
|---|---|
| 1 | shebang `#!/usr/bin/env python` -> `#!/usr/bin/env python3`. `dt_translator.py` rewrites it again to the absolute interpreter it runs under when it copies the script into `bin/` (what pip does on install). |
| 39 (after `this_dir = ...`) | added `sys.path.insert(0, str(this_dir))` + `import dt_datagen`, and two helpers `_drop_cache_enabled()` / `_drop_page_cache(fp)` (the `WFBENCH_DROP_CACHE` behaviour the builtin jarvis `wfcommons` package relies on; stock 1.2 has no such helper). |
| 208-210 (`io_read_benchmark_user_input_data_size`) | after the read loop: `_drop_page_cache(fp)`. |
| 225-226 (`io_write_benchmark_user_input_data_size`) | **the write site**: `fp.write(os.urandom(int(chunk_size)))` -> `dt_datagen.write_file(fp, str(file_name), int(chunk_size), offset=fp.tell())`, followed by `flush`+`fsync` when `WFBENCH_DROP_CACHE` is set and `_drop_page_cache(fp)`. `fp.tell()` on the append-mode handle is the number of bytes already in the file, which keeps the content a pure function of (name, offset) no matter how `io_alternate` chunks the writes. |
| 331 (`get_parser`) | added `--data-class {float_field,text_seq,text_table,mixed_binary,auto}` and `--data-noise FLOAT`. |
| 362-363 (`main`, after the `--debug` block) | the two new options are exported as `DT_DATA_CLASS` / `DT_DATA_NOISE` so the forked `io_alternate` process (and `dt_datagen`) see them. |
| 464-465 (`main`, after the cpu procs are waited for) | `cpu_queue.put(100.0)`: the I/O helper (`io_alternate`) blocks on `cpu_queue.get()` until a `Progress:` line arrives; a `cpu-benchmark` build that prints none (seen with an older binary on PATH) hangs every task forever. Once the CPU work has finished the I/O may complete regardless. |
| 474-475 (`main`) | `pkill -f stress-ng` now runs only when this task started stress-ng (`if mem_procs:`), so the evaluation's own `stress-ng` load injectors are not killed by every finishing task. |

Everything else (core locking, CPU/memory/GPU benchmark, Flowcept hooks,
argument parsing) is byte-identical to stock.

Note: the stock script has exactly one `os.urandom` write site (line 226).
The other two the evaluation plan refers to are in `bench.py` (below).

## `dt_bench.py` (subclass `DtWorkflowBenchmark(WorkflowBenchmark)`)

Stock `bench.py` writes `os.urandom` at three places:

| stock line | method | how it is handled |
|---|---|---|
| 336-337 | `create_benchmark` (workflow input files) | guarded by `if not file_path.is_file()`; the override of `_rename_files_to_wfbench_format` pre-creates those files with `dt_datagen` (`_dt_write_workflow_inputs`), so the stock branch never runs. |
| 598-599 | `_generate_data_for_root_nodes` (unused by `create_benchmark`; its call sites at 458/483 are commented out in stock) | overridden to use `dt_datagen`. |
| 763-764 | module-level `generate_sys_data` (only used by the test-only `run()`) | `dt_bench.generate_sys_data` drop-in using `dt_datagen`. |

Methods overridden, and why they are the smallest set:

* `_create_data_footprint(data, save_dir)` (stock 442-483): calls stock, then
  `_apply_extensions()`. Stock names every file `<task>_output.txt` /
  `<task>_input.txt` (lines 472, 539, 542, 563, 572, 578); the rename gives
  each file the producing task type's representative extension from the
  recipe's `task_type_stats.json` (e.g. montage `mProject` -> `.fits`,
  `mDiffFit` -> `.txt`, seismology `sG1IterDecon` -> `.stf`), which is what
  `dt_datagen` keys its data class on. Also remembers `save_dir`.
* `_rename_files_to_wfbench_format()` (stock 204-248): identical logic with
  the stock's own commented-out `extension = ''.join(...suffixes)` lines
  (222, 240) re-enabled so `<task>_outfile_0001.fits` /
  `workflow_infile_0001.fits` keep their suffix, plus the input-file
  pre-creation hook described above.
* `_generate_data_for_root_nodes()` (stock 583-600): generated data.

Constructor additions: `data_class`, `data_noise`, `write_inputs`.

## `dt_translator.py` (`DistributedTranslator(BashTranslator)`)

Subclass of `BashTranslator` (itself a `Translator`), reusing its
`__init__` level computation (`bash.py` 37-64, `task_level_map`). Overrides:

* `translate()` (bash.py 66-86): writes `placement.json`,
  `levels/L<i>/<node>.sh`, copies `run_dist.py`, then the two helpers below.
  Uses `mkdir(exist_ok=True)` (stock fails on an existing folder).
* `_copy_binary_files()` (abstract_translator.py 84-94): stages
  `wfbench_dt` (shebang rewritten to the running interpreter),
  `dt_datagen.py` and `cpu-benchmark` instead of `shutil.which("wfbench")`.
* `_generate_input_files()` (abstract_translator.py 96-119): hard-links
  the files `DtWorkflowBenchmark` already generated in `bench/`, or
  generates them with `dt_datagen`, instead of `os.urandom`.
* Task command lines follow `_bash_wftasks_codelines()` (bash.py 88-118)
  exactly, except: program `bin/wfbench_dt`, optional `clio::` prefix on
  every `data/...` path (same rewrite the builtin jarvis `run_wfbench.py`
  applies after the fact), optional `--data-class` / `--data-noise`.
* Constructor takes the in-memory `Workflow` from `DtWorkflowBenchmark`.
  Loading the JSON back through `wfcommons.wfinstances.Instance` fetches
  the newest WfFormat schema from GitHub (`schema.py` 67-78), which rejects
  the `schemaVersion: 1.5` documents 1.2 writes; a `schema_file` argument
  is provided for callers that do start from a JSON path.

## `dt_datagen.py`, `run_dist.py`

New files, no stock counterpart.
