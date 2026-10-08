#!/usr/bin/env python3
"""
Distributed WfBench translator for DTSchedule evaluations.

``DistributedTranslator`` levels the workflow DAG (reusing
``BashTranslator``'s topological level computation), assigns every task to
a node from a hostfile with one of three policies, and emits:

``placement.json``
    The DAG spec DTSchedule reads (``dag_path``): nodes, per-task node /
    level / inputs / outputs, and per-file producer / consumers /
    consumer nodes.
``levels/L<i>/<node>.sh``
    One bash script per (level, node): the same ``bin/wfbench_dt ...``
    invocations ``BashTranslator`` would emit, backgrounded with ``&`` and
    joined with ``wait``; each sources ``env.sh`` first.
``run_dist.py``
    The level-by-level ssh driver (copied from this directory).
``bin/``, ``data/``
    ``wfbench_dt`` + ``dt_datagen.py`` + ``cpu-benchmark``, and the
    workflow input files (generated with ``dt_datagen``).

Placement policies:
``round_robin``  every level is spread over all nodes.
``stage_sets``   nodes are split into ``k`` disjoint subsets; level ``i``
                 runs on subset ``i mod k``, so consecutive levels (and
                 therefore producer/consumer pairs) sit on disjoint nodes.
``single``       everything on the first node.

With ``--clio-prefix`` every file path handed to wfbench starts with
``clio::`` so the CTE POSIX interposer picks it up (same rewrite as the
builtin jarvis ``run_wfbench.py``).

With ``--api`` only ``placement.json`` is written (no scripts, binaries or
input data): ``dtschedule_wfrun`` (tools/wfrun) executes it over the CTE
API, staging the workflow inputs itself. Every task then also carries the
WfChef instance's ``runtime`` (s) and ``cores``; ``--size-mode instance``
rescales file sizes to the instance's per-task output (and root-input)
bytes, keeping the total footprint, clamped to ``[--min-file-kb,
--max-file-mb]``; each workflow input file names its staging node (the
first consumer's) as ``producer_node`` with ``"staged": true``.

CLI example::

    dt_translator.py --recipe montage --num-tasks 60 --data-footprint 30G \\
        --hostfile H --placement stage_sets --stage-sets 2 --out DIR \\
        --clio-prefix --data-class auto
"""
import argparse
import ast
import json
import logging
import os
import pathlib
import shutil
import stat
import sys
from typing import Dict, List, Optional, Tuple, Union

from wfcommons.common import Workflow
from wfcommons.wfbench.translator.bash import BashTranslator
from wfcommons.wfinstances.instance import Instance

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import dt_datagen  # noqa: E402
from dt_bench import DtWorkflowBenchmark, task_type_of  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

RECIPE_IMPORTS = {
    "montage": "MontageRecipe", "genome": "GenomeRecipe",
    "cycles": "CyclesRecipe", "blast": "BlastRecipe", "bwa": "BwaRecipe",
    "srasearch": "SrasearchRecipe", "epigenomics": "EpigenomicsRecipe",
    "seismology": "SeismologyRecipe", "soykb": "SoykbRecipe",
    "rnaseq": "RnaseqRecipe",
}
PLACEMENTS = ("round_robin", "stage_sets", "single")
SIZE_MODES = ("uniform", "instance")
DATA_DIR = "data"
PROGRAM = "wfbench_dt"


def load_recipe(name: str):
    """Resolve a recipe name to its wfcommons recipe class.

    :param name: Recipe name (case-insensitive), e.g. ``montage``.
    :return: The recipe class.
    """
    import wfcommons
    attr = RECIPE_IMPORTS.get(name.lower())
    if not attr or not hasattr(wfcommons, attr):
        raise SystemExit(f"unknown recipe '{name}'; choices: {sorted(RECIPE_IMPORTS)}")
    return getattr(wfcommons, attr)


def read_hostfile(path: Optional[str]) -> List[str]:
    """Read node names from a hostfile (one per line, ``#`` comments).

    :param path: Hostfile path; None or missing file means ``localhost``.
    :return: Unique node names in file order.
    """
    nodes: List[str] = []
    if path and pathlib.Path(path).is_file():
        for line in pathlib.Path(path).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line and line not in nodes:
                nodes.append(line)
    return nodes or ["localhost"]


def parse_size(text: Union[str, int, None]) -> int:
    """Parse ``100M``, ``1G``, ``512k`` or a plain integer into bytes.

    :param text: Size string or integer.
    :return: Size in bytes (0 for empty/None).
    """
    if text is None:
        return 0
    s = str(text).strip()
    if not s:
        return 0
    units = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}
    if s[-1].lower() in units:
        return int(float(s[:-1]) * units[s[-1].lower()])
    return int(s)


def _instance_weights(tasks: Dict[str, Dict], files: Dict[str, Dict],
                      info: Dict[str, Dict], out_ids: set) -> Dict[str, float]:
    """Weight of every placement file from the WfChef instance.

    A task's output file weighs the task's instance output bytes; a
    workflow input file weighs its first consumer's instance inputs that no
    instance task produces.

    :param tasks: Placement tasks.
    :param files: Placement files.
    :param info: ``DtWorkflowBenchmark.instance_info``.
    :param out_ids: ``DtWorkflowBenchmark.instance_out_ids``.
    :return: ``{file path: weight}`` (bytes, possibly 0).
    """
    weights: Dict[str, float] = {}
    for path, entry in files.items():
        if entry["producer"]:
            weights[path] = float(info.get(entry["producer"], {}).get("out_bytes", 0))
            continue
        first = entry["consumers"][0] if entry["consumers"] else None
        ins = info.get(first, {}).get("inputs", []) if first else []
        weights[path] = float(sum(sz for fid, sz in ins if fid not in out_ids))
    return weights


def rescale_sizes(weights: Dict[str, float], total: int, min_bytes: int,
                  max_bytes: int) -> Dict[str, int]:
    """Spread ``total`` bytes over files in proportion to their weights.

    Files are clamped to ``[min_bytes, max_bytes]`` and the remainder is
    re-spread over the unclamped ones (a few water-filling passes), so the
    footprint stays close to ``total`` unless the clamps forbid it.

    :param weights: ``{file: weight}``.
    :param total: Target total bytes.
    :param min_bytes: Smallest file size.
    :param max_bytes: Largest file size (0 = no cap).
    :return: ``{file: bytes}``.
    """
    cap = max_bytes if max_bytes > 0 else 1 << 62
    sizes: Dict[str, int] = {}
    free = dict(weights)
    budget = total
    for _ in range(8):
        wsum = sum(free.values())
        if not free or budget <= 0:
            break
        clamped = {}
        for f, w in free.items():
            want = budget * w / wsum if wsum > 0 else budget / len(free)
            if want <= min_bytes:
                clamped[f] = min_bytes
            elif want >= cap:
                clamped[f] = cap
        if not clamped:
            for f, w in free.items():
                sizes[f] = int(budget * w / wsum) if wsum > 0 else budget // len(free)
            free = {}
            break
        for f, sz in clamped.items():
            sizes[f] = sz
            budget -= sz
            free.pop(f)
    for f in free:
        sizes[f] = min_bytes
    return sizes


class DistributedTranslator(BashTranslator):
    """Translate a WfBench workflow into per-level, per-node scripts.

    :param workflow: Workflow object or path to the benchmark JSON.
    :param nodes: Node names to place tasks on.
    :param placement: One of ``PLACEMENTS``.
    :param stage_sets: Number of disjoint node subsets for ``stage_sets``.
    :param clio_prefix: Prefix every file path with ``clio::``.
    :param data_class: ``auto`` or a ``dt_datagen`` class for all files.
    :param data_noise: Float-field noise amplitude (None = default).
    :param bench_dir: Directory where ``DtWorkflowBenchmark`` already
        wrote input files (hard-linked instead of regenerated).
    :param python_exe: Interpreter for the ``bin/wfbench_dt`` shebang.
    :param schema_file: WfFormat JSON schema used to validate ``workflow``
        when it is a path (wfcommons 1.2 otherwise downloads the latest
        schema from GitHub, which rejects the 1.5 instances it writes).
    :param logger: Logger.
    :param api: Write only ``placement.json`` for ``dtschedule_wfrun``.
    :param size_mode: ``uniform`` (WfBench sizes) or ``instance`` (API only).
    :param instance_info: ``DtWorkflowBenchmark.instance_info``.
    :param instance_out_ids: ``DtWorkflowBenchmark.instance_out_ids``.
    :param min_file_bytes: Smallest file for ``size_mode=instance``.
    :param max_file_bytes: Largest file for ``size_mode=instance`` (0 = any).
    """

    def __init__(self, workflow: Union[Workflow, pathlib.Path],
                 nodes: List[str], placement: str = "round_robin",
                 stage_sets: int = 2, clio_prefix: bool = False,
                 data_class: str = "auto", data_noise: Optional[float] = None,
                 bench_dir: Optional[pathlib.Path] = None,
                 python_exe: Optional[str] = None,
                 schema_file: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 api: bool = False, size_mode: str = "uniform",
                 instance_info: Optional[Dict[str, Dict]] = None,
                 instance_out_ids: Optional[set] = None,
                 min_file_bytes: int = 1 << 20,
                 max_file_bytes: int = 0) -> None:
        """Build the translator and compute DAG levels (see class doc)."""
        if not isinstance(workflow, Workflow) and schema_file:
            workflow = Instance(pathlib.Path(workflow), schema_file=schema_file,
                                logger=logger).workflow
        super().__init__(workflow, logger)
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        if not nodes:
            raise ValueError("at least one node is required")
        self.nodes = list(nodes)
        self.placement = placement
        self.stage_sets = max(1, int(stage_sets))
        self.clio_prefix = clio_prefix
        self.data_class = data_class or "auto"
        self.data_noise = data_noise
        self.bench_dir = pathlib.Path(bench_dir) if bench_dir else None
        self.python_exe = python_exe or sys.executable
        if size_mode not in SIZE_MODES:
            raise ValueError(f"size_mode must be one of {SIZE_MODES}")
        self.api = api
        self.size_mode = size_mode
        self.instance_info = instance_info or {}
        self.instance_out_ids = instance_out_ids or set()
        self.min_file_bytes = int(min_file_bytes)
        self.max_file_bytes = int(max_file_bytes)

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def node_subsets(self) -> List[List[str]]:
        """Split the node list into ``stage_sets`` contiguous subsets.

        :return: Non-empty subsets; fewer than ``stage_sets`` if there are
            not enough nodes.
        """
        k = min(self.stage_sets, len(self.nodes))
        n = len(self.nodes)
        subsets = [self.nodes[j * n // k:(j + 1) * n // k] for j in range(k)]
        return [s for s in subsets if s]

    def assign_nodes(self) -> Dict[str, str]:
        """Assign every task to a node according to the placement policy.

        :return: ``{task name: node}``.
        """
        assignment: Dict[str, str] = {}
        subsets = self.node_subsets()
        for level in sorted(self.task_level_map):
            tasks = sorted(self.task_level_map[level])
            if self.placement == "single":
                pool = [self.nodes[0]]
            elif self.placement == "stage_sets":
                pool = subsets[level % len(subsets)]
            else:
                pool = self.nodes
            for i, name in enumerate(tasks):
                assignment[name] = pool[i % len(pool)]
        return assignment

    # ------------------------------------------------------------------
    # per-task command line (same shape as BashTranslator)
    # ------------------------------------------------------------------

    def _path(self, file_id: str) -> str:
        """Map a file id to the path handed to wfbench.

        :param file_id: Bare file name from the workflow.
        :return: ``data/<id>`` or ``clio::data/<id>``.
        """
        rel = f"{DATA_DIR}/{file_id}"
        if not self.clio_prefix:
            return rel
        # Root-marker absolute form: "/clio::<out>/data/<id>". The adapter
        # strips the marker and uses the absolute path inside the CTE
        # namespace; a marker in the middle of the path ("<out>/clio::data/
        # <id>") is rejected with ENOENT once a subdirectory follows it.
        # Flat, root-level names: the filesystem chimod's cross-node directory
        # lookup is unreliable for files inside subdirectories (ENOENT from
        # every node for some directories), while root-level files always
        # work. Files are named "<run>__<id>" so concurrent runs cannot clash.
        return f"/clio::{self._abs_out.name}__{file_id}"

    def _task_files(self, task) -> Tuple[List[str], Dict[str, int]]:
        """Extract the input list and output size dict from a task's args.

        :param task: Workflow task.
        :return: ``(input file ids, {output file id: bytes})``.
        """
        inputs: List[str] = []
        outputs: Dict[str, int] = {}
        for a in task.args:
            if a.startswith("--output-files"):
                outputs = dict(ast.literal_eval(a.split(" ", 1)[1]))
            elif a.startswith("--input-files"):
                inputs = list(ast.literal_eval(a.split(" ", 1)[1]))
        return inputs, outputs

    def _task_cmdline(self, task) -> str:
        """Build the ``bin/wfbench_dt ...`` line for one task.

        Mirrors ``BashTranslator._bash_wftasks_codelines`` except for the
        program name, the optional ``clio::`` prefix and ``--data-class``.

        :param task: Workflow task.
        :return: Shell command line (no trailing ``&``).
        """
        args = []
        for a in task.args:
            if a.startswith("--output-files"):
                flag, raw = a.split(" ", 1)
                files = {self._path(k): v for k, v in ast.literal_eval(raw).items()}
                a = f"{flag} '{json.dumps(files).replace(chr(34), chr(92) + chr(34))}'"
            elif a.startswith("--input-files"):
                flag, raw = a.split(" ", 1)
                files = [self._path(f) for f in ast.literal_eval(raw)]
                a = f"{flag} '{json.dumps(files).replace(chr(34), chr(92) + chr(34))}'"
            args.append(a)
        if self.data_class != "auto":
            args.append(f"--data-class {self.data_class}")
        if self.data_noise is not None:
            args.append(f"--data-noise {self.data_noise}")
        return f"bin/{PROGRAM} {' '.join(args)}"

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------

    def build_placement(self, assignment: Dict[str, str],
                        output_folder: pathlib.Path) -> Dict:
        """Build the ``placement.json`` document.

        :param assignment: ``{task: node}`` from :meth:`assign_nodes`.
        :param output_folder: Translator output directory.
        :return: The document (JSON-serialisable).
        """
        levels = {t: lvl for lvl, names in self.task_level_map.items() for t in names}
        tasks: Dict[str, Dict] = {}
        files: Dict[str, Dict] = {}
        for name in sorted(self.tasks):
            task = self.tasks[name]
            inputs, outputs = self._task_files(task)
            in_paths = [f"{DATA_DIR}/{f}" for f in inputs]
            out_paths = {f"{DATA_DIR}/{f}": int(sz) for f, sz in outputs.items()}
            tasks[name] = {"node": assignment[name], "level": levels[name],
                           "type": task_type_of(task), "inputs": in_paths,
                           "outputs": list(out_paths), "output_bytes": out_paths}
            for path, size in out_paths.items():
                entry = files.setdefault(path, {"producer": None, "producer_node": None,
                                                "consumers": [], "consumer_nodes": [],
                                                "size": size})
                entry["producer"], entry["producer_node"] = name, assignment[name]
                entry["size"] = size
            for path in in_paths:
                entry = files.setdefault(path, {"producer": None, "producer_node": None,
                                                "consumers": [], "consumer_nodes": [],
                                                "size": None})
                entry["consumers"].append(name)
                if assignment[name] not in entry["consumer_nodes"]:
                    entry["consumer_nodes"].append(assignment[name])
        if self.api:
            self._apply_api_fields(tasks, files)
        return {"recipe": self.workflow.name, "num_tasks": len(tasks),
                "placement": self.placement, "stage_sets": self.stage_sets,
                "path_prefix": "clio::" if self.clio_prefix or self.api else "",
                "api": self.api, "size_mode": self.size_mode if self.api else "uniform",
                "data_dir": DATA_DIR, "out_dir": str(output_folder),
                "nodes": self.nodes, "levels": len(self.task_level_map),
                "tasks": tasks, "files": files}

    def _apply_api_fields(self, tasks: Dict[str, Dict],
                          files: Dict[str, Dict]) -> None:
        """Add the fields ``dtschedule_wfrun`` needs to a placement.

        Per task: the instance ``runtime`` (s) and ``cores``. Per workflow
        input file: its staging node as ``producer_node`` (``staged``).
        With ``size_mode=instance``: instance-proportional sizes.

        :param tasks: Placement tasks (modified in place).
        :param files: Placement files (modified in place).
        """
        for name, entry in tasks.items():
            info = self.instance_info.get(name, {})
            entry["runtime"] = round(float(info.get("runtime", 0.0)), 3)
            entry["cores"] = int(info.get("cores", 1))
        in_sizes = {f"{DATA_DIR}/{f.file_id}": int(f.size)
                    for t in self.workflow.tasks.values() for f in t.input_files}
        for path, entry in files.items():
            if entry["size"] is None:
                entry["size"] = in_sizes.get(path, 0)
            if entry["producer"] is None and entry["consumers"]:
                entry["producer_node"] = tasks[entry["consumers"][0]]["node"]
                entry["staged"] = True
        if self.size_mode != "instance" or not self.instance_info:
            return
        total = sum(int(f["size"] or 0) for f in files.values())
        weights = _instance_weights(tasks, files, self.instance_info,
                                    self.instance_out_ids)
        sizes = rescale_sizes(weights, total, self.min_file_bytes,
                              self.max_file_bytes)
        for path, entry in files.items():
            entry["size"] = sizes[path]
            if entry["producer"]:
                tasks[entry["producer"]]["output_bytes"][path] = sizes[path]

    def _level_script(self, level: int, node: str, names: List[str],
                      output_folder: pathlib.Path) -> str:
        """Render the bash script for one (level, node).

        :param level: Level index.
        :param node: Node name.
        :param names: Task names placed on that node in that level.
        :param output_folder: Translator output directory (cwd of tasks).
        :return: Script text.
        """
        out = str(output_folder)
        lines = ["#!/bin/bash",
                 f"# level L{level} on {node}: {len(names)} task(s); generated by dt_translator.py",
                 "set -u",
                 f'cd "{out}" || exit 1',
                 f'if [ -f "{out}/env.sh" ]; then . "{out}/env.sh"; fi',
                 'mkdir -p logs', 'pids=()', 'names=()']
        for name in names:
            lines.append(f"{self._task_cmdline(self.tasks[name])} > logs/{name}.log 2>&1 &")
            lines.append('pids+=($!)')
            lines.append(f'names+=("{name}")')
        lines += ['rc=0',
                  'for i in "${!pids[@]}"; do',
                  '  if ! wait "${pids[$i]}"; then',
                  '    echo "FAILED: ${names[$i]} (see logs/${names[$i]}.log)" >&2',
                  '    rc=1',
                  '  fi',
                  'done',
                  'exit $rc']
        return "\n".join(lines) + "\n"

    def _write_level_scripts(self, assignment: Dict[str, str],
                             output_folder: pathlib.Path) -> int:
        """Write ``levels/L<i>/<node>.sh`` for every (level, node) pair.

        :param assignment: ``{task: node}``.
        :param output_folder: Translator output directory.
        :return: Number of scripts written.
        """
        levels_dir = output_folder / "levels"
        if levels_dir.exists():
            shutil.rmtree(levels_dir)
        count = 0
        for level in sorted(self.task_level_map):
            by_node: Dict[str, List[str]] = {}
            for name in sorted(self.task_level_map[level]):
                by_node.setdefault(assignment[name], []).append(name)
            for node, names in by_node.items():
                path = levels_dir / f"L{level}" / f"{node}.sh"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self._level_script(level, node, names, output_folder))
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
                count += 1
        return count

    def _copy_binary_files(self, output_folder: pathlib.Path) -> None:
        """Stage ``wfbench_dt``, ``dt_datagen.py`` and ``cpu-benchmark``.

        The ``wfbench_dt`` shebang is rewritten to ``python_exe`` so the
        script runs under the interpreter that has wfcommons installed.

        :param output_folder: Translator output directory.
        """
        bin_dir = output_folder / "bin"
        bin_dir.mkdir(exist_ok=True)
        src = (THIS_DIR / PROGRAM).read_text().splitlines(keepends=True)
        src[0] = f"#!{self.python_exe}\n"
        dst = bin_dir / PROGRAM
        dst.write_text("".join(src))
        dst.chmod(0o755)
        shutil.copy(THIS_DIR / "dt_datagen.py", bin_dir / "dt_datagen.py")
        # Prefer the binary that ships with the interpreter's wfcommons; an
        # older cpu-benchmark elsewhere on PATH (e.g. ~/.local/bin) may not
        # print "Progress:" lines, which leaves wfbench's I/O waiting forever.
        cpu = str(pathlib.Path(self.python_exe).parent / "cpu-benchmark")
        if not pathlib.Path(cpu).is_file():
            cpu = shutil.which("cpu-benchmark") or cpu
        if not pathlib.Path(cpu).is_file():
            raise SystemExit("cpu-benchmark not found; install wfcommons in this interpreter")
        shutil.copy(cpu, bin_dir / "cpu-benchmark")
        (bin_dir / "cpu-benchmark").chmod(0o755)

    def _generate_input_files(self, output_folder: pathlib.Path) -> None:
        """Create the workflow's root input files under ``data/``.

        Files already written by ``DtWorkflowBenchmark`` in ``bench_dir``
        are hard-linked (or copied) instead of regenerated; anything else
        is generated with ``dt_datagen``.

        :param output_folder: Translator output directory.
        """
        inputs, outputs = {}, set()
        for task in self.workflow.tasks.values():
            for f in task.input_files:
                inputs[f.file_id] = int(f.size)
            for f in task.output_files:
                outputs.add(f.file_id)
        data_dir = output_folder / DATA_DIR
        data_dir.mkdir(exist_ok=True)
        for file_id in sorted(set(inputs) - outputs):
            size = inputs[file_id]
            dst = data_dir / file_id
            if dst.is_file() and dst.stat().st_size == size:
                continue
            if dst.exists():
                dst.unlink()
            src = self.bench_dir / file_id if self.bench_dir else None
            if src and src.is_file() and src.stat().st_size == size:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy(src, dst)
                continue
            with open(dst, "wb") as fp:
                dt_datagen.write_file(fp, file_id, size,
                                      data_class=None if self.data_class == "auto"
                                      else self.data_class,
                                      noise=self.data_noise)

    def translate(self, output_folder: pathlib.Path) -> Dict:
        """Write everything the driver needs into ``output_folder``.

        :param output_folder: Output directory (created if missing).
        :return: The placement document that was written.
        """
        self._abs_out = pathlib.Path(output_folder).resolve()
        output_folder = pathlib.Path(output_folder).resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        assignment = self.assign_nodes()
        placement = self.build_placement(assignment, output_folder)
        (output_folder / "placement.json").write_text(json.dumps(placement, indent=2) + "\n")
        if self.api:
            self.logger.info(f"wrote API placement.json ({len(placement['tasks'])} "
                             f"tasks) under {output_folder}")
            return placement
        n_scripts = self._write_level_scripts(assignment, output_folder)
        shutil.copy(THIS_DIR / "run_dist.py", output_folder / "run_dist.py")
        (output_folder / "run_dist.py").chmod(0o755)
        self._copy_binary_files(output_folder)
        self._generate_input_files(output_folder)
        logs = output_folder / "logs"
        if logs.exists():
            shutil.rmtree(logs)
        self.logger.info(f"wrote placement.json ({len(placement['tasks'])} tasks, "
                         f"{placement['levels']} levels) and {n_scripts} level scripts "
                         f"under {output_folder}")
        return placement


def summarize(placement: Dict) -> None:
    """Print a per-level node/task summary of a placement.

    :param placement: Placement document.
    """
    per_level: Dict[int, Dict[str, int]] = {}
    for info in placement["tasks"].values():
        per_level.setdefault(info["level"], {}).setdefault(info["node"], 0)
        per_level[info["level"]][info["node"]] += 1
    cross = sum(1 for f in placement["files"].values()
                if f["producer_node"] and any(n != f["producer_node"]
                                              for n in f["consumer_nodes"]))
    print(f"[dt_translator] placement={placement['placement']} nodes={placement['nodes']}")
    for level in sorted(per_level):
        parts = ", ".join(f"{n}:{c}" for n, c in sorted(per_level[level].items()))
        print(f"[dt_translator]   L{level}: {parts}")
    print(f"[dt_translator] files={len(placement['files'])} "
          f"cross-node-consumed={cross}")


def main(argv=None) -> int:
    """CLI: generate the benchmark, translate it, print the placement.

    :param argv: Argument list (default ``sys.argv[1:]``).
    :return: Process exit code.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", required=True, choices=sorted(RECIPE_IMPORTS))
    p.add_argument("--num-tasks", type=int, required=True)
    p.add_argument("--data-footprint", default="0",
                   help="total workflow data footprint, e.g. 30G (0 = recipe default)")
    p.add_argument("--hostfile", default=None, help="one node per line (default localhost)")
    p.add_argument("--placement", default="round_robin", choices=PLACEMENTS)
    p.add_argument("--stage-sets", type=int, default=2)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--clio-prefix", action="store_true",
                   help="prefix every task path with clio:: for the CTE interposer")
    p.add_argument("--data-class", default="auto",
                   choices=["auto"] + list(dt_datagen.CLASSES))
    p.add_argument("--data-noise", type=float, default=None)
    p.add_argument("--cpu-work", type=int, default=1,
                   help="wfbench cpu_work (must be > 0 or wfbench does no I/O)")
    p.add_argument("--percent-cpu", type=float, default=1.0)
    p.add_argument("--python", default=sys.executable,
                   help="interpreter for bin/wfbench_dt (default: this one)")
    p.add_argument("--api", action="store_true",
                   help="write only placement.json for dtschedule_wfrun (CTE API)")
    p.add_argument("--size-mode", default="uniform", choices=SIZE_MODES,
                   help="--api file sizes: WfBench uniform or instance-proportional")
    p.add_argument("--min-file-kb", type=int, default=1024,
                   help="--size-mode instance: smallest file (KiB)")
    p.add_argument("--max-file-mb", type=int, default=0,
                   help="--size-mode instance: largest file (MiB, 0 = no cap)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed python/numpy RNGs so WfChef builds the same instance")
    args = p.parse_args(argv)
    if args.seed is not None:
        import random
        import numpy
        random.seed(args.seed)
        numpy.random.seed(args.seed)
    if args.cpu_work <= 0:
        print("[dt_translator] cpu_work <= 0 disables wfbench I/O; using 1")
        args.cpu_work = 1
    out = pathlib.Path(args.out).resolve()
    bench_dir = out / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    data_bytes = parse_size(args.data_footprint)
    data_mb = (data_bytes + (1 << 20) - 1) // (1 << 20) if data_bytes else 0
    nodes = read_hostfile(args.hostfile)
    print(f"[dt_translator] recipe={args.recipe} num_tasks={args.num_tasks} "
          f"footprint={data_bytes}B (={data_mb} MB) nodes={nodes}")
    bm = DtWorkflowBenchmark(recipe=load_recipe(args.recipe), num_tasks=args.num_tasks,
                             data_class=None if args.data_class == "auto" else args.data_class,
                             data_noise=args.data_noise, sparse_inputs=args.api)
    kwargs = dict(save_dir=bench_dir, cpu_work=args.cpu_work, percent_cpu=args.percent_cpu)
    if data_mb > 0:
        kwargs["data"] = int(data_mb)
    json_path = bm.create_benchmark(**kwargs)
    print(f"[dt_translator] benchmark JSON: {json_path}")
    # Pass the in-memory Workflow: loading the JSON back through
    # wfcommons.Instance would fetch the newest WfFormat schema from GitHub,
    # which rejects the schemaVersion 1.5 documents wfcommons 1.2 writes.
    tr = DistributedTranslator(bm.workflow, nodes, placement=args.placement,
                               stage_sets=args.stage_sets, clio_prefix=args.clio_prefix,
                               data_class=args.data_class, data_noise=args.data_noise,
                               bench_dir=bench_dir, python_exe=args.python,
                               api=args.api, size_mode=args.size_mode,
                               instance_info=bm.instance_info,
                               instance_out_ids=bm.instance_out_ids,
                               min_file_bytes=args.min_file_kb << 10,
                               max_file_bytes=args.max_file_mb << 20)
    placement = tr.translate(out)
    summarize(placement)
    if args.api:
        return 0
    print(f"[dt_translator] run with: {args.python} {out / 'run_dist.py'} --out {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
