#!/usr/bin/env python3
"""
``DtWorkflowBenchmark``: WfCommons ``WorkflowBenchmark`` with compressible
input data and recipe-derived file extensions.

Stock ``WorkflowBenchmark.create_benchmark(data=<int>)`` names every file
``<task>_output.txt`` / ``<task>_input.txt`` and then strips even that
suffix in ``_rename_files_to_wfbench_format`` (``<task>_outfile_0001``),
so nothing downstream can tell a FITS image from a VCF table. It also
fills workflow input files with ``os.urandom`` (bench.py line 337; the
unused helpers at lines 599 and 764 do the same).

This subclass overrides the smallest set of methods that fixes both:

``_create_data_footprint``
    Calls the stock method, then renames every file so its extension is
    the recipe's most representative one for the producing task type
    (from ``task_type_stats.json``). Root inputs take the task type's
    input extension.
``_rename_files_to_wfbench_format``
    Same renaming as stock but keeps the extension, and pre-creates the
    workflow input files with ``dt_datagen`` in ``save_dir`` so the stock
    ``os.urandom`` branch (guarded by ``if not file_path.is_file()``) is
    skipped.
``_generate_data_for_root_nodes``
    Stock helper (unused by ``create_benchmark``) rewritten to use
    ``dt_datagen`` for completeness.
"""
import inspect
import json
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple, Type, Union

from wfcommons.common import File, Task
from wfcommons.wfbench import WorkflowBenchmark
from wfcommons.wfchef.wfchef_abstract_recipe import WfChefWorkflowRecipe

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dt_datagen  # noqa: E402

STOCK_OUTPUT_SUFFIX = "_output.txt"
STOCK_INPUT_SUFFIX = "_input.txt"


def task_type_of(task: Task) -> str:
    """Return the recipe task type (e.g. ``mProject``) of a task.

    WfChef names tasks ``<type>_<8 digits>`` and stores the type in
    ``task.name``; ``task.category`` is unset for synthetic workflows.

    :param task: A workflow task.
    :return: The task type string.
    """
    if task.name and not task.name[-1].isdigit():
        return task.name
    return task.task_id.split("_0")[0]


def _pick_extension(stats: Dict, direction: str) -> Optional[str]:
    """Pick the representative extension from a task type's stats.

    :param stats: One entry of ``task_type_stats.json``.
    :param direction: ``"input"`` or ``"output"``.
    :return: The extension (with dot) carrying the most bytes, or None.
    """
    table = stats.get(direction) or {}
    best, best_max = None, -1.0
    for ext, info in table.items():
        if not ext.startswith(".") or ext == ".None":
            continue
        hi = float((info or {}).get("max", 0) or 0)
        if hi > best_max:
            best, best_max = ext, hi
    return best


def recipe_extensions(recipe: Type[WfChefWorkflowRecipe]
                      ) -> Dict[str, Tuple[str, str]]:
    """Map every task type of a recipe to its (input, output) extension.

    :param recipe: A WfChef recipe class (e.g. ``MontageRecipe``).
    :return: ``{type: (in_ext, out_ext)}``; missing ones fall back to
        ``.dat`` so the generator still picks a float class.
    """
    recipe_dir = pathlib.Path(inspect.getfile(recipe)).resolve().parent
    stats_path = recipe_dir / "task_type_stats.json"
    if not stats_path.is_file():
        return {}
    stats = json.loads(stats_path.read_text())
    out = {}
    for task_type, entry in stats.items():
        in_ext = _pick_extension(entry, "input")
        out_ext = _pick_extension(entry, "output")
        out[task_type] = (in_ext or out_ext or ".dat",
                          out_ext or in_ext or ".dat")
    return out


class DtWorkflowBenchmark(WorkflowBenchmark):
    """WorkflowBenchmark that emits typed file names and compressible data.

    :param recipe: A workflow recipe class.
    :param num_tasks: Total number of tasks in the benchmark workflow.
    :param with_flowcept: Enable Flowcept provenance (passed through).
    :param logger: Logger (passed through).
    :param data_class: Force one ``dt_datagen`` class for input files
        (default: by extension).
    :param data_noise: Noise amplitude for float data (default: env/0.05).
    :param write_inputs: Pre-create workflow input files in ``save_dir``
        with generated data (True) or leave that to the translator by
        creating nothing (False; stock would then write urandom, so the
        translator must generate them itself).
    :param sparse_inputs: Pre-create workflow input files as sparse
        placeholders of the right size instead (CTE-API runs stage their
        inputs themselves, so nothing is written to the shared FS and the
        stock ``os.urandom`` branch is still skipped).

    ``instance_info`` maps every task id to what the WfChef instance said
    about it before ``create_benchmark`` zeroes it: ``runtime`` (s),
    ``cores``, ``out_bytes`` (sum of its output files) and ``inputs``
    (``[(file id, bytes)]``), plus ``out_ids`` of all instance outputs.
    """

    def __init__(self, recipe: Type[WfChefWorkflowRecipe], num_tasks: int,
                 with_flowcept: bool = False,
                 logger: Optional[logging.Logger] = None,
                 data_class: Optional[str] = None,
                 data_noise: Optional[float] = None,
                 write_inputs: bool = True,
                 sparse_inputs: bool = False) -> None:
        """Create the benchmark generator; see class docstring."""
        super().__init__(recipe, num_tasks, with_flowcept, logger)
        self.data_class = data_class
        self.data_noise = data_noise
        self.write_inputs = write_inputs
        self.sparse_inputs = sparse_inputs
        self.instance_info: Dict[str, Dict] = {}
        self.instance_out_ids: set = set()
        self.extensions = recipe_extensions(recipe)
        self._dt_save_dir: Optional[pathlib.Path] = None

    def _set_argument_parameters(self, task: Task, *args, **kwargs) -> None:
        """Record the instance runtime/cores/file sizes, then call stock.

        Stock sets ``task.runtime = 0`` here and ``create_benchmark``
        empties the file lists right after, so this is the last point where
        the WfChef instance's values are visible.

        :param task: The task about to be turned into a wfbench task.
        :param args: Positional arguments of the stock method.
        :param kwargs: Keyword arguments of the stock method.
        """
        outs = [(f.file_id, int(f.size or 0)) for f in task.output_files]
        self.instance_info[task.task_id] = {
            "runtime": float(task.runtime or 0.0),
            "cores": int(task.cores or 1),
            "out_bytes": sum(sz for _, sz in outs),
            "inputs": [(f.file_id, int(f.size or 0)) for f in task.input_files],
        }
        self.instance_out_ids.update(fid for fid, _ in outs)
        super()._set_argument_parameters(task, *args, **kwargs)

    def _ext_for(self, task: Task, direction: str) -> str:
        """Look up the extension for a task's inputs or outputs.

        :param task: The task.
        :param direction: ``"input"`` or ``"output"``.
        :return: Extension with leading dot.
        """
        in_ext, out_ext = self.extensions.get(task_type_of(task),
                                              (".dat", ".dat"))
        return in_ext if direction == "input" else out_ext

    def _create_data_footprint(self, data, save_dir: pathlib.Path) -> None:
        """Stock footprint creation followed by extension assignment.

        :param data: Total footprint in MB (int) or per-type sizes (dict).
        :param save_dir: Benchmark save directory (remembered for later
            input-file generation).
        """
        self._dt_save_dir = pathlib.Path(save_dir)
        super()._create_data_footprint(data, save_dir)
        self._apply_extensions()

    def _apply_extensions(self) -> None:
        """Rename ``*_output.txt`` / ``*_input.txt`` files to typed names.

        Output files take the producer's output extension; root input
        files take the consumer's input extension; task args are
        rewritten to match.
        """
        mapping: Dict[str, str] = {}
        for task in self.workflow.tasks.values():
            for file in task.output_files:
                if file.file_id.endswith(STOCK_OUTPUT_SUFFIX):
                    stem = file.file_id[: -len(".txt")]
                    mapping[file.file_id] = stem + self._ext_for(task, "output")
        for task in self.workflow.tasks.values():
            for file in task.input_files:
                if file.file_id in mapping:
                    continue
                if file.file_id.endswith(STOCK_INPUT_SUFFIX):
                    stem = file.file_id[: -len(".txt")]
                    mapping[file.file_id] = stem + self._ext_for(task, "input")
        if not mapping:
            return
        ordered = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
        for task in self.workflow.tasks.values():
            for file in list(task.output_files) + list(task.input_files):
                if file.file_id in mapping:
                    file.file_id = mapping[file.file_id]
            for i, arg in enumerate(task.args):
                for old, new in ordered:
                    if old in arg:
                        arg = arg.replace(old, new)
                task.args[i] = arg

    def _rename_files_to_wfbench_format(self) -> List[File]:
        """Rename files to the wfbench scheme while keeping extensions.

        Mirrors the stock method (bench.py lines 204-248) with the
        commented-out ``extension`` lines re-enabled, then pre-creates
        the workflow input files with generated data.

        :return: The workflow input files that need to be generated.
        """
        new_file_names: Dict[str, str] = {}
        task_output_counter = 0
        workflow_inputs: List[File] = []
        for task in self.workflow.tasks.values():
            output_files = sorted(task.output_files, key=lambda x: -len(x.file_id))
            for file in output_files:
                if file.file_id in new_file_names:
                    raise ValueError(f"File name {file.file_id} already exists")
                task_output_counter += 1
                extension = "".join(pathlib.Path(file.file_id).suffixes)
                new_name = f"{task.task_id}_outfile_{task_output_counter:04d}{extension}"
                new_file_names[file.file_id] = new_name
                for i, item in enumerate(task.args):
                    if file.file_id in item:
                        task.args[i] = task.args[i].replace(file.file_id, new_name)
                file.file_id = new_name
        for task in self.workflow.tasks.values():
            input_files = sorted(task.input_files, key=lambda x: -len(x.file_id))
            for file in input_files:
                org_name = file.file_id
                if file.file_id in new_file_names:
                    file.file_id = new_file_names[file.file_id]
                else:
                    workflow_inputs.append(file)
                    extension = "".join(pathlib.Path(file.file_id).suffixes)
                    new_name = f"workflow_infile_{len(workflow_inputs):04d}{extension}"
                    new_file_names[file.file_id] = new_name
                    file.file_id = new_name
                for i, item in enumerate(task.args):
                    if org_name in item:
                        task.args[i] = task.args[i].replace(org_name, file.file_id)
        if self.sparse_inputs:
            self._dt_sparse_workflow_inputs(workflow_inputs)
        elif self.write_inputs:
            self._dt_write_workflow_inputs(workflow_inputs)
        return workflow_inputs

    def _dt_sparse_workflow_inputs(self, workflow_inputs: List[File]) -> None:
        """Create workflow input files as sparse placeholders in ``save_dir``.

        :param workflow_inputs: Files returned by the rename step.
        """
        if self._dt_save_dir is None:
            return
        self._dt_save_dir.mkdir(parents=True, exist_ok=True)
        for file in workflow_inputs:
            path = self._dt_save_dir / file.file_id
            with open(path, "ab") as fp:
                fp.truncate(int(file.size))

    def _dt_write_workflow_inputs(self, workflow_inputs: List[File]) -> None:
        """Create workflow input files with generated data in ``save_dir``.

        Existing files of the right size are kept, so re-running the
        generator with the same parameters is cheap.

        :param workflow_inputs: Files returned by the rename step.
        """
        if self._dt_save_dir is None:
            return
        self._dt_save_dir.mkdir(parents=True, exist_ok=True)
        for i, file in enumerate(workflow_inputs):
            path = self._dt_save_dir / file.file_id
            if path.is_file() and path.stat().st_size == int(file.size):
                continue
            self.logger.info(f"Generating input {path} ({file.size} bytes) "
                             f"{i + 1}/{len(workflow_inputs)}")
            with open(path, "wb") as fp:
                dt_datagen.write_file(fp, file.file_id, int(file.size),
                                      data_class=self.data_class,
                                      noise=self.data_noise)

    def _generate_data_for_root_nodes(self, save_dir: pathlib.Path,
                                      data: Union[int, Dict[str, str]]) -> None:
        """Stock helper (bench.py 583-600) with generated data.

        :param save_dir: Directory for the root input files.
        :param data: Per-type sizes or a single size.
        """
        for task in self.workflow.tasks.values():
            if self.workflow.tasks_parents[task.task_id]:
                continue
            file_size = data[task_type_of(task)] if isinstance(data, dict) else data
            name = f"{task.task_id}_input{self._ext_for(task, 'input')}"
            path = pathlib.Path(save_dir) / name
            if path.is_file():
                continue
            with open(path, "wb") as fp:
                dt_datagen.write_file(fp, name, int(file_size),
                                      data_class=self.data_class,
                                      noise=self.data_noise)
            self.logger.debug(f"Created file: {path}")


def generate_sys_data(num_files: int, tasks: Dict[str, int],
                      save_dir: pathlib.Path) -> List[str]:
    """Drop-in for bench.py ``generate_sys_data`` (line 747) with generated data.

    :param num_files: Number of copies of each file to generate.
    :param tasks: ``{file name: size in bytes}``.
    :param save_dir: Output directory.
    :return: The names written.
    """
    names = []
    for _ in range(num_files):
        for name, size in tasks.items():
            names.append(name)
            path = pathlib.Path(save_dir) / name
            with open(path, "wb") as fp:
                dt_datagen.write_file(fp, name, int(size))
    return names
