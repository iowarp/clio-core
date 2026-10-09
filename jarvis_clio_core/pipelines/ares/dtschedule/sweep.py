#!/usr/bin/env python3
"""
Sweep helper for the DTSchedule evaluation pipelines.

Runs one jarvis pipeline several times with per-run ``pkg_name.key=value``
overrides (the same variable spelling as a jarvis pipeline test's ``vars:``)
and collects, per run, every package's ``_get_stat`` output plus the
wfcommons ``makespan.json`` into

    ${HOME}/jarvis-runs/dtschedule-results/<exp>/<run>.json

and the run's dtschedule trace CSVs into ``<exp>/traces/<run>/`` (the
next run's configure resets them in place).

Each run is a plain ``jarvis ppl load yaml <run.yaml>`` followed by
``jarvis ppl run``, so it is meant to be driven from INSIDE an allocation
(``salloc`` or a ``jarvis ppl submit`` wrapper): the pipeline's
``scheduler:`` block is stripped from the per-run YAML and the hosts come
from ``--hostfile`` (or the pipeline's own ``hostfile:``).

Examples::

    # explicit runs on montage_2n.yaml
    python3 sweep.py --exp e5 --pipeline montage_2n.yaml \\
        --run auto:dtschedule.force_scenario=auto \\
        --run s1:dtschedule.force_scenario=1 \\
        --run s3:dtschedule.force_scenario=3,dtschedule.load_aware=false \\
        --hostfile ${HOME}/hostfile.txt

    # every combination of a pipeline test's vars/loop
    python3 sweep.py --exp e6 --from-test e6_component_ablation.yaml

    python3 sweep.py --exp e7 --from-test e7_qos_sweep.yaml --dry-run
"""
import argparse
import copy
import glob
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import yaml


DEFAULT_RESULTS_ROOT = os.path.join('${HOME}', 'jarvis-runs',
                                    'dtschedule-results')
WFCOMMONS_SUFFIX = '.clio_wfcommons_dist'
DTSCHEDULE_SUFFIX = '.clio_dtschedule'


# ----------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------

def parse_args(argv=None):
    """Build and parse the command line.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).
    :return: argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp', required=True,
                        help='experiment name, results go to <root>/<exp>/')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pipeline', help='plain pipeline YAML to run')
    source.add_argument('--from-test',
                        help='pipeline-test YAML: its config: is the base and '
                             'its vars:/loop: generate the runs')
    parser.add_argument('--run', action='append', default=[],
                        metavar='NAME:pkg.key=value[,pkg.key=value...]',
                        help='one run; repeatable. Values are parsed as YAML '
                             '(1e-3, true, [a,b]). NAME alone = no override')
    parser.add_argument('--hostfile',
                        help='hostfile for every run (replaces the scheduler '
                             'block of the pipeline)')
    parser.add_argument('--results-root', default=DEFAULT_RESULTS_ROOT,
                        help=f'default {DEFAULT_RESULTS_ROOT}')
    parser.add_argument('--jarvis', default=None,
                        help='jarvis executable (default: next to this '
                             'interpreter, else on PATH)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip runs whose result JSON already exists')
    parser.add_argument('--destroy', action='store_true',
                        help='jarvis ppl destroy each run\'s pipeline after '
                             'collecting its stats')
    parser.add_argument('--dry-run', action='store_true',
                        help='write the per-run YAMLs and print the commands '
                             'without running them')
    return parser.parse_args(argv)


def find_jarvis(explicit):
    """Locate the jarvis executable.

    :param explicit: ``--jarvis`` value or None.
    :return: Path string.
    :raises FileNotFoundError: when no executable is found.
    """
    if explicit:
        return explicit
    beside = pathlib.Path(sys.executable).parent / 'jarvis'
    if beside.is_file():
        return str(beside)
    found = shutil.which('jarvis')
    if found:
        return found
    raise FileNotFoundError('jarvis not found; pass --jarvis')


# ----------------------------------------------------------------------
# run list
# ----------------------------------------------------------------------

def parse_override(spec):
    """Parse ``pkg.key=value`` into (pkg, key, value).

    :param spec: Override string.
    :return: Tuple (package name, key, YAML-parsed value).
    """
    if '=' not in spec:
        raise ValueError(f"override '{spec}' is not pkg.key=value")
    var, raw = spec.split('=', 1)
    if '.' not in var:
        raise ValueError(f"override '{spec}' needs a pkg_name.key variable")
    pkg, key = var.split('.', 1)
    return pkg, key, yaml.safe_load(raw)


def parse_run_specs(specs):
    """Turn ``--run`` strings into (name, {var: value}) pairs.

    :param specs: List of ``NAME:pkg.key=value,...`` strings.
    :return: List of (run name, overrides dict keyed ``pkg.key``).
    """
    runs = []
    for spec in specs:
        name, _, rest = spec.partition(':')
        overrides = {}
        for item in filter(None, rest.split(',')):
            pkg, key, value = parse_override(item)
            overrides[f'{pkg}.{key}'] = value
        runs.append((name, overrides))
    return runs


def combinations_from_test(variables, loop):
    """Expand a pipeline test's ``vars``/``loop`` the way jarvis does.

    Variables in one loop group advance together (zip); groups are
    combined by cartesian product. No loop = one group per variable.

    :param variables: ``vars:`` mapping var -> list of values.
    :param loop: ``loop:`` list of lists of var names.
    :return: List of overrides dicts.
    """
    if not variables:
        return [{}]
    loop = loop or [[name] for name in variables]
    groups = []
    for group in loop:
        values = [variables[name] if isinstance(variables[name], list)
                  else [variables[name]] for name in group]
        lengths = {len(v) for v in values}
        if len(lengths) != 1:
            raise ValueError(f'loop group {group} has unequal lengths')
        groups.append([dict(zip(group, row)) for row in zip(*values)])
    combos = []
    for product in itertools.product(*groups):
        combo = {}
        for part in product:
            combo.update(part)
        combos.append(combo)
    return combos


def load_source(args):
    """Load the base pipeline config and the run list.

    :param args: Parsed arguments.
    :return: Tuple (base config dict, list of (run name, overrides)).
    """
    path = args.from_test or args.pipeline
    with open(path) as fp:
        doc = yaml.safe_load(fp)
    if 'config' in doc:
        base = doc['config']
        test_runs = [(f'run{i}', combo) for i, combo in enumerate(
            combinations_from_test(doc.get('vars', {}), doc.get('loop', [])))]
    else:
        base = doc
        test_runs = []
    runs = parse_run_specs(args.run) or test_runs or [('run0', {})]
    return base, runs


# ----------------------------------------------------------------------
# per-run YAML
# ----------------------------------------------------------------------

def apply_overrides(config, overrides):
    """Set ``pkg.key`` overrides on the matching ``pkgs:`` entries.

    :param config: Pipeline config (modified in place).
    :param overrides: Mapping ``pkg_name.key`` -> value.
    """
    for var, value in overrides.items():
        pkg_name, key = var.split('.', 1)
        for pkg in config.get('pkgs', []):
            name = pkg.get('pkg_name', pkg.get('pkg_type', '').split('.')[-1])
            if name == pkg_name:
                pkg[key] = value
                break
        else:
            raise ValueError(f"package '{pkg_name}' not in the pipeline")


def write_run_yaml(base, exp, run_name, overrides, hostfile, out_dir):
    """Write the per-run pipeline YAML.

    :param base: Base pipeline config.
    :param exp: Experiment name.
    :param run_name: Run name.
    :param overrides: Overrides for this run.
    :param hostfile: Hostfile path or None.
    :param out_dir: Directory receiving ``yaml/<run>.yaml``.
    :return: Tuple (yaml path, pipeline name).
    """
    config = copy.deepcopy(base)
    config.pop('scheduler', None)
    pipeline_name = f'{exp}_{run_name}'
    config['name'] = pipeline_name
    if hostfile:
        config['hostfile'] = os.path.abspath(os.path.expandvars(hostfile))
    apply_overrides(config, overrides)
    yaml_dir = pathlib.Path(out_dir) / 'yaml'
    yaml_dir.mkdir(parents=True, exist_ok=True)
    path = yaml_dir / f'{run_name}.yaml'
    with open(path, 'w') as fp:
        fp.write(f'# generated by sweep.py: exp={exp} run={run_name}\n')
        yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
    return str(path), pipeline_name


# ----------------------------------------------------------------------
# running and collecting
# ----------------------------------------------------------------------

def run_cmd(cmd, dry_run):
    """Run a command, streaming its output.

    :param cmd: Argument list.
    :param dry_run: Only print the command.
    :return: Exit code (0 for dry runs).
    """
    print('+ ' + ' '.join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def collect_stats(pipeline_name):
    """Collect every package's ``_get_stat`` output through the jarvis API.

    :param pipeline_name: Name of the loaded pipeline.
    :return: Flat stats dict (empty on any failure, with the error logged).
    """
    try:
        from jarvis_cd.core.pipeline import Pipeline
    except ImportError as err:
        print(f'sweep: jarvis_cd not importable, no stats: {err}')
        return {}
    stats = {}
    try:
        pipeline = Pipeline(pipeline_name)  # a named Pipeline auto-loads
        for pkg_def in pipeline.packages:
            try:
                instance = pipeline._load_package_instance(pkg_def, pipeline.env)
                if hasattr(instance, '_get_stat'):
                    instance._get_stat(stats)
            except Exception as err:  # one package must not lose the rest
                print(f"sweep: stats from {pkg_def.get('pkg_id')} failed: {err}")
    except Exception as err:
        print(f'sweep: could not load pipeline {pipeline_name}: {err}')
    return stats


def read_makespan(config):
    """Read the wfcommons makespan.json named by the pipeline config.

    :param config: Per-run pipeline config.
    :return: Parsed JSON dict, or None when absent.
    """
    for pkg in config.get('pkgs', []):
        if str(pkg.get('pkg_type', '')).endswith(
                (WFCOMMONS_SUFFIX, '.clio_prodcons')):
            out = os.path.expandvars(pkg.get('out', ''))
            path = os.path.join(out, 'makespan.json')
            if os.path.isfile(path):
                with open(path) as fp:
                    return json.load(fp)
    return None


def trace_base(pipeline_name, config):
    """Resolve the dtschedule ``trace_path`` of a run.

    Mirrors the package default (``${HOME}/jarvis-runs/<pipeline>/
    dtschedule_trace``) when the knob is unset.

    :param pipeline_name: Name of the run's pipeline.
    :param config: Per-run pipeline config.
    :return: Base path string, or None without a dtschedule package or
        with tracing turned off.
    """
    for pkg in config.get('pkgs', []):
        if str(pkg.get('pkg_type', '')).endswith(DTSCHEDULE_SUFFIX):
            knob = str(pkg.get('trace_path', '') or '')
            if knob.lower() == 'off':
                return None
            if knob:
                return os.path.expandvars(knob)
            return os.path.join(os.path.expandvars('${HOME}'), 'jarvis-runs',
                                pipeline_name, 'dtschedule_trace')
    return None


def copy_traces(pipeline_name, config, out_dir, run_name):
    """Copy a run's trace CSVs to ``<out_dir>/traces/<run>/``.

    The next run reconfigures the pipeline, which resets the trace files,
    so they are saved before that happens; the eval loaders read them
    from here.

    :param pipeline_name: Name of the run's pipeline.
    :param config: Per-run pipeline config.
    :param out_dir: Experiment results directory.
    :param run_name: Run name.
    :return: List of copied file paths.
    """
    base = trace_base(pipeline_name, config)
    if not base:
        return []
    dest = pathlib.Path(out_dir) / 'traces' / run_name
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(glob.glob(f'{base}.*')):
        target = dest / os.path.basename(src)
        shutil.copy2(src, target)
        copied.append(str(target))
    copied.extend(copy_workload_logs(config, dest))
    return copied


def copy_workload_logs(config, dest):
    """Copy the prodcons producer/consumer logs (and the CTE-API wfcommons
    run's wfrun.log / tasks.csv) next to the traces.

    The package deletes them at the next run's start, and they hold the
    only per-file read errors when a consumer reports bad files.

    :param config: Per-run pipeline config.
    :param dest: Destination directory (a pathlib.Path).
    :return: List of copied file paths.
    """
    copied = []
    for pkg in config.get('pkgs', []):
        ptype = str(pkg.get('pkg_type', ''))
        if ptype.endswith('.clio_prodcons'):
            names = ('producer.log', 'consumer.log')
        elif ptype.endswith(WFCOMMONS_SUFFIX) and pkg.get('api'):
            names = ('wfrun.log', 'tasks.csv')
        else:
            continue
        out = os.path.expandvars(pkg.get('out', ''))
        for name in names:
            src = os.path.join(out, name)
            if os.path.isfile(src):
                target = dest / name
                shutil.copy2(src, target)
                copied.append(str(target))
    return copied


def load_and_run(jarvis, yaml_path, dry_run):
    """``jarvis ppl load yaml`` then ``jarvis ppl run``, safe across concurrent sweeps.

    ``ppl run`` runs jarvis's single global *current* pipeline, which ``ppl
    load`` sets; two sweeps loading at nearly the same moment would run each
    other's pipeline. A file lock spans the load and the start of the run
    (``ppl run`` reads the current pipeline as it starts).

    :param jarvis: jarvis executable.
    :param yaml_path: Per-run pipeline YAML.
    :param dry_run: Print the commands only.
    :return: Exit code of the load, or of the run.
    """
    if dry_run:
        rc = run_cmd([jarvis, 'ppl', 'load', 'yaml', yaml_path], True)
        return rc or run_cmd([jarvis, 'ppl', 'run'], True)
    import fcntl
    lock = open(os.path.expanduser('~/.jarvis_sweep.lock'), 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        rc = run_cmd([jarvis, 'ppl', 'load', 'yaml', yaml_path], False)
        if rc != 0:
            return rc
        print(' '.join([jarvis, 'ppl', 'run']), flush=True)
        proc = subprocess.Popen([jarvis, 'ppl', 'run'])
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return proc.wait()


def execute_run(args, jarvis, base, run_name, overrides, out_dir):
    """Load, run and collect one sweep point.

    :param args: Parsed arguments.
    :param jarvis: jarvis executable.
    :param base: Base pipeline config.
    :param run_name: Run name.
    :param overrides: Overrides for this run.
    :param out_dir: Experiment results directory.
    :return: Result dict (also written to ``<out_dir>/<run>.json``).
    """
    yaml_path, pipeline_name = write_run_yaml(
        base, args.exp, run_name, overrides, args.hostfile, out_dir)
    with open(yaml_path) as fp:
        config = yaml.safe_load(fp)
    result = {'exp': args.exp, 'run': run_name, 'pipeline': pipeline_name,
              'yaml': yaml_path, 'overrides': overrides,
              'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
    t0 = time.time()
    rc = load_and_run(jarvis, yaml_path, args.dry_run)
    result['wall_s'] = round(time.time() - t0, 3)
    result['rc'] = rc
    result['status'] = 'dry-run' if args.dry_run else (
        'success' if rc == 0 else 'failed')
    if not args.dry_run:
        result['stats'] = collect_stats(pipeline_name)
        result['makespan'] = read_makespan(config)
        result['traces'] = copy_traces(pipeline_name, config, out_dir,
                                       run_name)
        if args.destroy:
            run_cmd([jarvis, 'ppl', 'destroy', pipeline_name], False)
    with open(os.path.join(out_dir, f'{run_name}.json'), 'w') as fp:
        json.dump(result, fp, indent=2, sort_keys=True, default=str)
    return result


def main(argv=None):
    """Entry point.

    :param argv: Argument list.
    :return: Process exit code (number of failed runs, capped at 1).
    """
    args = parse_args(argv)
    jarvis = find_jarvis(args.jarvis)
    base, runs = load_source(args)
    out_dir = os.path.join(os.path.expandvars(args.results_root), args.exp)
    os.makedirs(out_dir, exist_ok=True)
    failed = 0
    for run_name, overrides in runs:
        target = os.path.join(out_dir, f'{run_name}.json')
        if args.skip_existing and os.path.isfile(target):
            print(f'sweep: {run_name}: exists, skipped')
            continue
        print(f'sweep: === {args.exp}/{run_name} {overrides} ===', flush=True)
        result = execute_run(args, jarvis, base, run_name, overrides, out_dir)
        makespan = (result.get('makespan') or {}).get('total_ms')
        print(f"sweep: {run_name}: {result['status']} rc={result['rc']} "
              f"wall={result['wall_s']} s makespan_ms={makespan} -> {target}")
        failed += result['status'] == 'failed'
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
