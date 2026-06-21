"""
Per-node `dd` writer through the CTE FUSE mount.

Drop-in replacement for `builtin.ior` in the multi-node CTE pipeline. Where
ior wants an MPI runtime and a shared file layout, this just fans out one
`dd` invocation per node via pssh -- a much smaller blast radius for
debugging the clio_run 2n flow.

Each node writes <count>x<bs> to <mountpoint>/dd_<hostname>.bin (file-per-
process semantics, by construction).
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, PsshExecInfo


class ClioCteDd(Application):
    """Run `dd` through the FUSE mount on every node in the hostfile."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'mountpoint',
                'msg': 'Where the CTE FUSE filesystem is mounted on each node.',
                'type': str,
                'default': '${HOME}/cte_libfuse_sbatch',
            },
            {
                'name': 'bs',
                'msg': '`dd` block size (e.g. 1M).',
                'type': str,
                'default': '1M',
            },
            {
                'name': 'count',
                'msg': '`dd` count (blocks).',
                'type': int,
                'default': 64,
            },
            {
                'name': 'fdatasync',
                'msg': 'Pass `conv=fdatasync` to dd.',
                'type': bool,
                'default': True,
            },
        ]

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

    def start(self):
        import os
        mp = self.config['mountpoint']
        bs = self.config['bs']
        count = int(self.config['count'])
        conv = ' conv=fdatasync' if self.config.get('fdatasync', True) else ''

        log_dir = os.path.join(self.shared_dir, 'dd_logs')
        os.makedirs(log_dir, exist_ok=True)

        # jarvis-cd's pssh path prepends every env var as
        # ``KEY="val" KEY="val" <cmd>``. Bash only accepts that prefix
        # in front of a *simple* command, so anything starting with a
        # subshell (``(`` or ``{``) or a redirection (``> ...``) errors
        # out at ssh-time with "syntax error near unexpected token".
        # Workaround: emit a tiny script onto shared NFS and have pssh
        # run ``bash <script>``. The script is the simple command, so
        # bash happily applies the env prefix to it.
        script_path = os.path.join(self.shared_dir, 'dd_run.sh')
        script_body = (
            '#!/bin/bash\n'
            f'h=$(hostname)\n'
            f'log={log_dir}/dd_$h.log\n'
            f'mkdir -p {log_dir}\n'
            f'{{\n'
            f'  echo "=== start on $h $(date) ==="\n'
            f'  echo "mountpoint listing:"\n'
            f'  ls -la {mp}/ || true\n'
            f'  echo "running: dd if=/dev/zero of={mp}/dd_$h.bin '
            f'bs={bs} count={count}{conv}"\n'
            f'  dd if=/dev/zero of={mp}/dd_$h.bin bs={bs} '
            f'count={count}{conv}\n'
            f'  echo "dd_rc=$?"\n'
            f'  stat -c "size=%s %n" {mp}/dd_$h.bin || true\n'
            f'  echo "=== end on $h $(date) ==="\n'
            f'}} >"$log" 2>&1\n'
        )
        with open(script_path, 'w') as fh:
            fh.write(script_body)
        os.chmod(script_path, 0o755)

        cmd = f'bash {script_path}'
        self.log(f'Running per-host dd via {script_path}')
        self.log(f'  per-host logs in: {log_dir}/')
        Exec(cmd, PsshExecInfo(
            env=self.mod_env,
            hostfile=self.hostfile,
        )).run()

        # Aggregate per-host logs into the jarvis stdout so we can see
        # exactly what each node did, with throughput numbers.
        for f in sorted(os.listdir(log_dir)):
            path = os.path.join(log_dir, f)
            self.log(f'--- {f} ---')
            try:
                with open(path) as fh:
                    for line in fh:
                        self.log(f'    {line.rstrip()}')
            except Exception as e:
                self.log(f'    (read failed: {e})')

    def stop(self):
        pass

    def clean(self):
        pass
