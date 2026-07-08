"""
This module provides classes and methods to launch Redis.
Redis cluster is used if the hostfile has many hosts
"""
import time
from jarvis_cd.core.pkg import Application
from jarvis_cd.util.logger import Color
from jarvis_cd.util.hostfile import Hostfile
from jarvis_cd.shell import Exec, LocalExecInfo, PsshExecInfo
from jarvis_cd.shell.process import Kill
from jarvis_clio_core.container_utils import container_kwargs, all_hosts_ok


class Redis(Application):
    """
    This class provides methods to launch the Ior application.
    """
    def _init(self):
        """
        Initialize paths
        """
        pass

    def _configure_menu(self):
        """
        Create a CLI menu for the configurator method.
        For thorough documentation of these parameters, view:
        https://github.com/scs-lab/jarvis-util/wiki/3.-Argument-Parsing

        :return: List(dict)
        """
        return [
            {
                'name': 'port',
                'msg': 'The port to use for the cluster',
                'type': int,
                'default': 6379,
                'choices': [],
                'args': [],
            },
            {
                'name': 'single_instance',
                'msg': 'Force ONE redis server on the first host even when the '
                       'hostfile has >1 host (skip the cluster branch). Use '
                       'when redis is a metadata singleton — e.g. JuiceFS meta '
                       'over redis://.../1, which needs SELECT and so cannot '
                       'use DB0-only cluster mode. Default keeps the legacy '
                       'behaviour (cluster iff >1 host).',
                'type': bool,
                'default': False,
            },
        ]

    def _eff_hostfile(self):
        """The hostfile the redis server(s) actually run on.

        With single_instance set and a multi-host pipeline hostfile, collapse
        to just the FIRST host so redis stays a plain (non-cluster) server:
        cluster mode is DB0-only and would break a JuiceFS meta_url of
        redis://.../1 (SELECT 1), and 4 servers sharing one nodes.conf on the
        NFS shared dir collide anyway. Otherwise return the full hostfile
        (>1 host still takes the cluster branch)."""
        hf = self.hostfile
        if self.config.get('single_instance') and len(hf.hosts) > 1:
            return Hostfile(
                hosts=hf.hosts[:1],
                hosts_ip=hf.hosts_ip[:1] if hf.hosts_ip else None)
        return hf

    def _configure(self, **kwargs):
        """
        Converts the Jarvis configuration to application-specific configuration.
        E.g., OrangeFS produces an orangefs.xml file.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        # Create the redis hostfile
        self.copy_template_file(f'{self.pkg_dir}/config/redis.conf',
                                f'{self.shared_dir}/redis.conf',
                                {
                                    'PORT': self.config['port']
                                })

    def _redis_cli(self, args, expect=None, timeout_s=2):
        """Run ``redis-cli <args>`` in the deployment context (inside the
        apptainer instance for a container deploy — redis-cli need not exist
        host-side). ``timeout N`` bounds each attempt like the old
        subprocess timeout did; callers keep their own retry loops.

        :param args: redis-cli argument string (e.g. ``-p 6379 ping``)
        :param expect: optional substring required in every host's stdout
        :param timeout_s: per-attempt timeout in seconds
        :return: True iff every host exited 0 (and printed ``expect``)
        """
        res = Exec(f'timeout {timeout_s} redis-cli {args}',
                   PsshExecInfo(env=self.mod_env,
                                hostfile=self._eff_hostfile(),
                                collect_output=True,
                                **container_kwargs(self))).run()
        return all_hosts_ok(res, expect)

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """
        hostfile = self._eff_hostfile()
        host_str = [f'{host}:{self.config["port"]}' for host in hostfile.hosts]
        host_str = ' '.join(host_str)
        cluster_config_file = f'{self.private_dir}/nodes.conf'
        # Redis loads ./dump.rdb from its working directory at startup. The conf
        # sets `dir ./`, which under `apptainer exec` resolves to whatever CWD
        # the instance drops us in (e.g. $HOME) — and a stale dump.rdb left there
        # by a prior run or a newer redis crashes startup ("Can't handle RDB
        # format version 11 / Fatal error loading the DB. Exiting.") even though
        # save "" means we never write a fresh one. Pin --dir to THIS run's
        # private dir (container-visible: the cluster branch already read/wrote
        # nodes.conf there) and wipe any leftover dump so startup is clean.
        Exec(f'rm -f {self.private_dir}/dump.rdb',
             PsshExecInfo(env=self.mod_env, hostfile=hostfile,
                          **container_kwargs(self))).run()
        # Create redis servers
        self.log('Starting individual servers', color=Color.YELLOW)
        cmd = [
            'redis-server',
            f'{self.shared_dir}/redis.conf',
            f'--dir {self.private_dir}',
        ]
        if len(hostfile) > 1:
            cmd += [
                f'--cluster-enabled yes',
                f'--cluster-config-file {cluster_config_file}',
                f'--cluster-node-timeout 5000',
            ]

        cmd = ' '.join(cmd)
        # Wrapped: the SIF's redis-server, running inside the instance where
        # juicefs and clio_redis_bench connect to it. exec_async does NOT
        # skip the container wrap (ExecFactory.run always calls
        # _prepare_container before delegating); the daemon parents into the
        # instance and lives until instance stop.
        Exec(cmd,
             PsshExecInfo(env=self.mod_env,
                          hostfile=hostfile,
                          exec_async=True,
                          **container_kwargs(self))).run()
        self.log(f'Sleeping for {self.config["sleep"]} seconds', color=Color.YELLOW)
        time.sleep(self.config['sleep'])

        # Wait for Redis to be ready before proceeding
        port = self.config['port']
        self.log(f'Waiting for Redis to accept connections on port {port}', color=Color.YELLOW)
        for i in range(30):
            if self._redis_cli(f'-p {port} ping', expect='PONG'):
                break
            time.sleep(1)
        else:
            self.log('WARNING: Redis did not respond to PING after 30s', color=Color.RED)

        # #526: single-node hygiene — wipe ALL DBs so each sweep combo starts
        # from an empty server (DB0 for redis_bench, DB1 for the JuiceFS meta),
        # regardless of any dump.rdb a prior run left in redis's CWD. The
        # multi-host cluster branch below already flushall's per host.
        if len(hostfile) <= 1:
            self.log('Flushing all DBs (fresh slate for this run)',
                     color=Color.YELLOW)
            self._redis_cli(f'-p {port} flushall', timeout_s=5)

        # Create redis clients
        if len(hostfile) > 1:
            # One-shot cluster admin commands, wrapped into an instance
            # (LOCAL exec_info with a remote hostfile is promoted to SSH on
            # the first host, so these run inside the head node's instance).
            self.log('Flushing all data and resetting the cluster', color=Color.YELLOW)
            for host in hostfile.hosts:
                Exec(f'redis-cli -p {self.config["port"]} -h {host} flushall',
                     LocalExecInfo(env=self.mod_env,
                                   hostfile=hostfile,
                                   **container_kwargs(self))).run()
                Exec(f'redis-cli -p {self.config["port"]} -h {host} cluster reset',
                     LocalExecInfo(env=self.mod_env,
                                   hostfile=hostfile,
                                   **container_kwargs(self))).run()

            self.log('Creating the cluster', color=Color.YELLOW)
            cmd = [
                'redis-cli',
                f'--cluster create {host_str}',
                '--cluster-replicas 0',
                '--cluster-yes'
            ]
            cmd = ' '.join(cmd)
            print(cmd)
            Exec(cmd,
                 LocalExecInfo(env=self.mod_env,
                               hostfile=hostfile,
                               **container_kwargs(self))).run()
            self.log(f'Sleeping for {self.config["sleep"]} seconds', color=Color.YELLOW)
            time.sleep(self.config['sleep'])

    def stop(self):
        """
        Stop a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        port = self.config['port']
        eff_hostfile = self._eff_hostfile()
        # Gracefully shutdown redis on each host via redis-cli (wrapped: the
        # server lives inside the instance, and redis-cli comes from the SIF).
        # Target the SAME hosts redis was started on (single_instance => first
        # host only), else stop tries to reach servers that were never launched.
        Exec(f'redis-cli -p {port} shutdown nosave',
             PsshExecInfo(env=self.mod_env,
                          hostfile=eff_hostfile,
                          **container_kwargs(self))).run()
        # Fallback: force kill any remaining redis-server processes, scoped
        # to this run's instance (its PID namespace).
        # (No partial= kwarg: this jarvis's Kill doesn't accept it -> it raised
        # "Kill.__init__() got an unexpected keyword argument 'partial'", which
        # aborted stop() before the force-kill could run.)
        Kill('redis-server',
             PsshExecInfo(env=self.mod_env,
                          hostfile=eff_hostfile,
                          **container_kwargs(self))).run()
        # Wait for the port to be free before returning. If the instance is
        # already gone the wrapped probe fails -> treated as "port free",
        # which is the right best-effort answer at teardown.
        for i in range(10):
            if not self._redis_cli(f'-p {port} ping', expect='PONG'):
                break
            time.sleep(1)
        time.sleep(1)

    def clean(self):
        """
        Destroy all data for an application. E.g., OrangeFS will delete all
        metadata and data directories in addition to the orangefs.xml file.

        :return: None
        """
        pass
