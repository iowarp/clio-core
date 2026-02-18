"""
This module provides classes and methods to launch the Redis benchmark tool.
Redis cluster is used if the hostfile has many hosts
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.util.logger import Color
from jarvis_cd.shell import Exec, LocalExecInfo


class RedisBenchmark(Application):
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
                'default': 7000,
                'choices': [],
                'args': [],
            },
            {
                'name': 'count',
                'msg': 'Number of requests to generate',
                'type': int,
                'default': 1000,
                'choices': [],
                'args': [],
            },
            {
                'name': 'write',
                'msg': 'Perform writes',
                'type': bool,
                'default': True,
                'choices': [],
                'args': [],
            },
            {
                'name': 'read',
                'msg': 'Perform reads',
                'type': bool,
                'default': True,
                'choices': [],
                'args': [],
            },
            {
                'name': 'nthreads',
                'msg': 'Number of threads',
                'type': int,
                'default': 1,
                'choices': [],
                'args': [],
            },
            {
                'name': 'pipeline',
                'msg': 'Number of requests to pipeline',
                'type': int,
                'default': 1,
                'choices': [],
                'args': [],
            },
            {
                'name': 'req_size',
                'msg': 'Size of requests (bytes)',
                'type': int,
                'default': 3,
                'choices': [],
                'args': [],
            },
            {
                'name': 'node',
                'msg': 'The node id to use for the cluster',
                'type': int,
                'default': 0,
                'choices': [],
                'args': [],
            },
        ]

    def _configure(self, **kwargs):
        """
        Converts the Jarvis configuration to application-specific configuration.
        E.g., OrangeFS produces an orangefs.xml file.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        pass

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """

        hostfile = self.hostfile
        bench_type = [
            'set' if self.config['write'] else '',
            'get' if self.config['read'] else '',
        ]
        bench_type = ','.join([b for b in bench_type if b])
        cmd = [
            'redis-benchmark',
            f'-n {self.config["count"]}',
            f'-t {bench_type}',
            f'-P {self.config["pipeline"]}',
            f'--threads {self.config["nthreads"]}',
            f'-d {self.config["req_size"]}',
            f'-p {self.config["port"]}',
        ]
        if len(hostfile) > 1:
            cmd += [
                f'-h {hostfile.hosts[self.config["node"]]}',
                f'--cluster'
            ]
        self.log('Starting the cluster', color=Color.YELLOW)
        Exec(' '.join(cmd),
             LocalExecInfo(env=self.mod_env,
                           hostfile=hostfile)).run()

    def stop(self):
        """
        Stop a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        pass

    def clean(self):
        """
        Destroy all data for an application. E.g., OrangeFS will delete all
        metadata and data directories in addition to the orangefs.xml file.

        :return: None
        """
        hostfile = self.hostfile
        for host in range(hostfile.hosts):
            Exec(f'redis-cli -p {self.config["port"]} -h {host} flushall',
                 LocalExecInfo(env=self.mod_env,
                               hostfile=hostfile)).run()
            Exec(f'redis-cli -p {self.config["port"]} -h {host} cluster reset',
                 LocalExecInfo(env=self.mod_env,
                               hostfile=hostfile)).run()
