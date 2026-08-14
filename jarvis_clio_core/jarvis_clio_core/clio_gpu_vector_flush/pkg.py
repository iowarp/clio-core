"""
GPU Vector Tiered Flush Benchmark Package

Drives clio_gpu_vector_flush_bench, which measures how well a GPU kernel's
block-level page flushes overlap with its compute across a real CTE tier
ladder (VRAM -> host DRAM -> NVMe).

The binary is self-contained: it writes its own CLIO_SERVER_CONF and starts
the runtime internally, so no clio_runtime or clio_cte package belongs in the
pipeline alongside it.  The tier ladder is therefore configured through the
environment (GV_HBM_MB / GV_DRAM_MB / GV_NVME_MB) rather than by a separate
CTE package.

The parameter that makes this a tiering study is `pages_per_block`: the
per-block page cache.  Left at the binary's default the cache is sized to the
whole working set, nothing is ever evicted, and the lower tiers never see
traffic.  Sizing it below the working set forces each block to evict, which is
what pushes data down the ladder.

`total_io_per_block_mb` is the knob of record for how much each block moves;
the region flushed per iteration is derived as total / iters, so changing the
iteration count does not silently change the working set.

Parameters:
- blocks:                 CUDA blocks (each owns an independent page cache)
- threads:                threads per block
- iters:                  regions written+flushed per block
- spin_us:                simulated compute per iteration, microseconds
- page_kb:                page granularity
- total_io_per_block_mb:  total bytes each block writes per pass
- pages_per_block:        per-block page cache slots (0 = whole working set)
- repeat:                 timed repetitions; the best is reported
- read_mode:              run the read+prefetch benchmark instead of write+flush
- vram_mb / dram_mb / nvme_mb: tier capacities (0 omits the tier)
- nvme_path:              backing file for the NVMe tier
- tier_type:              host tier allocation, "ram" (pageable) or "pinned"
- busy_us:                runtime worker first_busy_wait
- output_dir:             directory for benchmark result files

Assumes clio_gpu_vector_flush_bench is installed and available in PATH.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os


class ClioGpuVectorFlush(Application):
    """
    GPU vector flush/prefetch overlap benchmark over a VRAM/DRAM/NVMe ladder.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'blocks',
             'msg': 'CUDA blocks (each owns an independent page cache)',
             'type': int, 'default': 64},
            {'name': 'threads',
             'msg': 'Threads per block',
             'type': int, 'default': 256},
            {'name': 'iters',
             'msg': 'Regions written and flushed per block',
             'type': int, 'default': 16},
            {'name': 'spin_us',
             'msg': 'Simulated compute per iteration (microseconds)',
             'type': int, 'default': 100},
            {'name': 'page_kb',
             'msg': 'Page granularity in KB',
             'type': int, 'default': 1024},
            {'name': 'total_io_per_block_mb',
             'msg': 'Total MB each block writes per pass '
                    '(region per iteration = this / iters)',
             'type': int, 'default': 64},
            {'name': 'pages_per_block',
             'msg': 'Per-block page cache slots '
                    '(0 = size to the whole working set, so nothing evicts)',
             'type': int, 'default': 0},
            {'name': 'repeat',
             'msg': 'Timed repetitions per phase (best is reported)',
             'type': int, 'default': 3},
            {'name': 'read_mode',
             'msg': 'Run the read+prefetch benchmark instead of write+flush',
             'type': bool, 'default': False},
            {'name': 'vram_mb',
             'msg': 'VRAM (kHBM) tier capacity in MB, 0 to omit',
             'type': int, 'default': 1024},
            {'name': 'dram_mb',
             'msg': 'Host DRAM tier capacity in MB, 0 to omit',
             'type': int, 'default': 256},
            {'name': 'nvme_mb',
             'msg': 'NVMe/file tier capacity in MB, 0 to omit',
             'type': int, 'default': 16384},
            {'name': 'nvme_path',
             'msg': 'Backing file for the NVMe tier',
             'type': str, 'default': '/tmp/gv_flush_nvme.dat'},
            {'name': 'tier_type',
             'msg': 'Host tier allocation: "ram" (pageable) or "pinned"',
             'type': str, 'default': 'pinned'},
            {'name': 'busy_us',
             'msg': 'Runtime worker first_busy_wait in microseconds',
             'type': int, 'default': 10000},
            {'name': 'output_dir',
             'msg': 'Output directory for benchmark results',
             'type': str, 'default': '/tmp/clio_gpu_vector_flush'},
        ]

    def _flush_mb(self):
        """Region flushed per iteration, derived from the per-block total.

        Kept as a whole number of MB: a fractional region would not be
        expressible on the command line, and silently rounding it would make
        the working set differ from what the pipeline asked for.
        """
        total = self.config['total_io_per_block_mb']
        iters = self.config['iters']
        if iters <= 0:
            raise Exception('iters must be positive')
        if total % iters != 0:
            raise Exception(
                f'total_io_per_block_mb ({total}) must divide evenly by '
                f'iters ({iters}); otherwise the per-iteration region is '
                f'fractional and the real working set would not match the '
                f'configured one')
        flush_mb = total // iters
        if flush_mb * 1024 < self.config['page_kb']:
            raise Exception(
                f'per-iteration region is {flush_mb}MB but the page is '
                f'{self.config["page_kb"]}KB; the region must hold at least '
                f'one page')
        return flush_mb

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # The tier ladder reaches the binary through the environment: it
        # generates its own CLIO_SERVER_CONF, so there is no config file for
        # a separate CTE package to own.
        self.setenv('GV_HBM_MB', str(self.config['vram_mb']))
        self.setenv('GV_DRAM_MB', str(self.config['dram_mb']))
        self.setenv('GV_NVME_MB', str(self.config['nvme_mb']))
        self.setenv('GV_NVME_PATH', self.config['nvme_path'])
        self.setenv('GV_TIER_TYPE', self.config['tier_type'])
        self.setenv('GV_BUSY_US', str(self.config['busy_us']))

        flush_mb = self._flush_mb()
        working_mb = self.config['blocks'] * self.config[
            'total_io_per_block_mb']
        cache_mb = (self.config['blocks'] * self.config['pages_per_block'] *
                    self.config['page_kb'] / 1024.0)
        tier_mb = (self.config['vram_mb'] + self.config['dram_mb'] +
                   self.config['nvme_mb'])

        self.log('GPU vector tiered flush benchmark configured')
        self.log(f'  blocks x threads:   {self.config["blocks"]} x '
                 f'{self.config["threads"]}')
        self.log(f'  iters:              {self.config["iters"]} '
                 f'(region {flush_mb}MB, page {self.config["page_kb"]}KB)')
        self.log(f'  spin:               {self.config["spin_us"]} us/iter')
        self.log(f'  working set:        {working_mb}MB '
                 f'({self.config["total_io_per_block_mb"]}MB per block)')
        if self.config['pages_per_block']:
            self.log(f'  cache:              '
                     f'{self.config["pages_per_block"]} pages/block '
                     f'({cache_mb:.0f}MB device total)')
        else:
            self.log('  cache:              whole working set (no eviction)')
        self.log(f'  tiers:              VRAM {self.config["vram_mb"]}MB / '
                 f'DRAM {self.config["dram_mb"]}MB / '
                 f'NVMe {self.config["nvme_mb"]}MB')

        # A working set larger than the whole ladder cannot be stored, and the
        # failure surfaces late as a confusing put error rather than as a
        # sizing mistake, so say so up front.
        if working_mb > tier_mb:
            self.log(f'  WARNING: working set {working_mb}MB exceeds total '
                     f'tier capacity {tier_mb}MB -- puts will fail')

    def start(self):
        Which('clio_gpu_vector_flush_bench',
              LocalExecInfo(env=self.mod_env)).run()

        os.makedirs(self.config['output_dir'], exist_ok=True)
        # EVERY swept parameter must appear in the name. With only the cache
        # size in it, a sweep that also varies spin_us writes all six compute
        # levels to the same file and leaves one survivor per cache size --
        # silent data loss that looks like a completed sweep.
        tag = (f'ppb{self.config["pages_per_block"]}'
               f'_spin{self.config["spin_us"]}us'
               f'_b{self.config["blocks"]}'
               f'_io{self.config["total_io_per_block_mb"]}mb')
        output_file = os.path.join(self.config['output_dir'],
                                   f'gv_flush_{tag}.log')

        cmd = [
            'clio_gpu_vector_flush_bench',
            f'--blocks {self.config["blocks"]}',
            f'--threads {self.config["threads"]}',
            f'--iters {self.config["iters"]}',
            f'--spin-us {self.config["spin_us"]}',
            f'--page-kb {self.config["page_kb"]}',
            f'--flush-mb {self._flush_mb()}',
            f'--repeat {self.config["repeat"]}',
        ]
        if self.config['pages_per_block']:
            cmd.append(f'--pages-per-block {self.config["pages_per_block"]}')
        if self.config['read_mode']:
            cmd.append('--read')

        self.log(f'Running: {" ".join(cmd)}')
        # Run from the output directory: the binary drops its generated
        # gpu_vector_flush.yaml in the CWD, and two concurrent runs in the
        # same directory would overwrite each other's config.
        Exec(f'{" ".join(cmd)} 2>&1 | tee {output_file}',
             LocalExecInfo(env=self.mod_env,
                           cwd=self.config['output_dir'])).run()

        self.log(f'Benchmark completed - results saved to {output_file}')

    def stop(self):
        """The benchmark runs to completion on its own."""
        pass

    def clean(self):
        self.log('Cleaning GPU vector flush benchmark data')
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            if name.startswith('gv_flush_') or name == 'gpu_vector_flush.yaml':
                os.remove(os.path.join(out, name))
        # The NVMe tier's backing file is the bulk of the footprint (up to
        # nvme_mb), so it is the one thing clean must not leave behind.
        if os.path.exists(self.config['nvme_path']):
            os.remove(self.config['nvme_path'])
        try:
            os.rmdir(self.config['output_dir'])
        except OSError:
            pass
        self.log('Cleanup completed')
