# External codecs: build setup

Builds the three GPU compressors NeuroPress **cannot select** — cuSZ, cuSZp v3
and ndzip — and turns them on in the Clio build. They are what
`../figures/fig9/figure_9.sh --panel b` runs as its `static-cusz`,
`static-cuszp` and `static-ndzip` arms.

Only setup lives here. The campaign that once drove these codecs through their
own Slurm phases has been removed; figure 9's panel (b) covers the same
comparison from one GPU.

## Why these codecs are baselines, not candidates

`kNeuroPressTrainedGpuBaseIds = {13,14,15,16,17,18,23,24}`
(`neuropress_bridge.cc`) filters cuSZ/cuSZp/ndzip out of the candidate set
before ranking, and the model has no output slot for them: its action space is
8 nvcomp algorithms x quantize x shuffle = 32. Upstream is the same — its own
trace campaign records ndzip as the per-chunk optimum on ~15% of chunks and
NeuroPress selecting it 0.0% of the time. So these arms measure what the
**action space** gives up, which is a separate question from whether the
**selector** is good.

## One-time setup

```bash
./install_codecs.sh --arch 80        # builds cuSZ, cuSZp v3 + ndzip, patches included
cp ../site.sh.example ../site.sh     # then edit it with your paths (gitignored)
```

Each codec goes in its own prefix under `$NPENV` (default `$HOME/np-env`) and
is skipped if already built.

**cuSZ** is v0.17.3, `PSZ_BACKEND=CUDA` with `PSZ_ACTIVATE_LC=ON`. Its clone
must be recursive: `PSZ_ACTIVATE_LC` compiles `third_party/lc_gen`, which
includes headers from the `third_party/lc` submodule, and without them the
build dies on `../lc/include/max_scan.h: No such file or directory`.

cuSZ and cuSZp install to `<prefix>/lib` on Debian/Ubuntu and `<prefix>/lib64`
elsewhere; `env.sh` spells them `lib64`, so the script symlinks the two.

`nvcomp` is not built here. The Debian package `libnvcomp5-dev-cuda-12` puts
its runtime in a directory the loader does not search — if `ldd` on
`libclio_cte_compressor_runtime.so` reports `libnvcomp.so.5 => not found`, add
it to `/etc/ld.so.conf.d/` and run `ldconfig`.

Then configure clio-core with the codecs on `CMAKE_PREFIX_PATH` — see the
message `install_codecs.sh` prints. **Verify `CLIO_CTP_ENABLE_{CUSZ,CUSZP,NDZIP}`
are all `ON`.** If a codec is missing the arm still runs: `WireIdForName` falls
back to zstd and produces a plausible but wrong result, so check the flags:

```bash
grep -E "CLIO_CTP_ENABLE_(CUSZ|CUSZP|NDZIP)" build/CMakeCache.txt
```

`cmake` must be >= 3.28 — clio-core uses `set_tests_properties(DIRECTORY ...)`,
and an older cmake fails configure with hundreds of "Can not find test to add
properties to".

## Two caveats before ranking anything against these arms

- **cuSZ and cuSZp do not hold the error bound.** Measured across this
  benchmark, they missed `eb=1e-3` on 151-276 of 300 chunks wherever it was
  checked. Any driver run at `--eb 1e-3 --check-bound` reproduces it
  (`../nyx/run_config.sh static-cusz --eb 1e-3 --check-bound --fields DIR`;
  figure 9 does not pass `--check-bound`, it only reports a verdict already
  in the log): both report `BOUND FAILED` while NeuroPress and the nvCOMP arms pass. The
  overshoot is small — worst case 47 ppm, inclusive-bound rounding rather than
  real accuracy loss — but it means a ratio comparison against them is not
  like-for-like unless the bound is verified.
- **Two cost models give opposite answers.** Ratio-only weights
  (`W_CT=0 W_DT=0 W_IO=1`) favour high-ratio slow codecs; time-inclusive
  (`W_CT=1`) favour fast ones. NeuroPress must be *run* under the weights it is
  then graded on, or it is being scored on an objective it was never given.
  cuSZ's viability in particular is entirely a bandwidth question.
