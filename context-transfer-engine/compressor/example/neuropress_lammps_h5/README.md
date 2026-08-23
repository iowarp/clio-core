# LAMMPS → HDF5 → Clio → NeuroPress

A **stock, unmodified LAMMPS** writing molecular dynamics through Clio's HDF5
VOL connector, compressed per chunk by NeuroPress, read back in a separate
process byte-identical from the compressed tier.

The application contains no Clio code and links no Clio library:

```console
$ ldd lmp | grep -c -E 'clio|gpucompress'
0
```

Compare the deck's one output line with upstream NeuroPress's LAMMPS
integration, which needs a patched LAMMPS carrying a custom fix style that
calls `gpucompress_*` directly:

| | |
|---|---|
| this example | `dump d1 all h5md 50 out.h5 position velocity force` |
| upstream | `fix gpuc all gpucompress 50 positions velocities forces` |

Everything Clio needs arrives through the environment.

(For the other direction -- LAMMPS linked *into* a program as `liblammps`,
with Clio's client called directly and no HDF5 in the path -- see
`../neuropress_lammps_lib`, which also documents where in the timestep the
atom arrays are complete and when each hand-over point fires.)

## Building LAMMPS

Nothing here is Clio-specific; it is a normal LAMMPS build that happens to
include the H5MD dump. H5MD is what makes the run interesting -- it emits real
HDF5, so the connector has something to intercept.

```bash
git clone --depth 1 --branch stable https://github.com/lammps/lammps.git ~/src/lammps
cd ~/src/lammps
cmake -S cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=OFF \
  -DPKG_H5MD=ON \
  -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
  -DCMAKE_CXX_STANDARD=17
cmake --build build -j
```

`BUILD_MPI=OFF` because H5MD prefers serial HDF5 and refuses to pair a serial
LAMMPS with a parallel HDF5. Set `Kokkos_ARCH_*` for your GPU.

Point the scripts at it with `LMP=/path/to/lmp` if it is not
`~/src/lammps/build/lmp`.

## Running

Two phases, two processes. The reader shares exactly one thing with the writer:
the store directory.

```bash
./lmp_write.sh --box 80 --steps 300 --gap 50 --chunk 4194304 --store /tmp/lmp_store
./lmp_read.sh  --store /tmp/lmp_store --chunk 4194304
```

`--box 80` is 2,048,000 atoms (the size upstream's own `in.melt_gpuc` uses);
`--steps 300 --gap 50` gives 7 frames, about 1 GB across the three fields.

Selection mode is chosen on the writer:

| mode | flag | what it does |
|------|------|--------------|
| inference | *(default)* | predict, compress, store; nothing measured back |
| learning | `--learn` | SGD on each chunk's real outcome |
| exploration | `--explore 31 --threshold 0` | also compress alternatives and adopt a winner |
| best | `--best` | exhaustive, ratio-only ranking; roughly 32x slower |

Cost-model weights are overridable for experiments:
`CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1`
gives a ratio-only objective. Per-chunk decisions land in a CSV via
`CLIO_NEUROPRESS_SELECTION_LOG`, and every explored candidate via
`CLIO_NEUROPRESS_EXPLORE_LOG`.

## Reading the result

`lmp_read.sh` asserts three things, and the second and third are the ones that
matter:

1. the bytes match a native read of the same file,
2. chunks were fetched from the tier and inverted by a codec,
3. no read fell back to the native file.

**Point 1 alone proves nothing.** The native HDF5 file is authoritative and
holds the same data uncompressed, so a read that misses the tier entirely still
returns correct bytes. During development a series of runs "passed" that way
while the compressed tier was completely unreachable. If the connector was
built without `-DCLIO_NEUROPRESS_PATH_TRACE=ON` there is no evidence either
way, and the script reports INCONCLUSIVE rather than PASS.

## What to expect

Melted Lennard-Jones coordinates are close to incompressible, and the headline
number is small on purpose -- this example demonstrates the path, not a
compression result:

- ~1.07x lossless on 1 GB, inference mode, 4 MiB chunks
- roughly two thirds of chunks compress; the rest the codec cannot shrink at
  all, and Clio stores them raw

The chunks that do compress are the opening frames, where the atoms are still
on their initial FCC lattice. Velocities are randomised at step 0 and never
compress at any point in the run.

## Notes

- h5md writes **float64**, and frames (`natoms*3*8` bytes) do not divide the
  chunk size, so chunk boundaries fall inside frames. Chunks are assembled
  across successive `H5Dwrite` calls -- this example is the reason that path
  exists.
- `CLIO_WITH_RUNTIME=1` is load-bearing: a compressor task sent to a runtime in
  another process arrives default-constructed, so the compressor has to live in
  whichever process is calling it.
- `CLIO_RESTART=1` on the reader replays the metadata log. Without it the
  tier's bytes are on disk but nothing remembers what they are, and every read
  misses.
- Nothing here is built by CMake; the scripts drive a prebuilt LAMMPS.
