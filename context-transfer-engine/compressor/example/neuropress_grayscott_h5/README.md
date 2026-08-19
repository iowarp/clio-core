# Gray-Scott -> HDF5 -> Clio -> NeuroPress (bare metal)

The first end-to-end run of the NeuroPress integration on a real workload:
an application that contains **no Clio code and links no Clio library**, yet
gets per-chunk NeuroPress selection and GPU compression.

`ldd` on the built binary shows zero Clio libraries. Clio enters the process
entirely through the environment:

| variable | what it does |
|---|---|
| `HDF5_VOL_CONNECTOR=clio` + `HDF5_PLUGIN_PATH` | HDF5 dlopens `libclio_hdf5_vol.so` into the app |
| `CLIO_WITH_RUNTIME=1` | the connector's `CLIO_INIT` hosts the runtime **in this process** |
| `CLIO_VOL_COMPRESSOR_POOL=512.0` | the connector builds a compressor client and calls `DynamicSchedule` |

`CLIO_WITH_RUNTIME=1` is the load-bearing one. Against a SEPARATE `clio_run`
daemon this does not work: `DynamicScheduleTask` arrives at the compressor
fully default-constructed (`size=0`, empty blob name, null `ShmPtr`), so the
write is rejected with "Invalid chunk data". `PutBlobTask` over the same
transport is unaffected. Every NeuroPress example and test in this tree
embeds the runtime, so nothing covered that path and the gap went unnoticed.

## Build

    nvcc -O2 -std=c++17 \
      -I../../generator/grayscott -I/usr/local/include \
      -o neuropress_grayscott_h5 neuropress_grayscott_h5.cu \
      ../../generator/grayscott/grayscott_sim.cu \
      -L/usr/local/lib -lhdf5

## Run

    mkdir -p /tmp/np_plugin
    ln -sf $CLIO_BUILD/bin/libclio_hdf5_vol.so /tmp/np_plugin/

    env CLIO_SERVER_CONF=$PWD/compose.yaml \
        CLIO_WITH_RUNTIME=1 \
        HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH=/tmp/np_plugin \
        CLIO_VOL_COMPRESSOR_POOL=512.0 CTP_LOG_LEVEL=debug \
        ./neuropress_grayscott_h5 /tmp/gs.h5 128 100 25

Edit `neuropress_model_path` in compose.yaml to point at this tree's
`context-transport-primitives/src/compress/model/weights`.

## Measured (A100, 128^3, 4 snapshots, 32 chunks)

    1048576 ->    732 B   ratio 1432x     near-uniform region
    1048576 ->  11296 B   ratio   92.8x
    1048576 -> 413592 B   ratio    2.5x   active pattern front

The spread is the result worth having: the model is choosing per chunk
against real reaction-diffusion data, not compressing a uniform field.

Uses the settings.json regime (Du=0.2 Dv=0.1 F=0.02 k=0.048 dt=1.0), NOT
the shipped defaults -- at those, V goes extinct by ~1000 steps and the
field becomes all zeros. See generator/grayscott/grayscott_sim.h.
