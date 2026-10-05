#!/usr/bin/env python3
"""Workloads with heterogeneity INSIDE one application, as native arrays.

    gen_intra_workloads.py {omics,mag,igbh,tpch,era5} [--out DIR]

Each writes one directory of raw array files named NNN_<array>.<type> (type
in u8/i32/i64/f32/...), NNN giving the order the application produces them;
stage_intra_workloads.py cuts them into one 4 MiB chunk stream.

  omics  omics-pbmc: 10x Genomics PBMC 10k (HDF5) and PBMC 68k (MatrixMarket):
         raw counts as CSC by cell (data i32, indices/indptr i64, as 10x
         writes them), barcodes and gene ids (bytes), then a standard scanpy
         analysis: QC metrics, normalized + log1p values, highly variable gene
         mask, PCA scores and loadings, kNN graph (CSR), UMAP, Leiden clusters
  mag    gnn-mag: OGB ogbn-mag (Microsoft Academic Graph): every relation's
         edge_index (int64 [2, E], as OGB stores it), paper features (f32
         [N, 128]), paper years and venue labels (int64)
  igbh   gnn-igbh: IGBH-small (Illinois Graph Benchmark, heterogeneous, built
         from the Microsoft Academic Graph; download
         igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/
         igb_heterogeneous_small.tar.gz into RAW/igbh): every relation's
         edge_index (int64 [E, 2], as IGB stores it), node features (f32
         [N, 1024]) of papers, fields of study, institutes, journals and
         conferences, then the 19- and 2,983-class paper labels (f32, as
         stored). Author features (7.9 GB) are left out to keep the workload
         at ~5.4 GB; like gnn-mag, the main features are the papers'.
  tpch   analytics-tpch: TPC-H SF 1 from DuckDB, every column of every table
         as DuckDB stores it: integers, DECIMAL(15,2) as int64 cents, DATE as
         int32 days, strings as offsets (i64) + bytes
  era5   climate-era5: ARCO-ERA5 (public, anonymous) 0.25 degree single-level
         fields, 10 variables x 48 consecutive hours, one 721 x 1440 f32
         field per file, hour by hour
"""
import argparse
import gzip
import os

import numpy as np

RAW = "/mnt/nvme0/np-data-new/raw"
OUT = "/mnt/nvme0/np-data-new"


class Writer:
    """Writes numbered native array files into one workload directory."""

    def __init__(self, path):
        self.path = path
        self.n = 0
        os.makedirs(path, exist_ok=True)

    def put(self, name, arr):
        """Write one array as NNN_<name>.<type>."""
        a = np.ascontiguousarray(arr)
        kind = {"b": "u8", "S": "u8", "U": "u8"}.get(a.dtype.kind)
        t = kind or {np.dtype(np.float16): "f16", np.dtype(np.float32): "f32",
                     np.dtype(np.float64): "f64", np.dtype(np.int8): "i8",
                     np.dtype(np.uint8): "u8", np.dtype(np.int16): "i16",
                     np.dtype(np.uint16): "u16", np.dtype(np.int32): "i32",
                     np.dtype(np.uint32): "u32", np.dtype(np.int64): "i64",
                     np.dtype(np.uint64): "u64"}[a.dtype]
        if a.dtype.kind == "U":
            a = np.char.encode(a, "utf-8")
        p = os.path.join(self.path, f"{self.n:03d}_{name}.{t}")
        a.tofile(p)
        self.n += 1
        print(f"  {os.path.basename(p):46s} {a.nbytes / 2**20:8.1f} MiB")


def omics(out):
    """omics-pbmc: raw counts + identifiers + scanpy analysis outputs."""
    import h5py
    import scanpy as sc
    import scipy.io
    import scipy.sparse as sp
    w = Writer(out)
    f = h5py.File(os.path.join(RAW, "omics", "pbmc_10k_v3.h5"))
    m = f["matrix"]
    n_genes, n_cells = (int(v) for v in m["shape"][:])
    x10 = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                        shape=(n_genes, n_cells))
    ids10 = m["features"]["id"][:]
    mex = os.path.join(RAW, "omics", "filtered_matrices_mex", "hg19")
    x68 = scipy.io.mmread(os.path.join(mex, "matrix.mtx")).tocsc().astype(np.int32)
    ids68 = np.loadtxt(os.path.join(mex, "genes.tsv"), dtype=str, usecols=0)
    bc68 = np.loadtxt(os.path.join(mex, "barcodes.tsv"), dtype=str)
    for tag, x, bc, ids in (("pbmc10k", x10, m["barcodes"][:], ids10),
                            ("pbmc68k", x68, bc68, ids68)):
        print(tag, x.shape, x.nnz)
        w.put(f"{tag}_counts_data", x.data.astype(np.int32))
        w.put(f"{tag}_counts_indices", x.indices.astype(np.int64))
        w.put(f"{tag}_counts_indptr", x.indptr.astype(np.int64))
        w.put(f"{tag}_barcodes", np.asarray(bc).astype("S"))
        w.put(f"{tag}_gene_ids", np.asarray(ids).astype("S"))
        analysis(w, tag, x.T.tocsr().astype(np.float32), np.asarray(ids).astype(str))


def analysis(w, tag, x, gene_ids):
    """A standard scanpy pass over cells x genes counts; writes its outputs."""
    import anndata
    import scanpy as sc
    ad = anndata.AnnData(x)
    ad.var_names = [str(g) for g in gene_ids]
    ad.var_names_make_unique()
    sc.pp.calculate_qc_metrics(ad, inplace=True)
    w.put(f"{tag}_qc_n_genes", ad.obs["n_genes_by_counts"].to_numpy().astype(np.int32))
    w.put(f"{tag}_qc_total_counts", ad.obs["total_counts"].to_numpy().astype(np.float32))
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    w.put(f"{tag}_lognorm_data", ad.X.data.astype(np.float32))
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    w.put(f"{tag}_hvg_mask", ad.var["highly_variable"].to_numpy().astype(np.uint8))
    sc.pp.pca(ad, n_comps=50, mask_var="highly_variable")
    w.put(f"{tag}_pca_scores", ad.obsm["X_pca"].astype(np.float32))
    w.put(f"{tag}_pca_loadings", ad.varm["PCs"].astype(np.float32))
    sc.pp.neighbors(ad, n_neighbors=15)
    g = ad.obsp["connectivities"].tocsr()
    w.put(f"{tag}_knn_data", g.data.astype(np.float32))
    w.put(f"{tag}_knn_indices", g.indices.astype(np.int32))
    w.put(f"{tag}_knn_indptr", g.indptr.astype(np.int32))
    sc.tl.umap(ad)
    w.put(f"{tag}_umap", ad.obsm["X_umap"].astype(np.float32))
    sc.tl.leiden(ad, flavor="igraph", n_iterations=2)
    w.put(f"{tag}_leiden", ad.obs["leiden"].cat.codes.to_numpy().astype(np.int32))


def mag(out):
    """gnn-mag: OGB ogbn-mag native arrays."""
    import pandas as pd
    w = Writer(out)
    base = os.path.join(RAW, "mag", "mag", "raw")
    rel = os.path.join(base, "relations")
    for r in sorted(os.listdir(rel)):
        e = pd.read_csv(os.path.join(rel, r, "edge.csv.gz"), header=None).to_numpy()
        w.put(f"edge_index_{r.replace('___', '-')}", e.T.astype(np.int64))
    feat = pd.read_csv(os.path.join(base, "node-feat", "paper", "node-feat.csv.gz"),
                       header=None, dtype=np.float32).to_numpy()
    w.put("paper_features", feat)
    for name, p in (("paper_year", "node-feat/paper/node_year.csv.gz"),
                    ("paper_label", "node-label/paper/node-label.csv.gz")):
        v = pd.read_csv(os.path.join(base, p), header=None).to_numpy().ravel()
        w.put(name, v.astype(np.int64))


def igbh(out):
    """gnn-igbh: IGBH-small native arrays (author features left out)."""
    w = Writer(out)
    base = os.path.join(RAW, "igbh", "small", "processed")
    for r in sorted(d for d in os.listdir(base) if "__" in d):
        w.put(f"edge_index_{r.replace('__', '-')}",
              np.load(os.path.join(base, r, "edge_index.npy"), mmap_mode="r"))
    for node in ("paper", "fos", "institute", "journal", "conference"):
        w.put(f"{node}_features", np.load(os.path.join(base, node, "node_feat.npy"),
                                          mmap_mode="r"))
    for k in ("19", "2K"):
        w.put(f"paper_label_{k}", np.load(os.path.join(base, "paper", f"node_label_{k}.npy")))


def tpch(out):
    """analytics-tpch: TPC-H SF 1 columns as DuckDB stores them."""
    import duckdb
    import pyarrow as pa
    w = Writer(out)
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf = 1)")
    for table in ("lineitem", "orders", "partsupp", "part", "customer", "supplier",
                  "nation", "region"):
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        for name, ctype, *_ in cols:
            expr = (f"CAST(round({name} * 100) AS BIGINT)" if ctype.startswith("DECIMAL")
                    else f"CAST({name} AS INTEGER)" if ctype == "INTEGER"
                    else f"CAST({name} - DATE '1970-01-01' AS INTEGER)" if ctype == "DATE"
                    else name)
            col = con.execute(f"SELECT {expr} FROM {table}").fetch_arrow_table().column(0)
            col = col.combine_chunks()
            if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                col = col.cast(pa.large_string())
                offs = np.frombuffer(col.buffers()[1], dtype=np.int64)[: len(col) + 1]
                data = np.frombuffer(col.buffers()[2], dtype=np.uint8)[: offs[-1]]
                w.put(f"{table}_{name}_offsets", offs)
                w.put(f"{table}_{name}_bytes", data)
            else:
                w.put(f"{table}_{name}", col.to_numpy(zero_copy_only=False))


def era5(out):
    """climate-era5: ARCO-ERA5 single-level fields, hour by hour."""
    import xarray as xr
    w = Writer(out)
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None, storage_options={"token": "anon"}, consolidated=True)
    names = ["2m_temperature", "surface_pressure", "mean_sea_level_pressure",
             "total_precipitation", "total_cloud_cover", "10m_u_component_of_wind",
             "10m_v_component_of_wind", "2m_dewpoint_temperature", "snow_depth",
             "sea_surface_temperature"]
    names = [v for v in names if v in ds]
    for t in ds.time.sel(time=slice("2020-07-01T00", "2020-07-02T23")).values:
        for v in names:
            field = ds[v].sel(time=t).values.astype(np.float32)
            w.put(f"{str(t)[:13].replace(':', '')}_{v}", field)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workload", choices=["omics", "mag", "igbh", "tpch", "era5"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = {"omics": "omics-pbmc", "mag": "gnn-mag", "igbh": "gnn-igbh", "tpch": "analytics-tpch",
            "era5": "climate-era5"}
    out = a.out or os.path.join(OUT, dirs[a.workload])
    {"omics": omics, "mag": mag, "igbh": igbh, "tpch": tpch, "era5": era5}[a.workload](out)
    print("wrote", out)


if __name__ == "__main__":
    main()
