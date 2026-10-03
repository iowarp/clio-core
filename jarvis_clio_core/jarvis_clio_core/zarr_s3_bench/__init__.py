"""
Zarr S3 Benchmark Application Package — drives scripts/zarr_s3_read.py or
scripts/zarr_s3_write.py depending on `mode`, moving a Zarr v3 store to or from
S3 via zarr-python + s3fs, and scrapes the results into the sweep's results.csv.
"""
