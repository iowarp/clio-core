"""
CLIO S3 Benchmark Application Package — drives clio_s3_read_bench or
clio_s3_write_bench depending on `mode`, timing CLIO's S3 read path (CAE
assimilator) or write path (the kS3 bdev tier) against real S3, and scrapes the
results into the sweep's results.csv.
"""
