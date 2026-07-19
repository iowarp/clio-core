/* Isolated C test for clio VOL Safe-mode flush (H5Fflush durability barrier).
 *
 * The connector caches whole-dataset writes as ASYNC CTE puts, drained on
 * dataset close. Safe mode additionally drains them on H5Fflush, so a flush is a
 * real durability barrier for the cache path: after it returns, no async put is
 * still in flight, and the dataset is readable through the VOL WITHOUT first
 * closing it. This test writes, flushes, then reads back the still-open dataset
 * (exercising the read-through cache after the flush drain), then reopens the
 * file and reads again. All values must match.
 *
 * Run with HDF5_VOL_CONNECTOR=clio (and a running clio_run). Exit 0 = pass.
 * Build: h5cc -o vol_c_safeflush_test vol_c_safeflush_test.c
 */
#include <hdf5.h>
#include <stdio.h>

#define N 1024

static int check(const int *got, const char *what) {
  for (int i = 0; i < N; ++i) {
    if (got[i] != i * 3 + 1) {
      fprintf(stderr, "%s: mismatch at %d: got %d want %d\n", what, i, got[i],
              i * 3 + 1);
      return 0;
    }
  }
  return 1;
}

int main(void) {
  const char *path = "/tmp/volc_safeflush.h5";
  int in[N], out[N];
  for (int i = 0; i < N; ++i) in[i] = i * 3 + 1;

  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) { fprintf(stderr, "Fcreate failed\n"); return 2; }
  hsize_t dims[1] = {N};
  hid_t s = H5Screate_simple(1, dims, NULL);
  hid_t d = H5Dcreate2(f, "vals", H5T_NATIVE_INT, s, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT);
  if (d < 0) { fprintf(stderr, "Dcreate failed\n"); return 2; }
  if (H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, in) < 0) {
    fprintf(stderr, "Dwrite failed\n"); return 2;
  }

  /* The durability barrier under test: must drain pending CTE puts. */
  herr_t fl = H5Fflush(f, H5F_SCOPE_GLOBAL);

  /* Read back the STILL-OPEN dataset (no close between write and read). */
  for (int i = 0; i < N; ++i) out[i] = -1;
  herr_t r1 = H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
  int ok_open = (r1 >= 0) && check(out, "read-after-flush(open)");

  H5Dclose(d);
  H5Fclose(f);

  /* Reopen and read again (post close/reopen). */
  hid_t f2 = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t d2 = H5Dopen2(f2, "vals", H5P_DEFAULT);
  for (int i = 0; i < N; ++i) out[i] = -1;
  herr_t r2 = H5Dread(d2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
  int ok_reopen = (r2 >= 0) && check(out, "read-after-reopen");
  H5Dclose(d2);
  H5Fclose(f2);
  H5Sclose(s);

  int ok = (fl >= 0) && ok_open && ok_reopen;
  printf("H5Fflush rc=%d, read-open=%s, read-reopen=%s: %s\n", (int)fl,
         ok_open ? "OK" : "BAD", ok_reopen ? "OK" : "BAD", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
