/* Isolated C test for clio VOL object/link ITERATION, using the modern
 * VOL-aware APIs (H5Ovisit3 / H5Literate2). This is the accurate way to test
 * VOL iteration: h5py cannot (it calls the deprecated *_by_name1 variants that
 * HDF5 restricts to the native VOL), so the h5py compat suite disables its
 * iteration case and defers to this test instead.
 *
 * Self-contained: builds a small grouped file through whatever VOL is active,
 * then visits it and checks the traversal matches the expected object/link set.
 * Run with HDF5_VOL_CONNECTOR=clio (and a running clio_run). Exit 0 = pass.
 *
 * Build: h5cc -o vol_c_iteration_test vol_c_iteration_test.c
 */
#include <hdf5.h>
#include <stdio.h>
#include <string.h>

#define MAXN 64
static char g_ovisit[MAXN][256];
static int  g_no;
static char g_liter[MAXN][256];
static int  g_nl;

static herr_t ovisit_cb(hid_t obj, const char *name, const H5O_info2_t *info,
                        void *data) {
  (void)obj; (void)info; (void)data;
  if (g_no < MAXN) snprintf(g_ovisit[g_no++], sizeof(g_ovisit[0]), "%s", name);
  return 0;
}
static herr_t liter_cb(hid_t g, const char *name, const H5L_info2_t *info,
                       void *data) {
  (void)g; (void)info; (void)data;
  if (g_nl < MAXN) snprintf(g_liter[g_nl++], sizeof(g_liter[0]), "%s", name);
  return 0;
}
static int seen(char arr[][256], int n, const char *s) {
  for (int i = 0; i < n; ++i) if (!strcmp(arr[i], s)) return 1;
  return 0;
}

int main(void) {
  const char *path = "/tmp/volc_iter.h5";
  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) { fprintf(stderr, "Fcreate failed\n"); return 2; }
  hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(lcpl, 1);   /* create "a" on the way to "a/b" */
  hid_t g = H5Gcreate2(f, "a/b", lcpl, H5P_DEFAULT, H5P_DEFAULT);
  H5Pclose(lcpl);
  hsize_t dims[1] = {4};
  hid_t s = H5Screate_simple(1, dims, NULL);
  hid_t d = H5Dcreate2(g, "leaf", H5T_NATIVE_INT, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(d); H5Sclose(s); H5Gclose(g);

  herr_t r1 = H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_NATIVE, ovisit_cb, NULL, H5O_INFO_BASIC);
  herr_t r2 = H5Literate2(f, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, liter_cb, NULL);
  H5Fclose(f);

  int ok = (r1 >= 0) && (r2 >= 0)
        && seen(g_ovisit, g_no, ".") && seen(g_ovisit, g_no, "a")
        && seen(g_ovisit, g_no, "a/b") && seen(g_ovisit, g_no, "a/b/leaf")
        && seen(g_liter, g_nl, "a");
  printf("H5Ovisit3 rc=%d (%d objs), H5Literate2 rc=%d (%d links): %s\n",
         (int)r1, g_no, (int)r2, g_nl, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
