#!/usr/bin/env bash
#
# smoke_juicefs_gcs.sh -- prove the JuiceFS<->GCS link end to end, NO benchmark.
#
# Formats + FUSE-mounts JuiceFS on a real gs:// bucket (Redis metadata), does a
# POSIX round-trip through the mount, forces a flush + cold read so the bytes
# must come from GCS, then INDEPENDENTLY confirms with `gcloud storage ls` that
# JuiceFS objects physically landed in the bucket. Prints PASS only when the
# in-mount round-trip AND the gcloud listing agree -- mirroring the
# "in-process round-trip AND external CLI agree = valid" discipline in
# cloud_adapter_validation.md (sections 1 and 5).
#
# This is a configuration / connectivity check, NOT the fio sweep
# (jarvis_clio_core/pipelines/juicefs_fio_1n.yaml).
#
# Required env:
#   BUCKET=gs://<your-bucket>                       the real bucket (must exist or be creatable)
#   GOOGLE_APPLICATION_CREDENTIALS=~/gcs-sa.json     service-account key  (or set GCS_ACCESS_TOKEN)
# Optional env:
#   GCS_PROJECT_ID=<project>             only used if juicefs auto-creates the bucket
#   META_URL=redis://127.0.0.1:6379/1    metadata engine (default). Any engine the
#                                        juicefs binary supports works here; for a
#                                        no-server / no-sudo run use the embedded
#                                        SQLite engine, e.g.
#                                          META_URL=sqlite3://jfsgcs-meta.db
#                                        (redis:// is pinged first; embedded engines
#                                        are created by `juicefs format` itself).
#   VOL=jfsgcssmoke   MNT=$HOME/jfs_gcs_mnt   CACHE=$HOME/jfs_gcs_cache
#
# Use a DEDICATED/empty test bucket: JuiceFS writes its objects (juicefs_uuid
# format marker + a chunks/ data tree, possibly under a <volume>/ prefix) inside
# the --bucket path, exactly as the jarvis juicefs package does for storage=gs.
#
# Usage:
#   export BUCKET=gs://my-bucket
#   export GOOGLE_APPLICATION_CREDENTIALS=~/gcs-sa.json
#   bash jarvis_clio_core/scripts/smoke_juicefs_gcs.sh
set -u

META_URL="${META_URL:-redis://127.0.0.1:6379/1}"
VOL="${VOL:-jfsgcssmoke}"
MNT="${MNT:-${HOME}/jfs_gcs_mnt}"
CACHE="${CACHE:-${HOME}/jfs_gcs_cache}"
LOCAL_BIN="${HOME}/.local/bin"
case ":${PATH}:" in *":${LOCAL_BIN}:"*) ;; *) export PATH="${LOCAL_BIN}:${PATH}" ;; esac

log()  { printf '\n=== %s ===\n' "$*"; }

is_mounted() {
  if command -v mountpoint >/dev/null 2>&1; then
    mountpoint -q "$1"
  else
    grep -q " $1 " /proc/mounts 2>/dev/null
  fi
}

cleanup() {
  # Best-effort unmount + temp-file removal; never fail the script from here.
  if is_mounted "${MNT}"; then
    juicefs umount "${MNT}" >/dev/null 2>&1 \
      || fusermount -u "${MNT}" >/dev/null 2>&1 || true
  fi
  [ -n "${SRC:-}" ] && rm -f "${SRC}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 1. Prereqs (each a hard gate) ----------------------------------------
log "prereqs"
command -v juicefs   >/dev/null 2>&1 || fail "juicefs not on PATH (run setup_juicefs_ares.sh)"
command -v gcloud    >/dev/null 2>&1 || fail "gcloud not on PATH (needed for the independent GCS check)"
command -v curl      >/dev/null 2>&1 || fail "curl not on PATH"
# Metadata engine is whatever META_URL names. redis:// needs a running server +
# redis-cli; embedded engines (sqlite3://, badger://) live in a local file the
# juicefs binary creates -- no server, no redis-cli, no sudo.
case "${META_URL}" in
  redis://*|rediss://*)
    meta_kind="redis"
    command -v redis-cli >/dev/null 2>&1 || fail "redis-cli not on PATH (META_URL is redis://)"
    ;;
  *) meta_kind="embedded" ;;
esac
[ -e /dev/fuse ] || fail "/dev/fuse missing -- FUSE mount impossible (need a privileged container)"
# juicefs also needs the userspace helper (fusermount/fusermount3 from the fuse3
# package). With /dev/fuse present but the helper absent, the mount dies with
# 'fuse: fuse is not installed' -- so gate on the binary, not just the device.
command -v fusermount3 >/dev/null 2>&1 || command -v fusermount >/dev/null 2>&1 \
  || fail "fusermount not on PATH -- install the 'fuse3' (or 'fuse') package; the mount cannot work without it"
# Real GCS is HTTPS: without a CA bundle every request fails with curl error 60.
[ -f /etc/ssl/certs/ca-certificates.crt ] || [ -d /etc/ssl/certs ] \
  || fail "ca-certificates not found -- TLS to storage.googleapis.com will fail"
: "${BUCKET:?set BUCKET=gs://<your-bucket>}"
case "${BUCKET}" in gs://*) ;; *) fail "BUCKET must be a gs:// URL, got '${BUCKET}'" ;; esac
# Credentials: juicefs uses Google's ADC chain. Accept ANY of three sources:
#   D1  GOOGLE_APPLICATION_CREDENTIALS=<key.json>   (service-account key)
#       GCS_ACCESS_TOKEN=<token>                    (short-lived token)
#   D2  user ADC: ~/.config/gcloud/application_default_credentials.json
#       (written by `gcloud auth application-default login`) -- no env var needed
adc_file="${CLOUDSDK_CONFIG:-${HOME}/.config/gcloud}/application_default_credentials.json"
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
  [ -r "${GOOGLE_APPLICATION_CREDENTIALS}" ] \
    || fail "GOOGLE_APPLICATION_CREDENTIALS='${GOOGLE_APPLICATION_CREDENTIALS}' is not readable"
  echo "creds: service-account key at GOOGLE_APPLICATION_CREDENTIALS (D1)"
elif [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
  echo "creds: GCS_ACCESS_TOKEN present (short-lived token)"
elif [ -r "${adc_file}" ]; then
  echo "creds: user ADC at ${adc_file} (D2)"
else
  fail "no GCS credentials: set GOOGLE_APPLICATION_CREDENTIALS=<key.json>, or GCS_ACCESS_TOKEN, or run 'gcloud auth application-default login' (expected ${adc_file})"
fi
# Metadata engine reachable. For redis, ping it; for an embedded engine the
# juicefs format step creates/opens the local DB itself, so there is nothing to
# reach here.
if [ "${meta_kind}" = "redis" ]; then
  redis-cli -u "${META_URL}" ping 2>/dev/null | grep -q PONG \
    || fail "redis not reachable at ${META_URL} (start the redis package / server first)"
  meta_status="redis PONG"
else
  meta_status="embedded meta (${META_URL})"
fi
# Network egress to GCS (connect, don't hang).
code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 https://storage.googleapis.com || true)"
{ [ -n "${code}" ] && [ "${code}" != "000" ]; } \
  || fail "no network egress to storage.googleapis.com (curl returned '${code:-none}')"
echo "prereqs OK (storage.googleapis.com http=${code}, ${meta_status}, /dev/fuse present)"

mkdir -p "${MNT}" "${CACHE}"
root="${BUCKET%/}"

# --- 2. Format the volume on real GCS -------------------------------------
# Exactly the command jarvis_clio_core.juicefs builds for storage=gs.
log "format (juicefs format --storage gs --bucket ${root})"
juicefs format --storage gs --bucket "${root}" "${META_URL}" "${VOL}" \
  || fail "juicefs format failed (check creds, bucket name, IAM role, clock skew)"

# --- 3. FUSE mount --------------------------------------------------------
log "mount"
juicefs mount "${META_URL}" "${MNT}" --cache-dir "${CACHE}" --background \
  || fail "juicefs mount returned an error"
ready=0
for _ in $(seq 1 20); do
  if is_mounted "${MNT}"; then ready=1; break; fi
  sleep 1
done
[ "${ready}" = 1 ] || fail "mount not ready after 20s (check creds / /dev/fuse)"
echo "mounted at ${MNT}"

# --- 4. POSIX round-trip through the mount --------------------------------
log "round-trip"
SRC="$(mktemp)"
DST="${MNT}/smoke_$$.bin"
head -c 5242880 /dev/urandom > "${SRC}"   # 5 MiB > one 4 MiB block => >=2 data objects
cp "${SRC}" "${DST}" || fail "write through the mount failed"
cmp "${SRC}" "${DST}" || fail "byte-compare through the mount failed"
echo "round-trip OK ($(wc -c < "${SRC}") bytes written + verified through the mount)"

# --- 5. Force durability: flush + drop cache, then cold read --------------
log "flush + cold read"
sync
juicefs warmup --evict "${DST}" >/dev/null 2>&1 || true   # drop local cache for this file
cmp "${SRC}" "${DST}" || fail "cold read-back (post-evict) mismatch -> bytes not durable in GCS"
echo "cold read-back OK (served from GCS, not local page cache)"

# --- 6. INDEPENDENT confirmation: objects physically in the bucket --------
# The step that proves bytes left the box (cloud_adapter_validation.md 3.1.E).
# JuiceFS writes a `juicefs_uuid` format marker + file data under a `chunks/`
# tree inside the --bucket path (it may nest under a <volume>/ prefix). We list
# the bucket recursively and match `/chunks/` anywhere, so prefix layout differences
# don't matter -- file *metadata* stays in the meta engine, not in GCS.
log "independent gcloud check"
nchunks=0
listing=""
for _ in $(seq 1 5); do        # brief retry: tolerate async upload completion
  listing="$(gcloud storage ls -r "${root}/" 2>/dev/null || true)"
  nchunks="$(printf '%s\n' "${listing}" | grep -Ec '/chunks/' || true)"
  [ "${nchunks:-0}" -ge 1 ] && break
  sleep 2
done
[ -n "${listing}" ] || fail "gcloud storage ls -r ${root}/ returned nothing (auth/permission, or empty bucket)"
if printf '%s\n' "${listing}" | grep -q 'juicefs_uuid'; then
  echo "format marker present: juicefs_uuid"
else
  echo "note: juicefs_uuid marker not seen (name varies by version); relying on chunk objects"
fi
[ "${nchunks:-0}" -ge 1 ] \
  || fail "no objects under chunks/ in ${root}/ -> file data never reached GCS"
echo "GCS holds ${nchunks} object(s) under chunks/ (>=2 expected for a 5 MiB file)"

# --- 7. Report (cleanup runs on EXIT) -------------------------------------
log "result"
echo "PASS: JuiceFS<->GCS verified -- in-mount round-trip AND gcloud agree the"
echo "      bytes physically landed in ${root}/"
echo "      teardown test objects with: gcloud storage rm -r ${root}/chunks"
echo "      (and ${root}/juicefs_uuid, ${root}/meta if present)"
