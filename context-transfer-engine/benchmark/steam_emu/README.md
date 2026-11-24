# Steam install emulator

`steam_emu.py` replays the file-system calls Steam makes when it installs a
game, so a file system can be measured against a real Windows workload
without Steam. It was built from a WinFsp request trace of Steam restoring
Age of Empires IV (48.6 GB, 15,423 files) onto clio-fs.

## What Steam does

Steam's content system runs an install in three phases:

| Phase | Threads | Per file |
|---|---|---|
| prealloc | 1 | open the staged name (fails); for the 8,271 large files: create, set EndOfFile, set Allocation, close |
| staging | ~9 in flight | open-if, set basic info (no change), write in 1 MiB chunks plus a short tail, query information, close |
| commit | 1 | open the installed name three times (each fails), open the staged file with DELETE, rename it into place, close; then remove `downloading/` |

Every call has Steam's access mask (`DesiredAccess` in the trace), so the
kernel adds what it adds for Steam: the target-directory open of a rename,
cache-manager I/O.

On the WinFsp volume each file Steam closes also gets two 64 KiB reads, at
the head and at `roundup(size, 4 KiB) - 64 KiB`. They are not Steam's own
calls (its handles have no read access) and files written by `copy`,
PowerShell or Python on the same volume get none, so they look like a
filter driver scanning Steam's files. `--scan-reads cached` reproduces them
through a second read-only handle. They are off by default: on NTFS that
extra open is itself scanned (about 7 ms a file), which Steam's own NTFS
install does not pay.

## Usage

```
python steam_emu.py ROOT [--scale S] [--files N] [--threads T]
                         [--phases prealloc,staging,commit]
                         [--scan-reads off|cached|unbuffered]
                         [--verify] [--json OUT]
```

`ROOT` stands for a Steam library (`SteamLibrary`); the game lands in
`ROOT/steamapps/common/Game`. `--scale` multiplies every file size (the
metadata pattern stays whole), `--files` keeps the first N files, `--verify`
reads every installed file back. Output: wall time per phase and per-call
latency (count, total, mean, p99).

## Making a manifest from another trace

1. Mount clio-fs with WinFsp tracing, each line stamped with milliseconds
   since mount: `clio_cte_fuse X: -f -d 2>&1 | python -u ts.py > trace.log`
   (`ts.py` prefixes `perf_counter` milliseconds).
2. Install the game with Steam. Note when Steam's
   `logs/content_log.txt` says `Preallocating`, `Staging`, `Committing` and
   `finished update`, and convert those times to trace milliseconds.
3. `python steam_trace.py trace.log game.json.gz --app-id ID
   --game-dir "Install Dir" --cuts T0,T1,T2,T3 --anonymize`

`steam_trace.py` also prints each phase's commonest per-file call sequences,
which is how the phases above were read off the AoE IV trace. Running it on
a trace of the emulator checks the emulator against Steam.

`aoe4_manifest.json.gz` holds the AoE IV install with every path component
hashed (extensions kept): `[path, size, preallocated]` per file, in the order
Steam first touched them.
