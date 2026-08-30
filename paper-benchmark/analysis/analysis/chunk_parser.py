#!/usr/bin/env python3
"""Pull scientific metadata out of a blob key, without ever depending on it.

A blob key is a runtime path, and each workload's writer builds it its own
way:

    Nyx / AMReX     plt00003/fab0000_comp04_rho_E/chunk_0
    WarpX / openPMD diag1/0000000100/fields/E/x/chunk_2
    VPIC            step_0040/field/ex/chunk_7
    LAMMPS          dump.0000200/xu/chunk_1

So this is a set of tolerant regexes over a normalised path, plus a fallback
that still recovers a timestep and a chunk id from almost anything. Nothing
here may raise: an unparsed blob keeps its original key and gets NaN metadata,
and every downstream analysis is written to survive that (a field-level
grouping simply has one "unknown" group).

The one field that MUST be right is `chunk_uid`, because every group-aware
split and every paired comparison keys on it. It is the raw blob string, not
anything parsed -- so a parser bug can lose a field label but can never leak
a chunk across a train/test boundary.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

#: Trailing "/chunk_<n>" (or "chunk<n>", "/c12"), the unit Clio compresses.
_CHUNK = re.compile(r"(?:^|[/_.])chunk[_-]?(\d+)$|(?:^|[/_.])c(\d+)$", re.I)

#: AMReX plotfile directory: plt00000, plt_000100.
_PLT = re.compile(r"^plt[_-]?(\d+)$", re.I)
#: AMReX FAB + component + name: fab0000_comp04_rho_E
_FAB = re.compile(r"^fab[_-]?(\d+)_comp[_-]?(\d+)_(.+)$", re.I)
#: A bare zero-padded step directory, as openPMD writes: 0000000100
_STEPDIR = re.compile(r"^(\d{4,})$")
#: step_0040 / t0040 / iter_100 / dump.0000200 / step-40
_STEPISH = re.compile(
    r"^(?:step|t|iter|iteration|dump|frame|out|snapshot)[_.\-]?(\d+)$", re.I)

#: openPMD-ish mesh path: .../fields/E/x  or .../meshes/rho
_MESHY = {"fields", "meshes", "field", "mesh", "particles", "data"}

#: Component names that are a vector index rather than a field of their own.
_VECTOR_COMPONENT = {"x", "y", "z", "r", "t", "theta", "phi",
                     "0", "1", "2", "u", "v", "w"}


def _to_int(s: Optional[str]) -> float:
    try:
        return float(int(s))
    except (TypeError, ValueError):
        return float("nan")


def parse_blob(blob: str) -> Dict[str, object]:
    """Best-effort metadata for one blob key. Never raises.

    Returns keys: timestep, plotfile, fab, component, field, chunk_id,
    field_path, parse_ok.
    """
    out: Dict[str, object] = {
        "timestep": float("nan"), "plotfile": None, "fab": float("nan"),
        "component": float("nan"), "field": None, "chunk_id": float("nan"),
        "field_path": None, "parse_ok": False,
    }
    if not isinstance(blob, str) or not blob:
        return out

    parts = [p for p in re.split(r"[/\\]", blob.strip()) if p]
    if not parts:
        return out

    # ---- trailing chunk id ----------------------------------------------
    m = _CHUNK.search(parts[-1])
    if m:
        out["chunk_id"] = _to_int(m.group(1) or m.group(2))
        parts = parts[:-1]
    if not parts:
        out["parse_ok"] = not np.isnan(out["chunk_id"])  # type: ignore[arg-type]
        return out

    # ---- timestep, from whichever leading component looks like one -------
    for i, p in enumerate(parts):
        mp = _PLT.match(p)
        if mp:
            out["timestep"] = _to_int(mp.group(1))
            out["plotfile"] = p
            break
        ms = _STEPISH.match(p) or _STEPDIR.match(p)
        if ms:
            out["timestep"] = _to_int(ms.group(1))
            out["plotfile"] = p
            break

    # ---- field, from the remaining components ----------------------------
    rest = [p for p in parts if p != out["plotfile"]]
    field = None
    for p in rest:
        mf = _FAB.match(p)
        if mf:
            out["fab"] = _to_int(mf.group(1))
            out["component"] = _to_int(mf.group(2))
            field = mf.group(3)
            break
    if field is None and rest:
        # openPMD / VPIC shape: drop container directories, then take the
        # tail. A trailing single-letter component is a VECTOR INDEX, so
        # "E/x" becomes field "E_x" rather than the field "x", which would
        # collide with every other vector's x across the run.
        meaningful = [p for p in rest if p.lower() not in _MESHY]
        if not meaningful:
            meaningful = rest
        if len(meaningful) >= 2 and \
                meaningful[-1].lower() in _VECTOR_COMPONENT:
            field = f"{meaningful[-2]}_{meaningful[-1]}"
        else:
            field = meaningful[-1]
    out["field"] = field
    out["field_path"] = "/".join(rest) if rest else None
    out["parse_ok"] = field is not None or not np.isnan(
        out["timestep"])  # type: ignore[arg-type]
    return out


def parse_blobs(blobs: pd.Series) -> pd.DataFrame:
    """Vectorised over the UNIQUE keys -- a blob repeats once per candidate,
    so parsing per row would redo the same regex 32 times."""
    uniq = pd.Index(blobs.dropna().unique())
    recs = {b: parse_blob(b) for b in uniq}
    tbl = pd.DataFrame.from_dict(recs, orient="index")
    tbl.index.name = "blob"
    out = tbl.reindex(blobs.to_numpy())
    out.index = blobs.index
    return out


def attach_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed metadata columns plus `chunk_uid` (the raw blob)."""
    meta = parse_blobs(df["blob"])
    df = df.copy()
    for c in meta.columns:
        df[c] = meta[c].to_numpy()
    # The grouping key for every split and every pairing. Deliberately the raw
    # key: a parser miss must not be able to merge two distinct chunks.
    df["chunk_uid"] = df["blob"].astype("string")
    if "field" in df:
        df["field"] = df["field"].astype("object").where(
            df["field"].notna(), "unknown")
    return df


def parse_quality(path_report: dict) -> dict:  # pragma: no cover - helper
    return path_report
