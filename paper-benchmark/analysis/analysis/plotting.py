#!/usr/bin/env python3
"""Figures. Every one of them is skipped rather than drawn empty when the
data behind it is too thin to read.

COLOR. Three categorical slots, in fixed order, never cycled -- blue, orange,
aqua. That is the number that validates for scatter and small multiples on
all pairs (CVD dE 9.2, normal-vision dE 24.0); a fourth slot puts yellow beside
orange and fails. There are up to eight codecs here, so per-codec views are
SMALL MULTIPLES -- one panel per codec, one hue inside each -- rather than
eight hues in one frame. Magnitude uses the single blue ramp; a signed
quantity (a shuffle or quantization delta, which can help or hurt) uses the
blue<->red diverging pair with a neutral gray midpoint, so the sign is visible
without reading the colorbar.

SCALE. Ratio spans 1x to 1400x within one sweep, so every ratio axis is
logarithmic. On a linear axis the forty chunks below 50x collapse onto the
baseline and the figure shows four points.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- tokens (see the dataviz reference palette) --------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Categorical slots, FIXED ORDER, never cycled: a ninth series folds into
#: "other" or small multiples rather than getting a generated hue.
#:
#: Validated as a 6-slot palette against this surface with the dataviz
#: validator (scripts/validate_palette.js, --mode light):
#:   lightness band   all inside L 0.43-0.77          PASS
#:   chroma floor     all >= 0.1                      PASS
#:   CVD separation   worst adjacent 9.1 protan, 5.8 tritan
#:   normal vision    worst adjacent 19.6             PASS
#:   contrast         aqua/yellow/magenta below 3:1 vs the surface
#:
#: The tritan 5.8 sits in the 6-8 floor band and the contrast check WARNs, so
#: BOTH are conditional passes: a figure using more than three of these owes
#: the reader a secondary identity channel. over_time() pays that with direct
#: end labels on every line, which is why they are drawn even when a legend is
#: present. Do not use four or more of these hues without one.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "clio_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "clio_div", ["#256abf", "#86b6ef", "#f0efec", "#ec835a", "#d03b3b"])

MIN_POINTS = 8          # below this a panel is not drawn
MIN_PANEL_POINTS = 5


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "medium", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.dpi": 130, "savefig.bbox": "tight",
    })


def _finish(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)


#: A feature axis goes log when the feature's positive values span more than
#: this many decades. Motivated by a real figure: VPIC's second_deriv runs
#: 1.4e-08 .. 0.42, and on a linear axis 93.3% of 3840 chunks pile into the
#: first tenth while a single field (rhof, 240 chunks at ~0.42) sits alone at
#: the right edge -- the plot reads as two or three distinct x values when
#: there are 3601. mad has the same shape. 2 decades is chosen to leave
#: genuinely narrow features (entropy, always ~7.3 bits) on a linear axis,
#: where log would just flatten them.
LOG_X_DECADES = 2.0


def _wants_log_x(v: "pd.Series") -> bool:
    """True when a log x-axis is the readable choice for these values.

    Requires strictly positive values -- a log axis silently DROPS zero and
    negative points, so a feature that has any would lose data rather than
    become legible, and stays linear.
    """
    x = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty or (x <= 0).any():
        return False
    lo, hi = float(x.min()), float(x.max())
    return lo > 0 and hi / lo >= 10.0 ** LOG_X_DECADES


class Figures:
    """Collects what was drawn and what was skipped, and why."""

    def __init__(self, outdir: str) -> None:
        self.dir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.made: List[str] = []
        self.skipped: List[Dict[str, str]] = []
        _style()

    def _save(self, fig, name: str) -> None:
        path = os.path.join(self.dir, name)
        fig.savefig(path)
        plt.close(fig)
        self.made.append(name)

    def _skip(self, name: str, why: str) -> None:
        self.skipped.append({"figure": name, "reason": why})

    # ---- property vs outcome ------------------------------------------
    def scatter_feature_target(self, chunks: pd.DataFrame, feature: str,
                               target: str, name: str, ylabel: str,
                               logy: bool = True,
                               color_by: Optional[str] = None) -> None:
        if feature not in chunks or target not in chunks:
            return self._skip(name, f"column {feature} or {target} absent")
        d = chunks[[c for c in (feature, target, color_by, "field")
                    if c and c in chunks]].replace(
            [np.inf, -np.inf], np.nan).dropna(subset=[feature, target])
        if logy:
            d = d[d[target] > 0]
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} chunks with both values")
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        # One field per marker face is the identity channel that does not cost
        # a colour slot; with more than three fields identity is carried by the
        # legend and the points share one hue.
        flds = sorted(d["field"].dropna().unique()) if "field" in d else []
        if 2 <= len(flds) <= 3:
            for i, f in enumerate(flds):
                s = d[d["field"] == f]
                ax.scatter(s[feature], s[target], s=26, alpha=0.85,
                           color=SERIES[i], edgecolor=SURFACE, linewidth=0.8,
                           label=str(f))
            ax.legend(title="field", loc="best", title_fontsize=8)
        else:
            ax.scatter(d[feature], d[target], s=26, alpha=0.8,
                       color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
        if logy:
            ax.set_yscale("log")
        # Log x when the feature spans decades, for the reason at
        # LOG_X_DECADES: a linear axis on a heavy-tailed feature collapses
        # thousands of distinct values onto the left edge and reads as though
        # the chunks share two or three values.
        logx = _wants_log_x(d[feature])
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(_axis_label(feature) + ("  (log scale)" if logx else ""))
        ax.set_ylabel(ylabel)
        ax.set_title(f"{_axis_label(feature)} vs {ylabel}\n"
                     f"n = {len(d)} chunks")
        _finish(ax)
        self._save(fig, name)

    def levels_vs_outcome(self, pc: pd.DataFrame, target: str, name: str,
                          ylabel: str, logy: bool,
                          xcol: str = "levels") -> None:
        """Outcome against the post-quantization level count L.

        L is closed-form in (data_range, eb) and codec-free, so a tight
        monotone band here is the quantization mechanism made visible: one
        number, derived before any codec runs, sets the outcome. Regime
        boundaries are drawn so the reader can see where the field stops
        being representable. Chunks whose bound was relaxed are hollow --
        their L is an upper bound and they must not be read as on-curve.
        """
        if pc.empty or target not in pc.columns or xcol not in pc.columns:
            return self._skip(name, "no per-chunk quantization table")
        d = pc[[xcol, target, "bound_relaxed"]].replace(
            [np.inf, -np.inf], np.nan).dropna(subset=[xcol, target])
        d = d.rename(columns={xcol: "levels"})
        if logy:
            d = d[d[target] > 0]
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} chunks")
        fig, ax = plt.subplots(figsize=(6.0, 3.9))
        ok, rl = d[~d["bound_relaxed"]], d[d["bound_relaxed"]]
        ax.scatter(ok["levels"], ok[target], s=22, alpha=0.75,
                   color=SERIES[0], edgecolor=SURFACE, linewidth=0.7,
                   label="bound honoured")
        if not rl.empty:
            ax.scatter(rl["levels"], rl[target], s=26, alpha=0.9,
                       facecolor="none", edgecolor=SERIES[1], linewidth=1.0,
                       label="bound RELAXED (L is an upper bound)")
        # Regime edges are defined on L (the range's rungs); on the bulk
        # axis they would mislead, so they are drawn only there.
        if xcol == "levels":
            from .quantization_mechanism import REGIMES
            for lo, _hi, label, _m in REGIMES[1:]:
                ax.axvline(lo, color=INK_2, lw=0.8, ls=":")
                ax.text(lo, 1.0, f" {label} →",
                        transform=ax.get_xaxis_transform(),
                        fontsize=7, color=INK_2, va="top")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(_axis_label(xcol) + "  (log scale)")
        ax.set_ylabel(ylabel)
        what = ("how many levels survive quantization" if xcol == "levels"
                else "how many grid steps the bulk of the data spans")
        ax.set_title(f"{ylabel} is set by {what}\nn = {len(d)} chunks; "
                     f"closed-form in the data and eb, no codec in it")
        if not rl.empty or True:
            ax.legend(loc="best", fontsize=7)
        _finish(ax)
        self._save(fig, name)

    def regime_bars(self, reg: pd.DataFrame, name: str) -> None:
        """Share of chunks, median ratio and median SSIM per L regime, side
        by side, so the trade the bound is making is one picture."""
        if reg.empty:
            return self._skip(name, "no regime table")
        cols = ["pct_of_chunks", "median_best_ratio"] + \
               [c for c in ("median_meas_ssim",) if c in reg.columns]
        labels = {"pct_of_chunks": "% of chunks",
                  "median_best_ratio": "median best ratio (log)",
                  "median_meas_ssim": "median SSIM"}
        fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.4))
        axes = np.atleast_1d(axes)
        x = np.arange(len(reg))
        for ax, c in zip(axes, cols):
            v = reg[c].to_numpy(dtype=float)
            ax.bar(x, v, color=SERIES[0], edgecolor=SURFACE)
            if c == "median_best_ratio":
                ax.set_yscale("log")
            if c == "median_meas_ssim":
                ax.set_ylim(0, 1.02)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{r}\n(L {lo:g}–{hi:g})" for r, lo, hi in
                                zip(reg["regime"], reg["levels_from"],
                                    reg["levels_to"])], fontsize=7)
            ax.set_title(labels[c], fontsize=9)
            for xi, vi in zip(x, v):
                if np.isfinite(vi):
                    ax.text(xi, vi, f"{vi:.3g}", ha="center", va="bottom",
                            fontsize=7, color=INK_2)
            _finish(ax)
        fig.suptitle("Regimes by levels after quantization: what each one "
                     "costs and buys", fontsize=10)
        fig.tight_layout()
        self._save(fig, name)

    def property_correlation_matrix(self, mat: pd.DataFrame, name: str,
                                    title: str = "") -> None:
        """The whole story in one square: every data property against every
        outcome, Spearman, with the property/outcome boundary drawn.

        The upper-left block is feature collinearity (what the features know
        about each other), the off-diagonal block is what a reader actually
        wants -- which property moves which metric, and in which direction --
        and the lower-right block is how the outcomes move together. Diverging
        colour around a neutral midpoint: sign is the message, so the eye must
        read zero as "nothing". Every cell prints its rho; a cell with too
        little support prints nothing rather than a colour.
        """
        if mat.empty:
            return self._skip(name, "no property/outcome matrix")
        order = [c for c in
                 ["entropy", "mad", "second_deriv", "data_range", "timestep",
                  "best_lossless_ratio", "best_quantized_ratio",
                  "fastest_ct_ms", "fastest_dt_ms", "quantized_ssim"]
                 if c in set(mat["a"]) | set(mat["b"])]
        labels = {**dict(zip(mat["a"], mat["a_label"])),
                  **dict(zip(mat["b"], mat["b_label"]))}
        k = len(order)
        M = np.full((k, k), np.nan)
        for _, r in mat.iterrows():
            i, j = order.index(r["a"]), order.index(r["b"])
            M[i, j] = M[j, i] = r["spearman_rho"]
        n_props = sum(1 for c in order if c in
                      ("entropy", "mad", "second_deriv", "data_range",
                       "timestep"))
        fig, ax = plt.subplots(figsize=(0.62 * k + 2.2, 0.62 * k + 1.4))
        ax.imshow(M, cmap=DIV, vmin=-1, vmax=1, aspect="equal")
        for i in range(k):
            for j in range(k):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color=SURFACE if abs(v) > 0.6 else INK)
        # The boundary between "what went in" and "what came out".
        ax.axhline(n_props - 0.5, color=INK_2, lw=1.1)
        ax.axvline(n_props - 0.5, color=INK_2, lw=1.1)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels([labels.get(c, c) for c in order], rotation=40,
                           ha="right", fontsize=8)
        ax.set_yticklabels([labels.get(c, c) for c in order], fontsize=8)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.text(-0.5, -0.9, "properties", color=INK_2, fontsize=8, va="bottom")
        ax.text(n_props - 0.5, -0.9, "outcomes", color=INK_2, fontsize=8,
                va="bottom")
        n = int(mat["n"].max()) if "n" in mat.columns else 0
        ax.set_title((title or "data properties vs outcomes") +
                     f"\nSpearman rho, {n} chunks; blue = together, "
                     f"red = opposite", pad=22)
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIV,
                          norm=plt.Normalize(-1, 1)), ax=ax, shrink=0.72)
        cb.set_label("Spearman rho", color=INK_2)
        cb.outline.set_visible(False)
        self._save(fig, name)

    def facet_by_codec(self, rows: pd.DataFrame, feature: str, target: str,
                       name: str, ylabel: str, logy: bool = True) -> None:
        """Small multiples: one panel per codec, one hue inside each.

        Eight codecs cannot be eight validated hues in one frame, and a
        legend of eight overlapping point clouds is unreadable even when they
        can. Shared axes make the panels comparable, which is the whole
        question being asked.
        """
        if feature not in rows or target not in rows:
            return self._skip(name, f"column {feature} or {target} absent")
        d = rows.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[feature, target])
        if logy:
            d = d[d[target] > 0]
        libs = sorted(d["lib_name"].dropna().unique())
        libs = [l for l in libs if len(d[d["lib_name"] == l]) >= MIN_PANEL_POINTS]
        if not libs:
            return self._skip(name, "no codec has enough points")
        ncol = min(4, len(libs))
        nrow = int(np.ceil(len(libs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        for k, lib in enumerate(libs):
            ax = axes[k // ncol][k % ncol]
            s = d[d["lib_name"] == lib]
            # Lossless and lossy are two different transforms of the chunk;
            # drawn as two series inside the panel rather than pooled, because
            # pooling them draws a cloud that describes neither.
            for i, (q, lab) in enumerate(((0, "lossless"), (1, "quantized"))):
                sq = s[s["quantize"] == q]
                if sq.empty:
                    continue
                ax.scatter(sq[feature], sq[target], s=12, alpha=0.7,
                           color=SERIES[i], edgecolor="none", label=lab)
            if logy:
                ax.set_yscale("log")
            ax.set_title(lib.replace("nvcomp-", ""), fontsize=9)
            _finish(ax)
        for k in range(len(libs), nrow * ncol):
            axes[k // ncol][k % ncol].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if len(labels) >= 2:
            fig.legend(handles, labels, loc="upper right", ncol=2,
                       bbox_to_anchor=(0.99, 1.02))
        fig.supxlabel(_axis_label(feature), color=INK_2, fontsize=9)
        fig.supylabel(ylabel, color=INK_2, fontsize=9)
        fig.suptitle(f"{_axis_label(feature)} vs {ylabel}, per codec",
                     x=0.01, ha="left", color=INK, fontsize=10)
        fig.tight_layout()
        self._save(fig, name)

    def heatmap_two_features(self, chunks: pd.DataFrame, fx: str, fy: str,
                             target: str, name: str, label: str,
                             nbins: int = 6) -> None:
        """Median target over a 2-D quantile grid of two properties."""
        cols = [fx, fy, target]
        if any(c not in chunks for c in cols):
            return self._skip(name, "a required column is absent")
        d = chunks[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < nbins * 3:
            return self._skip(name, f"only {len(d)} chunks for a "
                                    f"{nbins}x{nbins} grid")
        nb = min(nbins, d[fx].nunique(), d[fy].nunique())
        if nb < 2:
            return self._skip(name, "a feature is constant")
        try:
            bx = pd.qcut(d[fx], nb, duplicates="drop")
            by = pd.qcut(d[fy], nb, duplicates="drop")
        except ValueError:
            return self._skip(name, "cannot form quantile bins")
        piv = d.groupby([by, bx], observed=True)[target].median().unstack()
        cnt = d.groupby([by, bx], observed=True)[target].size().unstack()
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        vals = np.log10(piv.to_numpy(dtype=float)) if target.endswith("ratio") \
            or "ratio" in target else piv.to_numpy(dtype=float)
        im = ax.imshow(vals, origin="lower", cmap=SEQ, aspect="auto")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"median log10({label})" if "ratio" in target
                     else f"median {label}", color=INK_2)
        cb.outline.set_visible(False)
        # Support printed in every cell: an empty or one-chunk cell must not
        # read as a measurement.
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                n = cnt.to_numpy()[i, j]
                if np.isfinite(n) and n > 0:
                    ax.text(j, i, f"{int(n)}", ha="center", va="center",
                            fontsize=7, color=INK if np.isnan(
                                vals[i, j]) else "#ffffff")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{c.left:.3g}" for c in piv.columns],
                           rotation=45, ha="right")
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{c.left:.3g}" for c in piv.index])
        ax.set_xlabel(f"{_axis_label(fx)} (bin lower edge)")
        ax.set_ylabel(f"{_axis_label(fy)} (bin lower edge)")
        ax.set_title(f"median {label} over {_axis_label(fx)} x "
                     f"{_axis_label(fy)}\ncell numbers = chunks in the bin")
        ax.grid(False)
        _finish(ax)
        self._save(fig, name)

    def winner_property_space(self, chunks: pd.DataFrame, win: pd.DataFrame,
                              features: Sequence[str], name: str) -> None:
        """Which codec wins where, in the two most informative properties.

        Marker SHAPE carries the codec and colour carries nothing, because
        there are more codecs than validated hues. A single winner across the
        whole log is drawn anyway and said so in the title -- that is the
        finding.
        """
        if "best_ratio_lib" not in win or len(features) < 2:
            return self._skip(name, "no winner column or too few features")
        d = chunks.merge(win[["chunk_uid", "best_ratio_lib",
                              "best_ratio_nties"]], on="chunk_uid",
                         how="inner").replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=[features[0], features[1], "best_ratio_lib"])
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} chunks")
        libs = sorted(d["best_ratio_lib"].unique())
        marks = ["o", "s", "^", "D", "v", "P", "X", "*"]
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        for i, lib in enumerate(libs):
            s = d[d["best_ratio_lib"] == lib]
            ax.scatter(s[features[0]], s[features[1]], s=42,
                       marker=marks[i % len(marks)],
                       color=SERIES[i % len(SERIES)] if len(libs) <= 3
                       else SERIES[0],
                       edgecolor=SURFACE, linewidth=0.8, alpha=0.85,
                       label=lib.replace("nvcomp-", ""))
        ax.set_yscale("log" if (d[features[1]] > 0).all() else "linear")
        ax.set_xlabel(_axis_label(features[0]))
        ax.set_ylabel(_axis_label(features[1]))
        note = "" if len(libs) > 1 else "  -- a single codec wins every chunk"
        ax.set_title(f"highest-ratio codec in property space{note}")
        ax.legend(loc="best", title="best-ratio codec", title_fontsize=8)
        _finish(ax)
        self._save(fig, name)

    # ---- treatments -----------------------------------------------------
    def benefit_vs_feature(self, pairs: pd.DataFrame, feature: str,
                           name: str, label: str,
                           target: str = "ratio") -> None:
        """Signed change against a property, as a FRACTION of the chunk's own
        ratio -- not the absolute difference.

        The absolute delta is the right statistic for the summary table, where
        pairing cancels chunk difficulty. On a scatter ACROSS chunks it is
        not: these ratios span 1x to 6223x, so an absolute axis is set
        entirely by the handful of near-constant chunks. On the Nyx log a
        quantized chunk going 4120x -> 3173x plots at -947 and drags every
        ordinary chunk into a flat band at zero, reading as catastrophic
        damage when the change is -23%. rel_ratio is the same measurement in
        units that do not depend on how compressible the chunk happens to be;
        paired_analysis computes it already.
        """
        col = f"rel_{target}" if f"rel_{target}" in pairs else f"d_{target}"
        relative = col.startswith("rel_")
        if pairs.empty or col not in pairs or feature not in pairs:
            return self._skip(name, "no paired comparison available")
        d = pairs[[feature, col, "lib_name"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} pairs")
        v = d[col].to_numpy(dtype=float) * (100.0 if relative else 1.0)
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        # Sign is the question ("does it help?"), so sign is what carries the
        # colour -- two categorical slots with a legend. A continuous
        # colourbar here would re-encode the y position and buy nothing,
        # while costing the width the title needs.
        for m, lab, c in ((v > 0, "improves the ratio", SERIES[0]),
                          (v <= 0, "no gain or worse", SERIES[1])):
            if not m.any():
                continue
            ax.scatter(d[feature].to_numpy()[m], v[m], s=24, alpha=0.75,
                       color=c, edgecolor=SURFACE, linewidth=0.5, label=lab)
        ax.axhline(0.0, color=INK_2, linewidth=1.2, zorder=1)
        if relative:
            # Symlog: the gains run to thousands of percent on chunks the
            # setting rescues, the losses are bounded below by -100%, and the
            # bulk sits within a few percent of zero. A linear axis shows one
            # of those three; a plain log cannot cross zero.
            ax.set_yscale("symlog", linthresh=10, linscale=0.6)
        ax.set_xlabel(_axis_label(feature))
        ax.set_ylabel(f"change in {target}, % of the chunk's own {target}\n"
                      f"(setting on vs off, same chunk)" if relative else
                      f"change in {target}  (setting on - setting off,\n"
                      f"same chunk)")
        helps = float(100.0 * (v > 0).mean())
        med = float(np.median(v))
        ax.set_title(f"{label}: change in {target} vs {_axis_label(feature)}\n"
                     f"{helps:.0f}% of {len(d)} controlled pairs improve"
                     + (f"; median {med:+.1f}%" if relative else ""))
        ax.legend(loc="best")
        _finish(ax)
        self._save(fig, name)

    def error_bound_effect(self, rows: pd.DataFrame, name: str) -> None:
        lossy = rows[(rows["quantize"] == 1) & rows["error_bound"].notna()]
        if lossy.empty or lossy["error_bound"].nunique() < 2:
            return self._skip(name, "the sweep explored a single error bound; "
                                    "there is nothing to compare")
        d = lossy.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["error_bound", "ratio"])
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        bounds = sorted(d["error_bound"].unique())
        ax.boxplot([d[d["error_bound"] == b]["ratio"] for b in bounds],
                   labels=[f"{b:g}" for b in bounds],
                   patch_artist=True,
                   boxprops=dict(facecolor=SERIES[0], alpha=0.55,
                                 edgecolor=INK_2),
                   medianprops=dict(color=INK, linewidth=1.6),
                   whiskerprops=dict(color=AXIS),
                   capprops=dict(color=AXIS),
                   flierprops=dict(markersize=3, markerfacecolor=MUTED,
                                   markeredgecolor="none"))
        ax.set_yscale("log")
        ax.set_xlabel("requested error bound")
        ax.set_ylabel("compression ratio")
        ax.set_title("error bound vs compression ratio")
        _finish(ax)
        self._save(fig, name)

    # ---- tradeoff and prediction ----------------------------------------
    def pareto(self, rows: pd.DataFrame, xcol: str, name: str,
               xlabel: str) -> None:
        if xcol not in rows:
            return self._skip(name, f"{xcol} absent")
        d = rows.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[xcol, "ratio"])
        d = d[(d[xcol] > 0) & (d["ratio"] > 0)]
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} rows")
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        ax.scatter(d[xcol], d["ratio"], s=10, alpha=0.35, color=MUTED,
                   edgecolor="none", label="all candidates")
        # The frontier itself, in the leading slot: it is the only part of
        # this cloud a selector can ever choose from.
        f = _pareto_front(d[xcol].to_numpy(dtype=float),
                          d["ratio"].to_numpy(dtype=float))
        fr = d.iloc[f].sort_values(xcol)
        ax.plot(fr[xcol], fr["ratio"], color=SERIES[0], linewidth=2.0,
                marker="o", markersize=5, markeredgecolor=SURFACE,
                label="Pareto frontier")
        # One label per DISTINCT codec on the frontier, placed at that
        # codec's highest-ratio frontier point. Labelling every point stacks
        # the same name on itself six times and reads as a smudge.
        for lib, seg in fr.groupby("lib_name"):
            best = seg.loc[seg["ratio"].idxmax()]
            ax.annotate(str(lib).replace("nvcomp-", ""),
                        (best[xcol], best["ratio"]), fontsize=8, color=INK,
                        xytext=(6, 5), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("compression ratio")
        ax.set_title(f"compression ratio vs {xlabel}\n"
                     f"{len(d)} candidate measurements; the frontier is what "
                     f"a selector can actually choose from")
        ax.legend(loc="best")
        _finish(ax)
        self._save(fig, name)

    # ---- mechanism -----------------------------------------------------
    #: Fixed hue order for the three codec classes. Assigned by CLASS, never
    #: cycled, so a codec keeps its colour when the set of codecs changes.
    CLASS_ORDER = ("order-blind", "at the bound", "order-sensitive")
    #: The same three classes said in words a reader does not have to look up.
    #: The short names are what the TABLES use and what the code selects on;
    #: a legend is read once, in isolation, so it gets the long form.
    CLASS_LEGEND = {
        "order-blind": "never beats the prediction\n(reads the histogram only)",
        "at the bound": "beats it slightly\n(order buys a few percent)",
        "order-sensitive": "beats it substantially\n(order is worth a lot)",
    }

    def entropy_bound_scatter(self, bound_rows: pd.DataFrame,
                              order: pd.DataFrame, name: str) -> None:
        """Achieved ratio against the entropy bound 8/H, with the bound drawn.

        The one figure that carries the decomposition: everything on or below
        the diagonal is what a coder reading only the byte histogram can get,
        and every unit of height above it is contributed by order. Restricted
        to LOSSLESS rows, where 8/H is the right bound for the bytes the codec
        actually saw.
        """
        need = {"entropy", "ratio", "lib_name", "quantize"}
        if bound_rows is None or bound_rows.empty or \
                not need <= set(bound_rows.columns):
            return self._skip(name, "no entropy/ratio columns to compare")
        d = bound_rows.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["entropy", "ratio"])
        d = d[(d["quantize"] == 0) & (d["entropy"] > 0) & (d["ratio"] > 0)]
        if "shuffle" in d.columns:
            d = d[d["shuffle"] == 0]
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} lossless unshuffled rows")
        cls = {}
        if order is not None and not order.empty and "class" in order.columns:
            cls = dict(zip(order["lib_name"], order["class"]))
        d = d.assign(_cls=d["lib_name"].map(cls).fillna("order-sensitive"))
        d["_bound"] = 8.0 / d["entropy"]

        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        # Independent limits, not a forced square. The bound spans ~1 decade
        # here and the ratio ~3, so equal ranges would leave most of the
        # canvas empty; the diagonal is still exactly the locus ratio = 8/H,
        # and "above the line" reads the same whatever the aspect.
        xlo = float(d["_bound"].min()) * 0.8
        xhi = float(d["_bound"].max()) * 1.25
        ylo = float(d["ratio"].min()) * 0.7
        yhi = float(d["ratio"].max()) * 1.6
        ax.plot([xlo, xhi], [xlo, xhi], color=INK_2, linewidth=1.4,
                linestyle=(0, (5, 3)), zorder=1)
        for i, c in enumerate(self.CLASS_ORDER):
            g = d[d["_cls"] == c]
            if g.empty:
                continue
            ax.scatter(g["_bound"], g["ratio"], s=22, alpha=0.45,
                       color=SERIES[i], edgecolor="none",
                       label=self.CLASS_LEGEND.get(c, c), zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        rel = d["ratio"] / d["_bound"]
        med = float(np.median(rel))
        q1, q3 = (float(rel.quantile(0.25)), float(rel.quantile(0.75)))
        ax.set_xlabel("ratio ENTROPY PREDICTS for this chunk   "
                      "(8 / H, H = bits per byte)")
        ax.set_ylabel("ratio the codec ACTUALLY ACHIEVED")
        ax.set_title("What entropy predicts is what the codec gets"
                     f"\nachieved / predicted: median {med:.2f}, "
                     f"half of the rows in {q1:.2f}-{q3:.2f} · "
                     f"n = {len(d):,} (chunk, codec) rows")
        # Direct label ON the reference line, anchored to a point of the line
        # itself rather than floating in a corner, so the dashed guide is not
        # something the reader has to decode from the legend.
        xa = float(np.sqrt(xlo * xhi))
        if ylo < xa < yhi:
            # Label the diagonal ON the diagonal, rotated to lie along it.
            # A leader line from a corner had to cross the point cloud, so
            # the reader traced an arrow before the line meant anything.
            # The angle is measured in DISPLAY space: the axes are log-log
            # but not square, so the diagonal is not drawn at 45 degrees.
            fig.canvas.draw()
            p0 = ax.transData.transform((xa, xa))
            p1 = ax.transData.transform((xa * 4, xa * 4))
            ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
            ax.text(xa, xa, " achieved = predicted ",
                    rotation=ang, rotation_mode="anchor", ha="center",
                    va="center", fontsize=8.5, color=INK_2, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                              edgecolor="none", alpha=0.85))
        # Say what each HALF of the plot means. Without this a reader has to
        # derive "above the line = the codec found structure entropy cannot
        # see" from the axis names, which is the step that was being skipped.
        # Boxed, because both halves are occupied by points -- there is no
        # empty corner to retreat to.
        box = dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                   edgecolor=GRID, alpha=0.88)
        ax.annotate("ABOVE: the codec beat the prediction\n"
                    "by finding repeated SEQUENCES,\n"
                    "which a byte histogram cannot see",
                    xy=(0.02, 0.98), xycoords="axes fraction", fontsize=7.8,
                    color=INK, va="top", ha="left", zorder=5, bbox=box)
        ax.annotate("BELOW: it fell short of the prediction",
                    xy=(0.98, 0.03), xycoords="axes fraction", fontsize=7.8,
                    color=INK, va="bottom", ha="right", zorder=5, bbox=box)
        # Legend below the axes: the lower-right corner is where the reference
        # line's own caption has to go, and the two collided there.
        leg = ax.legend(title="codec behaviour", title_fontsize=8, ncol=3,
                        loc="upper center", bbox_to_anchor=(0.5, -0.17))
        for h in leg.legend_handles:
            h.set_alpha(1.0)
            h.set_sizes([44])
        _finish(ax)
        self._save(fig, name)

    def excess_over_bound(self, order: pd.DataFrame, name: str) -> None:
        """Ranked dot plot: how far past 8/H each codec gets, with its range."""
        if order is None or order.empty or \
                "median_excess_over_bound" not in order.columns:
            return self._skip(name, "no per-codec bound comparison available")
        d = order.sort_values("median_excess_over_bound").reset_index(
            drop=True)
        if len(d) < 2:
            return self._skip(name, "fewer than two codecs")
        fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(d) + 1.9))
        y = np.arange(len(d))
        ax.axvline(1.0, color=INK_2, linewidth=1.4, linestyle=(0, (5, 3)),
                   zorder=1)
        for i, (_, r) in enumerate(d.iterrows()):
            c = SERIES[self.CLASS_ORDER.index(r["class"])] \
                if r.get("class") in self.CLASS_ORDER else SERIES[0]
            lo = float(r.get("median_excess_over_bound", np.nan))
            hi = float(r.get("max_excess_over_bound", lo))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [i, i], color=c, linewidth=2.0, alpha=0.35,
                        solid_capstyle="round", zorder=2)
            ax.scatter([lo], [i], s=64, color=c, edgecolor=SURFACE,
                       linewidth=1.0, zorder=3)
            # Above the dot, on an opaque chip. Every median sits within a
            # few percent of 1.0, so the label lands on the reference line
            # whatever the offset; masking the line under the text is what
            # keeps it readable. (Sideways instead runs it into the codec
            # names on the y axis.)
            ax.annotate(f"{lo:.2f}x", xy=(lo, i), xytext=(0, 10),
                        textcoords="offset points", fontsize=8, color=INK,
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.16",
                                  facecolor=SURFACE, edgecolor="none"))
        ax.set_yticks(y)
        ax.set_yticklabels(d["lib_name"])
        ax.set_xscale("log")
        ax.set_xlabel("how many times its OWN entropy prediction the codec "
                      "reached\n(dot = typical chunk, bar out to its best "
                      "chunk)")
        ax.set_title("Can a codec beat what entropy predicts?  Barely.\n"
                     "1.0 = exactly the prediction · left of it = never "
                     "beats it · right = beats it")
        ax.margins(y=0.06)
        # Hue carries the class, so it needs a key: without one, identity
        # would rest on colour alone.
        present = [c for c in self.CLASS_ORDER if (d.get("class") == c).any()]
        if len(present) > 1:
            ax.legend(handles=[
                plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                           markerfacecolor=SERIES[self.CLASS_ORDER.index(c)],
                           markeredgecolor=SURFACE,
                           label=self.CLASS_LEGEND.get(c, c))
                for c in present],
                title="codec behaviour", title_fontsize=8, ncol=len(present),
                loc="upper center", bbox_to_anchor=(0.5, -0.20),
                fontsize=7.5)
        _finish(ax)
        self._save(fig, name)

    def timing_mediation(self, med: pd.DataFrame, name: str) -> None:
        """Per codec: does time follow the data, or follow the output size?

        A dumbbell per codec. When the size dot sits further right than the
        feature dot, the compressed size is the better account of the time --
        which is the mediation claim, drawn.
        """
        need = {"metric", "lib_name", "rho_time_vs_feature",
                "rho_time_vs_compressed_bytes"}
        if med is None or med.empty or not need <= set(med.columns):
            return self._skip(name, "no timing mediation table")
        d = med.copy()
        d["_f"] = d["rho_time_vs_feature"].abs()
        d["_s"] = d["rho_time_vs_compressed_bytes"].abs()
        metrics = [m for m in sorted(d["metric"].dropna().unique())
                   if d[d["metric"] == m]["_f"].notna().sum() >= 3]
        if not metrics:
            return self._skip(name, "too few usable timing rows")
        fig, axes = plt.subplots(1, len(metrics),
                                 figsize=(4.6 * len(metrics), 3.9),
                                 squeeze=False, sharex=True)
        for ax, m in zip(axes[0], metrics):
            # Strongest feature per codec: the fairest comparison for the
            # features, since a weak one would make size win by default.
            g = (d[d["metric"] == m].groupby("lib_name", sort=True)
                 .agg(f=("_f", "max"), s=("_s", "max")).reset_index()
                 .sort_values("s"))
            y = np.arange(len(g))
            for i, (_, r) in enumerate(g.iterrows()):
                ax.plot([r["f"], r["s"]], [i, i], color=MUTED, linewidth=1.6,
                        alpha=0.5, solid_capstyle="round", zorder=2)
            ax.scatter(g["f"], y, s=52, color=SERIES[0], edgecolor=SURFACE,
                       linewidth=1.0, zorder=3, label="best data feature")
            ax.scatter(g["s"], y, s=52, color=SERIES[1], edgecolor=SURFACE,
                       linewidth=1.0, zorder=3, label="compressed size")
            ax.set_yticks(y)
            ax.set_yticklabels(g["lib_name"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("|Spearman rho| with the time")
            ax.set_title(_axis_label(m))
            _finish(ax)
        # Below the panels, not inside one: in the left panel the lower-right
        # corner is crossed by the widest dumbbell.
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("The timings follow the OUTPUT SIZE, not the data "
                     "properties", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        self._save(fig, name)

    # ---- matched controls ---------------------------------------------
    def matched_pair_spread(self, pairs: pd.DataFrame, name: str) -> None:
        """Feature agreement on x, outcome disagreement on y.

        The figure the matched-control section exists to make: if the features
        determined the ratio, the cloud would funnel down to 1.0 as it moves
        left. Points at the left edge are pairs that are numerically identical
        in every feature, so their height is error no model can remove.
        """
        need = {"max_rel_feature_diff", "ratio_fold", "same_field"}
        if pairs is None or pairs.empty or not need <= set(pairs.columns):
            return self._skip(name, "no matched pairs in this log")
        d = pairs.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["max_rel_feature_diff", "ratio_fold"])
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} matched pairs")
        # Exact agreement is 0 and has no place on a log axis; it is clipped to
        # the left edge, which is labelled so the clip is not read as a value.
        floor = 1e-10
        x = d["max_rel_feature_diff"].to_numpy(dtype=float).clip(min=floor)
        y = d["ratio_fold"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        groups = [(False, "different field", SERIES[0]),
                  (True, "same field", SERIES[1])]
        drawn = 0
        for flag, label, colour in groups:
            m = (d["same_field"].to_numpy() == flag)
            if not m.any():
                continue
            drawn += 1
            ax.scatter(x[m], y[m], s=28, alpha=0.8, color=colour,
                       edgecolor=SURFACE, linewidth=0.8, label=label)
        ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_xlabel("largest relative disagreement in any feature\n"
                      "(left edge = numerically identical)")
        ax.set_ylabel("compression ratio fold  (higher / lower)")
        ax.set_title("Same moment, same size: what the statistics miss\n"
                     f"n = {len(d)} pairs over "
                     f"{d['group'].nunique() if 'group' in d else 0} moments")
        if drawn > 1:
            ax.legend(loc="best")
        _finish(ax)
        self._save(fig, name)

    def component_ratio_over_time(self, chunks: pd.DataFrame,
                                  fams: pd.DataFrame, target: str,
                                  name: str) -> None:
        """One panel per vector family, one line per component.

        Components of an isotropic field carry the same statistics; if the
        lines separate and stay separated, the separation is not in the
        statistics.
        """
        if fams is None or fams.empty or "timestep" not in chunks.columns:
            return self._skip(name, "no vector-component families detected "
                                    "in the field names")
        d = chunks.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["timestep", target, "field"])
        d = d[d[target] > 0]
        bases = [b for b, g in fams.groupby("base", sort=True)
                 if d["field"].isin(g["field"]).sum() >= MIN_PANEL_POINTS]
        if not bases:
            return self._skip(name, "too few chunks per component family")
        fig, axes = plt.subplots(1, len(bases), figsize=(4.4 * len(bases), 3.6),
                                 squeeze=False, sharey=True)
        for ax, b in zip(axes[0], bases):
            g = fams[fams["base"] == b].sort_values("axis")
            for i, (_, row) in enumerate(g.iterrows()):
                s = d[d["field"] == row["field"]].sort_values("timestep")
                if s.empty:
                    continue
                ax.plot(s["timestep"], s[target], marker="o", markersize=3.4,
                        linewidth=1.5, color=SERIES[i % len(SERIES)],
                        label=str(row["axis"]))
            ax.set_yscale("log")
            ax.set_xlabel("timestep")
            ax.set_title(f"`{b}` components")
            ax.legend(title="axis", loc="best", title_fontsize=8)
            _finish(ax)
        axes[0][0].set_ylabel("best achievable compression ratio")
        fig.suptitle("Same statistics, different compressibility",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        self._save(fig, name)

    def predicted_vs_actual(self, err: pd.DataFrame, name: str,
                            label: str) -> None:
        if err.empty:
            return self._skip(name, "no predictions available for this metric")
        d = err.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["predicted", "actual"])
        d = d[(d["predicted"] > 0) & (d["actual"] > 0)]
        if len(d) < MIN_POINTS:
            return self._skip(name, f"only {len(d)} usable rows")
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        for i, (role, lab) in enumerate((("alt", "explored alternative"),
                                         ("primary", "the model's own pick"))):
            s = d[d["role"] == role] if "role" in d else d
            if s.empty:
                continue
            ax.scatter(s["actual"], s["predicted"], s=14, alpha=0.5,
                       color=SERIES[i], edgecolor="none", label=lab)
        lo = float(min(d["actual"].min(), d["predicted"].min()))
        hi = float(max(d["actual"].max(), d["predicted"].max()))
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.2,
                linestyle="--", label="perfect prediction")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"measured {label}")
        ax.set_ylabel(f"predicted {label}")
        med = float(10 ** d["log10_ratio_error"].abs().median())
        ax.set_title(f"predicted vs measured {label}\n"
                     f"median error {med:.2f}x over {len(d)} candidates")
        ax.legend(loc="best")
        _finish(ax)
        self._save(fig, name)

    # ---- temporal --------------------------------------------------------
    def over_time(self, series: pd.DataFrame, col: str, name: str,
                  ylabel: str, logy: bool = False) -> None:
        if series.empty or col not in series:
            return self._skip(name, "no temporal series available")
        d = series.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["timestep", col])
        if logy:
            d = d[d[col] > 0]
        flds = sorted(d["field"].dropna().unique())
        if d["timestep"].nunique() < 3 or not flds:
            return self._skip(name, "fewer than three timesteps")
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        # One hue per field while the validated palette lasts; past it, a
        # single hue, because a generated ninth colour is not validated
        # against the others for CVD separation.
        one_hue = len(flds) > len(SERIES)
        # End labels are NOT optional above three hues: the palette's worst
        # adjacent pair is 5.8 tritan and three of its slots warn on contrast,
        # so colour alone is not a sufficient identity channel there.
        want_ends = one_hue or len(flds) > 3
        ends = []
        for i, f in enumerate(flds):
            s = d[d["field"] == f].sort_values("timestep")
            color = SERIES[0] if one_hue else SERIES[i]
            ax.plot(s["timestep"], s[col], color=color, linewidth=2.0,
                    marker="o", markersize=4, markeredgecolor=SURFACE,
                    alpha=0.7 if one_hue else 0.9,
                    label=None if one_hue else str(f))
            if want_ends and len(s):
                ends.append((float(s["timestep"].iloc[-1]),
                             float(s[col].iloc[-1]), str(f),
                             SERIES[0] if one_hue else SERIES[i]))
        if logy:
            ax.set_yscale("log")
        # Direct end labels are the identity channel once there are more
        # fields than validated hues, so they have to be legible: six fields
        # converging on the same value stack their labels on top of one
        # another. Push them apart in DISPLAY space, after the scale is set,
        # and tether each back to its own line.
        if ends:
            _label_ends(ax, ends)
        if not one_hue and len(flds) >= 2:
            ax.legend(title="field", loc="best", title_fontsize=8,
                      ncol=2 if len(flds) > 4 else 1, fontsize=7)
        ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} over simulation time")
        ax.margins(x=0.10)
        _finish(ax)
        self._save(fig, name)

    def winner_over_time(self, series: pd.DataFrame, name: str) -> None:
        col = "modal_best_ratio_lib"
        if series.empty or col not in series:
            return self._skip(name, "no winner series available")
        d = series.dropna(subset=["timestep", col])
        if d["timestep"].nunique() < 3:
            return self._skip(name, "fewer than three timesteps")
        libs = sorted(d[col].unique())
        idx = {l: i for i, l in enumerate(libs)}
        flds = sorted(d["field"].dropna().unique())
        fig, ax = plt.subplots(figsize=(5.8, 0.42 * len(flds) + 1.9))
        for row, f in enumerate(flds):
            s = d[d["field"] == f].sort_values("timestep")
            ax.scatter(s["timestep"], [row] * len(s), s=64,
                       c=[SERIES[idx[l] % len(SERIES)] for l in s[col]],
                       marker="s", edgecolor=SURFACE, linewidth=0.8)
        ax.set_yticks(range(len(flds)))
        ax.set_yticklabels(flds)
        ax.set_xlabel("timestep")
        ax.set_title("codec with the highest measured ratio, per field over time")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="s", linestyle="",
                                  markerfacecolor=SERIES[idx[l] % len(SERIES)],
                                  markeredgecolor=SURFACE,
                                  label=l.replace("nvcomp-", ""))
                           for l in libs],
                  loc="upper center", bbox_to_anchor=(0.5, -0.28),
                  ncol=min(4, len(libs)))
        ax.grid(axis="y", visible=False)
        _finish(ax)
        self._save(fig, name)


def _label_ends(ax, ends, min_gap_px: float = 14.0) -> None:
    """Place one label per line end, nudged apart so none overlaps another.

    Greedy in display coordinates: sort the ends by y, then walk upward
    pushing any label that lands within `min_gap_px` of the one below it. The
    nudge is applied as a PIXEL OFFSET on the annotation rather than as a new
    data position, so it survives the log scale and the tight bounding box;
    a leader line is drawn for any label that had to move, so it still points
    at the series it names.
    """
    ax.figure.canvas.draw()
    tr = ax.transData
    # An entry may carry its line's colour. When it does the label is drawn in
    # that colour and its leader matches, so the label is a real second
    # identity channel rather than grey text that still has to be traced back
    # to a line by eye -- which is the whole point of drawing it at all when
    # the palette's CVD separation is only a conditional pass.
    norm = [(e[0], e[1], e[2], e[3] if len(e) > 3 else None) for e in ends]
    pts = sorted(((tr.transform((x, y))[1], x, y, t, c)
                  for x, y, t, c in norm), key=lambda p: p[0])
    prev = -1e9
    for dy, x, y, t, c in pts:
        ny = max(dy, prev + min_gap_px)
        prev = ny
        shift = ny - dy
        ax.annotate(t, xy=(x, y), xytext=(9, shift),
                    textcoords="offset points", fontsize=7.5,
                    color=c or INK_2, va="center", annotation_clip=False,
                    arrowprops=(dict(arrowstyle="-", color=c or MUTED,
                                     linewidth=0.6, shrinkA=0, shrinkB=2)
                                if shift > 1.0 else None))
    ax.margins(x=0.18)


def _pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices minimising x and maximising y."""
    order = np.argsort(x)
    best, keep = -np.inf, []
    for i in order:
        if y[i] > best:
            keep.append(i)
            best = y[i]
    return np.array(keep, dtype=int)


_LABELS = {
    "entropy": "byte entropy (bits/byte)",
    "mad": "MAD (raw data units)",
    "second_deriv": "mean |2nd difference| (raw data units)",
    "ratio": "compression ratio",
    "ct_ms": "compression time (ms)",
    "dt_ms_measured": "decompression time (ms)",
    "levels": "levels after quantization  L = range / (2 * 0.95 * eb)",
    "bulk_levels": "bulk levels  L_bulk = MAD / (2 * 0.95 * eb)",
    "best_quantized_ratio": "best quantized compression ratio",
    "meas_ssim": "measured SSIM",
}


def _axis_label(col: str) -> str:
    return _LABELS.get(col, col)
