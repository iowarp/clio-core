"""Shared look of the parallel k-means figures (compare_parallel_runs.py,
plot_parallel_configs.py, plot_model_vs_measured.py)."""
import matplotlib
import matplotlib.colors as mcolors

# one colour per option, the same in every figure
OPTION_COLORS = {"fixed": "#8d99ae", "learn": "#d1495b", "oracle": "#2e86ab",
                 "hcompress": "#e09f3e", "xgboost": "#5e9e94"}
# measured read-side quantities (plot_model_vs_measured.py, panel C)
METRIC_COLORS = {"bytes": "#7b6fa8", "io": "#e09f3e", "dec": "#4f9d69", "wall": "#6c757d"}
TEXT_GREY = "#555555"


def apply():
    """Set the shared matplotlib defaults (call before making a figure)."""
    matplotlib.rcParams.update({
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10, "axes.labelcolor": "#333333",
        "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": "#e3e3e3", "grid.linewidth": 0.8,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.major.size": 0, "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })


def darker(color, f=0.62):
    """@return `color` darkened by factor f (for the write part of a bar)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * f, g * f, b * f)


def titles(fig, title, subtitle):
    """A bold figure title and a smaller grey line of detail under it."""
    fig.suptitle(title, x=0.01, y=1.0, ha="left", va="bottom", fontsize=14,
                 fontweight="bold")
    fig.text(0.01, 0.995, subtitle, ha="left", va="top", fontsize=9.5,
             color=TEXT_GREY)


def pct(v, digits=1):
    """@return v as a signed percentage; two decimals when it would round to 0."""
    if v != 0 and abs(v) < 0.5 * 10 ** -digits:
        return f"{v:+.2f}%"
    return f"{v:+.{digits}f}%"
