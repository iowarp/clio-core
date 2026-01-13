#!/usr/bin/env python3
"""
Compression Parameter Study - Detailed Statistics Report

Generates a comprehensive multi-page report with 5 figures per page for each
distribution showing how parameters affect compression performance.

Each page contains combo charts showing:
1. (Compress Library, Chunk size) vs Compress Time
2. (Compress Library, Chunk size) vs Decompress Time
3. (Compress Library, Chunk size) vs Compression Ratio
4. (Compress Library, Chunk size) vs Compress CPU %
5. (Compress Library, Chunk size) vs Decompress CPU %
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

# Base color palette for compressor families
BASE_LIBRARY_COLORS = {
    # Lossless compressors
    'BZIP2': '#1f77b4',
    'Blosc2': '#ff7f0e',
    'Brotli': '#2ca02c',
    'LZ4': '#d62728',
    'LZO': '#9467bd',
    'Lzma': '#8c564b',
    'Snappy': '#e377c2',
    'Zlib': '#7f7f7f',
    'Zstd': '#bcbd22',
    'LibPressio-BZIP2': '#17becf',
    # Lossy compressors (base names)
    'LibPressio-ZFP': '#ff9896',
    'LibPressio-BitGrooming': '#98df8a',
    'ZFP': '#3498db',        # Blue for ZFP direct wrappers
    'FPZIP': '#2ecc71',      # Green for FPZIP direct wrappers
    'BitGrooming': '#e67e22', # Orange for BitGrooming direct wrappers
}

def get_library_color(lib_name, all_libraries):
    """Get unique color for a library, generating variants for parameterized compressors.

    For parameterized compressors (e.g., BitGrooming_nsd_1, BitGrooming_nsd_2, BitGrooming_nsd_3),
    generates color variants by adjusting brightness.

    Args:
        lib_name: Library name (may include parameters)
        all_libraries: List of all library names to determine variant index

    Returns:
        Hex color string
    """
    # Check for parameter patterns
    base_name = None
    param_value = None

    if '_tol_' in lib_name:
        base_name = 'ZFP'
        param_value = lib_name.split('_tol_')[1]
    elif '_nsd_' in lib_name:
        base_name = 'BitGrooming'
        param_value = lib_name.split('_nsd_')[1]
    elif '_prec_' in lib_name:
        base_name = 'FPZIP'
        param_value = lib_name.split('_prec_')[1]
    else:
        # Non-parameterized library - use base color directly
        return BASE_LIBRARY_COLORS.get(lib_name, '#808080')

    # Get base color
    base_color = BASE_LIBRARY_COLORS.get(base_name, '#808080')

    # Find all variants of this compressor family
    variants = sorted([lib for lib in all_libraries if lib.startswith(base_name + '_')])
    if not variants:
        return base_color

    # Get variant index
    try:
        variant_idx = variants.index(lib_name)
    except ValueError:
        return base_color

    # Generate color variant by adjusting brightness
    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    r, g, b = int(base_color[0:2], 16), int(base_color[2:4], 16), int(base_color[4:6], 16)

    # Adjust brightness: darker for lower indices, lighter for higher indices
    num_variants = len(variants)
    if num_variants == 1:
        return f'#{r:02x}{g:02x}{b:02x}'

    # Scale from 0.6 (darkest) to 1.4 (lightest)
    brightness = 0.6 + (0.8 * variant_idx / (num_variants - 1))

    # Apply brightness adjustment
    r = int(min(255, r * brightness))
    g = int(min(255, g * brightness))
    b = int(min(255, b * brightness))

    return f'#{r:02x}{g:02x}{b:02x}'

def get_base_compressor_name(lib_name):
    """Extract base compressor name from parameter variants.

    Examples:
        'ZFP_tol_0.001000' -> 'ZFP'
        'BitGrooming_nsd_3' -> 'BitGrooming'
        'FPZIP_prec_16' -> 'FPZIP'
        'BZIP2' -> 'BZIP2'
    """
    # Check for parameter patterns
    if '_tol_' in lib_name:
        return 'ZFP'
    elif '_nsd_' in lib_name:
        return 'BitGrooming'
    elif '_prec_' in lib_name:
        return 'FPZIP'
    else:
        return lib_name  # Return as-is for non-parameterized names

# Output directory for plots
OUTPUT_DIR = Path(__file__).parent / 'benchmark_plots'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data(csv_path):
    """Load and preprocess benchmark data."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} benchmark records")

        # Convert chunk size to MB for readability
        df['Chunk Size (MB)'] = df['Chunk Size (bytes)'] / (1024**2)

        # Filter out invalid data
        df = df[df['Compress Time (ms)'] > 0]
        df = df[df['Decompress Time (ms)'] > 0]
        df = df[df['Compression Ratio'] > 0]

        # Check if we have Target CPU Util column (new version)
        has_cpu_util = 'Target CPU Util (%)' in df.columns

        print(f"✓ Processed {len(df)} valid records")
        if has_cpu_util:
            cpu_utils = sorted(df['Target CPU Util (%)'].unique())
            print(f"✓ Found CPU utilization levels: {cpu_utils}")
        return df
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        sys.exit(1)

def create_cpu_util_line_chart(ax, df_subset, libraries, metric, ylabel, title, use_log_scale=False):
    """
    Create a line chart showing how metric varies with CPU utilization.
    One line per library.

    Args:
        ax: Matplotlib axis
        df_subset: DataFrame subset for this distribution
        libraries: List of unique libraries (top N for readability)
        metric: Column name for the metric to plot
        ylabel: Y-axis label
        title: Chart title
        use_log_scale: Whether to use log scale for y-axis
    """
    # Get CPU utilization levels
    cpu_utils = sorted(df_subset['Target CPU Util (%)'].unique())

    # Plot line for each library (limit to top 10 for readability)
    for lib in libraries[:10]:
        lib_data = df_subset[df_subset['Library'] == lib]
        if len(lib_data) == 0:
            continue

        # Get values for each CPU util level
        values = []
        for cpu_util in cpu_utils:
            cpu_data = lib_data[lib_data['Target CPU Util (%)'] == cpu_util]
            if len(cpu_data) > 0:
                val = cpu_data[metric].mean()
                if use_log_scale and val <= 0:
                    val = 0.001
                values.append(val)
            else:
                values.append(None)

        # Get unique color for this library
        color = get_library_color(lib, libraries)

        # Plot line
        ax.plot(cpu_utils, values, marker='o', linewidth=2, markersize=6,
                label=lib, color=color, alpha=0.8)

    # Customize chart
    ax.set_xlabel('Target CPU Utilization (%)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    if use_log_scale:
        ax.set_yscale('log')

def create_simple_bar_chart(ax, df_subset, libraries, metric, ylabel, title, use_log_scale=False):
    """
    Create a simple bar chart for a fixed chunk size (64KB).

    Args:
        ax: Matplotlib axis
        df_subset: DataFrame subset for this distribution
        libraries: List of unique libraries
        metric: Column name for the metric to plot
        ylabel: Y-axis label
        title: Chart title
        use_log_scale: Whether to use log scale for y-axis
    """
    # Number of libraries
    n_libs = len(libraries)
    x_positions = np.arange(n_libs)

    values = []
    for lib in libraries:
        lib_data = df_subset[df_subset['Library'] == lib]
        if len(lib_data) > 0:
            val = lib_data[metric].mean()
            # For log scale, replace 0 or very small values with a minimum threshold
            if use_log_scale and val <= 0:
                val = 0.001
            values.append(val)
        else:
            values.append(0.001 if use_log_scale else 0)

    # Create color list using unique colors for each library variant
    colors = [get_library_color(lib, libraries) for lib in libraries]

    # Plot bars
    ax.bar(x_positions, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize chart
    ax.set_xlabel('Compression Library', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(libraries, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    if use_log_scale:
        ax.set_yscale('log')

def get_distribution_description(distribution):
    """
    Generate a descriptive subtitle for a distribution explaining the parameters.

    Args:
        distribution: Distribution name (e.g., 'gamma_high', 'exponential_medium')

    Returns:
        String with parameter description
    """
    # Uniform distributions - varying max value
    if distribution.startswith('uniform_'):
        if distribution == 'uniform_float':
            return "Float data: Uniform distribution [0.0, 1000.0]"
        max_val = distribution.split('_')[1]
        return f"Max value = {max_val} (controls entropy/bit usage)"

    # Normal distributions - varying standard deviation
    if distribution.startswith('normal_'):
        if distribution == 'normal_float':
            return "Float data: Normal distribution (μ=500, σ=200)"
        stddev = distribution.split('_')[1]
        return f"Standard deviation σ = {stddev} (controls clustering)"

    # Gamma distributions - varying shape/scale parameters
    if distribution.startswith('gamma_'):
        param_map = {
            'incomp': "Gamma(α=5, β=5) × 5 + noise: Wide spread, high entropy",
            'light': "Gamma(α=5, β=8) × 4: Moderate spread, some clustering",
            'medium': "Gamma(α=2, β=4) × 15: Medium clustering",
            'high': "Gamma(α=1, β=2) × 20: Tight clustering at low values"
        }
        suffix = distribution.split('_')[1]
        return param_map.get(suffix, "Gamma distribution")

    # Exponential distributions - varying rate parameter
    if distribution.startswith('exponential_'):
        param_map = {
            'incomp': "Exponential(λ=0.01) × 1.5 + noise: Slow decay, high entropy",
            'light': "Exponential(λ=0.012) × 2.5 + 10: Slow decay, wide spread",
            'medium': "Exponential(λ=0.02) × 3.0 + 5: Moderate decay",
            'high': "Exponential(λ=0.05) × 2.0: Fast decay, clustering near zero"
        }
        suffix = distribution.split('_')[1]
        return param_map.get(suffix, "Exponential distribution")

    # Repeating pattern
    if distribution == 'repeating':
        return "Deterministic pattern (AAABBBCCC...): Extremely compressible"

    return ""

def create_cpu_util_impact_page(df_subset, distribution, libraries):
    """
    Create a page showing CPU utilization impact for a specific distribution.
    Shows line charts of how compression metrics change with CPU load.

    Args:
        df_subset: DataFrame subset for this distribution
        distribution: Distribution name
        libraries: List of unique libraries (top 10 will be plotted)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get descriptive subtitle
    description = get_distribution_description(distribution)
    data_type = "Float Data Type" if distribution.endswith('_float') else "Char Data Type"

    # Create title
    if description:
        title = f'CPU Utilization Impact: {distribution}\n{description}\n{data_type}, 64KB Chunk Size'
    else:
        title = f'CPU Utilization Impact: {distribution}\n{data_type}, 64KB Chunk Size'

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    # 1. Compression Time vs CPU Util (log scale)
    create_cpu_util_line_chart(
        axes[0, 0], df_subset, libraries,
        'Compress Time (ms)', 'Time (ms, log scale)',
        'Compression Time vs CPU Utilization', use_log_scale=True
    )

    # 2. Decompression Time vs CPU Util (log scale)
    create_cpu_util_line_chart(
        axes[0, 1], df_subset, libraries,
        'Decompress Time (ms)', 'Time (ms, log scale)',
        'Decompression Time vs CPU Utilization', use_log_scale=True
    )

    # 3. Compression Ratio vs CPU Util (should be constant, no log scale)
    create_cpu_util_line_chart(
        axes[1, 0], df_subset, libraries,
        'Compression Ratio', 'Ratio (higher = better)',
        'Compression Ratio vs CPU Utilization', use_log_scale=False
    )

    # 4. Hide the last subplot (not needed)
    axes[1, 1].axis('off')

    # Add note about data statistics being constant
    axes[1, 1].text(0.5, 0.5,
                    'Data Statistics (Shannon Entropy, MAD, Second Derivative)\n'
                    'are constant per distribution and do not vary with CPU utilization.\n\n'
                    'These statistics are included in the CSV output for\n'
                    'training the dynamic compression selection model.',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig

def create_page(df_subset, distribution, libraries):
    """
    Create a page with bar charts for a specific distribution (64KB fixed size).
    For lossy compressors (with SNR data), creates 4x2 grid with quality metrics.
    For lossless compressors, creates 3x2 grid.

    If Target CPU Util column exists, only use data at 0% CPU utilization for comparison.

    Args:
        df_subset: DataFrame subset for this distribution
        distribution: Distribution name
        libraries: List of unique libraries
    """
    # Filter to 0% CPU utilization if the column exists
    if 'Target CPU Util (%)' in df_subset.columns:
        df_subset = df_subset[df_subset['Target CPU Util (%)'] == 0.0]
    # Check if this distribution has SNR data (lossy compressors)
    has_quality_metrics = 'SNR (dB)' in df_subset.columns and df_subset['SNR (dB)'].notna().any()

    if has_quality_metrics:
        # Extended layout for lossy compressors with quality metrics
        fig, axes = plt.subplots(4, 2, figsize=(11, 17))
    else:
        # Standard layout for lossless compressors
        fig, axes = plt.subplots(3, 2, figsize=(11, 14))

    # Get descriptive subtitle with parameters
    description = get_distribution_description(distribution)

    # Determine data type
    data_type = "Float Data Type" if distribution.endswith('_float') else "Char Data Type"

    # Create multi-line title
    if description:
        title = f'Parameter Study: {distribution}\n{description}\n{data_type}, 64KB Chunk Size'
    else:
        title = f'Parameter Study: {distribution}\n{data_type}, 64KB Chunk Size'

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    # 1. Compress Time (log scale)
    create_simple_bar_chart(
        axes[0, 0], df_subset, libraries,
        'Compress Time (ms)', 'Time (ms, log scale)',
        'Compression Time', use_log_scale=True
    )

    # 2. Decompress Time (log scale)
    create_simple_bar_chart(
        axes[0, 1], df_subset, libraries,
        'Decompress Time (ms)', 'Time (ms, log scale)',
        'Decompression Time', use_log_scale=True
    )

    # 3. Compression Ratio (log scale)
    create_simple_bar_chart(
        axes[1, 0], df_subset, libraries,
        'Compression Ratio', 'Ratio (log scale, higher = better)',
        'Compression Ratio', use_log_scale=True
    )

    # 4. Compress CPU %
    create_simple_bar_chart(
        axes[1, 1], df_subset, libraries,
        'Compress CPU %', 'CPU Utilization (%)',
        'Compression CPU Usage'
    )

    # 5. Decompress CPU %
    create_simple_bar_chart(
        axes[2, 0], df_subset, libraries,
        'Decompress CPU %', 'CPU Utilization (%)',
        'Decompression CPU Usage'
    )

    # 6. Summary statistics table
    axes[2, 1].axis('off')

    # Sort by compression ratio descending
    df_sorted = df_subset.sort_values('Compression Ratio', ascending=False)

    table_data = []
    for idx, row in df_sorted.iterrows():
        table_data.append([
            row['Library'],
            f"{row['Compression Ratio']:.2f}x",
            f"{row['Compress Time (ms)']:.1f}",
            f"{row['Compress CPU %']:.0f}%"
        ])

    table = axes[2, 1].table(
        cellText=table_data[:9],  # Top 9 libraries
        colLabels=['Library', 'Ratio', 'Time (ms)', 'CPU%'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color best result (first row)
    for i in range(4):
        table[(1, i)].set_facecolor('#E7E6E6')

    axes[2, 1].set_title('Performance Summary (64KB chunks)\nSorted by Compression Ratio',
                        fontweight='bold', pad=10)

    # Add quality metric plots for lossy compressors
    if has_quality_metrics:
        # 7. SNR (Signal-to-Noise Ratio)
        create_simple_bar_chart(
            axes[3, 0], df_subset, libraries,
            'SNR (dB)', 'SNR (dB, higher = better)',
            'Signal-to-Noise Ratio (SNR)', use_log_scale=False
        )

        # 8. PSNR (Peak Signal-to-Noise Ratio)
        create_simple_bar_chart(
            axes[3, 1], df_subset, libraries,
            'PSNR (dB)', 'PSNR (dB, higher = better)',
            'Peak Signal-to-Noise Ratio (PSNR)', use_log_scale=False
        )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig

def generate_statistics_report(df, output_file):
    """Generate detailed text statistics report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPRESSION PARAMETER STUDY - DETAILED STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        distributions = sorted(df['Distribution'].unique())
        libraries = sorted(df['Library'].unique())

        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Distributions: {len(distributions)}\n")
        f.write(f"Libraries: {len(libraries)}\n")
        f.write(f"Chunk Sizes: {len(df['Chunk Size (MB)'].unique())}\n\n")

        # Group distributions by type
        uniform_dists = sorted([d for d in distributions if d.startswith('uniform_')])
        normal_dists = sorted([d for d in distributions if d.startswith('normal_')])
        other_dists = sorted([d for d in distributions if not d.startswith('uniform_') and not d.startswith('normal_')])

        # Uniform distribution analysis
        if uniform_dists:
            f.write("=" * 80 + "\n")
            f.write("UNIFORM DISTRIBUTION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Parameter: Max value (controls entropy)\n")
            f.write("  Lower max value = lower entropy = better compression\n\n")

            for lib in libraries:
                f.write(f"{lib}:\n")
                f.write(f"  {'Distribution':<15} {'Max Val':<8} {'Avg Ratio':<12} {'Avg Time (ms)':<15} {'Avg CPU%':<10}\n")
                f.write(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*15} {'-'*10}\n")

                for dist in uniform_dists:
                    subset = df[(df['Library'] == lib) & (df['Distribution'] == dist)]
                    if len(subset) > 0:
                        max_val = dist.split('_')[1]
                        avg_ratio = subset['Compression Ratio'].mean()
                        avg_time = subset['Compress Time (ms)'].mean()
                        avg_cpu = subset['Compress CPU %'].mean()
                        f.write(f"  {dist:<15} {max_val:<8} {avg_ratio:<12.2f} {avg_time:<15.1f} {avg_cpu:<10.0f}\n")
                f.write("\n")

        # Normal distribution analysis
        if normal_dists:
            f.write("=" * 80 + "\n")
            f.write("NORMAL DISTRIBUTION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Parameter: Standard deviation (controls clustering)\n")
            f.write("  Lower stddev = tighter clustering = better compression\n\n")

            for lib in libraries:
                f.write(f"{lib}:\n")
                f.write(f"  {'Distribution':<15} {'StdDev':<8} {'Avg Ratio':<12} {'Avg Time (ms)':<15} {'Avg CPU%':<10}\n")
                f.write(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*15} {'-'*10}\n")

                # Sort normal distributions: numeric ones first by value, then 'float'
                def sort_key(x):
                    suffix = x.split('_')[1]
                    try:
                        return (0, int(suffix))  # Numeric distributions first
                    except ValueError:
                        return (1, suffix)  # 'float' comes last

                for dist in sorted(normal_dists, key=sort_key):
                    subset = df[(df['Library'] == lib) & (df['Distribution'] == dist)]
                    if len(subset) > 0:
                        stddev = dist.split('_')[1]
                        avg_ratio = subset['Compression Ratio'].mean()
                        avg_time = subset['Compress Time (ms)'].mean()
                        avg_cpu = subset['Compress CPU %'].mean()
                        f.write(f"  {dist:<15} {stddev:<8} {avg_ratio:<12.2f} {avg_time:<15.1f} {avg_cpu:<10.0f}\n")
                f.write("\n")

        # Best compressor analysis
        f.write("=" * 80 + "\n")
        f.write("BEST COMPRESSOR BY DISTRIBUTION (64KB chunks)\n")
        f.write("=" * 80 + "\n\n")

        # Use the chunk size that exists in the data (64KB = 0.0625 MB)
        chunk_size_mb = df['Chunk Size (MB)'].unique()[0]
        df_chunk = df[df['Chunk Size (MB)'] == chunk_size_mb]
        f.write(f"{'Distribution':<20} {'Best Library':<15} {'Ratio':<10} {'Time (ms)':<12}\n")
        f.write(f"{'-'*20} {'-'*15} {'-'*10} {'-'*12}\n")

        for dist in distributions:
            dist_data = df_chunk[df_chunk['Distribution'] == dist]
            if len(dist_data) > 0:
                best = dist_data.loc[dist_data['Compression Ratio'].idxmax()]
                f.write(f"{dist:<20} {best['Library']:<15} {best['Compression Ratio']:<10.2f} {best['Compress Time (ms)']:<12.1f}\n")

        # Parameter effect summary
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        if uniform_dists:
            f.write("1. ENTROPY EFFECT (Uniform Distribution):\n")
            # Compare uniform_15 vs uniform_255 for each library
            chunk_size_mb = df['Chunk Size (MB)'].unique()[0]
            for lib in libraries[:3]:  # Top 3
                u15 = df[(df['Library'] == lib) & (df['Distribution'] == 'uniform_15') & (df['Chunk Size (MB)'] == chunk_size_mb)]
                u255 = df[(df['Library'] == lib) & (df['Distribution'] == 'uniform_255') & (df['Chunk Size (MB)'] == chunk_size_mb)]
                if len(u15) > 0 and len(u255) > 0:
                    ratio_15 = u15['Compression Ratio'].iloc[0]
                    ratio_255 = u255['Compression Ratio'].iloc[0]
                    improvement = ((ratio_15 / ratio_255) - 1) * 100
                    f.write(f"   {lib}: uniform_15 compresses {improvement:.0f}% better than uniform_255\n")
            f.write("\n")

        if normal_dists:
            f.write("2. CLUSTERING EFFECT (Normal Distribution):\n")
            # Compare normal_10 vs normal_80 for each library
            chunk_size_mb = df['Chunk Size (MB)'].unique()[0]
            for lib in libraries[:3]:  # Top 3
                n10 = df[(df['Library'] == lib) & (df['Distribution'] == 'normal_10') & (df['Chunk Size (MB)'] == chunk_size_mb)]
                n80 = df[(df['Library'] == lib) & (df['Distribution'] == 'normal_80') & (df['Chunk Size (MB)'] == chunk_size_mb)]
                if len(n10) > 0 and len(n80) > 0:
                    ratio_10 = n10['Compression Ratio'].iloc[0]
                    ratio_80 = n80['Compression Ratio'].iloc[0]
                    improvement = ((ratio_10 / ratio_80) - 1) * 100
                    f.write(f"   {lib}: normal_10 compresses {improvement:.0f}% better than normal_80\n")
            f.write("\n")

        f.write("3. IMPLICATIONS FOR REAL-WORLD DATA:\n")
        f.write("   - Lower entropy data (limited value range) compresses much better\n")
        f.write("   - Clustered data (low variance) compresses much better\n")
        f.write("   - Random noise (uniform_255, normal_80) is nearly incompressible\n")
        f.write("   - Many scientific datasets have structure and compress well\n\n")

def generate_best_compressor_table(df, output_file):
    """Generate table showing best compressor for each distribution (64KB fixed)."""
    with open(output_file, 'w') as f:
        f.write("BEST COMPRESSOR BY DISTRIBUTION (64KB Chunk Size)\n")
        f.write("=" * 80 + "\n\n")

        distributions = sorted(df['Distribution'].unique())

        f.write(f"{'Distribution':<25} {'Best Library':<15} {'Ratio':<10} {'Time (ms)':<12} {'CPU %':<8}\n")
        f.write("-" * 80 + "\n")

        for dist in distributions:
            subset = df[df['Distribution'] == dist]
            if len(subset) > 0:
                best = subset.loc[subset['Compression Ratio'].idxmax()]
                f.write(f"{dist:<25} {best['Library']:<15} {best['Compression Ratio']:<10.2f} "
                       f"{best['Compress Time (ms)']:<12.1f} {best['Compress CPU %']:<8.0f}\n")

def main():
    """Main function to generate parameter study report."""
    print("=" * 80)
    print("COMPRESSION PARAMETER STUDY VISUALIZATION")
    print("=" * 80)
    print()

    # Find CSV files (both lossless and lossy)
    build_dir = Path(__file__).parent.parent.parent.parent.parent / 'build'
    lossless_csv_path = build_dir / 'compression_parameter_study_results.csv'
    lossy_csv_path = build_dir / 'compression_lossy_parameter_study_results.csv'

    # Check if at least one file exists
    if not lossless_csv_path.exists() and not lossy_csv_path.exists():
        print(f"✗ Error: No results files found")
        print(f"  Expected lossless: {lossless_csv_path}")
        print(f"  Expected lossy: {lossy_csv_path}")
        print("\nPlease run the parameter study benchmarks first:")
        print("  cd /workspace/build")
        print("  ./bin/benchmark_compress_parameter_study_exec")
        print("  ./bin/benchmark_compress_lossy_parameter_study_exec")
        sys.exit(1)

    # Load data separately (DO NOT combine - they test different data types)
    df_lossless = None
    df_lossy = None

    if lossless_csv_path.exists():
        print(f"Loading lossless data from: {lossless_csv_path}")
        df_lossless = load_data(lossless_csv_path)
        print(f"  ✓ Loaded {len(df_lossless)} lossless compression results (char data)")

    if lossy_csv_path.exists():
        print(f"Loading lossy data from: {lossy_csv_path}")
        df_lossy = load_data(lossy_csv_path)
        print(f"  ✓ Loaded {len(df_lossy)} lossy compression results (float data)")

    print()

    # Identify lossy compressors and float distributions
    lossy_compressors = ['LibPressio-ZFP', 'LibPressio-BitGrooming']
    float_distributions = ['uniform_float', 'normal_float']

    # Get unique values for each type
    lossless_distributions = sorted(df_lossless['Distribution'].unique()) if df_lossless is not None else []
    lossy_distributions = sorted(df_lossy['Distribution'].unique()) if df_lossy is not None else []

    # Separate distributions: lossless first, lossy at the end
    all_distributions = lossless_distributions + lossy_distributions

    lossless_libraries = sorted(df_lossless['Library'].unique()) if df_lossless is not None else []
    lossy_libraries = sorted(df_lossy['Library'].unique()) if df_lossy is not None else []

    print(f"✓ Found {len(lossless_distributions)} lossless distributions (char data)")
    print(f"✓ Found {len(lossy_distributions)} lossy distributions (float data)")
    print(f"✓ Found {len(lossless_libraries)} lossless compressors")
    print(f"✓ Found {len(lossy_libraries)} lossy compressors")
    print()

    # Check if we have CPU utilization data
    has_cpu_util = False
    if df_lossless is not None and 'Target CPU Util (%)' in df_lossless.columns:
        has_cpu_util = True
    elif df_lossy is not None and 'Target CPU Util (%)' in df_lossy.columns:
        has_cpu_util = True

    # Generate PDF report with lossless pages first, lossy pages at the end
    pdf_path = OUTPUT_DIR / 'parameter_study_full_report.pdf'
    print(f"Generating PDF report: {pdf_path}")
    print(f"  Layout: {len(lossless_distributions)} lossless pages, then {len(lossy_distributions)} lossy pages")
    if has_cpu_util:
        print(f"  Note: Bar charts use 0% CPU utilization data for comparison")
    print()

    with PdfPages(pdf_path) as pdf:
        page_num = 1

        # First, generate all lossless distribution pages
        for dist in lossless_distributions:
            print(f"  Page {page_num}/{len(all_distributions)}: {dist} (lossless)")

            if df_lossless is not None:
                df_dist = df_lossless[df_lossless['Distribution'] == dist]
                if len(df_dist) > 0:
                    fig = create_page(df_dist, dist, lossless_libraries)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    page_num += 1

        # Then, generate all lossy distribution pages at the end
        for dist in lossy_distributions:
            print(f"  Page {page_num}/{len(all_distributions)}: {dist} (lossy)")

            if df_lossy is not None:
                df_dist = df_lossy[df_lossy['Distribution'] == dist]
                if len(df_dist) > 0:
                    fig = create_page(df_dist, dist, lossy_libraries)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    page_num += 1

    print()
    print(f"✓ PDF report saved: {pdf_path.name} ({len(all_distributions)} pages)")
    print(f"  Pages 1-{len(lossless_distributions)}: Lossless compressors (char data)")
    print(f"  Pages {len(lossless_distributions)+1}-{len(all_distributions)}: Lossy compressors (float data)")
    print()

    # Generate CPU utilization impact report if we have the data
    if has_cpu_util:
        pdf_cpu_path = OUTPUT_DIR / 'cpu_utilization_impact_report.pdf'
        print(f"Generating CPU utilization impact report: {pdf_cpu_path}")
        print()

        with PdfPages(pdf_cpu_path) as pdf:
            page_num = 1

            # Generate CPU util impact pages for lossless distributions
            for dist in lossless_distributions:
                print(f"  Page {page_num}: {dist} (lossless CPU impact)")

                if df_lossless is not None:
                    df_dist = df_lossless[df_lossless['Distribution'] == dist]
                    if len(df_dist) > 0:
                        fig = create_cpu_util_impact_page(df_dist, dist, lossless_libraries)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        page_num += 1

            # Generate CPU util impact pages for lossy distributions
            for dist in lossy_distributions:
                print(f"  Page {page_num}: {dist} (lossy CPU impact)")

                if df_lossy is not None:
                    df_dist = df_lossy[df_lossy['Distribution'] == dist]
                    if len(df_dist) > 0:
                        fig = create_cpu_util_impact_page(df_dist, dist, lossy_libraries)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        page_num += 1

        print()
        print(f"✓ CPU utilization impact report saved: {pdf_cpu_path.name}")
        print(f"  Shows how compression performance changes with CPU load (0%-100%)")
        print()

    # Combine data for statistics (keep separate context)
    df_combined = pd.concat([df for df in [df_lossless, df_lossy] if df is not None], ignore_index=True)

    # Generate statistics report
    stats_path = OUTPUT_DIR / 'parameter_study_statistics.txt'
    print(f"Generating statistics report: {stats_path}")
    generate_statistics_report(df_combined, stats_path)
    print(f"✓ Statistics saved: {stats_path.name}")
    print()

    # Generate best compressor table
    table_path = OUTPUT_DIR / 'parameter_study_best_compressor.txt'
    print(f"Generating best compressor table: {table_path}")
    generate_best_compressor_table(df_combined, table_path)
    print(f"✓ Table saved: {table_path.name}")
    print()

    print("=" * 80)
    print("✓ REPORT GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  • {pdf_path.name} - {len(all_distributions)}-page detailed report (0% CPU baseline)")
    if has_cpu_util:
        print(f"  • cpu_utilization_impact_report.pdf - CPU load impact analysis")
    print(f"  • {stats_path.name} - Detailed statistics and analysis")
    print(f"  • {table_path.name} - Best compressor lookup table")
    print()
    if has_cpu_util:
        print("New metrics collected:")
        print("  • Target CPU Utilization (0%, 25%, 50%, 75%, 100%)")
        print("  • Shannon Entropy (bits per byte)")
        print("  • Data Variance")
        print("  • Second Derivative Mean (curvature)")
        print()

if __name__ == '__main__':
    main()
