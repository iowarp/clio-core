#!/usr/bin/env python3
"""
Compression Benchmark Results - Detailed Library Statistics Report

Generates a comprehensive multi-page report with 5 figures per page for each
(distribution, data_type) combination. Each page contains combo charts showing:
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

# Color palette for libraries - using distinct colors
LIBRARY_COLORS = {
    'BZIP2': '#1f77b4',
    'Blosc2': '#ff7f0e',
    'Brotli': '#2ca02c',
    'LZ4': '#d62728',
    'LZO': '#9467bd',
    'Lzma': '#8c564b',
    'Snappy': '#e377c2',
    'Zlib': '#7f7f7f',
    'Zstd': '#bcbd22',
    'LibPressio_SZ3': '#17becf',
    'LibPressio_ZFP': '#e377c2',
    'LibPressio_MGARD': '#8c6d31'
}

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

        print(f"✓ Processed {len(df)} valid records")
        return df
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        sys.exit(1)

def create_combo_chart(ax, df_subset, libraries, chunk_sizes, metric, ylabel, title, use_log_scale=False):
    """
    Create a combo chart with bars for each library grouped by chunk size.

    Args:
        ax: Matplotlib axis
        df_subset: DataFrame subset for this distribution/data_type
        libraries: List of unique libraries
        chunk_sizes: List of unique chunk sizes
        metric: Column name for the metric to plot
        ylabel: Y-axis label
        title: Chart title
        use_log_scale: Whether to use log scale for y-axis
    """
    # Number of libraries and chunk sizes
    n_libs = len(libraries)
    n_chunks = len(chunk_sizes)

    # Set up bar positions
    bar_width = 0.8 / n_chunks  # Leave some space between library groups
    x_positions = np.arange(n_libs)

    # Plot bars for each chunk size
    for i, chunk_size in enumerate(chunk_sizes):
        chunk_data = df_subset[df_subset['Chunk Size (MB)'] == chunk_size]
        values = []

        for lib in libraries:
            lib_data = chunk_data[chunk_data['Library'] == lib]
            if len(lib_data) > 0:
                val = lib_data[metric].mean()
                # For log scale, replace 0 or very small values with a minimum threshold
                if use_log_scale and val <= 0:
                    val = 0.001  # Minimum threshold for log scale
                values.append(val)
            else:
                values.append(0.001 if use_log_scale else 0)

        # Calculate offset for this chunk size's bars
        offset = (i - n_chunks/2 + 0.5) * bar_width

        # Create label for chunk size
        if chunk_size < 1:
            chunk_label = f"{chunk_size*1024:.0f}KB"
        else:
            chunk_label = f"{chunk_size:.1f}MB"

        # Plot bars
        ax.bar(x_positions + offset, values, bar_width,
               label=chunk_label, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(libraries, rotation=45, ha='right')
    ax.set_xlabel('Compression Library', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(title='Chunk Size', loc='best', fontsize=7, title_fontsize=8)

    # Apply log scale if requested
    if use_log_scale:
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    else:
        ax.grid(axis='y', alpha=0.3)

def plot_page_for_combination(df, distribution, data_type, page_num, total_pages):
    """
    Create a single page with 5 combo charts for a specific (distribution, data_type) combination.

    Args:
        df: Full DataFrame
        distribution: Distribution type (e.g., 'uniform', 'gaussian')
        data_type: Data type (e.g., 'int32', 'float')
        page_num: Current page number
        total_pages: Total number of pages

    Returns:
        matplotlib Figure
    """
    # Filter data for this combination
    df_subset = df[(df['Distribution'] == distribution) & (df['Data Type'] == data_type)]

    if len(df_subset) == 0:
        return None

    # Get unique libraries and chunk sizes for this subset
    libraries = sorted(df_subset['Library'].unique())
    chunk_sizes = sorted(df_subset['Chunk Size (MB)'].unique())

    # Create figure with 5 subplots (5 rows, 1 column)
    fig = plt.figure(figsize=(11, 14))
    fig.suptitle(f'Distribution: {distribution.upper()} | Data Type: {data_type.upper()}\n'
                 f'Page {page_num}/{total_pages}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Create 5 subplots
    ax1 = plt.subplot(5, 1, 1)
    ax2 = plt.subplot(5, 1, 2)
    ax3 = plt.subplot(5, 1, 3)
    ax4 = plt.subplot(5, 1, 4)
    ax5 = plt.subplot(5, 1, 5)

    # 1. Compress Time (log scale)
    create_combo_chart(ax1, df_subset, libraries, chunk_sizes,
                      'Compress Time (ms)', 'Compress Time (ms, log scale)',
                      'Compression Time by Library and Chunk Size', use_log_scale=True)

    # 2. Decompress Time (log scale)
    create_combo_chart(ax2, df_subset, libraries, chunk_sizes,
                      'Decompress Time (ms)', 'Decompress Time (ms, log scale)',
                      'Decompression Time by Library and Chunk Size', use_log_scale=True)

    # 3. Compression Ratio
    create_combo_chart(ax3, df_subset, libraries, chunk_sizes,
                      'Compression Ratio', 'Compression Ratio (Higher = Better)',
                      'Compression Ratio by Library and Chunk Size', use_log_scale=False)

    # 4. Compress CPU %
    create_combo_chart(ax4, df_subset, libraries, chunk_sizes,
                      'Compress CPU %', 'CPU Utilization (%)',
                      'Compression CPU Usage by Library and Chunk Size', use_log_scale=False)

    # 5. Decompress CPU %
    create_combo_chart(ax5, df_subset, libraries, chunk_sizes,
                      'Decompress CPU %', 'CPU Utilization (%)',
                      'Decompression CPU Usage by Library and Chunk Size', use_log_scale=False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    return fig

def generate_full_report(df, output_dir):
    """
    Generate the full multi-page PDF report with one page per (distribution, data_type) combination.

    Args:
        df: Full DataFrame
        output_dir: Output directory path
    """
    # Get all unique combinations of distribution and data type
    combinations = []
    for dist in sorted(df['Distribution'].unique()):
        for dtype in sorted(df['Data Type'].unique()):
            subset = df[(df['Distribution'] == dist) & (df['Data Type'] == dtype)]
            if len(subset) > 0:
                combinations.append((dist, dtype))

    total_pages = len(combinations)
    print(f"✓ Found {total_pages} unique (distribution, data_type) combinations")
    print(f"  Distributions: {sorted(df['Distribution'].unique())}")
    print(f"  Data Types: {sorted(df['Data Type'].unique())}")
    print()

    # Create PDF with all pages
    pdf_path = output_dir / 'compression_library_statistics_full_report.pdf'

    with PdfPages(pdf_path) as pdf:
        for page_num, (dist, dtype) in enumerate(combinations, 1):
            print(f"  Generating page {page_num}/{total_pages}: {dist} / {dtype}")

            fig = plot_page_for_combination(df, dist, dtype, page_num, total_pages)

            if fig is not None:
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)

    print()
    print(f"✓ Saved: {pdf_path.name}")
    print(f"  Total pages: {total_pages}")
    print(f"  Figures per page: 5")
    print(f"  Total figures: {total_pages * 5}")

def generate_summary_statistics(df, output_dir):
    """Generate a text file with summary statistics."""
    stats_path = output_dir / 'statistics_report.txt'

    with open(stats_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPRESSION LIBRARY STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total benchmark records: {len(df)}\n")
        f.write(f"Unique libraries: {df['Library'].nunique()}\n")
        f.write(f"Unique distributions: {df['Distribution'].nunique()}\n")
        f.write(f"Unique data types: {df['Data Type'].nunique()}\n")
        f.write(f"Unique chunk sizes: {df['Chunk Size (bytes)'].nunique()}\n\n")

        f.write("Libraries tested:\n")
        for lib in sorted(df['Library'].unique()):
            count = len(df[df['Library'] == lib])
            f.write(f"  • {lib}: {count} records\n")
        f.write("\n")

        f.write("Distributions tested:\n")
        for dist in sorted(df['Distribution'].unique()):
            count = len(df[df['Distribution'] == dist])
            f.write(f"  • {dist}: {count} records\n")
        f.write("\n")

        f.write("Data types tested:\n")
        for dtype in sorted(df['Data Type'].unique()):
            count = len(df[df['Data Type'] == dtype])
            f.write(f"  • {dtype}: {count} records\n")
        f.write("\n")

        f.write("Chunk sizes tested:\n")
        for size in sorted(df['Chunk Size (bytes)'].unique()):
            if size < 1024**2:
                size_str = f"{size/1024:.0f} KB"
            else:
                size_str = f"{size/(1024**2):.1f} MB"
            count = len(df[df['Chunk Size (bytes)'] == size])
            f.write(f"  • {size_str}: {count} records\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("AGGREGATE STATISTICS BY LIBRARY\n")
        f.write("=" * 80 + "\n\n")

        # Per-library statistics
        library_stats = df.groupby('Library').agg({
            'Compression Ratio': ['mean', 'std', 'min', 'max'],
            'Compress Time (ms)': ['mean', 'std', 'min', 'max'],
            'Decompress Time (ms)': ['mean', 'std', 'min', 'max'],
            'Compress CPU %': ['mean', 'std'],
            'Decompress CPU %': ['mean', 'std']
        }).round(2)

        for lib in sorted(library_stats.index):
            f.write(f"{lib}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Compression Ratio:\n")
            f.write(f"    Mean: {library_stats.loc[lib, ('Compression Ratio', 'mean')]:.2f}x\n")
            f.write(f"    Std:  {library_stats.loc[lib, ('Compression Ratio', 'std')]:.2f}\n")
            f.write(f"    Min:  {library_stats.loc[lib, ('Compression Ratio', 'min')]:.2f}x\n")
            f.write(f"    Max:  {library_stats.loc[lib, ('Compression Ratio', 'max')]:.2f}x\n")
            f.write(f"  Compress Time:\n")
            f.write(f"    Mean: {library_stats.loc[lib, ('Compress Time (ms)', 'mean')]:.2f} ms\n")
            f.write(f"    Std:  {library_stats.loc[lib, ('Compress Time (ms)', 'std')]:.2f} ms\n")
            f.write(f"  Decompress Time:\n")
            f.write(f"    Mean: {library_stats.loc[lib, ('Decompress Time (ms)', 'mean')]:.2f} ms\n")
            f.write(f"    Std:  {library_stats.loc[lib, ('Decompress Time (ms)', 'std')]:.2f} ms\n")
            f.write(f"  CPU Usage:\n")
            f.write(f"    Compress:   {library_stats.loc[lib, ('Compress CPU %', 'mean')]:.1f}%\n")
            f.write(f"    Decompress: {library_stats.loc[lib, ('Decompress CPU %', 'mean')]:.1f}%\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("REPORT STRUCTURE\n")
        f.write("=" * 80 + "\n\n")

        combinations = []
        for dist in sorted(df['Distribution'].unique()):
            for dtype in sorted(df['Data Type'].unique()):
                subset = df[(df['Distribution'] == dist) & (df['Data Type'] == dtype)]
                if len(subset) > 0:
                    combinations.append((dist, dtype, len(subset)))

        f.write(f"Total pages in report: {len(combinations)}\n")
        f.write(f"Figures per page: 5\n")
        f.write(f"Total figures: {len(combinations) * 5}\n\n")

        f.write("Page breakdown:\n")
        for i, (dist, dtype, count) in enumerate(combinations, 1):
            f.write(f"  Page {i:2d}: {dist:12s} / {dtype:8s} ({count:4d} records)\n")

    print(f"✓ Saved: {stats_path.name}")

def main():
    """Main function to generate the report."""
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'compression_benchmark_results.csv'
    output_dir = script_dir / 'benchmark_plots'

    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("COMPRESSION LIBRARY STATISTICS REPORT GENERATOR")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    df = load_data(csv_path)
    print()

    # Generate summary statistics
    print("Generating summary statistics...")
    generate_summary_statistics(df, output_dir)
    print()

    # Generate full PDF report
    print("Generating full PDF report...")
    generate_full_report(df, output_dir)
    print()

    print("=" * 80)
    print(f"✓ Report generation complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • compression_library_statistics_full_report.pdf")
    print("    → Multi-page report with 5 combo charts per (distribution, data_type)")
    print("  • statistics_report.txt")
    print("    → Summary statistics and report structure")

if __name__ == '__main__':
    main()
