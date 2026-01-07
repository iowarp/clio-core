#!/usr/bin/env python3
"""
Lossy Compression Benchmark Results Visualization

Generates comprehensive PDF reports with charts showing:
- Compression ratio
- Compression/decompression time
- CPU utilization
- Quality metrics (SNR, PSNR, MSE, Max Error)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from pathlib import Path
import numpy as np

# Output directory for plots
OUTPUT_DIR = Path(__file__).parent / 'benchmark_plots'
OUTPUT_DIR.mkdir(exist_ok=True)

# Color scheme for compressors
COLORS = {
    'zfp': '#1f77b4',
    'sz3': '#ff7f0e',
    'fpzip': '#2ca02c',
}

def load_data(csv_path):
    """Load benchmark results from CSV file."""
    df = pd.read_csv(csv_path)

    # Filter successful results only
    df = df[df['Success'] == 'YES'].copy()

    # Convert chunk size to KB for better readability
    df['Chunk Size (KB)'] = df['Chunk Size (bytes)'] / 1024

    # Replace 999.00 (lossless indicator) with NaN for quality metrics
    for col in ['SNR (dB)', 'PSNR (dB)', 'Max Error', 'MSE']:
        df[col] = df[col].replace(999.0, np.nan)

    return df

def format_chunk_size(kb):
    """Format chunk size for display."""
    if kb >= 1024:
        return f'{int(kb/1024)}M'
    return f'{int(kb)}K'

def create_line_chart(ax, df, compressors, distributions, metric_col, ylabel, title,
                     use_log_scale=False, invert_y=False):
    """Create a line chart with markers for a specific metric."""

    # Get unique chunk sizes and sort them
    chunk_sizes = sorted(df['Chunk Size (KB)'].unique())

    # Create x-axis positions
    x_pos = np.arange(len(chunk_sizes))

    # Plot each compressor-distribution combination
    for comp in compressors:
        for dist in distributions:
            subset = df[(df['Compressor'] == comp) & (df['Distribution'] == dist)]
            if len(subset) == 0:
                continue

            # Sort by chunk size
            subset = subset.sort_values('Chunk Size (KB)')

            # Get chunk sizes and metric values, ensuring they match
            subset_chunk_sizes = subset['Chunk Size (KB)'].values
            values = subset[metric_col].values

            # Skip if all values are NaN
            if np.all(np.isnan(values)):
                continue

            # Map chunk sizes to x-axis positions
            x_plot = []
            y_plot = []
            for i, cs in enumerate(chunk_sizes):
                # Find matching chunk size in subset
                idx = np.where(subset_chunk_sizes == cs)[0]
                if len(idx) > 0:
                    x_plot.append(i)
                    y_plot.append(values[idx[0]])

            if len(x_plot) == 0:
                continue

            # Create label
            label = f'{comp} - {dist}'

            # Plot line with markers
            color = COLORS.get(comp, '#808080')
            linestyle_map = {
                'uniform': '-',
                'normal': '--',
                'gamma': '-.',
                'exponential': ':',
                'repeating': '--',
                'sinusoidal': '-.',
                'gradient': ':'
            }
            linestyle = linestyle_map.get(dist, '-')
            ax.plot(x_plot, y_plot, marker='o', label=label, color=color,
                   linestyle=linestyle, linewidth=1.5, markersize=4)

    # Configure axes
    ax.set_xlabel('Chunk Size', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([format_chunk_size(kb) for kb in chunk_sizes])
    ax.grid(True, alpha=0.3, linestyle='--')

    if use_log_scale:
        ax.set_yscale('log')

    if invert_y:
        ax.invert_yaxis()

    # Add legend with smaller font
    ax.legend(fontsize=7, loc='best', ncol=2)

def create_bar_chart(ax, df, compressors, metric_col, ylabel, title, use_log_scale=False):
    """Create a grouped bar chart comparing compressors across distributions."""

    distributions = sorted(df['Distribution'].unique())
    x_pos = np.arange(len(distributions))
    width = 0.25

    for idx, comp in enumerate(compressors):
        subset = df[df['Compressor'] == comp]
        values = []

        for dist in distributions:
            dist_data = subset[subset['Distribution'] == dist]
            if len(dist_data) == 0:
                values.append(0)
            else:
                # Calculate mean, ignoring NaN
                mean_val = dist_data[metric_col].mean()
                values.append(mean_val if not np.isnan(mean_val) else 0)

        offset = (idx - len(compressors)/2 + 0.5) * width
        ax.bar(x_pos + offset, values, width, label=comp, color=COLORS.get(comp, '#808080'))

    ax.set_xlabel('Data Distribution', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(distributions, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(fontsize=8)

    if use_log_scale:
        ax.set_yscale('log')

def generate_report(csv_path):
    """Generate comprehensive PDF report with visualizations."""

    print("=" * 80)
    print("LOSSY COMPRESSION BENCHMARK VISUALIZATION")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    df = load_data(csv_path)
    print(f"✓ Loaded {len(df)} successful benchmark records")

    if len(df) == 0:
        print("ERROR: No successful benchmark results found!")
        return

    # Get unique values
    compressors = sorted(df['Compressor'].unique())
    distributions = sorted(df['Distribution'].unique())

    print(f"✓ Compressors: {', '.join(compressors)}")
    print(f"✓ Distributions: {', '.join(distributions)}")
    print()

    # Generate statistics report
    print("Generating statistics summary...")
    stats_file = OUTPUT_DIR / 'lossy_statistics_report.txt'
    with open(stats_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LOSSY COMPRESSION BENCHMARK STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total successful records: {len(df)}\n")
        f.write(f"Compressors tested: {', '.join(compressors)}\n")
        f.write(f"Distributions tested: {', '.join(distributions)}\n")
        f.write(f"Chunk sizes tested: {len(df['Chunk Size (KB)'].unique())}\n")
        f.write("\n")

        # Per-compressor statistics
        for comp in compressors:
            comp_data = df[df['Compressor'] == comp]
            f.write("=" * 80 + "\n")
            f.write(f"{comp.upper()} STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            # Compression ratio
            f.write("Compression Ratio:\n")
            f.write(f"  Mean:   {comp_data['Compression Ratio'].mean():.2f}x\n")
            f.write(f"  Median: {comp_data['Compression Ratio'].median():.2f}x\n")
            f.write(f"  Min:    {comp_data['Compression Ratio'].min():.2f}x\n")
            f.write(f"  Max:    {comp_data['Compression Ratio'].max():.2f}x\n\n")

            # Timing
            f.write("Compression Time:\n")
            f.write(f"  Mean:   {comp_data['Compress Time (ms)'].mean():.3f} ms\n")
            f.write(f"  Median: {comp_data['Compress Time (ms)'].median():.3f} ms\n")
            f.write(f"  Min:    {comp_data['Compress Time (ms)'].min():.3f} ms\n")
            f.write(f"  Max:    {comp_data['Compress Time (ms)'].max():.3f} ms\n\n")

            f.write("Decompression Time:\n")
            f.write(f"  Mean:   {comp_data['Decompress Time (ms)'].mean():.3f} ms\n")
            f.write(f"  Median: {comp_data['Decompress Time (ms)'].median():.3f} ms\n")
            f.write(f"  Min:    {comp_data['Decompress Time (ms)'].min():.3f} ms\n")
            f.write(f"  Max:    {comp_data['Decompress Time (ms)'].max():.3f} ms\n\n")

            # Quality metrics (excluding NaN/lossless)
            snr_data = comp_data['SNR (dB)'].dropna()
            if len(snr_data) > 0:
                f.write("Signal-to-Noise Ratio (SNR):\n")
                f.write(f"  Mean:   {snr_data.mean():.2f} dB\n")
                f.write(f"  Median: {snr_data.median():.2f} dB\n")
                f.write(f"  Min:    {snr_data.min():.2f} dB\n")
                f.write(f"  Max:    {snr_data.max():.2f} dB\n\n")

            psnr_data = comp_data['PSNR (dB)'].dropna()
            if len(psnr_data) > 0:
                f.write("Peak Signal-to-Noise Ratio (PSNR):\n")
                f.write(f"  Mean:   {psnr_data.mean():.2f} dB\n")
                f.write(f"  Median: {psnr_data.median():.2f} dB\n")
                f.write(f"  Min:    {psnr_data.min():.2f} dB\n")
                f.write(f"  Max:    {psnr_data.max():.2f} dB\n\n")

            mse_data = comp_data['MSE'].dropna()
            if len(mse_data) > 0:
                f.write("Mean Squared Error (MSE):\n")
                f.write(f"  Mean:   {mse_data.mean():.3e}\n")
                f.write(f"  Median: {mse_data.median():.3e}\n")
                f.write(f"  Min:    {mse_data.min():.3e}\n")
                f.write(f"  Max:    {mse_data.max():.3e}\n\n")

            max_err_data = comp_data['Max Error'].dropna()
            if len(max_err_data) > 0:
                f.write("Maximum Absolute Error:\n")
                f.write(f"  Mean:   {max_err_data.mean():.3e}\n")
                f.write(f"  Median: {max_err_data.median():.3e}\n")
                f.write(f"  Min:    {max_err_data.min():.3e}\n")
                f.write(f"  Max:    {max_err_data.max():.3e}\n\n")

    print(f"✓ Saved: {stats_file.name}")
    print()

    # Generate PDF report
    print("Generating PDF visualization report...")
    pdf_file = OUTPUT_DIR / 'lossy_compression_full_report.pdf'

    with matplotlib.backends.backend_pdf.PdfPages(pdf_file) as pdf:
        # Page 1: Performance Metrics (Timing and Ratio)
        print("  Generating page 1: Performance Metrics")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Lossy Compression Performance Metrics', fontsize=14, fontweight='bold')

        create_line_chart(axes[0, 0], df, compressors, distributions,
                         'Compress Time (ms)', 'Time (ms, log scale)',
                         'Compression Time by Chunk Size', use_log_scale=True)

        create_line_chart(axes[0, 1], df, compressors, distributions,
                         'Decompress Time (ms)', 'Time (ms, log scale)',
                         'Decompression Time by Chunk Size', use_log_scale=True)

        create_line_chart(axes[1, 0], df, compressors, distributions,
                         'Compression Ratio', 'Compression Ratio (Higher = Better)',
                         'Compression Ratio by Chunk Size', use_log_scale=True)

        create_bar_chart(axes[1, 1], df, compressors,
                        'Compression Ratio', 'Average Compression Ratio',
                        'Average Compression Ratio by Distribution', use_log_scale=True)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: CPU Utilization
        print("  Generating page 2: CPU Utilization")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Lossy Compression CPU Utilization', fontsize=14, fontweight='bold')

        create_line_chart(axes[0, 0], df, compressors, distributions,
                         'Compress CPU %', 'CPU Utilization (%)',
                         'Compression CPU Usage by Chunk Size')

        create_line_chart(axes[0, 1], df, compressors, distributions,
                         'Decompress CPU %', 'CPU Utilization (%)',
                         'Decompression CPU Usage by Chunk Size')

        create_bar_chart(axes[1, 0], df, compressors,
                        'Compress CPU %', 'Average CPU %',
                        'Average Compression CPU by Distribution')

        create_bar_chart(axes[1, 1], df, compressors,
                        'Decompress CPU %', 'Average CPU %',
                        'Average Decompression CPU by Distribution')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: Quality Metrics - SNR and PSNR
        print("  Generating page 3: Quality Metrics (SNR/PSNR)")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Lossy Compression Quality Metrics: SNR and PSNR', fontsize=14, fontweight='bold')

        create_line_chart(axes[0, 0], df, compressors, distributions,
                         'SNR (dB)', 'SNR (dB, Higher = Better)',
                         'Signal-to-Noise Ratio by Chunk Size')

        create_line_chart(axes[0, 1], df, compressors, distributions,
                         'PSNR (dB)', 'PSNR (dB, Higher = Better)',
                         'Peak Signal-to-Noise Ratio by Chunk Size')

        create_bar_chart(axes[1, 0], df, compressors,
                        'SNR (dB)', 'Average SNR (dB)',
                        'Average SNR by Distribution')

        create_bar_chart(axes[1, 1], df, compressors,
                        'PSNR (dB)', 'Average PSNR (dB)',
                        'Average PSNR by Distribution')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 4: Quality Metrics - Error Measures
        print("  Generating page 4: Quality Metrics (Error Measures)")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Lossy Compression Quality Metrics: Error Measures', fontsize=14, fontweight='bold')

        create_line_chart(axes[0, 0], df, compressors, distributions,
                         'Max Error', 'Max Error (log scale, Lower = Better)',
                         'Maximum Absolute Error by Chunk Size', use_log_scale=True)

        create_line_chart(axes[0, 1], df, compressors, distributions,
                         'MSE', 'MSE (log scale, Lower = Better)',
                         'Mean Squared Error by Chunk Size', use_log_scale=True)

        create_bar_chart(axes[1, 0], df, compressors,
                        'Max Error', 'Average Max Error',
                        'Average Max Error by Distribution', use_log_scale=True)

        create_bar_chart(axes[1, 1], df, compressors,
                        'MSE', 'Average MSE',
                        'Average MSE by Distribution', use_log_scale=True)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"✓ Saved: {pdf_file.name}")
    print(f"  Total pages: 4")
    print()

    print("=" * 80)
    print("✓ Report generation complete!")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  • {pdf_file.name}")
    print(f"    → 4-page report with performance and quality metrics")
    print(f"  • {stats_file.name}")
    print(f"    → Summary statistics for all compressors")
    print()

if __name__ == '__main__':
    csv_path = Path(__file__).parent / 'lossy_compression_benchmark_results.csv'

    if not csv_path.exists():
        print(f"ERROR: Benchmark results not found at {csv_path}")
        print("Please run the lossy compression benchmark first.")
        exit(1)

    generate_report(csv_path)
