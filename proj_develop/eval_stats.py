import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def load_json(file_path):
    """Load JSON file into a DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def print_stats_for_subset(df, subset_name="All"):
    """Print statistics for a subset of the data."""
    print(f"\n{'='*80}")
    print(f"Statistics for: {subset_name}")
    print(f"{'='*80}")
    print(f"Number of samples: {len(df)}")

    # ========== Regeneration Stats ==========
    print("\n========== Structure Regeneration Stats ==========")
    if 'n_regenerations' in df.columns:
        print(f"Mean structure regenerations: {df['n_regenerations'].mean():.2f}")

    # ========== Stability Stats ==========
    print("\n========== Stability Stats ==========")
    if 'mean_stability_score' in df.columns:
        print(f"Mean stability score: {df['mean_stability_score'].mean():.6f}")
        print(f"Max stability score: {df['mean_stability_score'].max():.6f}")
        print(f"Min stability score: {df['mean_stability_score'].min():.6f}")

    if 'max_stability_score' in df.columns:
        print(f"Mean of maximum stability scores: {df['max_stability_score'].mean():.6f}")
        print(f"Max of maximum stability scores: {df['max_stability_score'].max():.6f}")


def plot_total_rejections_by_structure(df, suffix, output_dir):
    """
    Plot mean number of regenerations by structure length bins.

    Args:
        df: DataFrame containing the evaluation results
        suffix: Suffix for the output filename (e.g., 'mixed_dataset', 'short_dataset', 'long_dataset')
        output_dir: Directory to save the plot
    """
    if 'n_bricks' not in df.columns or 'n_regenerations' not in df.columns:
        print(f"Missing required columns for regenerations by structure chart")
        return

    # Define brick count bins
    bins = [(0, 50), (51, 100), (101, 150), (151, 200), (201, float('inf'))]
    bin_labels = ['0-50', '51-100', '101-150', '151-200', '201+']

    # Calculate mean regenerations and counts for each bin
    mean_regenerations = []
    bin_counts = []
    for (min_bricks, max_bricks), label in zip(bins, bin_labels):
        # Filter data for this brick bin
        if max_bricks == float('inf'):
            bin_df = df[df['n_bricks'] >= min_bricks]
        else:
            bin_df = df[(df['n_bricks'] >= min_bricks) & (df['n_bricks'] <= max_bricks)]

        # Store count of structures in this bin
        bin_counts.append(len(bin_df))

        # Calculate mean regenerations in this bin
        if len(bin_df) > 0:
            mean_regen = bin_df['n_regenerations'].mean()
        else:
            mean_regen = 0
        mean_regenerations.append(mean_regen)

    # Create bar chart with improved styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a color gradient for better visual appeal
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bin_labels)))
    bars = ax.bar(bin_labels, mean_regenerations, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5, width=0.7)

    # Add count labels on top of each bar
    for bar, count in zip(bars, bin_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'n={count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(f'Mean Regenerations by Structure Length ({suffix.replace("_", " ").title()})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Structure Length (Number of Bricks)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Number of Regenerations', fontsize=12, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.3, axis='y', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'mean_regenerations_by_structure_{suffix}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved mean regenerations by structure chart to {output_path}")


def plot_structure_regeneration_frequency(df, suffix, output_dir):
    """
    Plot histogram of structure regeneration frequency.

    Args:
        df: DataFrame containing the evaluation results
        suffix: Suffix for the output filename (e.g., 'mixed_dataset', 'short_dataset', 'long_dataset')
        output_dir: Directory to save the plot
    """
    if 'n_regenerations' not in df.columns:
        print(f"Missing n_regenerations column for structure regeneration frequency histogram")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    regenerations = df['n_regenerations']
    max_regen = int(regenerations.max())
    min_regen = int(regenerations.min())

    # Create histogram with bins from 0 to max
    if max_regen > 0:
        bins = range(0, max_regen + 2)
    else:
        bins = 20

    # Calculate histogram data
    counts, bin_edges = np.histogram(regenerations, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Use a single non-blue color (teal/green)
    bar_color = '#2E8B57'  # SeaGreen
    
    # Create histogram bars with single color
    bars = ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], 
                  color=bar_color, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_title(f'Structure Regeneration Frequency ({suffix.replace("_", " ").title()})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Structure Regenerations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'structure_regeneration_frequency_{suffix}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved structure regeneration frequency histogram to {output_path}")


def analyze_stats(input_path, output_dir=None):
    """
    Analyze evaluation statistics from a JSON file.

    Args:
        input_path: Path to the JSON file containing evaluation results
        output_dir: Directory to save outputs (defaults to same directory as input file)
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    # Load file
    print(f"\nLoading evaluation results from {input_path}...")
    df = load_json(input_path)
    print(f"Loaded {len(df)} evaluation records")

    # Check if caption_type column exists
    has_caption_types = 'caption_type' in df.columns

    if has_caption_types:
        # Analyze overall statistics (mixed dataset)
        print_stats_for_subset(df, "Mixed Dataset (All)")
        plot_total_rejections_by_structure(df, "mixed_dataset", output_dir)
        plot_structure_regeneration_frequency(df, "mixed_dataset", output_dir)

        # Analyze statistics per caption type
        if 'short' in df['caption_type'].values:
            df_short = df[df['caption_type'] == 'short']
            print_stats_for_subset(df_short, "Short Prompts")
            plot_total_rejections_by_structure(df_short, "short_dataset", output_dir)
            plot_structure_regeneration_frequency(df_short, "short_dataset", output_dir)

        if 'long' in df['caption_type'].values:
            df_long = df[df['caption_type'] == 'long']
            print_stats_for_subset(df_long, "Long Prompts")
            plot_total_rejections_by_structure(df_long, "long_dataset", output_dir)
            plot_structure_regeneration_frequency(df_long, "long_dataset", output_dir)
    else:
        # No caption types - treat all as mixed dataset
        print_stats_for_subset(df, "Mixed Dataset")
        plot_total_rejections_by_structure(df, "mixed_dataset", output_dir)
        plot_structure_regeneration_frequency(df, "mixed_dataset", output_dir)

    print('\n========== Done ==========\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Analyze evaluation statistics from JSON file',
        usage='python eval_stats.py <input_path> [--output_dir OUTPUT_DIR]'
    )
    parser.add_argument('--input_path', type=str, help='Path to JSON file containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (defaults to same directory as input file)')
    args = parser.parse_args()

    analyze_stats(args.input_path, args.output_dir)
