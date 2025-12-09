import os
import json
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

    if 'max_stability_score' in df.columns:
        print(f"Mean of maximum stability scores: {df['max_stability_score'].mean():.6f}")


def plot_total_rejections_by_structure(df, suffix, output_dir):
    """
    Plot bar chart of total rejections by structure length bins.

    Args:
        df: DataFrame containing the evaluation results
        suffix: Suffix for the output filename (e.g., 'mixed_dataset', 'short_dataset', 'long_dataset')
        output_dir: Directory to save the plot
    """
    if 'n_bricks' not in df.columns or 'total_rejections' not in df.columns:
        print(f"Missing required columns for total rejections by structure chart")
        return

    # Define brick count bins
    bins = [(0, 50), (51, 100), (101, 150), (151, 200), (201, float('inf'))]
    bin_labels = ['0-50', '51-100', '101-150', '151-200', '201+']

    # Calculate total rejections for each bin
    total_rejections = []
    for (min_bricks, max_bricks), label in zip(bins, bin_labels):
        # Filter data for this brick bin
        if max_bricks == float('inf'):
            bin_df = df[df['n_bricks'] >= min_bricks]
        else:
            bin_df = df[(df['n_bricks'] >= min_bricks) & (df['n_bricks'] <= max_bricks)]

        # Sum total rejections in this bin
        total_rej = bin_df['total_rejections'].sum() if len(bin_df) > 0 else 0
        total_rejections.append(total_rej)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(bin_labels, total_rejections, alpha=0.7, edgecolor='black')

    plt.title(f'Total Rejections by Structure Length ({suffix.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Structure Length (Number of Bricks)', fontsize=12)
    plt.ylabel('Total Number of Rejections', fontsize=12)
    plt.ylim(bottom=0)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'total_rejections_by_structure_{suffix}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved total rejections by structure chart to {output_path}")


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

    plt.figure(figsize=(10, 6))

    regenerations = df['n_regenerations']
    max_regen = int(regenerations.max())
    min_regen = int(regenerations.min())

    # Create histogram with bins from 0 to max
    if max_regen > 0:
        bins = range(0, max_regen + 2)
    else:
        bins = 20

    plt.hist(regenerations, bins=bins, alpha=0.7, edgecolor='black')

    plt.title(f'Structure Regeneration Frequency ({suffix.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Number of Structure Regenerations', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.3)
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
    parser.add_argument('input_path', type=str, help='Path to JSON file containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (defaults to same directory as input file)')
    args = parser.parse_args()

    analyze_stats(args.input_path, args.output_dir)
