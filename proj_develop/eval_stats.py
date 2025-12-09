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
    print("\n========== Regeneration Stats ==========")
    if 'n_regenerations' in df.columns:
        print(f"Mean regenerations: {df['n_regenerations'].mean():.2f}")

    # ========== Stability Stats ==========
    print("\n========== Stability Stats ==========")
    if 'mean_stability_score' in df.columns:
        print(f"Mean stability score: {df['mean_stability_score'].mean():.6f}")

    if 'min_stability_score' in df.columns:
        print(f"Mean of minimum stability scores: {df['min_stability_score'].mean():.6f}")


def plot_regenerations_by_brick_bins(df, suffix, output_dir):
    """
    Plot bar chart of total regenerations for different brick count bins.

    Args:
        df: DataFrame containing the evaluation results
        suffix: Suffix for the output filename (e.g., 'mixed_dataset', 'short_dataset', 'long_dataset')
        output_dir: Directory to save the plot
    """
    if 'n_bricks' not in df.columns or 'n_regenerations' not in df.columns:
        print(f"Missing required columns for brick bin histogram")
        return

    # Define brick count bins
    bins = [(0, 50), (51, 100), (101, 150), (151, 200), (201, float('inf'))]
    bin_labels = ['0-50', '51-100', '101-150', '151-200', '201+']

    # Calculate total regenerations for each bin
    total_regenerations = []
    for (min_bricks, max_bricks), label in zip(bins, bin_labels):
        # Filter data for this brick bin
        if max_bricks == float('inf'):
            bin_df = df[df['n_bricks'] >= min_bricks]
        else:
            bin_df = df[(df['n_bricks'] >= min_bricks) & (df['n_bricks'] <= max_bricks)]

        # Sum total regenerations in this bin
        total_regen = bin_df['n_regenerations'].sum() if len(bin_df) > 0 else 0
        total_regenerations.append(total_regen)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, total_regenerations, alpha=0.7, edgecolor='black')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.title(f'Total Regenerations by Structure Length ({suffix.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Structure Length (Number of Bricks)', fontsize=12)
    plt.ylabel('Total Number of Regenerations', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'regenerations_by_brick_bins_{suffix}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved regenerations by brick bins chart to {output_path}")


def plot_rejection_frequency(df, suffix, output_dir):
    """
    Plot histogram of rejection frequency.

    Args:
        df: DataFrame containing the evaluation results
        suffix: Suffix for the output filename (e.g., 'mixed_dataset', 'short_dataset', 'long_dataset')
        output_dir: Directory to save the plot
    """
    if 'total_rejections' not in df.columns:
        print(f"Missing total_rejections column for rejection frequency histogram")
        return

    plt.figure(figsize=(10, 6))

    rejections = df['total_rejections']
    max_rej = int(rejections.max())
    min_rej = int(rejections.min())

    # Create histogram
    if max_rej > min_rej:
        bins = range(min_rej, max_rej + 2)
    else:
        bins = 20

    plt.hist(rejections, bins=bins, alpha=0.7, edgecolor='black')

    plt.title(f'Rejection Frequency ({suffix.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Number of Rejections', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'rejection_frequency_{suffix}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved rejection frequency histogram to {output_path}")


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
        plot_regenerations_by_brick_bins(df, "mixed_dataset", output_dir)
        plot_rejection_frequency(df, "mixed_dataset", output_dir)

        # Analyze statistics per caption type
        if 'short' in df['caption_type'].values:
            df_short = df[df['caption_type'] == 'short']
            print_stats_for_subset(df_short, "Short Prompts")
            plot_regenerations_by_brick_bins(df_short, "short_dataset", output_dir)
            plot_rejection_frequency(df_short, "short_dataset", output_dir)

        if 'long' in df['caption_type'].values:
            df_long = df[df['caption_type'] == 'long']
            print_stats_for_subset(df_long, "Long Prompts")
            plot_regenerations_by_brick_bins(df_long, "long_dataset", output_dir)
            plot_rejection_frequency(df_long, "long_dataset", output_dir)
    else:
        # No caption types - treat all as mixed dataset
        print_stats_for_subset(df, "Mixed Dataset")
        plot_regenerations_by_brick_bins(df, "mixed_dataset", output_dir)
        plot_rejection_frequency(df, "mixed_dataset", output_dir)

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
