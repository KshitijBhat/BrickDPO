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
    Plot histogram of regenerations for different brick count bins.

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

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot histogram for each bin
    for idx, ((min_bricks, max_bricks), label) in enumerate(zip(bins, bin_labels)):
        # Filter data for this brick bin
        if max_bricks == float('inf'):
            bin_df = df[df['n_bricks'] >= min_bricks]
        else:
            bin_df = df[(df['n_bricks'] >= min_bricks) & (df['n_bricks'] <= max_bricks)]

        # Plot histogram
        ax = axes[idx]
        if len(bin_df) > 0:
            regenerations = bin_df['n_regenerations']
            max_regen = int(regenerations.max())
            min_regen = int(regenerations.min())

            if max_regen > min_regen:
                ax.hist(regenerations, bins=range(min_regen, max_regen + 2),
                       alpha=0.7, edgecolor='black')
            else:
                ax.hist(regenerations, bins=10, alpha=0.7, edgecolor='black')

            ax.set_title(f'{label} bricks (n={len(bin_df)})')
            ax.set_xlabel('Number of Regenerations')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.3)

            # Add mean line
            mean_regen = regenerations.mean()
            ax.axvline(mean_regen, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_regen:.2f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} bricks (n=0)')

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle(f'Regenerations by Brick Count Bins ({suffix.replace("_", " ").title()})',
                 fontsize=16, y=1.00)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'regenerations_by_brick_bins_{suffix}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved regenerations by brick bins histogram to {output_path}")


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

        # Analyze statistics per caption type
        if 'short' in df['caption_type'].values:
            df_short = df[df['caption_type'] == 'short']
            print_stats_for_subset(df_short, "Short Prompts")
            plot_regenerations_by_brick_bins(df_short, "short_dataset", output_dir)

        if 'long' in df['caption_type'].values:
            df_long = df[df['caption_type'] == 'long']
            print_stats_for_subset(df_long, "Long Prompts")
            plot_regenerations_by_brick_bins(df_long, "long_dataset", output_dir)
    else:
        # No caption types - treat all as mixed dataset
        print_stats_for_subset(df, "Mixed Dataset")
        plot_regenerations_by_brick_bins(df, "mixed_dataset", output_dir)

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
