import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


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

    # Extract brick rejections data
    brick_rejections = df['total_rejections'].rename('brick_rejections')

    # Extract rejection reasons - expand dict into columns
    rejection_reasons_list = []
    for idx, row in df.iterrows():
        reasons_dict = row.get('rejection_reasons', {})
        reasons_dict['prompt'] = row.get('prompt', '')
        rejection_reasons_list.append(reasons_dict)

    reasons_df = pd.DataFrame(rejection_reasons_list)

    # Extract regeneration data
    regenerations = df['n_regenerations'].rename('regenerations')

    # ========== Brick Rejections Stats ==========
    print("\n========== Brick Rejections Stats ==========")
    print(brick_rejections.describe())

    # ========== Brick Rejection Reasons ==========
    print("\n========== Brick Rejection Reasons (Top 10) ==========")
    # Exclude prompt column
    reason_cols = [c for c in reasons_df.columns if c != 'prompt']
    reason_counts = reasons_df[reason_cols].sum().sort_values(ascending=False)
    print(reason_counts.head(10))

    # ========== Regeneration Stats ==========
    print("\n========== Regeneration Stats ==========")
    print(f"Mean regenerations: {regenerations.mean():.2f}")
    print(f"Std regenerations: {regenerations.std():.2f}")
    print(f"Regeneration distribution:")
    print(regenerations.value_counts().sort_index())

    # ========== Stability Stats ==========
    print("\n========== Stability Stats ==========")
    if 'mean_stability_score' in df.columns:
        print(f"Mean brick stability (across all structures): {df['mean_stability_score'].mean():.6f}")
        print(f"Std brick stability: {df['mean_stability_score'].std():.6f}")

    if 'min_stability_score' in df.columns:
        print(f"\nMean of lowest brick stability (across all structures): {df['min_stability_score'].mean():.6f}")
        print(f"Std of lowest brick stability: {df['min_stability_score'].std():.6f}")
        print(f"Overall lowest brick stability: {df['min_stability_score'].min():.6f}")

    if 'is_stable' in df.columns:
        stable_count = df['is_stable'].sum()
        total_count = len(df)
        print(f"\nStable structures: {stable_count}/{total_count} ({100*stable_count/total_count:.2f}%)")

    # ========== Generation Stats ==========
    print("\n========== Generation Stats ==========")
    if 'n_bricks' in df.columns:
        print(f"Mean number of bricks: {df['n_bricks'].mean():.2f}")
        print(f"Std number of bricks: {df['n_bricks'].std():.2f}")

    if 'inference_time_seconds' in df.columns:
        print(f"\nMean generation time: {df['inference_time_seconds'].mean():.2f}s")
        print(f"Std generation time: {df['inference_time_seconds'].std():.2f}s")
        print(f"Total generation time: {df['inference_time_seconds'].sum():.2f}s")


def plot_stats_for_subset(df, subset_name, output_dir):
    """Generate plots for a subset of the data."""
    # Create subdirectory for this subset
    subset_dir = os.path.join(output_dir, subset_name.lower().replace(' ', '_'))
    os.makedirs(subset_dir, exist_ok=True)

    # Extract brick rejections data
    brick_rejections = df['total_rejections'].rename('brick_rejections')

    # Extract rejection reasons
    rejection_reasons_list = []
    for idx, row in df.iterrows():
        reasons_dict = row.get('rejection_reasons', {})
        reasons_dict['prompt'] = row.get('prompt', '')
        rejection_reasons_list.append(reasons_dict)
    reasons_df = pd.DataFrame(rejection_reasons_list)

    # Extract regeneration data
    regenerations = df['n_regenerations'].rename('regenerations')

    # Plot brick rejections histogram
    try:
        plt.figure(figsize=(10, 6))
        brick_rejections.plot.hist(bins=20, alpha=0.7)
        plt.title(f'Histogram: Total Number of Brick Rejections per Structure ({subset_name})')
        plt.xlabel('Number of Brick Rejections')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(subset_dir, 'brick_rejections_histogram.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved brick rejections histogram to {output_path}")
    except Exception as e:
        print(f"Could not plot brick_rejections: {e}")

    # Plot rejection reasons
    try:
        reason_cols = [c for c in reasons_df.columns if c != 'prompt']
        reason_counts = reasons_df[reason_cols].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        reason_counts.head(10).plot.bar(alpha=0.7)
        plt.title(f'Total Count for Each Brick Rejection Reason ({subset_name})')
        plt.ylabel('Count')
        plt.xlabel('Reason')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = os.path.join(subset_dir, 'brick_rejection_reason_counts.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved rejection reasons bar chart to {output_path}")
    except Exception as e:
        print(f"Could not plot rejection reasons: {e}")

    # Plot regenerations histogram
    try:
        plt.figure(figsize=(10, 6))
        regenerations.plot.hist(bins=15, alpha=0.7)
        plt.title(f'Histogram: Number of Regenerations per Structure ({subset_name})')
        plt.xlabel('Number of Regenerations')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(subset_dir, 'regenerations_histogram.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved regenerations histogram to {output_path}")
    except Exception as e:
        print(f"Could not plot regenerations: {e}")


def analyze_stats(input_path, output_dir=None):
    """
    Analyze evaluation statistics from a JSON or JSONL file.

    Args:
        input_path: Path to the JSON/JSONL file containing evaluation results
        output_dir: Directory to save outputs (defaults to same directory as input file)
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    # Load file (detect JSON vs JSONL)
    print(f"\nLoading evaluation results from {input_path}...")
    if input_path.endswith('.jsonl'):
        df = load_jsonl(input_path)
    elif input_path.endswith('.json'):
        df = load_json(input_path)
    else:
        # Try JSON first, then JSONL
        try:
            df = load_json(input_path)
        except:
            df = load_jsonl(input_path)

    print(f"Loaded {len(df)} evaluation records")

    # Check if caption_type column exists
    has_caption_types = 'caption_type' in df.columns

    if has_caption_types:
        # Analyze overall statistics (merged)
        print_stats_for_subset(df, "All (Merged)")
        plot_stats_for_subset(df, "All_Merged", output_dir)

        # Analyze statistics per caption type
        caption_types = df['caption_type'].unique()
        for caption_type in sorted(caption_types):
            df_subset = df[df['caption_type'] == caption_type]
            print_stats_for_subset(df_subset, f"Caption Type: {caption_type}")
            plot_stats_for_subset(df_subset, f"Caption_{caption_type}", output_dir)
    else:
        # Legacy behavior: analyze all data as one group
        print_stats_for_subset(df, "All")
        plot_stats_for_subset(df, "All", output_dir)

    print('\n========== Done ==========\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Analyze evaluation statistics from JSON or JSONL file',
        usage='python eval_stats.py <input_path> [--output_dir OUTPUT_DIR]'
    )
    parser.add_argument('input_path', type=str, help='Path to JSON or JSONL file containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (defaults to same directory as input file)')
    args = parser.parse_args()

    analyze_stats(args.input_path, args.output_dir)
