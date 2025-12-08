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


def analyze_stats(jsonl_path, output_dir=None):
    """
    Analyze evaluation statistics from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file containing evaluation results
        output_dir: Directory to save outputs (defaults to same directory as input file)
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(jsonl_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSONL file
    print(f"\nLoading evaluation results from {jsonl_path}...")
    df = load_jsonl(jsonl_path)
    print(f"Loaded {len(df)} evaluation records")
    
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
    try:
        plt.figure(figsize=(10, 6))
        brick_rejections.plot.hist(bins=20, alpha=0.7)
        plt.title('Histogram: Total Number of Brick Rejections per Structure')
        plt.xlabel('Number of Brick Rejections')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'brick_rejections_histogram.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved brick rejections histogram to {output_path}")
    except Exception as e:
        print(f"Could not plot brick_rejections: {e}")
    
    # ========== Brick Rejection Reasons ==========
    print("\n========== Brick Rejection Reasons (Top 10) ==========")
    # Exclude prompt column
    reason_cols = [c for c in reasons_df.columns if c != 'prompt']
    reason_counts = reasons_df[reason_cols].sum().sort_values(ascending=False)
    print(reason_counts.head(10))
    try:
        plt.figure(figsize=(10, 6))
        reason_counts.head(10).plot.bar(alpha=0.7)
        plt.title('Total Count for Each Brick Rejection Reason')
        plt.ylabel('Count')
        plt.xlabel('Reason')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'brick_rejection_reason_counts.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved rejection reasons bar chart to {output_path}")
    except Exception as e:
        print(f"Could not plot rejection reasons: {e}")
    
    # ========== Regeneration Stats ==========
    print("\n========== Regeneration Stats ==========")
    print(regenerations.describe())
    try:
        plt.figure(figsize=(10, 6))
        regenerations.plot.hist(bins=15, alpha=0.7)
        plt.title('Histogram: Number of Regenerations per Structure')
        plt.xlabel('Number of Regenerations')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'regenerations_histogram.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved regenerations histogram to {output_path}")
    except Exception as e:
        print(f"Could not plot regenerations: {e}")
    
    # ========== Additional Stats ==========
    print("\n========== Additional Statistics ==========")
    if 'mean_stability_score' in df.columns:
        print(f"Mean Stability Score: {df['mean_stability_score'].mean():.6f}")
        print(f"Std Stability Score: {df['mean_stability_score'].std():.6f}")
    
    if 'is_stable' in df.columns:
        stable_count = df['is_stable'].sum()
        total_count = len(df)
        print(f"Stable structures: {stable_count}/{total_count} ({100*stable_count/total_count:.2f}%)")
    
    if 'n_bricks' in df.columns:
        print(f"\nAverage number of bricks: {df['n_bricks'].mean():.2f}")
        print(f"Std number of bricks: {df['n_bricks'].std():.2f}")
    
    if 'inference_time_seconds' in df.columns:
        print(f"\nAverage inference time: {df['inference_time_seconds'].mean():.2f}s")
        print(f"Total inference time: {df['inference_time_seconds'].sum():.2f}s")
    
    print('\n========== Done ==========\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze evaluation statistics from JSONL file')
    parser.add_argument('--jsonl_path', type=str, help='Path to JSONL file containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Directory to save outputs (defaults to same directory as input file)')
    args = parser.parse_args()
    
    analyze_stats(args.jsonl_path, args.output_dir)
