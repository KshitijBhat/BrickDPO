import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def analyze_stats(output_dir):
    # File paths
    brick_rejections_file = os.path.join(output_dir, 'brick_rejections_stats.csv')
    reasons_file = os.path.join(output_dir, 'brick_rejection_reasons.csv')
    regen_file = os.path.join(output_dir, 'regeneration_stats.csv')
    
    # Load as DataFrame
    print("\nLoading stats files...")
    brick_rej_df = pd.read_csv(brick_rejections_file)
    reasons_df = pd.read_csv(reasons_file)
    regen_df = pd.read_csv(regen_file)
    
    print("\n========== Brick Rejections Stats ==========")
    print(brick_rej_df.describe())
    try:
        plt.figure()
        brick_rej_df['brick_rejections'].plot.hist(bins=20, alpha=0.7)
        plt.title('Histogram: Total Number of Brick Rejections per Structure')
        plt.xlabel('Number of Brick Rejections')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'brick_rejections_histogram.png'))
        print(f"Saved brick rejections histogram to {output_dir}/brick_rejections_histogram.png")
    except Exception as e:
        print("Could not plot brick_rejections:", e)

    print("\n========== Brick Rejection Reasons (Top 10) ==========")
    # Exclude structure_id, prompt
    reason_cols = [c for c in reasons_df.columns if c not in ('structure_id', 'prompt')]
    reason_counts = reasons_df[reason_cols].sum().sort_values(ascending=False)
    print(reason_counts.head(10))
    try:
        plt.figure()
        reason_counts.plot.bar(alpha=0.7)
        plt.title('Total Count for Each Brick Rejection Reason')
        plt.ylabel('Count')
        plt.xlabel('Reason')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'brick_rejection_reason_counts.png'))
        print(f"Saved rejection reasons bar chart to {output_dir}/brick_rejection_reason_counts.png")
    except Exception as e:
        print("Could not plot rejection reasons:", e)
    
    print("\n========== Regeneration Stats ==========")
    print(regen_df['regenerations'].describe())
    try:
        plt.figure()
        regen_df['regenerations'].plot.hist(bins=15, alpha=0.7)
        plt.title('Histogram: Number of Regenerations per Structure')
        plt.xlabel('Number of Regenerations')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regenerations_histogram.png'))
        print(f"Saved regenerations histogram to {output_dir}/regenerations_histogram.png")
    except Exception as e:
        print("Could not plot regenerations:", e)

    print('\n========== Done ==========\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_prefix', type=str, default='proj_develop/batch_outputs', help='Base directory for all outputs')
    parser.add_argument('--experiment_name', type=str, default='', help='Output folder will have this as suffix')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir_prefix, args.experiment_name)
    analyze_stats(output_dir)