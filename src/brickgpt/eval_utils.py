"""Utility functions for analyzing evaluation inference results."""

import json
import pandas as pd
from pathlib import Path


def load_eval_results(jsonl_path):
    """
    Load evaluation results from a JSONL file into a pandas DataFrame.

    Args:
        jsonl_path: Path to the JSONL file containing evaluation results

    Returns:
        DataFrame with evaluation results
    """
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return pd.DataFrame(results)


def print_eval_summary(df):
    """
    Print a summary of evaluation results.

    Args:
        df: DataFrame from load_eval_results()
    """
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nTotal inferences: {len(df)}")
    print(f"Unique structures: {df['structure_id'].nunique()}")

    print("\n--- Timing Statistics ---")
    print(f"Total inference time: {df['inference_time_seconds'].sum():.2f}s")
    print(f"Mean inference time: {df['inference_time_seconds'].mean():.2f}s")
    print(f"Median inference time: {df['inference_time_seconds'].median():.2f}s")
    print(f"Min inference time: {df['inference_time_seconds'].min():.2f}s")
    print(f"Max inference time: {df['inference_time_seconds'].max():.2f}s")

    print("\n--- Generation Statistics ---")
    print(f"Mean regenerations: {df['n_regenerations'].mean():.2f}")
    print(f"Median regenerations: {df['n_regenerations'].median():.0f}")
    print(f"Max regenerations: {df['n_regenerations'].max()}")
    print(f"Structures with 0 regenerations: {(df['n_regenerations'] == 0).sum()} ({(df['n_regenerations'] == 0).mean()*100:.1f}%)")

    print("\n--- Brick Statistics ---")
    print(f"Mean bricks per structure: {df['n_bricks'].mean():.2f}")
    print(f"Median bricks per structure: {df['n_bricks'].median():.0f}")
    print(f"Min bricks: {df['n_bricks'].min()}")
    print(f"Max bricks: {df['n_bricks'].max()}")

    print("\n--- Rejection Statistics ---")
    print(f"Mean total rejections: {df['total_rejections'].mean():.2f}")
    print(f"Median total rejections: {df['total_rejections'].median():.0f}")
    print(f"Max total rejections: {df['total_rejections'].max()}")

    print("\n--- Stability Statistics ---")
    print(f"Stable structures: {df['is_stable'].sum()} ({df['is_stable'].mean()*100:.1f}%)")
    print(f"Mean stability score: {df['mean_stability_score'].mean():.4f}")
    print(f"Median stability score: {df['mean_stability_score'].median():.4f}")

    # Aggregate rejection reasons across all structures
    print("\n--- Rejection Reasons (Total across all inferences) ---")
    all_reasons = {}
    for reasons_dict in df['rejection_reasons']:
        for reason, count in reasons_dict.items():
            all_reasons[reason] = all_reasons.get(reason, 0) + count

    if all_reasons:
        for reason, count in sorted(all_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")
    else:
        print("  No rejections recorded")

    print("=" * 80)


def get_failed_structures(df, min_regenerations=None, unstable_only=False):
    """
    Get structures that failed or had issues during generation.

    Args:
        df: DataFrame from load_eval_results()
        min_regenerations: Minimum number of regenerations to be considered failed
        unstable_only: If True, only return unstable structures

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    if min_regenerations is not None:
        filtered = filtered[filtered['n_regenerations'] >= min_regenerations]

    if unstable_only:
        filtered = filtered[~filtered['is_stable']]

    return filtered


def export_to_csv(jsonl_path, csv_path=None):
    """
    Convert JSONL results to CSV format (flattens some fields).

    Args:
        jsonl_path: Path to input JSONL file
        csv_path: Path to output CSV file (if None, uses same name with .csv extension)
    """
    df = load_eval_results(jsonl_path)

    if csv_path is None:
        csv_path = str(Path(jsonl_path).with_suffix('.csv'))

    # Create a flattened version for CSV (excluding complex fields)
    csv_df = df.drop(columns=['stability_scores', 'rejection_reasons', 'final_sequence'])
    csv_df.to_csv(csv_path, index=False)
    print(f"Exported to: {csv_path}")

    return csv_path


if __name__ == '__main__':
    """Example usage of the utility functions."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('jsonl_file', type=str, help='Path to JSONL results file')
    parser.add_argument('--export-csv', action='store_true', help='Export to CSV format')
    parser.add_argument('--show-failed', action='store_true', help='Show failed structures')
    parser.add_argument('--min-regen', type=int, default=10, help='Minimum regenerations to consider failed')
    args = parser.parse_args()

    # Load and summarize
    df = load_eval_results(args.jsonl_file)
    print_eval_summary(df)

    # Show failed structures if requested
    if args.show_failed:
        print("\n")
        print("=" * 80)
        print(f"STRUCTURES WITH >={args.min_regen} REGENERATIONS")
        print("=" * 80)
        failed = get_failed_structures(df, min_regenerations=args.min_regen)
        print(failed[['structure_id', 'prompt', 'n_regenerations', 'is_stable', 'inference_time_seconds']])

    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(args.jsonl_file)
