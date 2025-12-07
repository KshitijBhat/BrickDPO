#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot distribution of sequence lengths in a parquet dataset.")
    parser.add_argument(
        "--file",
        type=str,
        default="combined_dataset/dpo_data.parquet",
        help="Parquet filename inside datasets/dpo_datasets/"
    )
    args = parser.parse_args()

    parent_dir = "datasets/dpo_datasets"
    file_path = os.path.join(parent_dir, args.file)

    print(f"Loading parquet file: {file_path}")
    df = pd.read_parquet(file_path)

    # Compute lengths of each sample
    lengths = []
    for index, row in df.iterrows():
        seqs = row.get("generated_bricks", [])
        for seq in seqs:
            lengths.append(len(seq))

    max_len = max(lengths) if lengths else 0
    mean_len = sum(lengths) / len(lengths) if lengths else 0
    std_len = np.array(lengths).std()

    print(f"Computed {len(lengths)} sequence lengths.")
    print("Max sequence length:", max_len)
    print("Mean sequence length:", mean_len)
    print("Standard deviation of sequence lengths:", std_len)

    # Create output dir
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "seq_len_histogram.png")

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save instead of show
    plt.savefig(output_path, dpi=300)
    print(f"Histogram saved to: {output_path}")


if __name__ == "__main__":
    main()
