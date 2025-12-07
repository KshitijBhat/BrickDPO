#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def main():
    # Argsument parser
    parser = argparse.ArgumentParser(description="Filter parquet rows based on max sequence length.")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum allowed length of the last generated_bricks element"
    )

    parser.add_argument(
        "--top_k_structures",
        type=int,
        default=None,
        help="Number of structures kept"
    )

    args = parser.parse_args()
    max_seq_len = args.max_seq_len

    # Load raw test set
    raw_test_set_filename = "datasets/4class_stablebrick2text_test.parquet"
    print(f"Loading: {raw_test_set_filename}")
    df = pd.read_parquet(raw_test_set_filename)


    # 1. Filter by car class
    original_len = len(df)
    df = df[df.category_id == "02958343"]
    print(f"Filtered from {original_len} → {len(df)} rows (car class == 02958343)")

    # 2. Filter by accepted seq len 
    print("Computing sequence lengths...")
    if max_seq_len is not None:
        df_filtered = df[df["bricks"].apply(len) <= max_seq_len].copy()
        print(f"Filtered from {len(df)} → {len(df_filtered)} rows (max_seq_len<={max_seq_len})")
    else:
        df_filtered = df.copy()

    # 3. Only take the top k rows (each row is a structure = 5 captions for that structure)
    if args.top_k_structures is not None:
        df_filtered_topk = df_filtered.iloc[: args.top_k_structures]
        print(f"Filtered to top {args.top_k_structures} structures → {len(df_filtered)} rows")
    else:
        df_filtered_topk = df_filtered.copy()
    
    # Save the output
    seq_len_suffix = "" if max_seq_len is None else f"_maxlen_{max_seq_len}"
    topk_suffix = "" if args.top_k_structures is None else f"_topk_{args.top_k_structures}"
    output_file = f"test_set{seq_len_suffix}{topk_suffix}.parquet"
    output_dir = "datasets/test_sets/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    df_filtered_topk.to_parquet(output_path)
    print(f"Saved filtered file to:\n{output_path}")

    # 🔥 Assertion for number of rows
    print(f"Final row count: {len(df_filtered_topk)}")


if __name__ == "__main__":
    main()
