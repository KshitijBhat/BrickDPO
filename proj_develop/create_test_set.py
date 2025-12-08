#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import ast


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

    parser.add_argument(
        "--use_2_prompts",
        type=int,
        default=True,
        help="Number of structures kept"
    )

    args = parser.parse_args()
    max_seq_len = args.max_seq_len
    use_2_prompts = args.use_2_prompts

    # Load raw test set
    raw_test_set_filename = "datasets/4class_stablebrick2text_test.parquet"
    print(f"Loading: {raw_test_set_filename}")
    df = pd.read_parquet(raw_test_set_filename)

    # 1. Filter by car class
    original_len = len(df)
    df = df[df.category_id == "02958343"].reset_index(drop=True)
    print(f"Filtered from {original_len} → {len(df)} rows (car class == 02958343)")

    # 2. Skip every other row
    df = df.iloc[::2]
    print(df.index)

    # 3. Filter by accepted seq len 
    print("Computing sequence lengths...")
    if max_seq_len is not None:
        df_filtered = df[df["bricks"].apply(len) <= max_seq_len].copy()
        print(f"Filtered from {len(df)} → {len(df_filtered)} rows (max_seq_len<={max_seq_len})")
    else:
        df_filtered = df.copy()

    # 4. Only take the top k rows (each row is a structure = 5 captions for that structure)
    if args.top_k_structures is not None:
        df_filtered_topk = df_filtered.iloc[: args.top_k_structures]
        print(f"Filtered to top {args.top_k_structures} structures → {len(df_filtered)} rows")
    else:
        df_filtered_topk = df_filtered.copy()
    
    # 5. Keep only 2 prompts per row
    def keep_first_and_last(captions_str):
        lst = captions_str.tolist() 
        new_lst = [lst[0], lst[-1]]
        return str(new_lst)
    
    if use_2_prompts:
        df_filtered_topk.loc[:, "captions"] = df_filtered_topk["captions"].apply(keep_first_and_last)

    # Save the output
    seq_len_suffix = "" if max_seq_len is None else f"_maxlen_{max_seq_len}"
    topk_suffix = "" if args.top_k_structures is None else f"_topk_{args.top_k_structures}"
    prompt_cnt_suffix = "" if not args.use_2_prompts else "_2prompts"

    output_file = f"test_set{seq_len_suffix}{topk_suffix}{prompt_cnt_suffix}.parquet"
    output_dir = "datasets/test_sets/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    df_filtered_topk.to_parquet(output_path)
    print(f"Saved filtered file to:\n{output_path}")

    # 🔥 Assertion for number of rows
    print(f"Final row count: {len(df_filtered_topk)}")


if __name__ == "__main__":
    main()
