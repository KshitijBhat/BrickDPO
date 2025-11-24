import os
import pandas as pd
import transformers
from tqdm import tqdm
import argparse

import sys
import os
from brickgpt.models import BrickGPT, BrickGPTConfig
from brickgpt.render_bricks import render_bricks
import multiprocessing


def load_dataset_from_parquet(dataset_path, start_idx=0, max_rows=None):
    df = pd.read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[start_idx : start_idx + max_rows]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='4class_stablebrick2text_train.parquet', help='Name of parquet file in proj_develop/datasets/')
    parser.add_argument('--output_dir_prefix', type=str, default='proj_develop/batch_outputs', help='Base directory for all outputs')
    parser.add_argument('--experiment_name', type=str, default='', help='Output folder will have this as suffix')
    parser.add_argument('--caption_column', type=str, default='captions', help='Column name holding list of prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--start_idx', type=int, default=None, help='Starting IDx for processing rows')
    parser.add_argument('--max_rows', type=int, default=None, help='Limit the number of rows processed')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir_prefix, args.experiment_name)

    print(output_dir)

    dataset_path = os.path.join('proj_develop/datasets', args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    transformers.set_seed(args.seed)

    # Prepare config and BrickGPT object
    cfg = BrickGPTConfig()
    brickgpt = BrickGPT(cfg)

    # Load data
    df = load_dataset_from_parquet(dataset_path, start_idx=args.start_idx, max_rows=args.max_rows)
    n_structures = df.shape[0]
    print(f"Running inference for {n_structures} rows from {dataset_path} starting at index {args.start_idx}")
    # breakpoint()


    dpo_csv_path = os.path.join(output_dir, f'dpo_data_process.csv')
    rejection_csv_path = os.path.join(output_dir, f'brick_rejections_stats_process.csv')
    reasons_csv_path = os.path.join(output_dir, f'brick_rejection_reasons_process.csv')
    regen_csv_path = os.path.join(output_dir, f'regeneration_stats_process.csv')

    # Write headers if files don't exist
    write_headers = not os.path.exists(dpo_csv_path)

    # DPO data collection
    dpo_data = []

    # Prepare dataframes for stats
    rejection_stats = []
    reasons_stats = []
    regen_stats = []

    # Main batch inference loop
    for idx, row in tqdm(df.iterrows(), total=n_structures):
        structure_id = row.get('structure_id', idx)
        object_id = row.get('object_id', idx)
        category_id = row.get('category_id', idx)
        prompts = row[args.caption_column]
        for prompt in prompts:
            output = brickgpt(prompt)

            # Collected DPO dataset
            dpo_data_point = output.get('dpo_response_dict', None)
            if dpo_data_point is not None:
                dpo_data_dict = {
                    'structure_id': structure_id,
                    'object_id': object_id,
                    'category_id': category_id,
                }
                dpo_data_dict.update(dpo_data_point)
                
                # Append to CSV immediately
                dpo_df_row = pd.DataFrame([dpo_data_dict])
                dpo_df_row.to_csv(dpo_csv_path, mode='a', header=write_headers, index=False)
                write_headers = False
            
            # Append rejection stats
            rejection_dict = {
                'structure_id': structure_id,
                'prompt': prompt,
                'brick_rejections': output['rejection_reasons'].total()
            }
            pd.DataFrame([rejection_dict]).to_csv(
                rejection_csv_path, mode='a', 
                header=not os.path.exists(rejection_csv_path), index=False
            )
            
            # Append reasons stats
            reasons_dict = {
                'structure_id': structure_id,
                'prompt': prompt,
                **output['rejection_reasons']
            }
            pd.DataFrame([reasons_dict]).to_csv(
                reasons_csv_path, mode='a',
                header=not os.path.exists(reasons_csv_path), index=False
            )
            
            # Append regeneration stats
            regen_dict = {
                'structure_id': structure_id,
                'prompt': prompt,
                'regenerations': output['n_regenerations']
            }
            pd.DataFrame([regen_dict]).to_csv(
                regen_csv_path, mode='a',
                header=not os.path.exists(regen_csv_path), index=False
            )

    print(f"Saved stats to {output_dir}/brick_rejections_stats.csv, brick_rejection_reasons.csv, regeneration_stats.csv")

    # convert dpo_data to parquet
    dpo_df = pd.DataFrame(dpo_data)
    dpo_parquet_path = os.path.join(output_dir, f'dpo_data_process.parquet')
    dpo_df.to_parquet(dpo_parquet_path, index=False)
    print(f"Saved DPO data to {dpo_parquet_path}")

if __name__ == '__main__':
    main()