import os
import pandas as pd
import transformers
from tqdm import tqdm
import argparse

import sys
import os
from brickgpt.models import BrickGPT, BrickGPTConfig
from brickgpt.render_bricks import render_bricks

def load_dataset_from_parquet(dataset_path, max_samples=None):
    df = pd.read_parquet(dataset_path)
    if max_samples is not None:
        df = df.iloc[:max_samples]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='4class_stablebrick2text_train.parquet', help='Name of parquet file in proj_develop/datasets/')
    parser.add_argument('--output_dir_prefix', type=str, default='proj_develop/batch_outputs', help='Base directory for all outputs')
    parser.add_argument('--experiment_name', type=str, default='', help='Output folder will have this as suffix')
    parser.add_argument('--caption_column', type=str, default='captions', help='Column name holding list of prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit the number of structures processed')
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
    df = load_dataset_from_parquet(dataset_path, args.max_samples)
    n_structures = df.shape[0]
    print(f"Running inference for {n_structures} structures")

    # Prepare dataframes for stats
    rejection_stats = []
    reasons_stats = []
    regen_stats = []

    # Main batch inference loop
    for idx, row in tqdm(df.iterrows(), total=n_structures):
        structure_id = row.get('structure_id', idx)
        prompts = row[args.caption_column]
        if not isinstance(prompts, list):
            prompts = [prompts]
        for prompt in prompts:
            # Run inference
            output = brickgpt(prompt)

            # Log stats 
            rejection_stats.append({'structure_id': structure_id, 'prompt': prompt, 'brick_rejections': output['rejection_reasons'].total()})
            reasons_stats.append({'structure_id': structure_id, 'prompt': prompt, **output['rejection_reasons']})
            regen_stats.append({'structure_id': structure_id, 'prompt': prompt, 'regenerations': output['n_regenerations']})

    # Save stats to CSV for easy later analysis
    pd.DataFrame(rejection_stats).to_csv(os.path.join(output_dir, 'brick_rejections_stats.csv'), index=False)
    pd.DataFrame(reasons_stats).to_csv(os.path.join(output_dir, 'brick_rejection_reasons.csv'), index=False)
    pd.DataFrame(regen_stats).to_csv(os.path.join(output_dir, 'regeneration_stats.csv'), index=False)

    print(f"Saved stats to {output_dir}/brick_rejections_stats.csv, brick_rejection_reasons.csv, regeneration_stats.csv")

if __name__ == '__main__':
    main()