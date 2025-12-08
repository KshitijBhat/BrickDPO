import os
import json
import time
import pandas as pd
import transformers
from tqdm import tqdm
import argparse
from datetime import datetime

from brickgpt.models import BrickGPT, BrickGPTConfig


def load_dataset_from_parquet(dataset_path, start_idx=0, max_rows=None):
    """Load dataset from parquet file with optional row limiting."""
    df = pd.read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[start_idx : start_idx + max_rows]
    elif start_idx > 0:
        df = df.iloc[start_idx:]
    return df


def main():
    parser = argparse.ArgumentParser(description='Fast evaluation inference script for baseline BrickGPT')
    parser.add_argument(
        '--dataset',
        type=str,
        default='4class_stablebrick2text_train.parquet',
        help='Name of parquet file in proj_develop/datasets/'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Output filename (without extension). If not provided, will use dataset name + timestamp'
    )
    parser.add_argument(
        '--caption_column',
        type=str,
        default='captions',
        help='Column name holding list of prompts'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting index for processing rows'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=None,
        help='Limit the number of rows processed'
    )
    args = parser.parse_args()

    # Setup paths
    dataset_path = os.path.join('proj_develop/datasets', args.dataset)
    output_dir = 'proj_develop/inference'
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    if args.output_name is None:
        dataset_basename = os.path.splitext(args.dataset)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'eval_baseline_{dataset_basename}_{timestamp}'
    else:
        output_name = args.output_name

    output_path = os.path.join(output_dir, f'{output_name}.jsonl')

    # Set seed (same as batch_infer.py)
    transformers.set_seed(args.seed)

    # Prepare config and BrickGPT object (same as batch_infer.py)
    cfg = BrickGPTConfig()
    brickgpt = BrickGPT(cfg)

    # Load dataset
    df = load_dataset_from_parquet(dataset_path, start_idx=args.start_idx, max_rows=args.max_rows)
    n_structures = df.shape[0]

    # Count total prompts across all rows
    total_prompts = df[args.caption_column].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 1).sum()

    print(f"Running evaluation inference on {n_structures} structures ({total_prompts} total prompts)")
    print(f"Model: {cfg.model_name_or_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Starting index: {args.start_idx}")
    print(f"Output file: {output_path}")
    print("-" * 80)

    # Track overall statistics
    total_inferences = 0
    total_time = 0

    # Open output file for writing
    with open(output_path, 'w') as f:
        # Create progress bar for total prompts
        pbar = tqdm(total=total_prompts, desc="Processing prompts")

        for idx, row in df.iterrows():
            structure_id = int(row.get('structure_id', idx))
            object_id = int(row.get('object_id', idx))
            category_id = int(row.get('category_id', idx))
            prompts = row[args.caption_column]

            if not isinstance(prompts, list):
                prompts = [prompts]

            for prompt_idx, prompt in enumerate(prompts):
                start_time = time.time()
                output = brickgpt(str(prompt))
                end_time = time.time()
                inference_time = end_time - start_time

                final_bricks = output['bricks']
                rejection_reasons = dict(output['rejection_reasons'])

                result = {
                    'model_name_or_path': cfg.model_name_or_path,
                    'structure_id': structure_id,
                    'object_id': object_id,
                    'category_id': category_id,
                    'prompt': str(prompt),
                    'prompt_idx': prompt_idx,
                    'final_sequence': final_bricks.to_txt(),
                    'n_bricks': len(final_bricks),
                    'n_regenerations': int(output['n_regenerations']),
                    'rejection_reasons': rejection_reasons,
                    'total_rejections': sum(rejection_reasons.values()),
                    'inference_time_seconds': inference_time,
                    'mean_stability_score': float(brickgpt._stability_scores(final_bricks).mean()),
                    'is_stable': bool(brickgpt._is_stable(final_bricks)),
                }

                f.write(json.dumps(result) + '\n')
                f.flush()

                total_inferences += 1
                total_time += inference_time
                pbar.update(1)

        # Close progress bar
        pbar.close()

    # Print summary statistics
    print("-" * 80)
    print(f"Evaluation complete!")
    print(f"Total inferences: {total_inferences}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per inference: {total_time/total_inferences:.2f}s")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
