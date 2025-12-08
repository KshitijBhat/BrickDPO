# Evaluation Inference Guide

Fast evaluation script for BrickGPT models that only stores final results (no intermediate generations).

## Quick Start

### Running from Project Root with UV

```bash
# Evaluate baseline model
uv run eval_infer --dataset your_dataset.parquet

# Evaluate HuggingFace merged model
uv run eval_infer \
    --dataset your_dataset.parquet \
    --model_name_or_path kshitij-hf/brickgpt-dpo-2048 \
    --use_hf
```

## Usage Examples

### 1. Baseline Model (Default)

Evaluates the default BrickGPT baseline model:

```bash
uv run eval_infer \
    --dataset test_set.parquet \
    --output_name baseline_eval
```

### 2. HuggingFace Merged Model

Evaluates a merged model from HuggingFace with subfolder structure:

```bash
uv run eval_infer \
    --dataset test_set.parquet \
    --model_name_or_path kshitij-hf/brickgpt-dpo-2048 \
    --use_hf \
    --output_name dpo_eval
```

The `--use_hf` flag tells the script to load from a subfolder (default: `merged_dpo_brickgpt`).

To change the subfolder:

```bash
uv run eval_infer \
    --dataset test_set.parquet \
    --model_name_or_path kshitij-hf/brickgpt-dpo-2048 \
    --use_hf \
    --hf_subfolder my_custom_subfolder \
    --output_name custom_eval
```

### 3. Quick Testing

Evaluate only first 100 rows:

```bash
uv run eval_infer \
    --dataset test_set.parquet \
    --max_rows 100 \
    --seed 42
```

## Command Line Arguments

- `--dataset`: Parquet file name in `proj_develop/datasets/` (default: `4class_stablebrick2text_train.parquet`)
- `--output_name`: Output filename without extension (default: auto-generated with timestamp)
- `--model_name_or_path`: Model checkpoint path (default: `AvaLovelace/BrickGPT`)
- `--use_hf`: Load merged model from HuggingFace with subfolder (flag)
- `--hf_subfolder`: Subfolder to use with `--use_hf` (default: `merged_dpo_brickgpt`)
- `--caption_column`: Column name with prompts (default: `captions`)
- `--seed`: Random seed (default: `42`)
- `--start_idx`: Starting row index (default: `0`)
- `--max_rows`: Limit number of rows (default: process all)

## Output Format

Results saved as JSONL (JSON Lines) in `proj_develop/inference/`.

Each line contains:

```json
{
  "structure_id": 12345,
  "prompt": "A red car",
  "final_sequence": "2x4 (0,0,0)\n4x2 (2,0,0)\n...",
  "n_bricks": 47,
  "n_regenerations": 3,
  "rejection_reasons": {"collision": 12, "out_of_bounds": 5},
  "total_rejections": 18,
  "inference_time_seconds": 12.34,
  "stability_scores": [0.0, 0.0, 0.1, ...],
  "mean_stability_score": 0.023,
  "is_stable": true
}
```

## Analysis

Use the `eval_utils.py` script to analyze results:

```bash
uv run python -m brickgpt.eval_utils proj_develop/inference/eval_*.jsonl
```

Or load into Python:

```python
from brickgpt.eval_utils import load_eval_results, print_eval_summary

df = load_eval_results('proj_develop/inference/eval_baseline_*.jsonl')
print_eval_summary(df)

# Compare models
baseline = load_eval_results('proj_develop/inference/eval_baseline_*.jsonl')
dpo = load_eval_results('proj_develop/inference/eval_dpo_*.jsonl')

print(f"Baseline stability: {baseline['is_stable'].mean():.2%}")
print(f"DPO stability: {dpo['is_stable'].mean():.2%}")
```

## Complete Workflow Example

```bash
# 1. Evaluate baseline model
uv run eval_infer \
    --dataset test_set.parquet \
    --output_name baseline \
    --seed 42

# 2. Evaluate your DPO model from HuggingFace
uv run eval_infer \
    --dataset test_set.parquet \
    --model_name_or_path kshitij-hf/brickgpt-dpo-2048 \
    --use_hf \
    --output_name dpo \
    --seed 42

# 3. Analyze and compare
uv run python -m brickgpt.eval_utils proj_develop/inference/eval_baseline_*.jsonl
uv run python -m brickgpt.eval_utils proj_develop/inference/eval_dpo_*.jsonl
```

## Notes

- The `--use_hf` flag loads merged weights from HuggingFace with `device_map="auto"` and `torch_dtype="auto"`
- Results are written incrementally - you can monitor progress in real-time
- To evaluate different checkpoints, change `--model_name_or_path` to point to the new location
- To change the subfolder name, use `--hf_subfolder`
- All paths are relative to the project root
