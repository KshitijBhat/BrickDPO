# Baseline Evaluation Guide

Simple evaluation script for the baseline BrickGPT model. Loads the model exactly like `batch_infer.py` but only stores final results.

## Quick Start

```bash
uv run eval_baseline --dataset test_sets/test_set_maxlen_2048_topk_100.parquet
```

## Usage Examples

### Basic Evaluation (100 structures × 5 prompts = 500 total)

```bash
uv run eval_baseline \
    --dataset test_sets/test_set_maxlen_2048_topk_100.parquet \
    --output_name baseline_test
```

### Quick Test (First 5 structures = 25 prompts)

```bash
uv run eval_baseline \
    --dataset test_sets/test_set_maxlen_2048_topk_100.parquet \
    --max_rows 5 \
    --output_name quick_test
```

### Evaluate with Specific Settings

```bash
uv run eval_baseline \
    --dataset test_set.parquet \
    --output_name my_eval \
    --seed 42 \
    --max_rows 100
```

### Resume from Specific Index

```bash
uv run eval_baseline \
    --dataset large_dataset.parquet \
    --start_idx 1000 \
    --max_rows 500
```

## Command Line Arguments

- `--dataset`: Parquet file in `proj_develop/datasets/` (default: `4class_stablebrick2text_train.parquet`)
- `--output_name`: Output filename without extension (default: auto-generated with timestamp)
- `--caption_column`: Column name with prompts (default: `captions`)
- `--seed`: Random seed (default: `42`)
- `--start_idx`: Starting row index (default: `0`)
- `--max_rows`: Limit number of rows (default: all)

## Output Format

Results saved as JSONL in `proj_develop/inference/`.

Each line (one per prompt):

```json
{
  "model_name_or_path": "AvaLovelace/BrickGPT",
  "structure_id": 12345,
  "object_id": 67890,
  "category_id": 3,
  "prompt": "A red car",
  "prompt_idx": 0,
  "final_sequence": "2x4 (0,0,0)\n4x2 (2,0,0)\n...",
  "n_bricks": 47,
  "n_regenerations": 3,
  "rejection_reasons": {"collision": 12, "out_of_bounds": 5},
  "total_rejections": 18,
  "inference_time_seconds": 12.34,
  "mean_stability_score": 0.023,
  "is_stable": true
}
```

**Note**: Individual brick stability scores are NOT saved (only mean) to reduce file size.

## Analysis

Use `eval_utils.py`:

```bash
uv run python -m brickgpt.eval_utils proj_develop/inference/eval_baseline_*.jsonl
```

Or in Python:

```python
from brickgpt.eval_utils import load_eval_results, print_eval_summary

df = load_eval_results('proj_develop/inference/eval_baseline_*.jsonl')
print_eval_summary(df)
```

## Performance Notes

**For test_set_maxlen_2048_topk_100.parquet (100 structures × 5 prompts = 500 total):**
- **Per prompt**: ~5-15 seconds (2048 tokens ≈ 200 bricks)
- **Full evaluation**: ~40-125 minutes (0.7-2 hours)
- **Quick test** (5 structures): ~2-6 minutes

**Main bottlenecks:**
1. Long sequence generation (2048 tokens = ~200 bricks)
2. Rejection sampling (max 500 attempts per brick)
3. Physics-informed rollback (up to 100 regenerations)
4. Gurobi stability solver

See `PERFORMANCE_NOTES.md` for optimization options.

## Notes

- Loads model exactly like `batch_infer.py` (no custom loading)
- Only stores final results (no DPO intermediate data)
- Progress bar shows total prompts (not structures)
- Results written incrementally - can monitor in real-time
- Dataset loading is NOT a bottleneck (only 100 rows)
- All paths relative to project root
