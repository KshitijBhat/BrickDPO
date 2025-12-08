# Performance Analysis for eval_baseline

## Key Performance Bottlenecks

### 1. **Sequence Length (2048 tokens = ~200 bricks)**
- Each brick is ~10 tokens: `"2x4 (0,1,2)\n"`
- For 2048 tokens, you're generating ~200-250 bricks per structure
- **This is the main bottleneck** - generation is O(n²) in sequence length due to attention
- **Expected time**: ~5-15 seconds per prompt on GPU for this length

### 2. **Rejection Sampling (max_brick_rejections=500)**
- Each rejected brick rolls back the KV cache and regenerates
- For long structures, rejection rate increases (more collisions, out of bounds)
- Can add 2-5x overhead depending on rejection rate

### 3. **Physics-Informed Rollback (max_regenerations=100)**
- Stability checking runs after each full generation
- If unstable, removes bricks and regenerates
- Gurobi solver adds overhead for each stability check
- Can trigger multiple full regenerations

### 4. **Memory - NOT an issue for this dataset**
- Loading 100 rows into memory is negligible (~few MB)
- Parquet is efficient and only loads used columns
- **Dataset loading is not the bottleneck**

## Current Configuration (from BrickGPTConfig defaults)

```python
max_bricks: 2000          # Can generate up to 2000 bricks
max_brick_rejections: 500 # Up to 500 attempts per brick
max_regenerations: 100    # Up to 100 full regenerations
use_gurobi: True          # Expensive stability solver
temperature: 0.6          # Sampling temperature
```

## Performance Optimizations (if needed)

### Quick Wins:
1. **Reduce max_brick_rejections**: Lower from 500 to 100-200
2. **Reduce max_regenerations**: Lower from 100 to 20-50
3. **Use connectivity check**: Set `use_gurobi=False` (faster but less accurate)

### Advanced:
4. **Batch inference**: Not applicable - each structure is independent
5. **Use smaller context**: Not applicable - need full structure
6. **Reduce temperature**: Lower temperature = less diversity = fewer rejections

## Expected Performance

For your test set (100 structures × 5 prompts = 500 total):
- **Per prompt**: 5-15 seconds (for 2048 token sequences = ~200 bricks)
- **Total time**: 40-125 minutes (0.7-2 hours) for full evaluation
- **With optimizations**: 20-60 minutes

## Recommendations

1. **Start with small sample** (--max_rows 5 = 25 prompts) to test
2. **Monitor rejection rates** in output to see if rejection sampling is the issue
3. **Check regeneration counts** - if high, structures are unstable
4. **Consider reducing max_bricks** to 1500 if 2048 isn't critical

## Not Performance Issues

✅ Loading dataset into memory (only 100 rows)
✅ Writing JSONL incrementally (minimal overhead)
✅ Computing mean stability score (fast)
