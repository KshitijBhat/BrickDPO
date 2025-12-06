"""
Script to prepare a DPO (Direct Preference Optimization) dataset in HuggingFace format.
Fixed to output proper format for TRL DPOTrainer.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any
from brickgpt.models import create_instruction

def create_dpo_entry(prompt: List[Dict[str, str]], chosen: List[Dict[str, str]], rejected: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Create a single DPO dataset entry in chat format.
    Returns plain Python lists (not numpy arrays).
    """
    return {
        "prompt": prompt,  
        "chosen": chosen,  
        "rejected": rejected  
    }

def main():
    input_file = "datasets/dpo_datasets/combined_dataset/dpo_data.parquet"
    output_file = "datasets/dpo_datasets/combined_dataset/dpo_hf.parquet"
    
    # Load the parquet file
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    dpo_data = []

    for index, row in df.iterrows():
        caption = row.get('caption', '')
        bricks_candidates = row.get('generated_bricks', '[]')

        # The last entry of the array is the chosen response
        chosen_content = bricks_candidates[-1]
        
        # All the entries df['generated_bricks'][0:-1] are the rejected ones
        rejected_candidates = bricks_candidates[:-1]

        # Pair each rejected candidate with the single chosen response
        for rejected_content in rejected_candidates:
            
            # Construct messages in the chat format
            # Prompt contains the conversation history (system + user)
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": create_instruction(caption)}
            ]
            
            # Chosen/rejected contain only the assistant response
            chosen_messages = [
                {"role": "assistant", "content": chosen_content}
            ]
            
            rejected_messages = [
                {"role": "assistant", "content": rejected_content}
            ]

            entry = create_dpo_entry(
                prompt=prompt_messages,
                chosen=chosen_messages,
                rejected=rejected_messages
            )
            
            dpo_data.append(entry)

    print(f"Created {len(dpo_data)} DPO training pairs")
    
    # Convert to DataFrame with explicit dtype to avoid numpy array wrapping
    dataset = pd.DataFrame(dpo_data)
    
    # Option 1: Save as Parquet with PyArrow (better type handling)
    table = pa.Table.from_pandas(dataset, preserve_index=False)
    pq.write_table(table, output_file)
    
    print(f"Successfully processed data. Saved {len(dataset)} DPO entries to {output_file}")
    
    # Verify the format
    print("\nVerifying format...")
    test_df = pd.read_parquet(output_file)
    print(f"First entry 'prompt' type: {type(test_df['prompt'].iloc[0])}")
    print(f"First entry 'chosen' type: {type(test_df['chosen'].iloc[0])}")
    print(f"First entry 'rejected' type: {type(test_df['rejected'].iloc[0])}")
    print(f"\nSample entry:")
    print(f"prompt: {test_df['prompt'].iloc[0]}")
    print(f"chosen: {test_df['chosen'].iloc[0][:100]}...")  # First 100 chars
    print(f"rejected: {test_df['rejected'].iloc[0][:100]}...")  # First 100 chars

if __name__ == "__main__":
    main()