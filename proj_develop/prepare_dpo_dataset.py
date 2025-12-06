"""
Script to prepare a DPO (Direct Preference Optimization) dataset in HuggingFace format.
"""

import pandas as pd
import ast
import pyarrow.parquet as pq
from typing import Dict, List, Any
from brickgpt.models import create_instruction

def create_dpo_entry(prompt: List[Dict[str, str]], chosen: List[Dict[str, str]], rejected: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Create a single DPO dataset entry in chat format.
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

        # The user specified: "The last entry of the array is the chosen response"
        chosen_content = bricks_candidates[-1]
        
        # "all the entries df['generated_bricks'][0:-1] are the rejected ones"
        rejected_candidates = bricks_candidates[:-1]

        # Pair each rejected candidate with the single chosen response
        for rejected_content in rejected_candidates:
            
            # Construct messages in the chat format
            # System prompt is implied or added to prompt depending on specific DPO trainer requirements.
            # Here we separate the user prompt history from the assistant response.
            
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": create_instruction(caption)}
            ]
            
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

    # Convert to DataFrame
    dataset = pd.DataFrame(dpo_data)
    
    # Save to Parquet
    dataset.to_parquet(output_file)
    print(f"Successfully processed data. Saved {len(dataset)} DPO entries to {output_file}")

if __name__ == "__main__":
    main()
    output_file = "datasets/dpo_datasets/combined_dataset/dpo_hf.parquet"
    # Load and verify the saved dataset
    loaded_dataset = pd.read_parquet(output_file)
    # breakpoint()
    print(f"\nLoaded dataset shape: {loaded_dataset.shape}")
    print(f"Columns: {loaded_dataset.columns.tolist()}")
    print(f"\nFirst entry:")
    print(loaded_dataset.iloc[0])

