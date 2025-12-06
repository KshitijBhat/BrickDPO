from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("AvaLovelace/BrickGPT")

# Load the SFT model with its LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B-Instruct',  # Original base model
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load as PEFT model
sft_model = PeftModel.from_pretrained(base_model, "AvaLovelace/BrickGPT")
merged_model = sft_model.merge_and_unload()

# Save both model AND tokenizer
output_path = "sft_brickgpt"
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)