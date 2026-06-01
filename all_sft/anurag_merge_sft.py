import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_PATH = "./sft_checkpoints/checkpoint-740"
REPO_ID = "anuragc14653/qwen_sft_fixed"

print("Load base model")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Loading LoRA weights from {ADAPTER_PATH}...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Fusing weights permanently...")
merged_model = peft_model.merge_and_unload()

print(f"Upload to HF Hub at {REPO_ID}")
merged_model.push_to_hub(REPO_ID, private=False)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.push_to_hub(REPO_ID)

print("Finished merge and upload")
