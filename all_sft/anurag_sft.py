import os
import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_DIR = "./sft_checkpoints"
DATA_PATH = "data/sft_train_fixed.jsonl" 
MAX_TOKENS = 8192

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# for extra vram space
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def load_format_and_split_data(file_path):
    with open(file_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
    
    formatted_data = []
    dropped = 0
    
    for row in raw_data:
        full_conversation = row["prompt"] + row["completion"]
        text_string = tokenizer.apply_chat_template(full_conversation, tokenize=False)
        
        # drop any rows with more than 8192 tokens
        token_ids = tokenizer(text_string)["input_ids"]
        if len(token_ids) <= MAX_TOKENS:
            formatted_data.append({"text": text_string})
        else:
            dropped += 1
            
    print(f"Loaded {len(formatted_data)} rows. Dropped {dropped} rows exceeding {MAX_TOKENS} tokens.")
    
    full_dataset = Dataset.from_list(formatted_data)
    
    # validation set (quick)
    split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
    
    return split_dataset["train"], split_dataset["test"]

train_dataset, eval_dataset = load_format_and_split_data(DATA_PATH)

# train args
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    
    # SFT-Specific Arguments
    dataset_text_field="text",
    max_length=MAX_TOKENS,

    # Memory Management 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    bf16=True,
    loss_type="chunked_nll",
    #activation_offloading=True,
    
    # Validation & Logging
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    logging_dir="./sft_logs",
    
    # Checkpointing Logic 
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1, 
    
    # Hyperparameters
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=2,
    report_to="none"
)

# SFT tariner
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

# re-running
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = True
        print("Resuming from latest checkpoint")

trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.model.save_pretrained("./qwen-lora-final")
print("Training Complete!")
