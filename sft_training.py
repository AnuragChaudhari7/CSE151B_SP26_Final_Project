!source ./.venv/bin/activate

import json
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/openr1_math_7_5k_stratified.jsonl"
OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS  = 32768

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# from vllm import LLM, SamplingParams
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
from datasets import load_dataset

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0" 
DATA_PATH   = "data/openr1_math_7_5k_stratified.jsonl"
OUTPUT_PATH = "results/sft_experiments.jsonl"
MAX_TOKENS  = 32768

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

data = [json.loads(line) for line in open(DATA_PATH)]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(MODEL_ID)
gen_config = GenerationConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    tokenizer=tokenizer
)
llm.generation_config = gen_config

trainer = SFTTrainer(
    model=llm,
    args=SFTConfig(
        max_length=16384
    ),
    tokenizer=tokenizer,
    train_dataset=[data[0]],
)
trainer.train()

trainer.save_model("./sft_outputs/final_model")
tokenizer.save_pretrained("./sft_outputs/final_model")
trainer.state.save_to_json("./sft_outputs/trainer_state.json")
metrics = trainer.state.log_history
with open("./sft_outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open("./sft_outputs/training_args.json", "w") as f:
    json.dump(training_args.to_dict(), f, indent=2)