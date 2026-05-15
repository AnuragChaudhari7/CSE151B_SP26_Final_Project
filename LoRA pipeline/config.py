# config.py — edit MODEL_NAME and INPUT_JSON before running

import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"   # replace with your exact HF model ID

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(OUTPUT_DIR, "results")

# ── Data ──────────────────────────────────────────────────────────────────────
INPUT_JSON  = os.path.join(DATA_DIR, "training_data.json")
TRAIN_FILE  = os.path.join(PROCESSED_DIR, "train.jsonl")
VAL_FILE    = os.path.join(PROCESSED_DIR, "val.jsonl")
TEST_FILE   = os.path.join(PROCESSED_DIR, "test.jsonl")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # remainder after train+val
RANDOM_SEED = 42
MAX_SEQ_LEN = 2_048

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS        = 3
LEARNING_RATE     = 1e-4
PER_DEVICE_BATCH  = 2
GRAD_ACCUMULATION = 8
LR_SCHEDULER      = "cosine"
WARMUP_RATIO      = 0.05
BF16              = True

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_MAX_NEW_TOKENS = 1_024
