"""
02_train.py

LoRA fine-tuning on the train split.
Evaluates on the val split at the end of each epoch.
Saves the checkpoint with the lowest val loss.
The test split is never touched here.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer

import config


def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


class BestValCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float("inf")
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        val_loss = metrics.get("eval_loss", float("inf"))
        print(f"\n[Epoch {state.epoch:.1f}]  val loss = {val_loss:.4f}")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_checkpoint = state.best_model_checkpoint
            print("  ✓ New best checkpoint")


def main():
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {config.MODEL_NAME}…")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {config.MODEL_NAME}…")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.BF16 else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets…")
    train_dataset = load_jsonl(config.TRAIN_FILE)
    val_dataset   = load_jsonl(config.VAL_FILE)
    print(f"  train: {len(train_dataset):,}   val: {len(val_dataset):,}")
    print(f"  (test split is untouched until 03_evaluate.py)")

    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        bf16=config.BF16,
        fp16=not config.BF16,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.NUM_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )

    best_ckpt_callback = BestValCheckpointCallback()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LEN,
        callbacks=[best_ckpt_callback],
    )

    print("\nStarting training…")
    trainer.train()

    # Write a pointer to the best checkpoint so 03_evaluate.py picks it up
    best_path_file = Path(config.OUTPUT_DIR) / "best_checkpoint.txt"
    with best_path_file.open("w") as f:
        f.write(best_ckpt_callback.best_checkpoint or config.CHECKPOINT_DIR)

    print(f"\nBest val loss: {best_ckpt_callback.best_loss:.4f}")
    print(f"Best checkpoint path saved → {best_path_file}")


if __name__ == "__main__":
    main()
