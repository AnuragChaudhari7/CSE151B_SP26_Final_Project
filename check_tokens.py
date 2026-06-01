import json
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuration
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/8k_max_openr1_math_7_5k_stratified.jsonl"
MAX_TOKENS = 8192

def analyze_dataset():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    total_rows = 0
    kept_rows = 0
    dropped_rows = 0
    
    print(f"Analyzing token counts in {DATA_PATH}...")
    
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
        
        for line in tqdm(lines, desc="Tokenizing"):
            total_rows += 1
            row = json.loads(line)
            
            full_conversation = row["prompt"] + row["completion"]
            
            text_string = tokenizer.apply_chat_template(full_conversation, tokenize=False)
            
            token_ids = tokenizer(text_string)["input_ids"]
            token_count = len(token_ids)
            
            if token_count <= MAX_TOKENS:
                kept_rows += 1
            else:
                dropped_rows += 1

    # Summary
    print("TOKEN LIMIT ANALYSIS REPORT:")
    print()
    print(f"Total rows in dataset   : {total_rows}")
    print(f"Rows > {MAX_TOKENS} tokens  : {dropped_rows}")
    print(f"Rows <= {MAX_TOKENS} tokens : {kept_rows}")
    print(f"FINAL TRAINING SET SIZE : {kept_rows} rows")

if __name__ == "__main__":
    analyze_dataset()