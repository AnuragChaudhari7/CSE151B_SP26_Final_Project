# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |


## Runtime

Model training took approximately \[training time\] hours (16k_max_openr1_math_7_5k_stratified.jsonl) on a \[type\] GPU. Running our trained model on public.jsonl for evaluation and private.jsonl for our final inference took 2 hours on each dataset.

## Setup


## Inference

The simplest way to run with all defaults (baseline Qwen model, data/private.jsonl input, output to results/):

    python run_inference.py

To call it with custom arguments:

    from run_inference import run_inference

    # Baseline model
    run_inference()

    # Custom paths or GPU
    run_inference(
        model_path="Qwen/Qwen3-4B-Thinking-2507",
        data_path="data/public.jsonl",
        output_path="results/my_run.csv",
        gpu_id="0",
    )

Parameters:

  model_path   - (Default: "Qwen/Qwen3-4B-Thinking-2507")

  data_path    - (Default: "data/private.jsonl")

  output_path  - (Default: "results/run_inference_results_temp_0_25.csv" (NEED TO CHANGE NAME))
                 The output directory is created automatically if it does not exist.

  gpu_id       - (Default: "0")
                 GPU index passed to CUDA_VISIBLE_DEVICES.

The following modules must be present in the same directory:

  apply_bandaid.py  - Fixes multiple boxed{} and box-inside-think formatting issues.
  judger.py         - Required only when running on the public (labeled) dataset.
