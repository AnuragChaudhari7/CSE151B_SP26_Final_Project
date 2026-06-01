# git_commitment_issues: Experiments in LLM Mathematical Reasoning

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

Note: If attempting to run any files found in folders `all_sft/` or `experiment_notebooks/`, be advised that these files have all changed locations within the repository. As such, certain relative paths or variables may not have been updated correctly and may or may not function as they did during experimentation and model training. These files are not necessary for replication, testing, and inference.


## Runtime

Using an A5000 GPU, supervised fine-tuning took approximately 14 hours on 16k_max_openr1_math_7_5k_stratified.jsonl, our synthetic dataset which was generated using stratified sampling from the Open-R1 dataset. Our stratified data keeps the ratio of MCQs to FRQs as well as the distribution of question categories (e.g., algebra, calculus, etc.) the same as public.jsonl, making it suitable for training. For our final submission, however, we opted to use the baseline model with tuned hyperparameters. Running our model on public.jsonl for evaluation and private.jsonl for our final inference took 3-4 hours on each dataset.

## Inference

Inference should be done using A5000 or L40S GPUs. We set up the virtual environment as follows:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

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

Where:

| Parameter | Description |
|---|---|
| `model_path` | (Default: "Qwen/Qwen3-4B-Thinking-2507") |
| `data_path` | (Default: "data/private.jsonl") |
| `output_path` | (Default: "results/run_inference_results_temp_0_25.csv" (NEED TO CHANGE NAME)) <br> The output directory is created automatically if it does not exist. |
| `gpu_id` | (Default: "0") <br> GPU index passed to CUDA_VISIBLE_DEVICES. |
                 

The following modules must be present in the same directory:

| Prerequisite | Description |
|---|---|
| `apply_bandaid.py` | Fixes multiple boxed{} and box-inside-think formatting issues. |
| `judger.py` | Required only when running on the public (labeled) dataset. |
