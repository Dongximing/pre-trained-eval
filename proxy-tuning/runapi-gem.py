import json
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_gem import load_lm_and_tokenizer, load_dexperts_model_and_tokenizer, generate_completions
# ==================================================
# Argument Parsing
# ==================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--start_id",
    type=int,
    default=0,
    help="start index (inclusive)",
)
parser.add_argument(
    "--end_id",
    type=int,
    default=1,
    help="end index (exclusive)",
)
parser.add_argument(
    "--output",
    type=str,
    default="qwen1.5-14b.json",
)
args = parser.parse_args()

start_id = args.start_id
end_id = args.end_id
output_file = args.output

print(f"▶ Running from {start_id} to {end_id - 1}")

print("🚀 Loading DExperts...")
model, tokenizer = load_dexperts_model_and_tokenizer(
    base_model_name_or_path="/home/original_models/gemma-2-27b",
    expert_model_name_or_path="/home/original_models/gemma-2-9b-it",
    antiexpert_model_name_or_path="/home/original_models/gemma-2-9b",
    alpha=1.0,
)
print("🔥 DExperts loaded!")


# ==================================================
# Dataset Loading
# ==================================================
dataset = load_dataset("HuggingFaceH4/math-500")["test"]
print("Total dataset size:", len(dataset))

assert 0 <= start_id < len(dataset), "start_id out of range"
assert 0 < end_id <= len(dataset), "end_id out of range"
assert start_id < end_id, "start_id must be < end_id"

dataset = load_dataset("HuggingFaceH4/math-500")["test"]
print("Total dataset size:", len(dataset))

assert 0 <= start_id < len(dataset), "start_id out of range"
assert 0 < end_id <= len(dataset), "end_id out of range"
assert start_id < end_id, "start_id must be < end_id"


# ==================================================
# Inference
# ==================================================
results = []

for idx in range(start_id, end_id):
    sample = dataset[idx]
    prompt = sample["problem"]

    messages = (
        prompt
        + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    responses = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts_an=([messages], [""]),
        batch_size=1,
        max_new_tokens=4096,
        temperature=1.0,
        top_p=0.9,
        disable_tqdm=True,
        run_id=idx,
    )

    results.append(
        {
            "current_id": idx,
            "pure_input": prompt,
            "input": messages,
            "output": [responses],
        }
    )

    print(f"[{idx}/{end_id - 1}] done")


# ==================================================
# Save Results
# ==================================================
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved →", output_file)

