import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start_id", type=int, default=0, help="start index (inclusive)")
parser.add_argument("--end_id", type=int, default=100, help="end index (exclusive)")
parser.add_argument("--output", type=str, default="baa-27b-math500.json")
args = parser.parse_args()

start_id = args.start_id
end_id = args.end_id
output_file = args.output

print(f"▶ Running from {start_id} to {end_id - 1}")

# -----------------------------
# Load model
# -----------------------------
model_name = "/home/original_models/gemma-2-9b-it"
# device = "cuda:5,6"
MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}.\n"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("HuggingFaceH4/math-500")["test"]
print("Total dataset size:", len(dataset))

assert 0 <= start_id < len(dataset), "start_id out of range"
assert 0 < end_id <= len(dataset), "end_id out of range"
assert start_id < end_id, "start_id must be < end_id"

# -----------------------------
# Run inference only for given range
# -----------------------------
results = []

for idx in range(start_id, end_id):
    sample = dataset[idx]
    prompt = sample["problem"]

    # messages = prompt+"Please reason step by step, and put your final answer within \\boxed{}.\n"
    messages = [
                {"role": "user", "content": prompt + MATH_PROMPT}
            ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print('text',text)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = outputs[0][inputs.input_ids.shape[1]:]
    print(gen)
    response = tokenizer.decode(gen, skip_special_tokens=True)
    print(response)

    results.append({
        "current_id": idx,
        "pure_input": prompt,
        "input": messages,
        "output": [response]
    })

    print(f"[{idx}/{end_id - 1}] done")

# -----------------------------
# Save results
# -----------------------------
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved →", output_file)
