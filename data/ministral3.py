import json
import torch
import argparse
from datasets import load_dataset
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config


# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start_id", type=int, default=0, help="start index (inclusive)")
parser.add_argument("--end_id", type=int, default=100, help="end index (exclusive)")
parser.add_argument("--output", type=str, default="ministral-3-8B-Instruct-math500.json")
args = parser.parse_args()

start_id = args.start_id
end_id = args.end_id
output_file = args.output

print(f"▶ Running from {start_id} to {end_id - 1}")

# -----------------------------
# Load model
# -----------------------------
model_name = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
device = "cuda:2"

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={"": device},
)
tokenizer = MistralCommonBackend.from_pretrained(model_name)
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

    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}."
        },
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            # temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(gen, skip_special_tokens=True)

    results.append({
        "current_id": idx,
        "pure_input": prompt,
        "input": text,
        "output": [response]
    })

    print(f"[{idx}/{end_id - 1}] done")

# -----------------------------
# Save results
# -----------------------------
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved →", output_file)
