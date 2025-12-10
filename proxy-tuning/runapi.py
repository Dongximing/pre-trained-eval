import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import torch

# === 1. import your existing code ===
from generation import generate_completions
from dexperts import load_dexperts_model_and_tokenizer
# 或 from your file that loads normal model
# from somefile import load_lm_and_tokenizer

app = FastAPI()

print("🚀 Loading models ONCE on GPU ...")

# === 2. Load model ONCE when the server starts ===
model, tokenizer = load_dexperts_model_and_tokenizer(
    base_model_path="meta-llama/Meta-Llama-3-8B",
    expert_model_path="expert/path",
    antiexpert_model_path="antiexpert/path",
    alpha=2.0,
    torch_dtype=torch.float16
)

model.eval()  # 不训练，只推理
print("🔥 Models loaded and ready!")


# === 3. Request body ===
class GenRequest(BaseModel):
    prompts: List[str]
    answers: List[str] = []  # 可选，不要求你改逻辑
    max_new_tokens: int = 300
    temperature: float = 1.0
    top_p: float = 1.0


# === 4. API endpoint ===
@app.post("/generate")
def generate(req: GenRequest):
    """
    调用你原来的 generation.py
    没有改你的生成逻辑！
    """
    results = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts_an=(req.prompts, req.answers),
        batch_size=1,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        save_dir=None,         # 不写磁盘也可以
        disable_tqdm=True
    )
    return {"results": results}


# === 5. Entry ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
