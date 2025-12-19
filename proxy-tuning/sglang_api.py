# server.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import sglang as sgl
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# =========================
# Prompt
# =========================
MATH_PROMPT = (
    "You are an expert python programmer. you will be given a question "
    "(problem specification) and will generate a correct python program "
    "that matches the specification and passes all tests."
)

# =========================
# DExpertsMultiGPU (你的逻辑，不改)
# =========================
class DExpertsMultiGPU:
    def __init__(
        self,
        base_model_path: str,
        expert_model_path: str,
        anti_model_path: Optional[str],
        tokenizer: PreTrainedTokenizer,
        alpha: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.alpha = alpha

        self.base_model_path = base_model_path
        self.expert_model_path = expert_model_path
        self.anti_model_path = anti_model_path

        self.llm_base = None
        self.expert_model = None
        self.antiexpert_model = None
    #     self._engine_inited = False

    # # ⚠️ 只负责「用」，不在这里创建
    # def _assert_inited(self):
    #     if not self._engine_inited:
    #         raise RuntimeError("Engines not initialized")

    def _get_tokenized_chat_inputs(self, input_text: str) -> str:
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        messages = [{"role": "user", "content": input_text}]
        return qwen_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def generate(
        self,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int,
        sampling_params: Dict[str, Any],
    ) -> str:
        # self._assert_inited()

        chat_prompt = self._get_tokenized_chat_inputs(prompt)
        original_prompt = prompt

        for _ in range(max_new_tokens):
            base_output = self.llm_base.generate(original_prompt, sampling_params)
            base_logits = base_output[0]["meta_info"]["output_token_logits"]

            expert_output = self.expert_model.generate(chat_prompt, sampling_params)
            expert_logits = expert_output[0]["meta_info"]["output_token_logits"]

            if self.antiexpert_model:
                anti_output = self.antiexpert_model.generate(original_prompt, sampling_params)
                anti_logits = anti_output[0]["meta_info"]["output_token_logits"]
            else:
                anti_logits = torch.zeros_like(base_logits)

            final_logits = base_logits + self.alpha * (expert_logits - anti_logits)
            next_token = torch.argmax(final_logits, dim=1)

            if next_token.item() in (151643, 151645):
                break

            next_text = tokenizer.decode(next_token)
            original_prompt += next_text
            chat_prompt += next_text

        return chat_prompt


# =========================
# Utils
# =========================
def load_dexperts_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    antiexpert_model_name_or_path: Optional[str],
    alpha: float,
):
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer_expert = AutoTokenizer.from_pretrained(expert_model_name_or_path)

    model = DExpertsMultiGPU(
        base_model_path=base_model_name_or_path,
        expert_model_path=expert_model_name_or_path,
        anti_model_path=antiexpert_model_name_or_path,
        tokenizer=tokenizer_base,
        alpha=alpha,
    )
    return model, tokenizer_base, tokenizer_expert


def generate_completions(
    model: DExpertsMultiGPU,
    prompt: str,
    tokenizer_base: PreTrainedTokenizer,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
):
    sampling_params = {
        "max_new_tokens": 1,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    return model.generate(
        prompt=prompt,
        tokenizer=tokenizer_base,
        max_new_tokens=max_new_tokens,
        sampling_params=sampling_params,
    )


# =========================
# FastAPI
# =========================
app = FastAPI()

model: DExpertsMultiGPU = None
tokenizer_base = None
tokenizer_expert = None


@app.on_event("startup")
def startup_event():
    global model, tokenizer_base, tokenizer_expert

    print("🚀 Loading DExperts (main thread)...")

    model, tokenizer_base, tokenizer_expert = load_dexperts_model_and_tokenizer(
        base_model_name_or_path="/home/original_models/Qwen3-8B-base",
        expert_model_name_or_path="/home/original_models/Qwen3-4B",
        antiexpert_model_name_or_path="/home/original_models/Qwen3-4B-base",
        alpha=1.0,
    )

    # ⭐⭐⭐ 在主线程创建 sgl.Engine ⭐⭐⭐
    model.llm_base = sgl.Engine(
        model_path=model.base_model_path,
        mem_fraction_static=0.3,
        tp_size=1,
    )
    model.expert_model = sgl.Engine(
        model_path=model.expert_model_path,
        mem_fraction_static=0.3,
        tp_size=1,
    )
    if model.anti_model_path:
        model.antiexpert_model = sgl.Engine(
            model_path=model.anti_model_path,
            mem_fraction_static=0.3,
            tp_size=1,
        )

    # model._engine_inited = True
    print("🔥 DExperts engines initialized!")


# =========================
# OpenAI-compatible schema
# =========================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "dexperts"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 16000
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20


@app.post("/v1/chat/completions")
def chat_completion(req: ChatCompletionRequest):
    prompt = MATH_PROMPT + "\n".join(m.content for m in req.messages)

    text = generate_completions(
        model=model,
        prompt=prompt,
        tokenizer_base=tokenizer_base,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_new_tokens=req.max_tokens,
    )

    return {
        "id": "chatcmpl-local-001",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8333)
