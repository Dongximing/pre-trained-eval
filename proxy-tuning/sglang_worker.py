# sglang_worker.py
import os


import torch
import sglang as sgl
from transformers import AutoTokenizer
from typing import Dict, Any
import multiprocessing as mp

# ===== 你的原始逻辑（不改） =====
class DExpertsMultiGPU:
    def __init__(self, base_model_path, expert_model_path, anti_model_path, tokenizer, alpha=1.0):
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.base_model_path = base_model_path
        self.expert_model_path = expert_model_path
        self.anti_model_path = anti_model_path
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        self.llm_base = sgl.Engine(model_path=base_model_path, tp_size=1,mem_fraction_static=0.4,)
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        self.expert_model = sgl.Engine(model_path=expert_model_path, tp_size=1,mem_fraction_static=0.4,)
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        self.antiexpert_model = (
            sgl.Engine(model_path=anti_model_path, tp_size=1,mem_fraction_static=0.4,)
            if anti_model_path else None
        )
    def _get_tokenized_chat_inputs(self, input_text):
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

        messages = [
            {"role": "user", "content": input_text},
        ]
        chat_text = qwen_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return chat_text
    def generate(self, prompt, tokenizer, max_new_tokens, sampling_params):
        original_prompt = prompt
        chat_prompt = self._get_tokenized_chat_inputs(prompt)
        # print(chat_prompt)
        import time
        time_state = time.time()
        for i in range(10000):
            print(i)
            base_output = self.llm_base.generate(original_prompt, sampling_params)

            base_logits=base_output["meta_info"]["output_token_logits"]
            base_logits = base_logits[-1:]
            # print("base_logits")
            expert_output = self.expert_model.generate(chat_prompt, sampling_params)
            expert_logits = expert_output["meta_info"]["output_token_logits"]
            expert_logits = expert_logits[-1:]
            anti_output =self.antiexpert_model.generate(original_prompt, sampling_params)
            anti_logits = anti_output["meta_info"]["output_token_logits"]
            anti_logits = anti_logits[-1:]
           

            final_logits = base_logits + self.alpha * (expert_logits - anti_logits)

            next_token = torch.argmax(final_logits, dim=1)

            if next_token.item() in (151643, 151645):
                print("end token ",next_token)
                break

            token_text = tokenizer.decode(next_token.tolist())
            original_prompt += token_text
            chat_prompt += token_text
        print("time",time.time()-time_state)
        return original_prompt
# ===== 你的逻辑结束 =====


def worker_loop(conn):
    tokenizer = AutoTokenizer.from_pretrained("/home/original_models/Qwen3-8B-base")

    model = DExpertsMultiGPU(
        base_model_path="/home/original_models/Qwen3-8B-base",
        expert_model_path="/home/original_models/Qwen3-4B",
        anti_model_path="/home/original_models/Qwen3-4B-base",
        tokenizer=tokenizer,
        alpha=1.0,
    )

    while True:
        data = conn.recv()
        if data is None:
            break

        prompt, params = data
        result = model.generate(
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=params["max_new_tokens"],
            sampling_params=params,
        )
        conn.send(result)


if __name__ == "__main__":
    parent, child = mp.Pipe()
    p = mp.Process(target=worker_loop, args=(child,))
    p.start()

    parent.send(("Hello", {"max_new_tokens": 1, "temperature": 0.6, "top_k": 20, "top_p": 0.95}))
    print(parent.recv())
