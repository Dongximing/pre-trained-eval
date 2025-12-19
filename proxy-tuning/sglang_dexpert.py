import torch
import sglang as sgl
import torch.nn.functional as F
from typing import Optional, Dict, Any
from collections import defaultdict
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

MATH_PROMPT = (
    "You are an expert python programmer. you will be given a question "
    "(problem specification) and will generate a correct python program "
    "that matches the specification and passes all tests."
)


class DExpertsMultiGPU:
    """
    多 GPU 版本：
    - base        → 一个 GPU
    - expert      → 一个 GPU
    - anti-expert → 一个 GPU
    """

    def __init__(
        self,
        base_model_path: str,
        expert_model_path: str,
        anti_model_path: str,
        tokenizer: PreTrainedTokenizer,
        alpha: float = 1.0,
        base_gpu: int = 0,
        expert_gpu: int = 1,
        anti_gpu: int = 2,
        model_kwargs: Dict[str, Any] = None,
    ):
        self.tokenizer = tokenizer
        self.alpha = alpha

        # ===== 只改这里：不在 __init__ 创建 Engine =====
        self.base_model_path = base_model_path
        self.expert_model_path = expert_model_path
        self.anti_model_path = anti_model_path

        self.llm_base = None
        self.expert_model = None
        self.antiexpert_model = None
        self._engine_inited = False
        # ==============================================

        self.sampling_params = {
            "temperature": 0.6,
            "max_new_tokens": 1,
            "top_p": 0.95,
            "top_k": 20,
        }

    # ✅ 新增：延迟初始化（唯一新增函数）
    def _init_engines_if_needed(self):
        if self._engine_inited:
            return

        self.llm_base = sgl.Engine(
            model_path=self.base_model_path,
            mem_fraction_static=0.3,
            tp_size=1,
        )
        self.expert_model = sgl.Engine(
            model_path=self.expert_model_path,
            mem_fraction_static=0.3,
            tp_size=1,
        )
        if self.anti_model_path:
            self.antiexpert_model = sgl.Engine(
                model_path=self.anti_model_path,
                mem_fraction_static=0.3,
                tp_size=1,
            )

        self._engine_inited = True

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

    def generate(
        self,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int,
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:

        # ✅ 关键：只在真正 generate 时才起 Engine
        self._init_engines_if_needed()

        chat_prompt = self._get_tokenized_chat_inputs(prompt)
        original_prompt = prompt

        for step in range(max_new_tokens):
            base_output = self.llm_base.generate(
                original_prompt,
                sampling_params,
            )
            base_logits = base_output[0]["meta_info"]["output_token_logits"]

            expert_output = self.expert_model.generate(
                chat_prompt,
                sampling_params,
            )
            expert_logits = expert_output[0]["meta_info"]["output_token_logits"]

            if self.antiexpert_model:
                anti_output = self.antiexpert_model.generate(
                    original_prompt,
                    sampling_params,
                )
                anti_logits = anti_output[0]["meta_info"]["output_token_logits"]
            else:
                anti_logits = torch.zeros_like(base_logits)

            final_logits = base_logits + self.alpha * (expert_logits - anti_logits)
            next_token = torch.argmax(final_logits, axis=1)

            if next_token.item() in (151643, 151645):
                break

            next_text = tokenizer.decode(next_token)
            original_prompt += next_text
            chat_prompt += next_text

        return chat_prompt
