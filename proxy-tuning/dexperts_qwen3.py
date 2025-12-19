import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers.generation.utils import (
    StoppingCriteriaList,
    LogitsProcessorList,
    # top_k_top_p_filtering
)
from typing import Optional, Dict, Any
from collections import defaultdict
THINK_END_ID = 151668   # </think> token id（Qwen3）
HARD_ENTROPY_TH = 6.0 
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

MATH_PROMPT = "You are an expert python programmer. you will be given a question (problem specification) and will generate a correct python program that matches the specification and passes all tests."
import torch


THINK_END_ID = 151668  # </think> for Qwen3

class DExpertsMultiGPU:
    """
    Multi-GPU DExperts with entropy-based forced </think>
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

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_path, device_map={"": base_gpu}, **(model_kwargs or {})
        ).eval()
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_path, device_map={"": expert_gpu}, **(model_kwargs or {})
        ).eval()
        self.anti = AutoModelForCausalLM.from_pretrained(
            anti_model_path, device_map={"": anti_gpu}, **(model_kwargs or {})
        ).eval()

        self.base_device = torch.device(f"cuda:{base_gpu}")
        self.expert_device = torch.device(f"cuda:{expert_gpu}")
        self.anti_device = torch.device(f"cuda:{anti_gpu}")

        # entropy control
        self.entropy_th = 5.0
        self.entropy_patience = 3
        self.high_entropy_steps = 0
        self.abnormal_stop = False

    # --------------------------------------------------

    def entropy_from_logits(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        top1_prob = probs.max(dim=-1).values
        return top1_prob.item()

    # --------------------------------------------------

    def _get_tokenized_chat_inputs(self, input_ids: torch.Tensor):
        from transformers import AutoTokenizer

        prompts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        qwen_tok = AutoTokenizer.from_pretrained("/home/original_models/Qwen3-8B")

        chat_prompts = []
        for p in prompts:
            messages = [
         
                {"role": "user", "content": p},
            ]
            chat_prompts.append(
                qwen_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        chat_inputs = self.tokenizer(
            chat_prompts,
            padding="longest",
            return_tensors="pt",
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.expert_device)
        if "attention_mask" in chat_inputs:
            chat_inputs.attention_mask = chat_inputs.attention_mask.to(
                self.expert_device
            )
        return chat_inputs

    # --------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask:torch.Tensor,
        max_new_tokens: int =16000,
        do_sample: bool = False,
        temperature: float = 0.6,
    ):
        assert input_ids.size(0) == 1

        # reset state per generation
        self.high_entropy_steps = 0
        self.abnormal_stop = False

        base_ids = input_ids.to(self.base_device)
        anti_ids = input_ids.to(self.anti_device)
        expert_ids = self._get_tokenized_chat_inputs(input_ids).input_ids

        base_past = expert_past = anti_past = None
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):

            # ---------- base ----------
            base_out = self.base(
                input_ids=base_ids if base_past is None else base_ids[:, -1:],
                past_key_values=base_past,
                use_cache=True,
            )
            base_past = base_out.past_key_values
            b = base_out.logits[:, -1, :]

            # ---------- expert ----------
            expert_out = self.expert(
                input_ids=expert_ids if expert_past is None else expert_ids[:, -1:],
                past_key_values=expert_past,
                use_cache=True,
            )
            expert_past = expert_out.past_key_values
            e = expert_out.logits[:, -1, :].to(self.base_device)

            # ---------- anti ----------
            anti_out = self.anti(
                input_ids=anti_ids if anti_past is None else anti_ids[:, -1:],
                past_key_values=anti_past,
                use_cache=True,
            )
            anti_past = anti_out.past_key_values
            a = anti_out.logits[:, -1, :].to(self.base_device)

            vocab = b.size(-1)
            e = e[:, :vocab]
            a = a[:, :vocab]

            # ---------- DExperts fusion (RAW logits) ----------
            logits_raw = b + self.alpha * (e - a)

            # ---------- entropy check (NO temperature) ----------
            entropy = self.entropy_from_logits(logits_raw)
            print(f"[entropy] {entropy:.3f}")

            if entropy <0.5:
                self.high_entropy_steps += 1
            else:
                self.high_entropy_steps = 0

            # ---------- FORCE </think> ----------
            print("high_entropy_steps:", self.high_entropy_steps)
            if (
                self.high_entropy_steps >= self.entropy_patience
                and THINK_END_ID not in base_ids[0].tolist()
            ):
                print(
                    f"🔥 FORCE </think> | entropy={entropy:.2f}, "
                    f"steps={self.high_entropy_steps}"
                )
                next_token = torch.tensor(
                    [[THINK_END_ID]],
                    device=self.base_device,
                    dtype=torch.long,
                )
                self.abnormal_stop = True
            else:
                logits = logits_raw
                if temperature != 1.0:
                    logits = logits / temperature
                warpers = LogitsProcessorList([
                TopKLogitsWarper(top_k=20),
                TopPLogitsWarper(top_p=0.95),
            ])
                logits = warpers(input_ids, logits)

                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    print('sampling...')
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_id = next_token.item()
            print("next_token:", next_id)

            # ---------- stop ----------
            if next_id in {eos_id, 151645}:
                break

            # ---------- append ----------
            base_ids = torch.cat([base_ids, next_token.to(self.base_device)], dim=-1)
            expert_ids = torch.cat(
                [expert_ids, next_token.to(self.expert_device)], dim=-1
            )
            anti_ids = torch.cat(
                [anti_ids, next_token.to(self.anti_device)], dim=-1
            )

        return base_ids.cpu()
