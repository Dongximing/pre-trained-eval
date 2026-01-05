from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from collections import defaultdict

from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import logging

logging.basicConfig(
    filename="entropy_07_math_gem.log",      # 日志文件名
    filemode="a",                # 追加写入
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class DExpertsLlama:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        alpha: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs,device_map="auto"
        )
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path, **model_kwargs,device_map="auto"
        )
        self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path, **model_kwargs,device_map="auto"
        )

        self.base.eval()
        self.expert.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.device = self.base.device
        self.base_device = self.base.device
        self.expert_device = self.expert.device
        self.anti_device = self.antiexpert.device
    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs, antiexpert_outputs

    def _get_tokenized_chat_inputs(self, input_ids: torch.Tensor):
            """
            把 base 的 input_ids decode 成文本，再用 Qwen2.5-Coder-14B-Instruct 的
            apply_chat_template 构造 expert 的 chat 格式输入，然后用 base tokenizer 重新 encode。
            """
            # 用 base tokenizer 还原文本
            prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            print("base prompts", prompts)

            # 用 Qwen2.5-Coder-14B-Instruct 的 tokenizer 来套 chat_template
            from transformers import AutoModelForCausalLM, AutoTokenizer
            qwen_tok = AutoTokenizer.from_pretrained(
                "/home/original_models/gemma-2-9b-it"
            )

            chat_prompts = []
            for p in prompts:
                problem = p
                messages = [
                    # {"role": "system", "content": MATH_PROMPT},
                    {"role": "user", "content": problem},
                ]
                chat_text = qwen_tok.apply_chat_template(
                    messages,
                    tokenize=False,        
                    add_generation_prompt=True,
                )
                chat_prompts.append(chat_text)

            # print("chat_prompts", chat_prompts)
            # print("token_end id ", self.tokenizer.eos_token_id)

            # 再用 base tokenizer 编码（假设 base/expert 共享 vocab）
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


    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0,
        stopping_criteria: StoppingCriteriaList | None = None,
        run_id=None,
        **kwargs
    ):
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()

        input_ids = input_ids.to(self.base_device)

        chat_inputs = self._get_tokenized_chat_inputs(input_ids)
        expert_input_ids = chat_inputs.input_ids.to(self.expert_device)

        base_past = None
        expert_past = None
        anti_past = None

        bsz = input_ids.size(0)
        unfinished_sequences = torch.ones(bsz, dtype=torch.long, device=self.base_device)

        pad_id = self.tokenizer.pad_token_id

        # 你指定的“结束 token 集合”
        eos_ids = torch.tensor([1], device=self.base_device)

        warpers = LogitsProcessorList([
            TopKLogitsWarper(top_k=20),
            TopPLogitsWarper(top_p=0.8),
        ])

        def _per_sample_stopped(ids_on_base: torch.Tensor) -> torch.Tensor:
            """把 transformers 的 stopping_criteria（全局 bool）变成 per-sample 的 [B] bool。"""
            if len(stopping_criteria) == 0:
                return torch.zeros(ids_on_base.size(0), device=ids_on_base.device, dtype=torch.bool)

            stopped_list = []
            for i in range(ids_on_base.size(0)):
                # 注意：这里返回的是 python bool（transformers 默认行为）
                s = stopping_criteria(ids_on_base[i:i+1], None)
                # 强制转成 bool
                stopped_list.append(bool(s))
            return torch.tensor(stopped_list, device=ids_on_base.device, dtype=torch.bool)
        entropy_a_list = []
        entropy_b_list = []
        entropy_e_list = []
        top1_b_list = []
        top1_e_list = []
        top1_a_list = []


        for step in range(max_new_tokens):
            # -------- BASE --------
            base_out = self.base(
                input_ids=input_ids if base_past is None else input_ids[:, -1:],
                past_key_values=base_past,
                use_cache=True,
            )
            base_past = base_out.past_key_values
            b = base_out.logits[:, -1, :]  # [B, V]

            # -------- EXPERT --------
            expert_out = self.expert(
                input_ids=expert_input_ids if expert_past is None else expert_input_ids[:, -1:],
                past_key_values=expert_past,
                use_cache=True,
            )
            expert_past = expert_out.past_key_values
            e = expert_out.logits[:, -1, :].to(self.base_device)

            # -------- ANTI --------
            anti_out = self.antiexpert(
                input_ids=input_ids if anti_past is None else input_ids[:, -1:],
                past_key_values=anti_past,
                use_cache=True,
            )
            anti_past = anti_out.past_key_values
            a = anti_out.logits[:, -1, :].to(self.base_device)

            # vocab align
            e = e[:, : b.size(-1)]
            a = a[:, : b.size(-1)]
            entropy_a = -(F.softmax(a, dim=-1) * F.log_softmax(a, dim=-1)).sum(dim=-1)
            entropy_b = -(F.softmax(b, dim=-1) * F.log_softmax(b, dim=-1)).sum(dim=-1)
            entropy_e = -(F.softmax(e, dim=-1) * F.log_softmax(e, dim=-1)).sum(dim=-1)
  
            # ===== top-1 probability =====
            prob_b = F.softmax(b, dim=-1)
            prob_e = F.softmax(e, dim=-1)
            prob_a = F.softmax(a, dim=-1)
            K = 3

            topk_prob_b, topk_id_b = torch.topk(prob_b, K, dim=-1)
            topk_prob_e, topk_id_e = torch.topk(prob_e, K, dim=-1)
            topk_prob_a, topk_id_a = torch.topk(prob_a, K, dim=-1)
            tokens_b = self.tokenizer.convert_ids_to_tokens(topk_id_b[0].tolist())
            tokens_e = self.tokenizer.convert_ids_to_tokens(topk_id_e[0].tolist())
            tokens_a = self.tokenizer.convert_ids_to_tokens(topk_id_a[0].tolist())
            def fmt_topk(tokens, probs, ids):
                return ", ".join(
                    [f"{i}:{tok}({probs[i].item():.3f})"
                    for i, tok in enumerate(tokens)]
                )

            logger.info(
                f"[run_id={run_id}] step={step} | "
                f"BASE top3: {fmt_topk(tokens_b, topk_prob_b[0], topk_id_b[0])} | "
                f"EXPERT top3: {fmt_topk(tokens_e, topk_prob_e[0], topk_id_e[0])} | "
                f"ANTI top3: {fmt_topk(tokens_a, topk_prob_a[0], topk_id_a[0])}"
            )




       
            
            entropy_b_list.append(entropy_b.mean().item())
            entropy_e_list.append(entropy_e.mean().item())
            entropy_a_list.append(entropy_a.mean().item())

            top1_b_list.append(topk_prob_b[0][0].mean().item())
            top1_e_list.append(topk_prob_e[0][0].mean().item())
            top1_a_list.append(topk_prob_a[0][0].mean().item())


            # -------- DExperts fusion --------
            if topk_prob_b[0][0] > topk_prob_e[0][0]:
                logits = b 
            else:
                logits = b + self.alpha * (e - a)
            logits = logits / 0.7
            logits = warpers(input_ids, logits)

            # -------- decode --------
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)   # [B, 1]
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]

            # ⭐ 已结束的 sample 强制 PAD
            next_tokens = torch.where(
                unfinished_sequences[:, None].bool(),
                next_tokens,
                torch.full_like(next_tokens, pad_id),
            )

            # append（注意：expert 在另外的 device 就要搬过去）
            input_ids = torch.cat([input_ids, next_tokens.to(self.base_device)], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens.to(self.expert_device)], dim=-1)
            text = self.tokenizer.decode(
    input_ids[0].tolist(),
    skip_special_tokens=False
)

            logger.info(f"[run_id={run_id}] |step={step}| input_text: {text}")


            # ⭐ 多 EOS：151643/151645 任一命中就结束该 sample
            next_tok = next_tokens.squeeze(-1).to(self.base_device)   # [B]
            is_eos = torch.isin(next_tok, eos_ids)                    # [B] bool
            unfinished_sequences = unfinished_sequences.mul((~is_eos).long())

            # ⭐ per-sample stopping_criteria（谁触发谁结束）
            stopped = _per_sample_stopped(input_ids)                 # [B] bool
            unfinished_sequences = unfinished_sequences.mul((~stopped).long())

            # ⭐ 全部结束才 break
            if unfinished_sequences.max() == 0:
                break



    



        

        
        return input_ids