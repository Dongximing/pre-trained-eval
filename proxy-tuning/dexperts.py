# import torch
# import torch.nn.functional as F
# import transformers
# from transformers import AutoModelForCausalLM, PreTrainedTokenizer
# from transformers.generation.utils import (
#     StoppingCriteriaList,
#     LogitsProcessorList,
#     # top_k_top_p_filtering
# )

# from transformers.generation.logits_process import (
#     LogitsProcessorList,
#     TopKLogitsWarper,
#     TopPLogitsWarper,
# )

# from typing import Optional, Dict, Any
# from collections import defaultdict

# MATH_PROMPT = "You are an expert python programmer. you will be given a question (problem specification) and will generate a correct python program that matches the specification and passes all tests."
# class DExpertsMultiGPU:
#     """
#     多 GPU 版本：
#     - base        → 一个 GPU
#     - expert      → 一个 GPU
#     - anti-expert → 一个 GPU
#     支持 KV cache、高速生成、无 NVLink 交叉访问。
#     """

#     def __init__(
#         self,
#         base_model_path: str,
#         expert_model_path: str,
#         anti_model_path: str,
#         tokenizer: PreTrainedTokenizer,
#         alpha: float = 1.0,
#         base_gpu: int = 0,
#         expert_gpu: int = 0,
#         anti_gpu: int = 0,
#         model_kwargs: Dict[str, Any] = None
#     ):
#         self.tokenizer = tokenizer
#         self.alpha = alpha

#         # --------------------------
#         # Load models to specific GPUs
#         # --------------------------
#         self.base = AutoModelForCausalLM.from_pretrained(
#             base_model_path, device_map={"": base_gpu}, **model_kwargs,
#         ).eval()

#         self.expert = AutoModelForCausalLM.from_pretrained(
#             expert_model_path, device_map={"": expert_gpu}, **model_kwargs,
#         ).eval()

#         self.anti = AutoModelForCausalLM.from_pretrained(
#             anti_model_path, device_map={"": anti_gpu}, **model_kwargs,
#         ).eval()

#         # Save devices
#         self.base_device = torch.device(f"cuda:{base_gpu}")
#         self.expert_device = torch.device(f"cuda:{expert_gpu}")
#         self.anti_device = torch.device(f"cuda:{anti_gpu}")

#         print("🚀 Models loaded successfully on GPUs:",
#               self.base_device, self.expert_device, self.anti_device)

#     # ----------------------------------------------
#     # Helper: update KV-cache for next step
#     # ----------------------------------------------
#     def _update_kv(self, outputs, kwargs):
#         kwargs["past_key_values"] = outputs.past_key_values
#         if "attention_mask" in kwargs:
#             attn = kwargs["attention_mask"]
#             kwargs["attention_mask"] = torch.cat(
#                 [attn, attn.new_ones((attn.size(0), 1))],
#                 dim=-1
#             )
#         return kwargs
#         # ----------------------------------------------
#     def detect_repetition(self,ids, min_block=10, tolerance=0):
#         """
#         动态窗口重复检测：自动从 10 到 length//2 尝试匹配重复块。
#         tolerance=0 表示完全相同（推荐）
#         """

#         seq = ids[0].tolist()
#         L = len(seq)

#         # 最小长度不足时直接返回
#         if L < min_block * 2:
#             return False

#         # 尝试不同窗口大小
#         max_block = L // 2

#         for k in range(min_block, max_block + 1):

#             A = seq[-k:]          # 最后 k token
#             B = seq[-2*k : -k]    # 再前面 k token

#             # 完全相等 → 重复
#             if A == B:
#                 print(f"[REPEAT DETECTED] block size={k}")
#                 return True

#             # 如果你想允许一点点不同，可以使用 tolerance
#             if tolerance > 0:
#                 diff = sum(a != b for a, b in zip(A, B))
#                 if diff <= tolerance:
#                     print(f"[REPEAT DETECTED] block size={k}, diff={diff}")
#                     return True

#         return False
#     def _get_tokenized_chat_inputs(self, input_ids: torch.Tensor):
#         """
#         把 base 的 input_ids decode 成文本，再用 Qwen2.5-Coder-14B-Instruct 的
#         apply_chat_template 构造 expert 的 chat 格式输入，然后用 base tokenizer 重新 encode。
#         """
#         # 用 base tokenizer 还原文本
#         prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         print("base prompts", prompts)

#         # 用 Qwen2.5-Coder-14B-Instruct 的 tokenizer 来套 chat_template
#         from transformers import AutoModelForCausalLM, AutoTokenizer
#         qwen_tok = AutoTokenizer.from_pretrained(
#             "/home/original_models/Qwen3-14B"
#         )

#         chat_prompts = []
#         for p in prompts:
#             problem = p
#             messages = [
#                 # {"role": "system", "content": MATH_PROMPT},
#                 {"role": "user", "content": problem},
#             ]
#             chat_text = qwen_tok.apply_chat_template(
#                 messages,
#                 tokenize=False,        # 先生成字符串
#                 add_generation_prompt=True,
#                 enable_thinking=False,
#             )
#             chat_prompts.append(chat_text)

#         print("chat_prompts", chat_prompts)
#         print("token_end id ", self.tokenizer.eos_token_id)

#         # 再用 base tokenizer 编码（假设 base/expert 共享 vocab）
#         chat_inputs = self.tokenizer(
#             chat_prompts,
#             padding="longest",
#             return_tensors="pt",
#         )

#         chat_inputs.input_ids = chat_inputs.input_ids.to(self.expert_device)
#         if "attention_mask" in chat_inputs:
#             chat_inputs.attention_mask = chat_inputs.attention_mask.to(
#                 self.expert_device
#             )

#         return chat_inputs

#     # ----------------------------------------------
#     # Main KV-based autoregressive generation
#     # ----------------------------------------------
#     @torch.inference_mode()
#     def generate(
#             self,
#             input_ids: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             max_new_tokens: int = 256,
#             do_sample: bool = False,
#             temperature: float = 1.0,
#             top_p: float = 1.0,
#             stopping_criteria: Optional[StoppingCriteriaList] = None,
#             logits_processor: Optional[LogitsProcessorList] = None,
#     ):
#         # batch_size = input_ids.size(0)
#         # assert batch_size == 1, "Only batch=1 supported"

#         # move prompt to the 3 GPUs
#         base_ids = input_ids.to(self.base_device)
#         # expert_ids = input_ids.to(self.expert_device)
#         anti_ids = input_ids.to(self.anti_device)
#         chat_inputs = self._get_tokenized_chat_inputs(input_ids)  # CPU/任意 → expert_device
#         expert_ids = chat_inputs.input_ids 
#         print(base_ids)
#         print("---------------------------")

#         # KV cache placeholders
#         base_past = None
#         expert_past = None
#         anti_past = None

#         eos_id = self.tokenizer.eos_token_id
#         unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
#         eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)
#         import time
#         time_state = time.time()
#         if stopping_criteria is not None:
#             print(stopping_criteria)
#         for step in range(1600):
#             # print(step)

#             # ================= BASE =================
#             base_out = self.base(
#                 input_ids=base_ids if base_past is None else base_ids[:, -1:],
#                 past_key_values=base_past,
#                 use_cache=True,
#             )
#             base_past = base_out.past_key_values
#             b = base_out.logits[:, -1, :]  # (1, vocab)

#             # ================= EXPERT =================
#             expert_out = self.expert(
#                 input_ids=expert_ids if expert_past is None else expert_ids[:, -1:],
#                 past_key_values=expert_past,
#                 use_cache=True,
#             )
#             expert_past = expert_out.past_key_values
#             e = expert_out.logits[:, -1, :].to(self.base_device)

#             # ================= ANTI =================
#             anti_out = self.anti(
#                 input_ids=anti_ids if anti_past is None else anti_ids[:, -1:],
#                 past_key_values=anti_past,
#                 use_cache=True,
#             )
#             anti_past = anti_out.past_key_values
#             a = anti_out.logits[:, -1, :].to(self.base_device)

#             # align vocab

#             # ================== DEXPERTS FUSION ==================
#             logits = b + self.alpha * (e - a)
#             temperature = 0.6
#             if temperature != 1.0:
#                 # print("debug: temperature filtering applied")
#                 logits = logits / temperature

#             # 3. top-k / top-p warpers
            
#             warpers = LogitsProcessorList([
#                 TopKLogitsWarper(top_k=20),
#                 TopPLogitsWarper(top_p=0.95),
#             ])
#             logits = warpers(input_ids, logits)
#             if do_sample:
#                 probs = torch.softmax(logits, dim=-1)
#                 next_tokens = torch.multinomial(probs, 1)
#             else:
#                 next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
#             print("next_tokensnext_tokens",next_tokens)
#             next_tokens = (
#                 next_tokens * unfinished_sequences[:, None] +
#                 self.tokenizer.pad_token_id * (1 - unfinished_sequences[:, None])
#             )
#             base_ids = torch.cat([base_ids, next_tokens.to(self.base_device)], dim=-1)
#             expert_ids = torch.cat([expert_ids, next_tokens.to(self.expert_device)], dim=-1)
#             anti_ids = torch.cat([anti_ids, next_tokens.to(self.anti_device)], dim=-1)
#             print("next_tokens:\n",next_tokens)
#             if stopping_criteria(base_ids, None).all():
#                 break
#             unfinished_sequences = unfinished_sequences.mul(
#                 next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
#             )
#             print(' unfinished_sequences',unfinished_sequences)

#             # stop when each sentence is finished
#             if unfinished_sequences.max() == 0:
#                 break

            
#             # if self.detect_repetition(base_ids):
#             #     print("STOP: repetition detected")
#             #     break

            
#         print("time",time.time()-time_state)
#         return base_ids.cpu()



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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
                "/home/original_models/Qwen3-14B"
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
                    tokenize=False,        # 先生成字符串
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                chat_prompts.append(chat_text)

            print("chat_prompts", chat_prompts)
            print("token_end id ", self.tokenizer.eos_token_id)

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
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ):
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()

        # ===== devices =====
        input_ids = input_ids.to(self.base_device)

        chat_inputs = self._get_tokenized_chat_inputs(input_ids)
        expert_input_ids = chat_inputs.input_ids.to(self.expert_device)

        # ===== KV cache =====
        base_past = None
        expert_past = None
        anti_past = None

        # ===== unfinished =====
        bsz = input_ids.size(0)
        unfinished_sequences = torch.ones(
            bsz, dtype=torch.long, device=self.base_device
        )
        eos_id = self.tokenizer.eos_token_id

        # ===== warpers =====
        warpers = LogitsProcessorList([
            TopKLogitsWarper(top_k=20),
            TopPLogitsWarper(top_p=0.95),
        ])

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

            # -------- DExperts --------
            logits = b + self.alpha * (e - a)
            logits = logits / temperature
            logits = warpers(input_ids, logits)

            # -------- decode --------
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)  # [B,1]
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]

            # -------- mask finished (SAFE) --------
            next_tokens = torch.where(
                unfinished_sequences[:, None].bool(),
                next_tokens,
                torch.full_like(next_tokens, self.tokenizer.pad_token_id),
            )

            # -------- append --------
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens], dim=-1)

            # -------- stopping --------
            if stopping_criteria(input_ids, None).all():
                break

            # -------- update unfinished --------
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.squeeze(-1).ne(eos_id).long()
            )

            if unfinished_sequences.max() == 0:
                break

        return input_ids
