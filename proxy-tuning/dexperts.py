from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer
from transformers.generation.utils import (
    StoppingCriteriaList,
    LogitsProcessorList,
)
from collections import defaultdict

MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}.\n"  # 需要的话可以改成数学 prompt


class DExpertsLlama:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,          # 保留参数以兼容原 load_dexperts_model_and_tokenizer
        alpha: float = 3.0,
        chat_response_prefix: str = None,   # 同上
        model_kwargs: Dict[str, Any] = None
    ):
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **(model_kwargs or {})
        )
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path, **(model_kwargs or {})
        )
        self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path, **(model_kwargs or {})
        )

        self.base.eval()
        self.expert.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.device = self.base.device

        # 你这里就是想强制 expert 用 chat 格式
        self.use_chat_format_for_expert = True

    # -------------------------------
    # chat expert 输入构造
    # -------------------------------
    def _get_tokenized_chat_inputs(self, input_ids: torch.Tensor):
        """
        把 base 的 input_ids decode 成文本，再用 apply_chat_template
        构造 expert 的 chat 格式输入。
        """
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        tokenizer = AutoTokenizer.from_pretrained("/home/original_models/gemma-2-9b-it")
        chat_prompts = []
        for p in prompts:
            problem = p
            messages = [
                {"role": "user", "content": problem }
            ]
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,           # 先要字符串
                add_generation_prompt=True
            )
            chat_prompts.append(chat_text)
        print('chat_prompts',chat_prompts)
        print('toke_end id ',self.tokenizer.eos_token_id)
        chat_inputs = self.tokenizer(
            chat_prompts,
            # padding="longest",
            return_tensors="pt",
            # add_special_tokens=True,
        )

        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)
        return chat_inputs

    # -------------------------------
    # 只是方便同时 forward 三个模型
    # -------------------------------
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

    # -------------------------------
    # 可选：分析用的记录
    # -------------------------------
    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data["tokens"].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data["token_ids"].append(next_tokens)
        for name, logits in next_token_logits_dict.items():
            analysis_data[f"logits_{name}"].append(logits.unsqueeze(1))
        return analysis_data

    # -------------------------------
    # 生成主循环：不使用 KV cache
    # 和 eval/utils.generate_completions 参数逻辑兼容
    # -------------------------------
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs,
    ):
        # 保证在同一设备
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        # expert 初始输入：chat 格式
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids
            expert_attention_mask = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.clone()
            expert_attention_mask = attention_mask.clone()

        # antiexpert 初始输入和 base 一样
        antiexpert_input_ids = input_ids.clone()
        antiexpert_attention_mask = attention_mask.clone()

        # 跟原 dexperts 一样：追踪哪些句子还没结束
        unfinished_sequences = torch.ones(input_ids.size(0), dtype=torch.long, device=self.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(self.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        for _ in range(max_new_tokens):
            # ---- 不用 prepare_inputs_for_generation，不用 cache ----
            base_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            expert_inputs = {
                "input_ids": expert_input_ids,
                "attention_mask": expert_attention_mask,
            }
            antiexpert_inputs = {
                "input_ids": antiexpert_input_ids,
                "attention_mask": antiexpert_attention_mask,
            }

            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]

            # 对齐 vocab（有些 expert 会多一些无用 token）
            vocab_size = base_next_token_logits.size(-1)
            expert_next_token_logits = expert_next_token_logits[:, :vocab_size]
            antiexpert_next_token_logits = antiexpert_next_token_logits[:, :vocab_size]

            # DExperts 合成
            # print('base_next_token_logits',base_next_token_logits)
            # print('expert_next_token_logits',expert_next_token_logits)
            # print('antiexpert_next_token_logits',antiexpert_next_token_logits)
            next_token_logits = (
                base_next_token_logits
                + self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)
            )
            # print(next_token_logits)

            # 额外的 logits_processor（比如禁止某些 token）
            if logits_processor is not None:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 采样 / greedy + 简单 top-p
            if do_sample or top_p < 1.0:
                probs = F.softmax(next_token_logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    # mask 掉 cumulative_prob > top_p 的部分
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    # 置 0 再 scatter 回去
                    filtered_sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
                    probs = torch.zeros_like(probs).scatter(1, sorted_indices, filtered_sorted_probs)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            # print('next_tokens',next_tokens)
            # 已经结束的序列用 pad 补
            next_tokens = (
                next_tokens * unfinished_sequences
                + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )
            print('next_tokens',next_tokens)

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    "dexperts": next_token_logits,
                    "base": base_next_token_logits,
                    "expert": expert_next_token_logits,
                    "antiexpert": antiexpert_next_token_logits,
                }
                analysis_data = self.update_analysis_data(
                    analysis_data, next_tokens, next_token_logits_dict
                )

            # 把新 token 拼到各自 input_ids / attention_mask 后面
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))],
                dim=-1,
            )

            expert_input_ids = torch.cat([expert_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            expert_attention_mask = torch.cat(
                [expert_attention_mask, expert_attention_mask.new_ones((expert_attention_mask.size(0), 1))],
                dim=-1,
            )

            antiexpert_input_ids = torch.cat(
                [antiexpert_input_ids, next_tokens.unsqueeze(-1)], dim=-1
            )
            antiexpert_attention_mask = torch.cat(
                [antiexpert_attention_mask, antiexpert_attention_mask.new_ones((antiexpert_attention_mask.size(0), 1))],
                dim=-1,
            )

            # eval/utils 里传进来的 stopping_criteria（比如关键字停止）
            if stopping_criteria is not None and stopping_criteria(input_ids, None):
                print('stopping_criteria triggered')
                break

            # 如果 eos 出现，则标为结束
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
            if unfinished_sequences.max() == 0:
                break

        if return_logits_for_analysis:
            for k in list(analysis_data.keys()):
                if k.startswith("logits"):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
                if k == "token_ids":
                    analysis_data[k] = torch.stack(analysis_data[k], dim=1)
            print(analysis_data)
            return input_ids, analysis_data
        print('input_ids',input_ids)
        return input_ids
