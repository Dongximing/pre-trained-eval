"""
DExperts Model Implementation with configurable GPU allocation and generation parameters
"""
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from transformers.generation.logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


class DExpertsLlama:
    """
    DExperts: Dynamically combining expert and anti-expert models with a base model

    This implementation supports:
    - Flexible GPU allocation for each model
    - Configurable generation parameters
    - Chat template support for expert models
    - Entropy-based and probability-based gating
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: Optional[str] = None,
        alpha: float = 1.0,
        chat_response_prefix: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        base_devices: Optional[List[int]] = None,
        expert_devices: Optional[List[int]] = None,
        antiexpert_devices: Optional[List[int]] = None,
        config_path: str = "config.yaml",
    ):
        """
        Initialize DExperts model

        Args:
            base_model_name_or_path: Path to base model
            expert_model_name_or_path: Path to expert model
            antiexpert_model_name_or_path: Path to antiexpert model
            tokenizer: Tokenizer instance
            system_prompt: Optional system prompt for chat models
            alpha: DExperts interpolation coefficient
            chat_response_prefix: Prefix for chat responses
            model_kwargs: Additional kwargs for model loading
            base_devices: GPU devices for base model (e.g., [0, 1])
            expert_devices: GPU devices for expert model (e.g., [2, 3])
            antiexpert_devices: GPU devices for antiexpert model (e.g., [4, 5])
            config_path: Path to config file
        """
        # Load configuration
        self.config = load_config(config_path)

        # Set generation parameters from config
        gen_config = self.config.get('generation', {})
        self.default_temperature = 0.7
        self.default_top_k = 20
        self.default_top_p = 0.8
        self.default_max_new_tokens = gen_config.get('max_new_tokens', 16000)

        # Get expert tokenizer path from config
        expert_config = self.config.get('models', {}).get('expert', {})
        self.expert_tokenizer_path = expert_config.get('tokenizer_path', expert_model_name_or_path)

        if model_kwargs is None:
            model_kwargs = {}

        # Load each model with constrained GPU visibility
        logger.info(f"Loading base model: {base_model_name_or_path}")
        logger.info(f"  Target devices: {base_devices}")
        self.base = self._load_model_on_devices(
            base_model_name_or_path,
            base_devices,
            model_kwargs
        )

        logger.info(f"Loading expert model: {expert_model_name_or_path}")
        logger.info(f"  Target devices: {expert_devices}")
        self.expert = self._load_model_on_devices(
            expert_model_name_or_path,
            expert_devices,
            model_kwargs
        )

        logger.info(f"Loading antiexpert model: {antiexpert_model_name_or_path}")
        logger.info(f"  Target devices: {antiexpert_devices}")
        self.antiexpert = self._load_model_on_devices(
            antiexpert_model_name_or_path,
            antiexpert_devices,
            model_kwargs
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

        logger.info(f"DExperts initialized with alpha={alpha}")
        logger.info(f"  Base device: {self.base_device}")
        logger.info(f"  Expert device: {self.expert_device}")
        logger.info(f"  Antiexpert device: {self.anti_device}")

    def _load_model_on_devices(
        self,
        model_path: str,
        devices: Optional[List[int]],
        model_kwargs: Dict[str, Any]
    ):
        """
        Load model (CUDA_VISIBLE_DEVICES already set by runapi.py)

        Args:
            model_path: Path to model
            devices: Not used (CUDA_VISIBLE_DEVICES controls GPU)
            model_kwargs: Model loading kwargs

        Returns:
            Loaded model
        """
        logger.info(f"  Loading with device_map='auto'")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
            device_map="auto"
        )

    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        """Forward pass through all three models"""
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs, antiexpert_outputs

    def _get_tokenized_chat_inputs(self, input_ids: torch.Tensor):
        """
        Convert base input_ids to chat format for expert model

        This decodes the base input, applies chat template, and re-encodes
        for the expert model.
        """
        # Decode base input to text
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        logger.debug(f"Base prompts: {prompts[0][:100]}...")

        # Load expert tokenizer for chat template
        from transformers import AutoTokenizer
        expert_tokenizer = AutoTokenizer.from_pretrained(self.expert_tokenizer_path)

        chat_prompts = []
        for p in prompts:
            messages = [
                {"role": "user", "content": p},
            ]
            chat_text = expert_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            chat_prompts.append(chat_text)

        logger.debug(f"Chat prompt: {chat_prompts[0][:100]}...")

        # Re-encode with base tokenizer
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
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        run_id: Optional[str] = None,
        use_entropy_gating: bool = False,
        entropy_threshold: float = 5.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using DExperts

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature (overrides config)
            top_k: Top-k sampling (overrides config)
            top_p: Top-p sampling (overrides config)
            stopping_criteria: Stopping criteria
            run_id: Run ID for logging
            use_entropy_gating: Whether to use entropy-based gating
            entropy_threshold: Entropy threshold for gating
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # Use config defaults if not specified
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if temperature is None:
            temperature = self.default_temperature
        if top_k is None:
            top_k = self.default_top_k
        if top_p is None:
            top_p = self.default_top_p

        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()

        input_ids = input_ids.to(self.base_device)

        # Get chat-formatted inputs for expert
        chat_inputs = self._get_tokenized_chat_inputs(input_ids)
        expert_input_ids = chat_inputs.input_ids.to(self.expert_device)

        # Initialize KV caches
        base_past = None
        expert_past = None
        anti_past = None

        bsz = input_ids.size(0)
        unfinished_sequences = torch.ones(bsz, dtype=torch.long, device=self.base_device)

        pad_id = self.tokenizer.pad_token_id

        # EOS token IDs (Qwen-specific)
        eos_ids = torch.tensor([151643, 151645], device=self.base_device)

        # Create logits warpers with configurable parameters
        warpers = LogitsProcessorList([
            TopKLogitsWarper(top_k=top_k),
            TopPLogitsWarper(top_p=top_p),
        ])

        def _per_sample_stopped(ids_on_base: torch.Tensor) -> torch.Tensor:
            """Convert global stopping criteria to per-sample bool tensor"""
            if len(stopping_criteria) == 0:
                return torch.zeros(ids_on_base.size(0), device=ids_on_base.device, dtype=torch.bool)

            stopped_list = []
            for i in range(ids_on_base.size(0)):
                s = stopping_criteria(ids_on_base[i:i+1], None)
                stopped_list.append(bool(s))
            return torch.tensor(stopped_list, device=ids_on_base.device, dtype=torch.bool)

        # Generation loop
        for step in range(max_new_tokens):
            # -------- BASE MODEL --------
            base_out = self.base(
                input_ids=input_ids if base_past is None else input_ids[:, -1:],
                past_key_values=base_past,
                use_cache=True,
            )
            base_past = base_out.past_key_values
            b = base_out.logits[:, -1, :]  # [B, V]

            # -------- EXPERT MODEL --------
            expert_out = self.expert(
                input_ids=expert_input_ids if expert_past is None else expert_input_ids[:, -1:],
                past_key_values=expert_past,
                use_cache=True,
            )
            expert_past = expert_out.past_key_values
            e = expert_out.logits[:, -1, :].to(self.base_device)

            # -------- ANTIEXPERT MODEL --------
            anti_out = self.antiexpert(
                input_ids=input_ids if anti_past is None else input_ids[:, -1:],
                past_key_values=anti_past,
                use_cache=True,
            )
            anti_past = anti_out.past_key_values
            a = anti_out.logits[:, -1, :].to(self.base_device)

            # Align vocabularies
            b = b[:, : e.size(-1)]

            # -------- DEXPERTS FUSION --------
            if use_entropy_gating:
                # Entropy-based gating
                entropy_b = -(F.softmax(b, dim=-1) * F.log_softmax(b, dim=-1)).sum(dim=-1)
                entropy_e = -(F.softmax(e, dim=-1) * F.log_softmax(e, dim=-1)).sum(dim=-1)

                # Use base model when it's confident (low entropy)
                use_base = entropy_b < entropy_threshold
                logits = torch.where(
                    use_base.unsqueeze(-1),
                    b,
                    b + self.alpha * (e - a)
                )
            else:
                # Standard DExperts combination
                logits = b + self.alpha * (e - a)

            # Apply temperature and warpers
            logits = logits / 0.7
            logits = warpers(input_ids, logits)

            # -------- SAMPLING --------
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)   # [B, 1]
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]

            # Mask finished sequences with padding
            next_tokens = torch.where(
                unfinished_sequences[:, None].bool(),
                next_tokens,
                torch.full_like(next_tokens, pad_id),
            )

            # Append new tokens
            input_ids = torch.cat([input_ids, next_tokens.to(self.base_device)], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens.to(self.expert_device)], dim=-1)

            # Check for EOS tokens
            next_tok = next_tokens.squeeze(-1).to(self.base_device)   # [B]
            is_eos = torch.isin(next_tok, eos_ids)                    # [B] bool
            unfinished_sequences = unfinished_sequences.mul((~is_eos).long())

            # Check stopping criteria
            stopped = _per_sample_stopped(input_ids)                 # [B] bool
            unfinished_sequences = unfinished_sequences.mul((~stopped).long())

            # Break if all sequences finished
            if unfinished_sequences.max() == 0:
                break

        return input_ids
