"""
Generation utilities with configurable parameters and flexible GPU allocation
"""
import torch
import tqdm
import os
from importlib import import_module
from typing import List, Tuple, Optional, Dict, Any
import yaml
from pathlib import Path
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)
from accelerate import infer_auto_device_map
import json


def ensure_dir(d):
    """Create directory if it doesn't exist"""
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


class KeyWordsCriteria(StoppingCriteria):
    """Stopping criteria based on keyword sequences"""
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts_an: Tuple[List[str], List[str]],
    batch_size: int = 2,
    stop_id_sequences: Optional[List[List[int]]] = None,
    banned_id_sequences: Optional[List[List[int]]] = None,
    banned_begin_ids: Optional[List[int]] = None,
    add_special_tokens: bool = True,
    disable_tqdm: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    run_id: Optional[str] = None,
    **generation_kwargs
) -> List[str]:
    """
    Generate completions for a list of prompts using the model

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        prompts_an: Tuple of (prompts, answers)
        batch_size: Batch size for generation (default: 2)
        stop_id_sequences: List of token ID sequences to stop generation
        banned_id_sequences: List of token ID sequences to ban
        banned_begin_ids: List of token IDs to ban at the beginning
        add_special_tokens: Whether to add special tokens
        disable_tqdm: Whether to disable progress bar
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        run_id: Optional run ID for logging
        **generation_kwargs: Additional generation parameters

    Returns:
        List of generated text completions
    """
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts_an[0]), desc="Generating Completions")

    prompts = prompts_an[0]
    answers = prompts_an[1]

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)

    # Default stop sequences (EOS token + common stop tokens for Qwen models)
    if stop_id_sequences is None:
        stop_id_sequences = [
            [tokenizer.eos_token_id],
            [151645]  # Qwen-specific stop token
        ]

    stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize batch
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts['input_ids']
        attention_mask = tokenized_prompts['attention_mask']

        # Move to correct device
        if model.base_device.type == "cuda":
            if isinstance(batch_input_ids, dict):
                for k in batch_input_ids:
                    batch_input_ids[k] = batch_input_ids[k].to(model.base.base_device)
                    attention_mask[k] = attention_mask[k].to(model.base.base_device)
            else:
                first_device = next(model.base.parameters()).device
                batch_input_ids = batch_input_ids.to(first_device)
                attention_mask = attention_mask.to(first_device)

        # Prepare logits processor
        logits_processor = None
        if banned_id_sequences or banned_begin_ids:
            logits_processor = LogitsProcessorList()
            if banned_id_sequences:
                logits_processor.append(NoBadWordsLogitsProcessor(banned_id_sequences, tokenizer.eos_token_id))
            if banned_begin_ids:
                logits_processor.append(SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1]))

        # Generate
        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            run_id=run_id if run_id else f"batch_{i}",
            **generation_kwargs
        )

        # Handle mixed tokenizers (for DExperts)
        if isinstance(batch_input_ids, dict):
            batch_input_ids = batch_input_ids.get('llama', batch_input_ids.get('base', list(batch_input_ids.values())[0]))

        # Remove stop sequences from outputs
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence
                           for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # Decode outputs
        batch_outputs_text = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts_text = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # Duplicate prompts to match return sequences
        batch_prompts_text = [prompt for prompt in batch_prompts_text for _ in range(num_return_sequences)]

        # Extract generated text (remove prompt)
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts_text, batch_outputs_text)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    if not disable_tqdm:
        progress.close()

    assert len(generations) == len(prompts) * num_return_sequences, \
        f"Number of generations ({len(generations)}) should equal number of prompts ({len(prompts)}) * num_return_sequences ({num_return_sequences})"

    return generations


def load_lm_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    device_map: str = "auto",
    devices: Optional[List[int]] = None,
    load_in_8bit: bool = False,
    convert_to_half: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
    torch_dtype: str = "auto",
) -> Tuple[Any, Any]:
    """
    Load a language model and tokenizer

    Args:
        model_name_or_path: Path to the model
        tokenizer_name_or_path: Path to the tokenizer (defaults to model path)
        device_map: Device map strategy ("auto" or custom dict)
        devices: List of GPU device IDs to use (e.g., [0, 1, 2])
        load_in_8bit: Whether to load in 8-bit precision
        convert_to_half: Whether to convert to half precision
        use_fast_tokenizer: Whether to use fast tokenizer
        padding_side: Padding side ("left" or "right")
        torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Convert torch_dtype string to actual dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype_actual = dtype_map.get(torch_dtype, "auto")

    # Build device map
    if devices and len(devices) > 0:
        # Set CUDA_VISIBLE_DEVICES if not already set
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        device_map = "auto"

    model_kwargs = {
        'device_map': device_map,
        'torch_dtype': torch_dtype_actual,
        'offload_folder': 'offload_folder',
        'offload_state_dict': True,
    }

    if load_in_8bit:
        model_kwargs['load_in_8bit'] = True

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if convert_to_half and not load_in_8bit:
        model = model.half()

    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    return model, tokenizer


def add_pad_token(tokenizer, padding_side: str = "left"):
    """Add padding token to tokenizer if it doesn't have one"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def load_dexperts_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    antiexpert_model_name_or_path: str = None,
    device_map: str = "auto",
    base_devices: Optional[List[int]] = None,
    expert_devices: Optional[List[int]] = None,
    antiexpert_devices: Optional[List[int]] = None,
    system_prompt: Optional[str] = None,
    alpha: float = 1.0,
    chat_response_prefix: Optional[str] = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
    torch_dtype: str = "bfloat16",
    config_path: str = "config.yaml",
) -> Tuple[Any, Any]:
    """
    Load DExperts model with three components: base, expert, and antiexpert

    Args:
        base_model_name_or_path: Path to base model
        expert_model_name_or_path: Path to expert model
        antiexpert_model_name_or_path: Path to antiexpert model
        device_map: Device map strategy
        base_devices: GPU devices for base model (e.g., [0, 1])
        expert_devices: GPU devices for expert model (e.g., [2, 3])
        antiexpert_devices: GPU devices for antiexpert model (e.g., [4, 5])
        system_prompt: System prompt for chat models
        alpha: DExperts alpha parameter
        chat_response_prefix: Chat response prefix
        load_in_8bit: Whether to load in 8-bit
        use_fast_tokenizer: Whether to use fast tokenizer
        padding_side: Padding side
        torch_dtype: Torch dtype
        config_path: Path to config file

    Returns:
        Tuple of (dexperts_model, tokenizer)
    """
    from transformers import AutoTokenizer
    from dexperts import DExpertsLlama

    # Load config if not provided with device info
    config = load_config(config_path)
    if config and 'models' in config:
        model_config = config['models']
        if base_devices is None:
            base_devices = model_config.get('base', {}).get('devices', None)
        if expert_devices is None:
            expert_devices = model_config.get('expert', {}).get('devices', None)
        if antiexpert_devices is None:
            antiexpert_devices = model_config.get('antiexpert', {}).get('devices', None)

        # Get advanced settings
        advanced = config.get('advanced', {})
        torch_dtype = advanced.get('torch_dtype', torch_dtype)

    # Convert torch_dtype string to actual dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype_actual = dtype_map.get(torch_dtype, torch.bfloat16)

    model_kwargs = {
        'torch_dtype': torch_dtype_actual,
        'offload_folder': 'offload_folder',
        'offload_state_dict': True,
    }

    if load_in_8bit:
        model_kwargs['load_in_8bit'] = True

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    if not antiexpert_model_name_or_path:
        antiexpert_model_name_or_path = 'meta-llama/Llama-2-7b-hf'

    # Create DExperts model with device configuration
    model = DExpertsLlama(
        base_model_name_or_path,
        expert_model_name_or_path,
        antiexpert_model_name_or_path,
        tokenizer,
        alpha=alpha,
        model_kwargs=model_kwargs,
        base_devices=base_devices,
        expert_devices=expert_devices,
        antiexpert_devices=antiexpert_devices,
    )

    return model, tokenizer


def dynamic_import_function(function_path: str):
    """
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    """
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
