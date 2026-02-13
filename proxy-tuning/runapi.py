"""
DExperts Model API Server with configurable models, endpoints, and GPU allocation
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import yaml
import logging
from pathlib import Path

from generation import generate_completions, load_dexperts_model_and_tokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DExperts API Server")


class Config:
    """Configuration loader from YAML file"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config

    def _default_config(self) -> dict:
        """Default configuration fallback"""
        return {
            'models': {
                'base': {
                    'path': '/storage/data/original_models/Qwen2.5-32B',
                    'devices': [0, 1]
                },
                'expert': {
                    'path': '/storage/data/original_models/Qwen3-8B',
                    'devices': [0, 3]
                },
                'antiexpert': {
                    'path': '/storage/data/original_models/Qwen3-8B-base',
                    'devices': [4, 5]
                }
            },
            'api': {'host': '0.0.0.0', 'port': 8402},
            'generation': {
                'max_new_tokens': 16000,
                'temperature': 1.0,
                'top_p': 1.0,
                'batch_size': 1,
                'alpha': 1.0
            }
        }

    def get_gpu_device_string(self) -> str:
        """Get comma-separated GPU device string for CUDA_VISIBLE_DEVICES"""
        devices = set()
        for model_type in ['base', 'expert', 'antiexpert']:
            devices.update(self.config['models'][model_type]['devices'])
        return ','.join(map(str, sorted(devices)))

    def get_device_map(self, model_type: str) -> dict:
        """Get device map for specific model"""
        devices = self.config['models'][model_type]['devices']
        if len(devices) == 1:
            return {"": f"cuda:{devices[0]}"}
        return "auto"


# Load configuration
config = Config()

# Log available GPUs (don't set CUDA_VISIBLE_DEVICES globally)
gpu_devices = config.get_gpu_device_string()
logger.info(f"Will use GPUs: {gpu_devices}")


# OpenAI-compatible API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "dexperts"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict


# Global model and tokenizer (loaded once at startup)
model = None
tokenizer = None


@app.on_event("startup")
async def load_models():
    """Load models once when server starts"""
    global model, tokenizer

    try:
        logger.info("🔄 Loading DExperts models...")

        model_config = config.config['models']
        gen_config = config.config['generation']

        # Set CUDA_VISIBLE_DEVICES based on all GPUs needed
        all_gpus = set()
        all_gpus.update(model_config['base']['devices'])
        all_gpus.update(model_config['expert']['devices'])
        all_gpus.update(model_config['antiexpert']['devices'])
        gpu_str = ','.join(map(str, sorted(all_gpus)))

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        logger.info(f"  Setting CUDA_VISIBLE_DEVICES={gpu_str}")

        logger.info(f"  Base model: {model_config['base']['path']} on GPUs {model_config['base']['devices']}")
        logger.info(f"  Expert model: {model_config['expert']['path']} on GPUs {model_config['expert']['devices']}")
        logger.info(f"  Anti-expert model: {model_config['antiexpert']['path']} on GPUs {model_config['antiexpert']['devices']}")

        model, tokenizer = load_dexperts_model_and_tokenizer(
            base_model_name_or_path=model_config['base']['path'],
            expert_model_name_or_path=model_config['expert']['path'],
            antiexpert_model_name_or_path=model_config['antiexpert']['path'],
            alpha=gen_config.get('alpha', 1.0),
        )

        logger.info("✅ DExperts models loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        raise


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "DExperts API Server",
        "config": {
            "models": {
                "base": config.config['models']['base']['path'],
                "expert": config.config['models']['expert']['path'],
                "antiexpert": config.config['models']['antiexpert']['path']
            },
            "gpu_devices": gpu_devices
        }
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": model is not None,
        "gpu_devices": gpu_devices,
        "config_path": str(config.config_path)
    }


@app.post("/v1/chat/completions")
async def chat_completion(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        # Merge messages into single prompt
        prompt = " ".join([msg.content for msg in req.messages])
        logger.info(f"Received request with prompt length: {len(prompt)} chars")
        logger.info(f"Full prompt: {prompt[:500]}...")  # Print first 500 chars
        logger.info(f"Request params: model={req.model}, max_tokens={req.max_tokens}, temp={req.temperature}, top_p={req.top_p}")

        # Get generation parameters (request overrides config)
        gen_config = config.config['generation']
        max_tokens = req.max_tokens or gen_config.get('max_new_tokens', 16000)
        temperature = req.temperature if req.temperature is not None else gen_config.get('temperature', 1.0)
        top_p = req.top_p if req.top_p is not None else gen_config.get('top_p', 1.0)

        logger.info(f"Generation params: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")

        # Generate completion
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts_an=([prompt], [""]),
            batch_size=gen_config.get('batch_size', 1),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            disable_tqdm=True
        )

        # Construct OpenAI-compatible response
        import time
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": outputs[0],
                        "refusal": None
                    },
                    "finish_reason": "stop",
                    "logprobs": None
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(outputs[0].split()),
                "total_tokens": len(prompt.split()) + len(outputs[0].split())
            }
        }

        logger.info(f"Generated response with {len(outputs[0])} chars")
        return response

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate")
async def generate(req: dict):
    """
    Legacy generation endpoint (for backwards compatibility)
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        prompts = req.get("prompts", [])
        answers = req.get("answers", [""] * len(prompts))
        max_new_tokens = req.get("max_new_tokens", config.config['generation']['max_new_tokens'])
        temperature = req.get("temperature", config.config['generation']['temperature'])
        top_p = req.get("top_p", config.config['generation']['top_p'])

        results = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts_an=(prompts, answers),
            batch_size=config.config['generation']['batch_size'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            disable_tqdm=True
        )

        return {"results": results}

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import sys

    # 支持命令行参数指定配置文件
    # 用法: python runapi.py config_service1.yaml
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        logger.info(f"Using config file: {config_file}")
        # 重新加载配置
        config.config_path = Path(config_file)
        config.config = config._load_config()

    api_config = config.config['api']
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8402)

    logger.info(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
