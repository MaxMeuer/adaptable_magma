import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2LMHeadModel, AutoModelForCausalLM, GPTJForCausalLM, GPTJConfig
from .utils import print_main
from pathlib import Path
from magma.config import MultimodalConfig

LANGUAGE_MODELS = [
    "gptj",
]



def get_gptj(config: MultimodalConfig,
    gradient_checkpointing: bool = True,
    from_pretrained="EleutherAI/gpt-j-6B",
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    print_main("Loading GPTJ language model...")
    print_main("From", from_pretrained)
    gptj_config = GPTJConfig.from_pretrained(from_pretrained)
    gptj_config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        gptj_config.use_cache = False

    if config.deepspeed_config_params['fp16']['enabled'] is True:
        model = GPTJForCausalLM.from_pretrained(
            from_pretrained, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, config=gptj_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(from_pretrained, config=gptj_config)

    return model
