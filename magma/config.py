from dataclasses import dataclass, asdict
import yaml
from pprint import pprint
from .utils import is_main
import os
from pathlib import Path
import uuid


def load_config(path, config_dir=Path("configs")):
    if not path.endswith(".yml"):
        path += ".yml"
    if not os.path.exists(path):
        path = config_dir / path
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


@dataclass
class MultimodalConfig:

    # Training:
    # ------------------------------------------------------------

    batch_size: int
    train_steps: int
    train_micro_batch_size_per_gpu: int
    optimizer_name: str = "AdamW"
    lr: float = 8.0e-4
    image_enc_lr: float = None
    rationals_lr: float = None
    mlp_switch_lr: float = None
    attn_switch_lr: float = None
    min_lr: float = 0.0
    lr_decay_iters: int = None
    gradient_accumulation_steps: int = 1
    image_size: int = 256
    eval_every: int = 250
    eval_steps: int = 25
    zero_stage: int = 2
    gradient_clipping: float = 1.0
    warmup_num_steps: int = 100
    weight_decay: float = 0.00
    run_blind: bool = False
    fine_tune: bool = False
    load_optimizer: bool = True
    rational_image_encoder: bool = False
    loss_scale_window: int = 1000
    fp16_enabled: bool = True
    min_loss_scale: float = 1e-6
    initial_scale_power: int = 16
    load_from_fp32_weights: int = False

    # Chefer:
    # ------------------------------------------------------------
    chefer: bool = False

    # Checkpointing:
    # ------------------------------------------------------------
    save_every: int = 500
    save: str = None
    load: str = None

    # Data:
    # ------------------------------------------------------------
    train_dataset_name: str = "conceptual_captions"
    eval_dataset_name: str = "/data/conceptual_captions"
    train_dataset_dir: str = "/data/coco_data"
    eval_dataset_dir: str = "/data/coco_data"
    eval_dataset_pct: float = 0.1

    # Model architecture:
    # ------------------------------------------------------------
    encoder_name: str = "clip"
    tokenizer_name: str = "gpt2"
    lm_name: str = "EleutherAI/gpt-j-6B"
    from_checkpoint: bool = False
    magma_checkpoint_path: str = None
    image_seq_len: int = 2
    pretrained_img_encoder: bool = False
    seq_len: int = None
    perceiver: bool = False
    dtype: str = "float32"

    # Layer Freezing settings:
    # ------------------------------------------------------------
    cross_attention_config: dict = None

    # Layer Freezing settings:
    # ------------------------------------------------------------
    freeze_lm: bool = True
    freeze_img_encoder: bool = True

    image_embed_dropout_prob: float = 0.0
    use_image_embed_layernorm: bool = False

    # Adapter settings:
    # ------------------------------------------------------------
    adapter_config: dict = None

    # Rational settings:
    # ------------------------------------------------------------
    use_cuda_kernels: bool = False

    # Classification Finetuning settings:
    # ------------------------------------------------------------
    # {num_classes: .., ckpt_path: .., classifier_type:, .., interface_type: .., interface_position: .., freeze_model: ..}
    class_dict: dict = None

    # Logging settings:
    # ------------------------------------------------------------
    name: str = None  # name, just used for wandb logging
    log_every: int = 1
    wandb_project: str = "magma"

    def print(self):
        if is_main():
            print("-" * 100)
            pprint(self.__dict__, indent=4)
            print("-" * 100)

    def __post_init__(self):
        self.is_classifier = self.class_dict is not None
        if self.adapter_config is None:
            self.adapter_config = {}

        # Deepspeed Settings:
        # ------------------------------------------------------------
        if self.lr_decay_iters is None:
            self.lr_scheduler = "WarmupLR"
            self.scheduler_dict = {
                "type": self.lr_scheduler,
                "params": {
                    "warmup_min_lr": self.min_lr,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_num_steps,
                },
            }
        else:
            self.lr_scheduler = "WarmupDecayLR"
            self.scheduler_dict = {
                "type": self.lr_scheduler,
                "params": {
                    "total_num_steps": self.lr_decay_iters,
                    "warmup_min_lr": self.min_lr,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_num_steps,
                },
            }
        self.deepspeed_config_params = {
            "train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "train__micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_clipping": self.gradient_clipping,
            "fp16": {"enabled": self.fp16_enabled, "loss_scale_window": self.loss_scale_window, 'min_loss_scale': self.min_loss_scale, 'initial_scale_power': self.initial_scale_power},
            "scheduler": self.scheduler_dict,
            "zero_optimization": {
                "stage": self.zero_stage,
                "load_from_fp32_weights": True,
                "offload_optimizer": {
                    "device": "cpu",
                },
            },

        }

        if self.name is None:
            self.name = str(uuid.uuid4())[:8]

    @classmethod
    def from_yml(cls, path):
        return cls(**load_config(path))

    def to_dict(self):
        return asdict(self)
