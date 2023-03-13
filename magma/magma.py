from pathlib import Path
from os.path import exists
import torch
import torch.nn as nn
from copy import deepcopy
from einops import rearrange
from typing import Literal, Optional, List
from torchtyping import TensorType
from transformers.file_utils import ModelOutput
from magma.config import MultimodalConfig
from .perceiver_resampler import PerceiverResampler
from .cross_attention import GatedCrossAttentionBlock
from torch.nn.modules.container import ModuleList, Sequential
from transformers import AutoModelForCausalLM
from torch.nn.parameter import Parameter
from activations.torch import Rational
from collections import OrderedDict

from magma.utils import get_tokenizer
from .language_model import get_gptj
from .adapters import (
    Adapter,
    ParallelAdapter,
    AdapterWrapper,
    ParallelAdapterWrapper,
)
from .image_prefix import ImagePrefix
from .sampling import generate
from .utils import build_labels, is_url, print_main, download_checkpoint, freeze_rational_clip, get_world_info
from .image_input import ImageInput
from .transforms import get_transforms
import inspect
# ------------------------- Magma main class ----------------------------------


class Magma(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, (str, Path)):
            config = MultimodalConfig.from_yml(
                config
            )  # load config from yml file if config is a string
        else:
            assert isinstance(config, MultimodalConfig)
        self.config = config

        # .to(self.device)
        self.lm = get_gptj(config, from_pretrained=config.lm_name)
        self.seq_len = self.lm.config.max_position_embeddings

        self.tokenizer = get_tokenizer(
            config.tokenizer_name, sequence_length=self.seq_len)

        self.image_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.lm.config.pad_token_id = self.tokenizer.eos_token_id

        self.lm.resize_token_embeddings(len(self.tokenizer))
        self.word_embedding = self.lm.transformer.wte  # .to(device)
        self.transformer = self.lm.transformer.h

        # adapter settings
        self.mlp_adapter_added, self.attn_adapter_added = False, False

        self.image_prefix = ImagePrefix(
            config=config,
            out_dim=self.lm.config.hidden_size,
        )  # .to(self.device)

        # might change based on the type of image encoder, so get from prefix instead of config
        self.image_prefix_seq_len = self.image_prefix.out_seq_len
        self.transforms = get_transforms(
            config.image_size,
            config.encoder_name,
            input_resolution=self.image_prefix.enc.input_resolution,
        )

        for name, param in self.named_parameters():
            if param.is_contiguous() is False:
                path, param = name.rsplit(".", 1)
                path = path.split('.')
                ref = self
                while path:
                    element, path = path[0], path[1:]
                    if type(ref) in {Sequential, ModuleList}:
                        ref = ref[int(element)]
                    else:
                        ref = getattr(ref, element)
                setattr(ref, param, Parameter(
                    getattr(ref, param).contiguous()))

        # add cross attention
        if config.cross_attention_config:
            self.cross_attention_layers = []
            self.lm.resize_token_embeddings(len(self.tokenizer))
            self.word_embedding = self.lm.transformer.wte

            self.perceiver_resampler = PerceiverResampler(
                self.image_prefix.encoder_out_dim,
                n_latents=config.cross_attention_config['n_latents'],
                num_layers=config.cross_attention_config["num_layers"]
            )

            self.add_cross_attention_modules()

        # add adapters
        if config.adapter_config:
            mlp_config = deepcopy(config.adapter_config.get("mlp", None))
            if mlp_config:
                assert mlp_config.get("adapter_type") is not None
                self.add_adapters(
                    location="mlp",
                    adapter_type=mlp_config.pop("adapter_type"),
                    downsample_factor=mlp_config.pop(
                        "downsample_factor", 4),
                    **mlp_config,
                )
            attn_config = deepcopy(
                config.adapter_config.get("attention", None))
            if attn_config:
                assert attn_config.get("adapter_type") is not None
                self.add_adapters(
                    location="attention",
                    adapter_type=attn_config.pop("adapter_type"),
                    **attn_config,
                )

        # freeze parameters
        if config.freeze_lm:
            for name, param in self.lm.named_parameters():  # freeze lm weights
                if config.adapter_config and not config.adapter_config.get('freeze', False):
                    if any(map(name.__contains__, ['adapter', 'switch_logits', "gated_x_attn"])):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if any(map(name.__contains__, ["ln_1.1."])):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


        if config.freeze_img_encoder:
            if config.rational_image_encoder:
                self.image_prefix.enc = freeze_rational_clip(
                    self.image_prefix.enc)
            else:
                for param in self.image_prefix.enc.parameters():
                    param.requires_grad = False

        # I need this to be able to load the model from a checkpoint
        # for name, param in self.named_parameters():
        #     if param.is_contiguous() is False:
        #         path, param = name.rsplit(".", 1)
        #         path = path.split('.')
        #         ref = self
        #         while path:
        #             element, path = path[0], path[1:]
        #             if type(ref) in {Sequential, ModuleList}:
        #                 ref = ref[int(element)]
        #             else:
        #                 ref = getattr(ref, element)
        #         setattr(ref, param, Parameter(
        #             getattr(ref, param).contiguous()))

    def add_cross_attention_modules(self):
        for l in range(len(self.lm.transformer.h)):
            if l % self.config.cross_attention_config['cadence'] != 0:
                continue

            pretrained_block = getattr(self.lm.transformer.h[l], "ln_1")
            x_attn_block = GatedCrossAttentionBlock(
                config=self.config,
                text_token_dim=self.lm.config.hidden_size,
                visual_token_dim=self.image_prefix.encoder_out_dim
            )
            self.cross_attention_layers.append(l)

            # setattr(self.lm.transformer.h, f'{l}', nn.Sequential(
            #     OrderedDict([("self_attn",pretrained_block),("gated_x_attn" ,x_attn_block)])))
            setattr(self.lm.transformer.h[l], "ln_1", nn.Sequential(*[pretrained_block,x_attn_block]))

    def add_adapters(
        self,
        downsample_factor: int = 4,
        adapter_type: Literal["normal", "parallel",
                              "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
        ff_attr: str = "mlp",
        attn_attr: str = "attn",

        **adapter_kwargs,
    ):
        """
        Adds an adapter layer to `self` at the specified location
        """
        assert adapter_type in [
            "normal",
            "parallel",
            "scaled_parallel",
        ], "adapter_type must be one of 'normal', 'parallel', or 'scaled_parallel'"
        assert location in [
            "mlp",
            "attention",
        ], "location must be one of 'mlp' or 'attention'"

        for l in range(len(self.transformer)):
            if location == "mlp":
                if self.mlp_adapter_added:
                    raise ValueError("Adapter layer already added")
                mlp = getattr(self.transformer[l], ff_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adapter_layer = ParallelAdapter(
                        module=mlp,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled=adapter_type == "scaled_parallel",
                        hidden_act=self.config.adapter_config.get(
                            'hidden_act', False),
                        adapter_switch=self.config.adapter_config.get(
                            'adapter_switch', False),
                        switch_temp=self.config.adapter_config.get(
                            'switch_temp', None),
                        use_cuda_kernels=self.config.use_cuda_kernels,
                        **adapter_kwargs,
                    )
                else:
                    adpt = Adapter(
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        hidden_act=self.config.adapter_config.get(
                            'hidden_act', False),
                        adapter_switch=self.config.adapter_config.get(
                            'adapter_switch', False),
                        switch_temp=self.config.adapter_config.get(
                            'switch_temp', None),
                        use_cuda_kernels=self.config.use_cuda_kernels,
                        ** adapter_kwargs,
                    )
                    adapter_layer = nn.Sequential(
                        *[
                            mlp,
                            adpt,
                        ]
                    )
                setattr(self.transformer[l], ff_attr, adapter_layer)
            else:
                if self.attn_adapter_added:
                    raise ValueError("Adapter layer already added")
                attn = getattr(self.transformer[l], attn_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adapter_layer = ParallelAdapterWrapper(
                        module=attn,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled="scaled" in adapter_type,
                        hidden_act=self.config.adapter_config.get(
                            'hidden_act', False),
                        adapter_switch=self.config.adapter_config.get(
                            'adapter_switch', False),
                        switch_temp=self.config.adapter_config.get(
                            'switch_temp', None),
                        use_cuda_kernels=self.config.use_cuda_kernels,
                        **adapter_kwargs,
                    )
                else:
                    adapter_layer = AdapterWrapper(
                        attn_block=attn,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        hidden_act=self.config.adapter_config.get(
                            'hidden_act', False),
                        adapter_switch=self.config.adapter_config.get(
                            'adapter_switch', False),
                        switch_temp=self.config.adapter_config.get(
                            'switch_temp', None),
                        use_cuda_kernels=self.config.use_cuda_kernels,
                        **adapter_kwargs,
                    )
                setattr(self.transformer[l], attn_attr, adapter_layer)

        if location == "mlp":
            self.mlp_adapter_added = True
        else:
            self.attn_adapter_added = True

    def preprocess_inputs(self, input_list: list, embed=True) -> List[torch.Tensor]:
        """
        Expects a list of strings and instances of ImageInput
        Converts them into a list of tensors and then optionally runs self.embed over it
        """
        for i in range(len(input_list)):
            inp = input_list[i]
            if isinstance(inp, str):
                input_list[i] = self.tokenizer.encode(inp, return_tensors="pt")
            elif isinstance(inp, ImageInput):
                input_list[i] = inp.get_transformed_image(
                    transform_fn=self.transforms)
            else:
                raise Exception(f'Invalid input type:{type(inp)}')

        if embed == True:
            return self.embed(input_list)
        else:
            return input_list

    def embed(self, inputs: List[torch.Tensor]) -> TensorType["b", "s", "d"]:
        """
        Embeds a list of tensors In the correct format to input into the LM (b, s, d).
        For each tensor, if it's 2d assume it's text and use word embedding,
        if it's 4d, assume it's an image, and use image_prefix to embed.
        """
        emb_list = []
        for x in inputs:
            if x.ndim == 2:
                x = x.cuda()  # .to(self.device) #.half()
                emb_list.append(self.word_embedding(x))
            elif x.ndim == 4:
                x = x.cuda().half()  # to(self.device).half()
                image_embeddings = self.image_prefix(x)
                emb_list.append(image_embeddings)
            else:
                raise ValueError(f"Expected 2d or 4d tensor, got {x.ndim}d")
        return torch.cat(emb_list, dim=1)

    @torch.no_grad()
    def generate(
        self,
        embeddings: TensorType["b", "s", "d"],
        max_steps: int = 100,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.9,
        decode: bool = True,
    ):
        """
        Generates captions for a batch of embeddings.
        """

        return generate(
            self,
            embeddings=embeddings,
            max_steps=max_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            decode=decode,
        )

    def forward(
        self,
        images: TensorType["b", "n", "t", "c", "h", "w"] = None,
        captions: Optional[TensorType["b", "seq"]] = None,
        output_hidden_states: bool = False,
        input_embeddings: TensorType["b", 'n', 't', "s", "d"] = None,
    ) -> ModelOutput:

        assert captions is not None, "Must provide captions in training"
        assert any([i is not None for i in [images, input_embeddings]]) and not all(
            [i is not None for i in [images, input_embeddings]]
        ), "Pass in either images, or input embeddings, not both."
        assert (
            captions.shape[1] == self.seq_len
        ), f"in training, captions should be padded to sequence length ({self.seq_len}), but are length {captions.shape[1]}"

        #print("input" , images.shape, captions.shape)
        #print("captions"  , captions.shape)

        if input_embeddings is None:
            input_embeddings = self.image_prefix(images)
            input_embeddings = rearrange(
                input_embeddings, '(b n) s d -> b n s d', n=self.config.few_shot)

        #print("embeds", input_embeddings.shape)

        word_embeddings = self.word_embedding(captions)
        labels = build_labels(
            input_embeddings, captions, self.eos_token
        )
        #import pdb
        #pdb.set_trace()
        #print("labels", labels.shape)

        if self.config.cross_attention_config is not None:

            media_pos = captions == self.tokenizer.cls_token_id
            media_mask = media_pos.cumsum(dim=-1)
            # add cross attention
            visual_features = self.perceiver_resampler(
                input_embeddings
            )

            labels = torch.clone(captions)
            for label in labels:
                for k, token in enumerate(label):
                    if token == self.eos_token:
                        label[k + 1:] = -100
                        break

            for l in self.cross_attention_layers:
                x_attn_block = getattr(self.transformer[l], 'ln_1')[1]

                x_attn_block.perceiver_pipe(
                    visual_features, media_mask=media_mask)

            lm_outputs = self.lm(
                inputs_embeds=word_embeddings,
                labels=labels,
                output_hidden_states=output_hidden_states,
            )

        else:

            # forward joined embeddings through lm
            input_embeddings = torch.cat(
                (
                    input_embeddings,
                    word_embeddings[:, : -input_embeddings.shape[1], :],
                ),  # remove padding in the word embedding before concatenating
                dim=1,
            )

            lm_outputs = self.lm(
                inputs_embeds=input_embeddings,
                labels=labels,
                output_hidden_states=output_hidden_states,
            )

        return lm_outputs

    @classmethod
    def from_checkpoint(cls, config_path, checkpoint_path, device="cpu"):
        """
        Loads a model checkpoint from disk / downloads from url if not present
        """

        checkpoint_url = 'https://bit.ly/aleph-alpha-download'

        if exists(checkpoint_path) == False:
            print_main(
                f'checkpoint: {checkpoint_path} does not exist, downloading model')
            download_checkpoint(checkpoint_url=checkpoint_url,
                                save_as=checkpoint_path)

        model = cls(config=config_path)
        sd = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if "module" in sd.keys():
            sd = sd["module"]
        distributions = sd.pop("distributions")

        print_main(f'loading magma checkpoint from: {checkpoint_path}')
        model.load_state_dict(sd)
        print_main("magma successfully loaded")

        model.half()  # .half()  # .eval()  # .to(device).eval()
        return model
