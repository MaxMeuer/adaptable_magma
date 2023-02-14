# %%
from atman_magma.magma  import Magma
from magma.utils import configure_param_groups, load_model
from train import get_pretraining_datasets
from atman_magma.explainer import Explainer
from atman_magma.utils import split_str_into_tokens
from atman_magma.logit_parsing import get_delta_cross_entropies
from magma.image_input import ImageInput
import matplotlib.pyplot as plt
import numpy as np
import cv2
import deepspeed
import os 
# print("CUDA_VISIBLE_DEVICES: ", torch.cuda.device_count())
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
with torch.cuda.device(1):
    print("CUDA_VISIBLE_DEVICES2: ", torch.cuda.device_count())
    device = 'cuda:0'
    model = Magma.from_checkpoint(
                config= '/home/ml-mmeuer/adaptable_magma/fb20-dgx2-configs/dev.yml',
                checkpoint_path='/home/ml-mmeuer/adaptable_magma/model_checkpoints/rationals-no-switch/mp_rank_00_model_states.pt',
                device=device
            )
    tokenizer, config, transforms = model.tokenizer, model.config, model.transforms
    trainable_parameters = configure_param_groups(model, config)

    # load data:
    train_dataset, eval_dataset = get_pretraining_datasets(
        config, tokenizer, transforms
    )

    ex = Explainer(
        model = model, 
        device = device, 
        tokenizer = model.tokenizer, 
    #     conceptual_suppression_threshold = None
        conceptual_suppression_threshold = 0.95
    )

    prompt  = [
            '''Suggest suitable categories for the following piece of text:
    State University and I possess a common vision. I, like State University, constantly work to explore the limits of nature by exceeding expectations. Long an amateur scientist, it was this drive that brought me to the University of Texas for its Student Science Training Program in 2013.
    tags:'''
    ]
    embeddings = model.preprocess_inputs(prompt)

    ## generate completion
    output = model.generate(
        embeddings = embeddings,
        max_steps = 10,
        temperature = 0.00001,
        top_k = 1,
        top_p = 0.0,
    )
    completion = output[0]

    logit_outputs = ex.collect_logits_by_manipulating_attention(
        prompt = prompt.copy(),
        target = completion,
        max_batch_size=1,
    )

    results = get_delta_cross_entropies(
        output = logit_outputs,
        square_outputs=True
    )

    print(results)

    prompt =[
        ## supports urls and path/to/image
        ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
        'This is a picture of a'
    ]

    ## returns a tensor of shape: (1, 149, 4096)
    embeddings = model.preprocess_inputs(prompt.copy())
    answer = ' log cabin in the woods'
    ## returns a list of length embeddings.shape[0] (batch size)
    output = model.generate(
        embeddings = embeddings,
        max_steps = 5,
        temperature = 0.001,
        top_k = 1,
        top_p = 0.0,
    )  
    completion = output[0]

    logit_outputs = ex.collect_logits_by_manipulating_attention(
        prompt = prompt.copy(),
        target = completion,
        max_batch_size=1,
        # prompt_explain_indices=[i for i in range(10)]
    )

    results = get_delta_cross_entropies(
        output = logit_outputs
    )



    results.save('output.json')
    # %%
