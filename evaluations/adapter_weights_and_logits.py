#  %%
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import json
import tqdm
import pandas as pd

from magma.datasets import ImgCptDataset
from activations.torch import Rational
from magma.magma import Magma
from rtpt import RTPT

sns.set_style("darkgrid")
palette = sns.color_palette('hls', 8)
# %env CUDA_VISIBLE_DEVICES = 6
# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--model_names', type=str, default=None)
parser.add_argument('--config_path', type=str,
                    default='./../fb20-dgx2-configs/dev.yml')
parser.add_argument("--dataset_path", type=str,
                    default='/storage-01/ml-mmeuer/datasets/coco_converted/coco_converted_index_dataset')
parser.add_argument('--model_name', type=str, default='Some Title')
parser.add_argument('--steps', type=str, default='')
parser.add_argument('--double', type=bool, default=False)


def return_dynamic_fig(nr_subplots, cols=2, nr_types=1, title='Dynamic Softmax'):
    i = 0
    for p in range(nr_subplots):
        i += nr_types
    rows = i // cols + 1
    rest = i % cols
    if rest != 0:
        rows += 1
    Position = range(1, i + 1)
    fig = plt.figure(1, figsize=(cols*10, rows*10))
    fig.suptitle(title, fontsize=20, )
    return fig, Position, rows, cols


def get_adapter_logits(module):
    mlp_weights = []
    attn_weights = []
    for name, value in tqdm.tqdm(module.items()):
        if 'switch' in name and 'lm' in name:
            if 'attn' in name:
                attn_weights.append(value.cpu().tolist())
            if 'mlp' in name:
                mlp_weights.append(value.cpu().tolist())
    return np.array(attn_weights), np.array(mlp_weights)


def get_adapter_weights(module):
    g = torch.distributions.Gumbel(0, 1)
    switch_temp = 0.1
    attn_res = []
    mlp_res = []
    for name, value in tqdm.tqdm(module.items()):
        if 'switch' in name and 'lm' in name:
            layer = []
            g_sample_list = []
            p = value.cpu()
            for i in range(1000):
                g_sample = g.sample(p.shape).to('cpu')
                sim = torch.softmax((p + g_sample)/switch_temp, dim=-1)
                g_sample_list.append(g_sample)
                layer.append(sim.tolist())
            if 'attn' in name:
                attn_res.append(layer)

            elif 'mlp' in name:
                mlp_res.append(layer)
    return np.array(attn_res), np.array(mlp_res)


def get_logits_and_weights(model_name, model_path):
    model = torch.load(f'{model_path}{model_name}')
    attn_logits, mlp_logits = get_adapter_logits(
        model['module'])
    attn_weights, mlp_weights = get_adapter_weights(
        model['module'])
    torch.cuda.empty_cache()
    return attn_logits, mlp_logits, attn_weights, mlp_weights


def plot_logits(attn_logits, mlp_logits, steps, title='Adapter Logits'):
    data = []
    fig, ax = plt.subplots(figsize=(10, 10))
    for model_idx in range(len(steps)):
        for layer_idx in range(len(mlp_logits[1])):
            data.append([f'MLP Gumbel Weight', f'{steps[model_idx]} Steps',
                        mlp_logits[model_idx][layer_idx][0],  f'MLP Layer {layer_idx}'])
        for layer_idx in range(len(attn_logits[1])):
            data.append([f'Attention Gumbel Weight',
                        f'{steps[model_idx]} Steps', attn_logits[model_idx][layer_idx][0],  f'Attention Layer {layer_idx}'])
    df = pd.DataFrame(data, columns=['Type', 'Steps', 'Logit', 'Layer'])
    ax.title.set_text(title)
    l_plt = sns.lineplot(x='Steps', y='Logit', hue='Layer',
                         data=df, marker="o", markersize=7, ax=ax)

    sns.move_legend(l_plt, 'upper left', bbox_to_anchor=(1, 1), ncol=3)
    plt.tight_layout()
    plt.savefig(f'./../plots/{title.replace(" ","_")}.png', dpi=100)
    plt.close()


def plot_distributions(attn_weights, mlp_weights, steps, title='Adapter Weights', double=False):
    nr_subplots = len(mlp_weights[0]) + len(attn_weights[0])
    fig, Position, rows, cols = return_dynamic_fig(
        nr_subplots, cols=len(steps), nr_types=len(steps), title=title)
    mlp_idx = 0
    attn_idx = len(steps)
    for layer_idx in range(len(mlp_weights[1])):
        for step_idx in range(len(steps)):
            if len(mlp_weights):
                ax1 = fig.add_subplot(rows, cols, Position[mlp_idx])
                ax1.title.set_text(
                    f'MLP Adapter Gumbel Weights Distribution at Layer {layer_idx} at Step {steps[step_idx]}')
                hst_plt = sns.histplot(
                    mlp_weights[step_idx][layer_idx][..., 0], ax=ax1)
                mlp_idx += 1
            if double:
                ax2 = plt.subplot(rows, cols, Position[attn_idx])
                ax2.title.set_text(
                    f'Attention Adapter Gumbel Weights Distribution at Layer {layer_idx} at Step {steps[step_idx]}')
                sns.histplot(attn_weights[step_idx][layer_idx][..., 0], ax=ax2)

                attn_idx += 1
        mlp_idx += len(steps)
        attn_idx += len(steps)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(f'./../plots/{title.replace(" ","_")}.png', dpi=100)


if __name__ == "__main__":
    # %%

    # %%
    #  Load the data
    args = parser.parse_args()
    models = args.model_names.split(',')
    steps = args.steps.split(',')
    rtpt = RTPT(name_initials='MM',
                experiment_name='Visualize Magma', max_iterations=len(models))
    rtpt.start()
    attn_logits = []
    mlp_logits = []
    attn_weights = []
    mlp_weights = []
    for model in models:
        rtpt.start
        attn_logit_model, mlp_logits_model, attn_weights_model, mlp_weights_model = get_logits_and_weights(
            model, args.model_path)
        attn_logits.append(attn_logit_model)
        mlp_logits.append(mlp_logits_model)
        attn_weights.append(attn_weights_model)
        mlp_weights.append(mlp_weights_model)
    plot_logits = plot_logits(attn_logits, mlp_logits,
                              steps, title=f'{args.model_name} Logits')
    plot_distributions = plot_distributions(
        attn_weights, mlp_weights, steps, title=f'{args.model_name} Weights', double=args.double)
