#  %%
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import json
import tqdm
from magma.datasets import ImgCptDataset
from magma.adapters import Activation_Function_Class
from activations.torch import Rational
from magma.magma import Magma
from rtpt import RTPT
sns.set_style("darkgrid")
palette = sns.color_palette('hls', 8)
# %env CUDA_VISIBLE_DEVICES = 6
# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--config_path', type=str,
                    default='./../fb20-dgx2-configs/dev.yml')
parser.add_argument("--dataset_path", type=str,
                    default='/storage-01/ml-mmeuer/datasets/coco_converted/coco_converted_index_dataset')
parser.add_argument('--title', type=str, default='Some Title')
parser.add_argument('--plain', type=bool, default=False)


def return_dynamic_fig(model, cols=2, nr_figures=2, title='Dynamic Softmax', type=Rational, plain=False):
    i = 0
    for n, p in model.named_modules():
        if (isinstance(p, Rational) and not plain) or (isinstance(p, Activation_Function_Class) and plain):
            i += cols*nr_figures
    rows = i // cols + 1
    rest = i % cols
    if rest != 0:
        rows += 1

    Position = range(1, i + 1)

    fig = plt.figure(1, figsize=(20, 160))

    fig.suptitle(f'Input and Output Distributions of {title}', fontsize=20, )
    return fig, Position, rows, cols


def dist_forwards(model, dataset, num_samples=10, cuda=True, plain=False):
    def grad_hook(module, input, output):
        module.input = torch.cat((module.input, input[0]), dim=0)
        module.output = torch.cat((module.output, output[0]), dim=0)

    for i, (n, m) in enumerate(model.named_modules()):
        if (isinstance(m, Rational) and not plain) or (isinstance(m, Activation_Function_Class) and plain):
            setattr(m, 'input', torch.empty(0).cuda())
            setattr(m, 'output', torch.empty(0).cuda())
            m.register_forward_hook(grad_hook)

    with torch.no_grad():
        for i in range(num_samples):
            images, captions = dataset[i]
            if cuda:
                images, captions = images.cuda(), captions.cuda()
            model.forward(images, captions)
            torch.cuda.empty_cache()


def plot_input_dist(model, title='Some Title', cols=2, nr_figures=2, cuda=True, plain=False):
    fig, Position, rows, cols = return_dynamic_fig(
        model, cols=cols, nr_figures=nr_figures, title=title, plain=plain)
    j = 0
    for n, m in tqdm.tqdm(model.named_modules()):
        print(type(m))
        if (isinstance(m, Rational) and not plain) or (isinstance(m, Activation_Function_Class) and plain):
            print("HUHU")
            linsp = torch.linspace(-3, 3, 100)
            if cuda:
                linsp = linsp.cuda()
            if plain == True:
                y = torch.nn.ReLU()(linsp)
            else:
                y = m.forward(linsp)
            ax1 = fig.add_subplot(rows, cols, Position[j*2])
            ax1.grid(False)
            text = f'Input Distribution at Layer {j} in Attention Adapter' if j % 2 and nr_figures == 2 else f'Input Distribution at Layer {j} in MLP Adapter'
            ax1.title.set_text(text)
            ax2 = ax1.twinx()
            input_data = m.input.view(-1)
            input_filter_range = input_data[~torch.where(
                (input_data > -3) & (input_data < 3), input_data, np.nan).isnan()]
            sns.histplot(input_filter_range.cpu().numpy(),
                         ax=ax1, color=palette[0])
            sns.lineplot(x=linsp.detach().cpu().numpy(),
                         y=y.detach().cpu().numpy(), ax=ax2, color=palette[4])
            if nr_figures == 2:
                ax3 = fig.add_subplot(rows, cols, Position[(j*2)+1])
                text = f'Output Distribution at Layer {j} in Attention Adapter' if j % 2 and nr_figures == 2 else f'Output Distribution at Layer {j} in MLP Adapter'
                ax3.title.set_text(text)
                output_data = m.output.view(-1)
                output_filter_range = output_data[~torch.where(
                    (output_data > -2) & (output_data < 2), output_data, np.nan).isnan()]
                # ax3.set_xlim(-1, 1)
                # ax3.set_ylim(0, 150000)

                sns.histplot(output_filter_range.cpu().numpy(),
                             ax=ax3, color=palette[0])

                # sns.lineplot(x=linsp.detach().cpu().numpy(),
                #              y=y.detach().cpu().numpy(), ax=ax4, color=palette[4])

            j += 1
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig(
        f'./../plots/Input_Distributions_{title.replace(" ", "_")}.png', dpi=300)
    plt.show()


# %%
if __name__ == "__main__":
    # %%
    #  Load the data
    rtpt = RTPT(name_initials='MM',
                experiment_name='Eval Notebook Magma', max_iterations=2)
    rtpt.start()
    args = parser.parse_args()
    if args.model_path is None:
        model = Magma(args.config_path).cuda()
    else:
        model = Magma.from_checkpoint(
            config_path=args.config_path, checkpoint_path=args.model_path).cuda()
    for n, p in model.named_modules():
        if n in 'lm.transformer.h.19.mlp.1.adapter.1':
            print(p)
    # %%
    dataset = ImgCptDataset(args.dataset_path,
                            model.tokenizer, model.transforms)
    # %%
    dist_forwards(model, dataset, num_samples=10, cuda=True, plain=args.plain)
    plot_input_dist(model, title=args.title, cols=2,
                    cuda=True, plain=args.plain)

# %%
