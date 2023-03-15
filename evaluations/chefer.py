# %%
import torch
import numpy as np
import cv2

from magma.magma import Magma
from magma.image_input import ImageInput
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from PIL import Image
from atman_magma.explainer import Explainer
import os
parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str,
                    default='/storage-01/ml-mmeuer/switch_relu_rationals/relu6_5250.pt')
parser.add_argument('--model_name', type=str,
                    default='switch_relu_rationals')
parser.add_argument('--steps', type=str, default='5260')
parser.add_argument('--gumbel', type=str, default=True)
parser.add_argument('--config_path', type=str,
                    default='/home/ml-mmeuer/adaptable_magma/fb20-dgx2-configs/single.yml')
parser.add_argument('--device', type=str, default='cuda:0')


# %%
# create heatmap from mask on image
def show_cam_on_image_old(
    img,
    mask,
    mask_interpolation=cv2.INTER_NEAREST,
    cmap=cv2.COLORMAP_JET
):
    '''
    example usage:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    for i in range(len(relevance_maps)):

        cam = show_cam_on_image(
            img = np.array(prompt[0].pil_image),
            mask = relevance_maps[i]['relevance_map'].reshape(12,12).numpy()
        )

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (10 , 4))
        fig.suptitle(
            'Target token:'+ cm.magma.tokenizer.decode([relevance_maps[i]['target_token_id']]))
        ax[0].imshow(prompt[0].pil_image)
        ax[1].imshow(cam)
        ax[2].imshow(relevance_maps[i]['relevance_map'].reshape(12,12).numpy())
        plt.show()
    ```
    '''

    if mask.shape == (12, 12):
        mask = cv2.resize(
            mask,
            dsize=(img.shape[1], img.shape[0]),
            interpolation=mask_interpolation
        )

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cmap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    return heatmap, cam


# %%
def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=384, mode='bilinear')
    image_relevance = image_relevance.reshape(
        384, 384).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / \
        (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

# %%


class CheferMethod:
    '''
    Example usage:
    ```python
    cm = CheferMethod(
        magma = model,
        device = 'cuda:0'
    )

    prompt = [
        # supports urls and path/to/image
        ImageInput('./samples/el2.png'),
        'A picture of an'
    ]
    embeddings = model.preprocess_inputs(prompt.copy())
    target = ' Elephant and a zebra'

    relevance_maps = cm.run(
        embeddings = embeddings,
        target = target
    )
    ```
    '''

    def __init__(self, magma: Magma, device: str, args: argparse.Namespace):
        # CheferMagma is a modified version of magma which contains backward hooks on attention
        self.magma = magma
        self.device = device
        self.magma = self.magma.to(self.device).eval()
        self.args = args

    # rule 5 from paper
    def avg_heads(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam

        # they are clamping values between zero and +ve values (kinda like a relu)
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    # rule 6 from paper
    def apply_self_attention_rules(self, R_ss, cam_ss):
        R_ss_addition = torch.matmul(cam_ss, R_ss)
        return R_ss_addition

    def default_index_fn(self, relevance_matrix, target_token_index, num_target_token_ids):
        '''
        this is the indexing we use by default :)
        '''
        idx = -num_target_token_ids + target_token_index
        # print(f'target_token_index: {target_token_index} num_target_token_ids: {num_target_token_ids} idx: {idx}')
        return relevance_matrix[idx, :144]

    def run(self, embeddings, target: str, custom_index_fn=None):
        '''
        Steps:
        0. make sure input embeddings batch size is 1
        1. forward pass through model (with grads)
        2. convert target string to token ids using tokenizer
        3. calculate relevance for each target token
        4. return nice list where each item is a dictionary containing:
            - target token index (int)
            - target token id (int)
            - relevance map (numpy array)
        '''

        # step 0 safety first
        assert embeddings.shape[
            0] == 1, f'Expected batch size to be 1 but got: {embeddings.shape[0]}'

        # find num tokens in prompt
        num_tokens_in_prompt = embeddings.shape[1]

        # target embeddings
        target_embeddings = self.magma.preprocess_inputs(
            [target]
        ).to(self.device)

        # combine prompt and target embeddings along seq dim
        combined_embeddings = torch.cat(
            [
                embeddings.to(self.device),
                # exclude last item in seq dim
                # see: https://github.com/Mayukhdeb/atman-magma/blob/master/atman_magma/explainer.py#L98
                target_embeddings[:, :-1, :]
            ],
            dim=1
        )

        # step 1
        # output_logits.shape: (batch, seq, vocabulary)
        output_logits = self.magma.forward(
            input_embeddings=combined_embeddings,
        ).logits

        # step 2
        target_token_ids = self.magma.tokenizer.encode(target)
        num_target_token_ids = len(target_token_ids)

        # completion_logits.shape: (num_target_token_ids, vocabulary)
        # -1 because next token
        completion_logits = output_logits[0, num_tokens_in_prompt-1:, :]
        assert completion_logits.shape[
            0] == num_target_token_ids, f'Expected completion_logits.shape[0] to be: {num_target_token_ids} but got shape: {completion_logits.shape}'

        relevance_maps_per_layer = []
        accumulated_maps = []
        print('Calculating relevance maps...', num_target_token_ids)
        for target_token_index in range(num_target_token_ids):

            # make a onehot vector of shape (batch_size, vocabulary)
            # set the index of the target token id as 1
            one_hot = np.zeros((1, output_logits.shape[-1]), dtype=np.float32)
            one_hot[0, target_token_ids[target_token_index]] = 1.

            # convert it to a torch tensor
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)

            # dot product (?)
            one_hot = torch.sum(one_hot.to(
                self.device) * completion_logits[target_token_index, :].unsqueeze(0))

            # shortcut one-hot, possibly doing the same thing, but will NOT use it for now
            # one_hot = completion_logits[target_token_index, target_token_ids[target_token_index]]

            # make sure there are no older grads which might accumulate
            self.magma.zero_grad()
            one_hot.backward(retain_graph=True)
            num_tokens = self.magma.lm.transformer.h[0].attn.get_attention_map(
            ).shape[-1]

            big_R = torch.eye(num_tokens, num_tokens).to(self.device)

            per_layer = []

            for blk in self.magma.lm.transformer.h:

                grad = blk.attn.get_attn_gradients().detach()
                cam = blk.attn.get_attention_map().detach()
                cam = self.avg_heads(cam, grad)
                small_R = self.apply_self_attention_rules(
                    big_R.to(self.device).float(), cam.to(self.device).float())
                big_R += small_R

            # apply custom indexing on relevance "matrix"(?)
            # dont know which is the correct way to index this for decoder models
                if custom_index_fn is not None:
                    relevance_map = custom_index_fn(
                        relevance_matrix=small_R,
                        target_token_index=target_token_index,
                        num_target_token_ids=num_target_token_ids
                    ).cpu().detach()

                else:
                    relevance_map = self.default_index_fn(
                        relevance_matrix=small_R,
                        target_token_index=target_token_index,
                        num_target_token_ids=num_target_token_ids
                    ).cpu().detach()
                per_layer.append(relevance_map)

            accumulated_map = self.default_index_fn(
                relevance_matrix=big_R,
                target_token_index=target_token_index,
                num_target_token_ids=num_target_token_ids
            ).float().cpu().detach()
            accumulated_maps.append(accumulated_map)
            relevance_maps_per_layer.append(per_layer)

        results_per_layer = []
        result_accumulated = []

        for i in range(len(relevance_maps_per_layer)):
            for j in range(len(relevance_maps_per_layer[i])):
                data = {
                    'target_token_index': i,
                    'target_token_id': target_token_ids[i],
                    'relevance_map': relevance_maps_per_layer[i][j],
                    'layer': j,
                }
                if self.args.gumbel:
                    data['gumbel_weight'] = self.magma.lm.transformer.h[j].mlp[1].switch_logits[0].item(
                    )
                results_per_layer.append(data)

            data = {
                'target_token_index': i,
                'target_token_id': target_token_ids[i],
                'relevance_map': accumulated_maps[i],
            }
            result_accumulated.append(data)
        return results_per_layer, result_accumulated

    def build_relevance_maps(self, relevance_maps, path, prompt, layerwise=False):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        transformed_img = magma.transforms(
            prompt[0].pil_image)[0].unsqueeze(0)
        for i in range(len(relevance_maps)):
            heatmap = show_image_relevance(
                relevance_maps[i]['relevance_map'], transformed_img, prompt[0].pil_image)

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
            target_token = self.magma.tokenizer.decode(
                [relevance_maps[i]["target_token_id"]]).strip()

            layer_text = f'At Layer {relevance_maps[i]["layer"]}' if layerwise else ""
            switch_weight = relevance_maps[i]['gumbel_weight'] if 'gumbel_weight' in relevance_maps[i] else ""
            layer_text = f'{layer_text} Switch Weight: {switch_weight}' if 'gumbel_weight' in relevance_maps[
                i] else layer_text

            fig.suptitle(
                f'Target token: "{target_token}" {layer_text}')
            im0 = ax[0].imshow(prompt[0].pil_image)
            im1 = ax[1].imshow(heatmap)
            im2 = ax[2].imshow(
                relevance_maps[i]['relevance_map'].reshape(12, 12).numpy())
            fig.subplots_adjust(right=0.85)
            cax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            npath = f'{path}token_{target_token}_layer_{relevance_maps[i]["layer"]}.png' if layerwise else f'{path}token_{target_token}.png'
            fig.colorbar(im2, cax=cax)
            plt.savefig(npath)
            plt.close()

    def log_pdist(self, relevance, relevance_layerwise):
        pdist = torch.nn.PairwiseDistance()
        for map in relevance:
            below = []
            above = []
            for layer in relevance_layerwise:

                if map['target_token_id'] != layer['target_token_id']:
                    continue
                coeff = pdist(map['relevance_map'], layer['relevance_map'])
                if layer['gumbel_weight'] >= 0.5:
                    above.append(coeff)
                else:
                    below.append(coeff)
            print('---------')
            print("Token", self.magma.tokenizer.decode(
                [map["target_token_id"]]).strip())
            print("Below", sum(below)/len(below) if len(below)
                  > 0 else 0, 'Layers', len(below))
            print("Above", sum(above)/len(above) if len(below)
                  > 0 else 0, 'Layers', len(above))


# %%
if __name__ == '__main__':
    args = parser.parse_args(args=[])

    if args.model_path:
        magma = Magma.from_checkpoint(
            config_path=args.config_path, checkpoint_path=args.model_path)
    else:
        magma = Magma(config_path=args.config_path)
    chefer = CheferMethod(magma, device=args.device, args=args)
    prompt = [
        # supports urls and path/to/image
        ImageInput(
            'https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
        'This is a picture of a'
    ]

    embeddings = magma.preprocess_inputs(prompt)

    # %%
    relevance_maps_per_layer, relevance_maps = chefer.run(
        embeddings=embeddings,
        target=' cabin in the woods'
    )

    path = f'chefer_plots/{args.model_name}_{args.steps}/woods/layerwise/'
    chefer.build_relevance_maps(
        relevance_maps_per_layer, path, prompt, layerwise=True)

    path = f'chefer_plots/{args.model_name}_{args.steps}/woods/total/'
    chefer.build_relevance_maps(relevance_maps, path, prompt)

    chefer.log_pdist(relevance_maps, relevance_maps_per_layer)


# %%
