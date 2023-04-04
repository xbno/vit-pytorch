# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import pathlib

# import skimage.io
# from skimage.measure import find_contours
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import transforms as pth_transforms
# import numpy as np
# from PIL import Image

# import utils
# import vision_transformer as vits

import torch
from torch.utils.data import DataLoader, Dataset

from vit_pytorch.vit_3d import ViT as ViT3D
from vit_pytorch.vit_3d import ViTOC
from vit_pytorch.dino import Dino3D, Dino3DOC
import contrastive


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = (
            image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        )
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(
    image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5
):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis("off")
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect="auto")
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument("--symbols", default=None, type=str, help="Symbols to load.")
    parser.add_argument(
        "--image_size", default=(480, 480), type=int, nargs="+", help="Resize image."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path where to save visualizations."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------------ #
    # My shit
    # ------------------------------------------------------------------------ #

    # build model
    model = ViTOC(
        image_height=200,
        image_width=20,
        patch_height=10,
        patch_width=4,
        frames=10,  # number of frames
        frame_patch_size=2,  # frame patch size
        channels=18,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    # Test settings
    batch_size = 16
    num_workers = 2
    epochs = 20
    lr = 3e-5
    gamma = 0.7
    seed = 42
    device = "mps"

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            args.pretrained_weights, msg
        )
    )

    test_ds = contrastive.OcDataset(
        data_dir="/Users/xbno/Downloads/vit_20201123_to_20230328/npy",
        pretext_task="spatiotemporal",
        num_frames=10,  # frames per window
        stride=3,  # num frames between next window set
        max_delta=5,
        keep_symbols=args.symbols,
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # collate_fn=contrastive.pad_collate,
    )

    # model pass thrus
    b = next(iter(test_dl))
    w1, w2 = b["window_one"].to(device), b["window_two"].to(device)
    e1 = w1[:1]  # pull single example from batch

    # make the image divisible by the patch size
    patch_height = 10
    patch_width = 4
    frame_patch_size = 2
    f, h, w = (
        e1.shape[2] - e1.shape[2] % frame_patch_size,
        e1.shape[3] - e1.shape[3] % patch_height,
        e1.shape[4] - e1.shape[4] % patch_width,
    )
    e1 = e1[:, :, :f, :h, :w].unsqueeze(0)

    model(e1).shape

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = (
            nn.functional.interpolate(
                th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        os.path.join(args.output_dir, "img.png"),
    )
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format="png")
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(
                image,
                th_attn[j],
                fname=os.path.join(
                    args.output_dir,
                    "mask_th" + str(args.threshold) + "_head" + str(j) + ".png",
                ),
                blur=False,
            )
