from __future__ import print_function

import glob
from itertools import chain
import os
import sys
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# from tqdm.notebook import tqdm
from tqdm import tqdm

import torch
from vit_pytorch.vit_3d import ViT as ViT3D
from vit_pytorch.vit_3d import ViTOC
from vit_pytorch.dino import Dino3D, Dino3DOC

# sys.path.append("/Users/xbno/ml/options_trading")
# from lambdas import yf_utils, utils
# from pl_cnn import data, contrastive
import contrastive
import pathlib


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    # Training settings
    batch_size = 8
    num_workers = 4
    epochs = 20
    lr = 3e-5
    gamma = 0.7
    seed = 42
    device = "mps"

    seed_everything(seed)

    # data
    train_ds = oc_ds = contrastive.OcDataset(
        data_dir="/Users/xbno/Downloads/vit_20201123_to_20230328/npy",
        pretext_task="spatiotemporal",
        num_frames=10,  # frames per window
        stride=3,  # num frames between next window set
        max_delta=5,
        skip_symbols=["MSFT"],
    )

    # TODO create spatio temporal dates as a collate func so
    # each new epoch has a different slice/slices of date windows
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # collate_fn=contrastive.pad_collate,
    )

    print(len(train_ds), len(train_dl))

    def mean_std(loader):
        images = next(iter(loader))
        # shape of images = [b,c,w,h]
        mean = images.mean([0, 2, 3])
        std = images.std([0, 2, 3])
        print("mean and std: \n", mean, std)
        return mean, std

    mean_std(train_dl)

    model = ViTOC(
        image_height=200,
        image_width=20,
        patch_height=10,
        patch_width=2,
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

    # train_ds[0]

    # model pass thrus
    # b = next(iter(train_dl))
    # model(b[:4]).shape

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    count_parameters(model)

    learner = Dino3DOC(
        model,
        frames=10,
        image_height=200,
        image_width=20,
        channels=18,
        hidden_layer="to_latent",  # hidden layer name or index, from which to extract the embedding
        projection_hidden_size=256,  # projector network hidden dimension
        projection_layers=4,  # number of layers in projection network
        num_classes_K=65336,  # output logits dimensions when pretraining complex large datasets (referenced as K in paper)
        student_temp=0.9,  # student temperature
        teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
        local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
        global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
        moving_average_decay=0.99,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
        center_moving_average_decay=0.9995,  # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        augment_fn=torch.nn.Sequential(),
        augment_fn2=torch.nn.Sequential(),  # add pass throughs
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    learner = learner.to(device)

    count_parameters(learner)

    # bs = 16, 200x20
    # num_wrokers=8 @ ~4:40 per epoch, 2.15s/it, ~100%gpu no peaking
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0

        for images in tqdm(train_dl):
            images["window_one"] = images["window_one"].to(device)
            images["window_two"] = images["window_two"].to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of teacher encoder and teacher centers
            epoch_loss += loss / len(train_dl)

        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f}\n")

    torch.save(model.state_dict(), "./dino_vit_3d_oc_32e.pt")
