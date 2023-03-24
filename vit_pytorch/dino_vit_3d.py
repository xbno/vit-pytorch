from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# from tqdm.notebook import tqdm
from tqdm import tqdm

import torch
from vit_pytorch.vit_3d import ViT as ViT3D
from vit_pytorch.dino import Dino3D


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, frames=10, transform=None):
        self.file_list = file_list
        self.frames = frames
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        if idx >= self.filelength - self.frames:
            idx = self.filelength - 75
        img_paths = self.file_list[idx : idx + self.frames]
        #         print(img_paths)
        imgs = []
        for img_path in img_paths:
            imgs.append(self.transform(Image.open(img_path)))
        video_transformed = torch.stack(imgs, 1)
        #         print(video_transformed.shape)
        return video_transformed


if __name__ == "__main__":

    # Training settings
    batch_size = 32
    epochs = 20
    lr = 3e-5
    gamma = 0.7
    seed = 42
    device = "mps"

    seed_everything(seed)

    train_dir = "/Users/xbno/ml/options_trading/data/train"
    test_dir = "/Users/xbno/ml/options_trading/data/test"

    train_list = glob.glob(os.path.join(train_dir, "*.jpg"))
    test_list = glob.glob(os.path.join(test_dir, "*.jpg"))

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")

    labels = [path.split("/")[-1].split(".")[0] for path in train_list]

    # random_idx = np.random.randint(1, len(train_list), size=9)
    # fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # for idx, ax in enumerate(axes.ravel()):
    #     img = Image.open(train_list[idx])
    #     ax.set_title(labels[idx])
    #     ax.imshow(img)

    train_list, valid_list = train_test_split(
        train_list, test_size=0.2, stratify=labels, random_state=seed
    )

    print(f"Train Data: {len(train_list)}")
    print(f"Validation Data: {len(valid_list)}")
    print(f"Test Data: {len(test_list)}")

    transforms = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    train_data = CatsDogsDataset(train_list, transform=transforms)
    valid_data = CatsDogsDataset(valid_list, transform=transforms)
    test_data = CatsDogsDataset(test_list, transform=transforms)

    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valid_loader = DataLoader(
        dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=8
    )

    print(len(train_data), len(train_loader))
    print(len(valid_data), len(valid_loader))

    model = ViT3D(
        image_size=128,  # image size
        frames=10,  # number of frames
        image_patch_size=32,  # image patch size
        frame_patch_size=2,  # frame patch size
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    # model pass thrus
    # b = next(iter(valid_loader))
    # model(b[:4]).shape

    # wrapped_model = NetWrapper(model, 256, 4, 256)  # ,"mlp_head")
    # [_.shape for _ in wrapped_model(b[:4])]

    learner = Dino3D(
        model,
        frames=10,
        image_size=128,
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

    # bs = 32, 256img
    # num_wrokers=0 @ 7.5min per epoch, 2.99s/it, very bursty gpu useage
    # num_wrokers=8 @ ~6.25in per epoch, 1.93s/it, ~100%gpu no peaking

    # bs = 32, 128img
    # num_wrokers=8 @ ~2.5in per epoch, 1.97s/it, ~100%gpu no peaking
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0

        for images in tqdm(valid_loader):
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of teacher encoder and teacher centers
            epoch_loss += loss / len(valid_loader)

        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f}\n")

    torch.save(model.state_dict(), "./dino_vit_3d_catdogs_100e.pt")

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:24<00:00,  1.08it/s]
    # Epoch : 1 - loss : 10.3333

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:24<00:00,  1.09it/s]
    # Epoch : 2 - loss : 7.8434

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:24<00:00,  1.09it/s]
    # Epoch : 3 - loss : 4.5750

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:23<00:00,  1.09it/s]
    # Epoch : 4 - loss : 2.4068

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:26<00:00,  1.07it/s]
    # Epoch : 5 - loss : 1.1146

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:26<00:00,  1.07it/s]
    # Epoch : 6 - loss : 0.5165

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:26<00:00,  1.07it/s]
    # Epoch : 7 - loss : 0.2818

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:23<00:00,  1.09it/s]
    # Epoch : 8 - loss : 0.1781

    # .....

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:23<00:00,  1.09it/s]
    # Epoch : 29 - loss : 0.0062

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:23<00:00,  1.09it/s]
    # Epoch : 30 - loss : 0.0056

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:24<00:00,  1.09it/s]
    # Epoch : 31 - loss : 0.0051

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:25<00:00,  1.08it/s]
    # Epoch : 32 - loss : 0.0046

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:23<00:00,  1.09it/s]
    # Epoch : 33 - loss : 0.0042

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:24<00:00,  1.09it/s]
    # Epoch : 34 - loss : 0.0039

    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:24<00:00,  1.09it/s]
    # Epoch : 35 - loss : 0.0035
