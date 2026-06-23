# -*- coding: utf-8 -*-
# @Time    : 2026/4/28 16:12
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : Inference_2.py
# @Software: PyCharm

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from smallRegionFilter import smallRegionFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnp_model import CNPNode
from spikingjelly.clock_driven import functional




import torch
import torch.nn as nn
from pcnn_model import CNPNode


class SiameseCNPNetwork(nn.Module):
    def __init__(self, batch_size: int):
        super(SiameseCNPNetwork, self).__init__()

        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.cnp1 = CNPNode(
            0.5, 0.5, 0.7, 0.5, 1,
            32, 32, 3, 1, 1,
            batch_size, (32, 16, 16),
            local=True, shared=True, linking=True
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.cnp2 = CNPNode(
            0.5, 0.5, 0.7, 0.5, 1,
            64, 64, 3, 1, 1,
            batch_size, (64, 16, 16),
            local=True, shared=True, linking=True
        )

        # MUST match ANN exactly
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.cnp3 = CNPNode(
            0.5, 0.5, 0.7, 0.5, 1,
            128, 128, 3, 1, 1,
            batch_size, (128, 8, 8),
            local=True, shared=True, linking=True
        )

        # self.upsample = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)
        self.fc1 = nn.Linear(128 * 8 * 8 * 2, 128)
        self.cnp_fc1 = CNPNode(
            0.5, 0.5, 0.7, 0.5, 1,
            1, 1, 3, 1, 1,
            batch_size, (128,),
            local=False, shared=False, linking=False
        )

        self.fc2 = nn.Linear(128, 2)
        self.cnp_fc2 = CNPNode(
            0.5, 0.5, 0.7, 0.5, 1,
            1, 1, 3, 1, 1,
            batch_size, (2,),
            local=False, shared=False, linking=False
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 15, 15)
        return: (B, 256, 15, 15)
        """
        x = self.conv1(x)
        x = self.cnp1(x)

        x = self.conv2(x)
        x = self.cnp2(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.cnp3(x)

        # x = self.upsample(x)
        return x

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        input1: (B, 1, 15, 15)
        input2: (B, 1, 15, 15)
        return: (B, 2)
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        combined = torch.cat([output1, output2], dim=1)   # (B, 512, 15, 15)
        combined = combined.view(combined.size(0), -1)    # (B, 256*15*30)

        combined = self.fc1(combined)
        combined = self.cnp_fc1(combined)

        output = self.fc2(combined)
        output = self.cnp_fc2(output)

        return output

@torch.no_grad()
def forward_spike_freq_padded(net: nn.Module,
                              patch1: torch.Tensor,
                              patch2: torch.Tensor,
                              num_timesteps: int,
                              fixed_batch_size: int) -> torch.Tensor:
    """
    patch1: (B,1,15,15), B <= fixed_batch_size
    patch2: (B,1,15,15), B <= fixed_batch_size
    return: (B,2)
    """
    net.eval()

    B = patch1.shape[0]
    assert patch2.shape[0] == B
    assert B <= fixed_batch_size

    if B < fixed_batch_size:
        pad_num = fixed_batch_size - B
        pad_patch1 = patch1[-1:].repeat(pad_num, 1, 1, 1)
        pad_patch2 = patch2[-1:].repeat(pad_num, 1, 1, 1)

        patch1 = torch.cat([patch1, pad_patch1], dim=0)
        patch2 = torch.cat([patch2, pad_patch2], dim=0)

    for t in range(num_timesteps):
        y = net(patch1, patch2)
        if t == 0:
            spike_count = torch.zeros_like(y)
        else:
            spike_count += y

    spike_freq = spike_count / (num_timesteps - 1)
    functional.reset_net(net)
    return spike_freq[:B]

from torchvision import transforms
from PIL import Image
# =========================================================
# 3. Utils
# =========================================================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


def load_gray_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path)
    img = transform(img)   # (1,H,W)
    return img.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "./weights/epoch-527.pth"  # 527 0.02
    for i in range(1, 21):
        imageA_path = f"./Lytro/near/{i}.jpg"
        imageB_path = f"./Lytro/far/{i}.jpg"
        net = SiameseCNPNetwork(batch_size=infer_batch_size).to(device)

        state_dict = torch.load(ckpt_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            net.load_state_dict(state_dict["model_state_dict"])
        else:
            net.load_state_dict(state_dict)

        net.eval()

        imageA = load_gray_tensor(imageA_path, device)
        imageB = load_gray_tensor(imageB_path, device)

        if imageA.shape != imageB.shape:
            raise ValueError(f"Image shape mismatch: {imageA.shape} vs {imageB.shape}")

        """
        The core code will be updated
        """

if __name__ == "__main__":
    main()

