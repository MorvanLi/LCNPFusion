# -*- coding: utf-8 -*-
# @Time    : 2026/4/28 10:16
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : train_2.py
# @Software: PyCharm


# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from PIL import Image
import json

from cnp_model import CNPNode
from spikingjelly.clock_driven import functional

# -------------------------
# Reproducibility
# -------------------------
torch.manual_seed(9)
torch.set_printoptions(precision=10)

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

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.cnp3 = CNPNode(
            0.5, 0.5, 0.7, 0.5, 1,
            128, 128, 3, 1, 1,
            batch_size, (128, 8, 8),
            local=True, shared=True, linking=True
        )

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




# 自定义数据集
class PatchDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 读取标签文件
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        self.folder_list = list(self.labels.keys())

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        imgA_path = f'{self.root_dir}/{folder_name}/A.jpeg'
        imgB_path = f'{self.root_dir}/{folder_name}/B.jpeg'
        imageA = Image.open(imgA_path)
        imageB = Image.open(imgB_path)
        label = self.labels[folder_name]

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        return imageA, imageB, label

def forward_spike_freq(net: nn.Module,
                       patch1: torch.Tensor,
                       patch2: torch.Tensor,
                       num_timesteps: int) -> torch.Tensor:
    """
    Run net for num_timesteps and return spike frequency.

    patch1: (B,1,15,15)
    patch2: (B,1,15,15)
    return: (B,2)
    """
    for t in range(num_timesteps):
        if t == 0:
            spike_count = torch.zeros_like(net(patch1, patch2))
        else:
            spike_count += net(patch1, patch2)

    spike_freq = spike_count / (num_timesteps - 1)
    return spike_freq
    # spike_count = 0.
    # for _ in range(num_timesteps):
    #     spike_count = spike_count + net(patch1, patch2)
    # return spike_count / num_timesteps


# 数据加载和预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Pad((0, 0, 1, 1), padding_mode='reflect'),
    transforms.ToTensor()
])



root_dir = './dataset/data'
labels_file = './dataset/label/labels.json'
epoch = 1000


batch_size = 256
lr = 0.0001
momentum = 0.9
weight_decay = 0.0001

dataset = PatchDataset(
    root_dir=root_dir, labels_file=labels_file, transform=transform)
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net = SiameseCNPNetwork(batch_size=batch_size).to(device)


import copy
optimizer = optim.Adam(net.parameters(), lr=lr)

min_loss = float('inf')
weight = None
num_timesteps = 8
best_acc = 0.0
best_weight = None
for e in range(epoch):
    start_time = time.time()
    net.train()

    epoch_loss = 0.0
    correct_sum = 0
    total = 0
    batch_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Train Epoch {e+1}", dynamic_ncols=True)

    for i, data in enumerate(pbar, 0):
        patch1, patch2, labels = data
        patch1 = patch1.to(device, non_blocking=True)
        patch2 = patch2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = forward_spike_freq(net, patch1, patch2, num_timesteps)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()
        functional.reset_net(net)

        bs = labels.size(0)
        epoch_loss += loss.item() * bs
        batch_loss += loss.item()

        pred = outputs.argmax(dim=1)
        correct_sum += (pred == labels).sum().item()
        total += bs

        if i % 10 == 9:
            print(f"[Epoch {e+1}, Batch {i+1}] Loss: {batch_loss / 10:.4f}")
            batch_loss = 0.0

        pbar.set_postfix(
            loss=f"{epoch_loss / total:.4f}",
            acc=f"{correct_sum / total:.4f}"
        )

    epoch_loss = epoch_loss / total
    epoch_acc = correct_sum / total
    print(f"Epoch {e+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, time={time.time()-start_time:.1f}s")

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = e + 1
        best_weight = copy.deepcopy(net.state_dict())

    torch.save(net.state_dict(), f'./weight/all_weights_8/epoch-{e + 1}.pth')

torch.save(best_weight, './weight/all_weights_8/siamese_cnp_model_weights.pth')

print('Finished Training')
print(f'Best Acc: {best_acc:.4f} (Epoch {best_epoch})')


