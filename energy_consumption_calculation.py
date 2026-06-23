# -*- coding: utf-8 -*-
import os
import csv
import time
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from spikingjelly.clock_driven import functional

from pcnn_model import CNPNode


# =========================================================
# 1. Network
# =========================================================
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
        x = self.conv1(x)
        x = self.cnp1(x)

        x = self.conv2(x)
        x = self.cnp2(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.cnp3(x)

        return x

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        combined = torch.cat([output1, output2], dim=1)
        combined = combined.view(combined.size(0), -1)

        combined = self.fc1(combined)
        combined = self.cnp_fc1(combined)

        output = self.fc2(combined)
        output = self.cnp_fc2(output)

        return output


# =========================================================
# 2. Spiking-rate meter
# =========================================================
class SpikeRateMeter:
    """
    统计每个 CNPNode 输出的 spiking rate。

    Conv1 的 spiking rate 对应 cnp1 输出；
    Conv2 的 spiking rate 对应 cnp2 输出；
    Conv3 的 spiking rate 对应 cnp3 输出；
    FC1 的 spiking rate 对应 cnp_fc1 输出；
    FC2 的 spiking rate 对应 cnp_fc2 输出。
    """

    def __init__(self, net: nn.Module, count_first_timestep: bool = True):
        self.net = net
        self.count_first_timestep = count_first_timestep

        self.layer_map = {
            "cnp1": "Conv1",
            "cnp2": "Conv2",
            "cnp3": "Conv3",
            "cnp_fc1": "FC1",
            "cnp_fc2": "FC2",
        }

        self.spike_sum = {v: 0.0 for v in self.layer_map.values()}
        self.elem_sum = {v: 0.0 for v in self.layer_map.values()}
        self.call_sum = {v: 0 for v in self.layer_map.values()}

        self.hooks = []

    def _make_hook(self, layer_name):
        def hook(module, inputs, output):
            current_t = getattr(self.net, "_current_timestep_for_meter", 0)

            if (not self.count_first_timestep) and current_t == 0:
                return

            if isinstance(output, (tuple, list)):
                output_tensor = output[0]
            else:
                output_tensor = output

            if not torch.is_tensor(output_tensor):
                return

            valid_b = getattr(
                self.net,
                "_valid_batch_size_for_meter",
                output_tensor.shape[0]
            )

            output_tensor = output_tensor[:valid_b]

            spikes = (output_tensor.detach() > 0).float()

            self.spike_sum[layer_name] += spikes.sum().item()
            self.elem_sum[layer_name] += spikes.numel()
            self.call_sum[layer_name] += 1

        return hook

    def register(self):
        named_modules = dict(self.net.named_modules())

        for module_name, layer_name in self.layer_map.items():
            if module_name not in named_modules:
                raise KeyError(f"Cannot find module: {module_name}")

            module = named_modules[module_name]
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self.hooks.append(handle)

    def remove(self):
        for h in self.hooks:
            h.remove()

    def rates(self):
        result = {}

        for k in self.spike_sum:
            if self.elem_sum[k] == 0:
                result[k] = 0.0
            else:
                result[k] = self.spike_sum[k] / self.elem_sum[k]

        return result


# =========================================================
# 3. FLOPs and energy estimation
# =========================================================
def count_conv_macs(h, w, cin, cout, k, branches=1):
    """
    按 MACs 计算。
    如果你想按照 1 MAC = 2 FLOPs，可在最终结果整体乘以 2。
    """
    return branches * h * w * cin * cout * k * k


def count_fc_macs(in_features, out_features):
    return in_features * out_features


def get_layer_flops():
    """
    单个 patch、单个时间步的操作量。

    Siamese 结构中：
    Conv1/Conv2/Conv3 对 near/far 两个分支都执行，因此 branches=2；
    FC1/FC2 在拼接后执行一次。
    """
    flops = {}

    flops["Conv1"] = count_conv_macs(
        h=16, w=16, cin=1, cout=32, k=3, branches=2
    )

    flops["Conv2"] = count_conv_macs(
        h=16, w=16, cin=32, cout=64, k=3, branches=2
    )

    flops["Conv3"] = count_conv_macs(
        h=8, w=8, cin=64, cout=128, k=3, branches=2
    )

    flops["FC1"] = count_fc_macs(
        in_features=128 * 8 * 8 * 2,
        out_features=128
    )

    flops["FC2"] = count_fc_macs(
        in_features=128,
        out_features=2
    )

    return flops


def format_flops(x):
    if x >= 1e9:
        return f"{x / 1e9:.2f}G"
    elif x >= 1e6:
        return f"{x / 1e6:.2f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.2f}K"
    else:
        return f"{x:.0f}"


def calculate_energy_rows(spike_rates, flops_per_patch, total_patches, num_timesteps):
    """
    45-nm CMOS 参考能耗：
        ANN MAC energy: 4.6 pJ
        CNP/SNN spike operation energy: 0.9 pJ

    注意：
        Conv1 直接接收连续输入图像，因此按 dense MAC 计算，使用 4.6 pJ。
        Conv2/Conv3/FC1/FC2 的输入来自前一 CNP 节点的脉冲输出，
        因此按 spiking rate 和 0.9 pJ 估计。
    """
    E_MAC = 4.6e-12
    E_SOP = 0.9e-12

    rows = []

    for layer in ["Conv1", "Conv2", "Conv3", "FC1", "FC2"]:
        flops_1 = flops_per_patch[layer]
        rate = spike_rates[layer]

        total_flops = flops_1 * total_patches * num_timesteps

        ann_energy = total_flops * E_MAC

        if layer == "Conv1":
            # 第一层是连续图像输入，不使用 spike sparsity
            cnp_energy = total_flops * E_MAC
            energy_per_op_pj = 4.6
            used_rate = 1.0
        else:
            cnp_energy = total_flops * rate * E_SOP
            energy_per_op_pj = 0.9
            used_rate = rate

        rows.append({
            "Layer": layer,
            "Spiking Rate": rate,
            "Used Spiking Rate": used_rate,
            "FLOPs per Patch": flops_1,
            "Total FLOPs": total_flops,
            "Energy Cost in CNP Single Operation": energy_per_op_pj,
            "ANN Energy (J)": ann_energy,
            "CNP Energy (J)": cnp_energy,
        })

    return rows


# =========================================================
# 4. Image loading
# =========================================================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


def load_gray_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path)
    img = transform(img)
    return img.to(device)


def get_num_patches(image_tensor, window_size=16, stride=4):
    """
    计算 unfold 后的 patch 数。
    image_tensor: (1, H, W)
    """
    _, H, W = image_tensor.shape
    pad = window_size // 2

    out_h = (H + 2 * pad - window_size) // stride + 1
    out_w = (W + 2 * pad - window_size) // stride + 1

    return out_h * out_w


# =========================================================
# 5. Forward with fixed batch padding
# =========================================================
@torch.no_grad()
def forward_spike_freq_padded(
    net: nn.Module,
    patch1: torch.Tensor,
    patch2: torch.Tensor,
    num_timesteps: int,
    fixed_batch_size: int
) -> torch.Tensor:
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

    net._valid_batch_size_for_meter = B

    spike_count = None

    for t in range(num_timesteps):
        net._current_timestep_for_meter = t

        y = net(patch1, patch2)

        if t == 0:
            spike_count = torch.zeros_like(y)
        else:
            spike_count += y

    spike_freq = spike_count / (num_timesteps - 1)

    functional.reset_net(net)

    if hasattr(net, "_valid_batch_size_for_meter"):
        delattr(net, "_valid_batch_size_for_meter")

    if hasattr(net, "_current_timestep_for_meter"):
        delattr(net, "_current_timestep_for_meter")

    return spike_freq[:B]


# =========================================================
# 6. Unfold-based sliding-window inference
# =========================================================
@torch.no_grad()
def sliding_window_inference_unfold(
    net: nn.Module,
    imageA: torch.Tensor,
    imageB: torch.Tensor,
    num_timesteps: int = 4,
    window_size: int = 16,
    stride: int = 4,
    infer_batch_size: int = 256
) -> np.ndarray:
    assert imageA.shape == imageB.shape
    assert imageA.dim() == 3 and imageA.shape[0] == 1

    _, H, W = imageA.shape
    pad = window_size // 2

    imageA_4d = imageA.unsqueeze(0)
    imageB_4d = imageB.unsqueeze(0)

    imageA_pad = F.pad(imageA_4d, (pad, pad, pad, pad), mode="reflect")
    imageB_pad = F.pad(imageB_4d, (pad, pad, pad, pad), mode="reflect")

    colsA = F.unfold(imageA_pad, kernel_size=window_size, stride=stride)
    colsB = F.unfold(imageB_pad, kernel_size=window_size, stride=stride)

    L = colsA.shape[-1]

    out_h = (H + 2 * pad - window_size) // stride + 1
    out_w = (W + 2 * pad - window_size) // stride + 1

    assert L == out_h * out_w

    patchesA = colsA.squeeze(0).transpose(0, 1).contiguous().view(
        L, 1, window_size, window_size
    )

    patchesB = colsB.squeeze(0).transpose(0, 1).contiguous().view(
        L, 1, window_size, window_size
    )

    probs_all = np.zeros(L, dtype=np.float32)

    for s in tqdm(range(0, L, infer_batch_size), desc="Unfold Inference", dynamic_ncols=True):
        e = min(s + infer_batch_size, L)

        patchA_batch = patchesA[s:e]
        patchB_batch = patchesB[s:e]

        logits = forward_spike_freq_padded(
            net=net,
            patch1=patchA_batch,
            patch2=patchB_batch,
            num_timesteps=num_timesteps,
            fixed_batch_size=infer_batch_size
        )

        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs_all[s:e] = probs

    score_map = probs_all.reshape(out_h, out_w)

    if score_map.shape != (H, W):
        score_map = cv2.resize(score_map, (W, H), interpolation=cv2.INTER_LINEAR)

    return score_map


# =========================================================
# 7. Print and save results
# =========================================================
def print_table(spike_rates, flops_per_patch, energy_rows):
    layers = ["Conv1", "Conv2", "Conv3", "FC1", "FC2"]

    energy_cost_map = {
        row["Layer"]: row["Energy Cost in CNP Single Operation"]
        for row in energy_rows
    }

    print("\n===== Spiking Rate / FLOPs / Energy Table =====")
    print("Layer\t" + "\t".join(layers))

    print(
        "Spiking Rate\t" +
        "\t".join([f"{spike_rates[layer] * 100:.2f}%" for layer in layers])
    )

    print(
        "FLOPs per Patch\t" +
        "\t".join([format_flops(flops_per_patch[layer]) for layer in layers])
    )

    print(
        "Energy Cost in CNP\n(Single Operation)\t" +
        "\t".join([f"{energy_cost_map[layer]:.1f}pJ" for layer in layers])
    )

    print("\n===== LaTeX Table Rows =====")
    print("Layer & " + " & ".join(layers) + r" \\")

    print(
        "Spiking Rate & " +
        " & ".join([f"{spike_rates[layer] * 100:.2f}\\%" for layer in layers]) +
        r" \\"
    )

    print(
        "FLOPs & " +
        " & ".join([format_flops(flops_per_patch[layer]) for layer in layers]) +
        r" \\"
    )

    print(
        "Energy Cost in CNP (Single Operation) & " +
        " & ".join([f"{energy_cost_map[layer]:.1f}pJ" for layer in layers]) +
        r" \\"
    )

    print("\n===== Energy Estimation =====")
    total_ann = 0.0
    total_cnp = 0.0

    for row in energy_rows:
        total_ann += row["ANN Energy (J)"]
        total_cnp += row["CNP Energy (J)"]

        print(
            f"{row['Layer']:5s} | "
            f"Rate={row['Spiking Rate'] * 100:.2f}% | "
            f"UsedRate={row['Used Spiking Rate'] * 100:.2f}% | "
            f"Total FLOPs={format_flops(row['Total FLOPs'])} | "
            f"OpEnergy={row['Energy Cost in CNP Single Operation']:.1f}pJ | "
            f"ANN={row['ANN Energy (J)']:.4e} J | "
            f"CNP={row['CNP Energy (J)']:.4e} J"
        )

    print("\nTotal ANN Energy:", f"{total_ann:.4e}", "J")
    print("Total CNP Energy:", f"{total_cnp:.4e}", "J")

    if total_cnp > 0:
        print("Energy Reduction:", f"{total_ann / total_cnp:.2f}x")


def save_csv(spike_rates, flops_per_patch, energy_rows, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Layer",
            "Spiking Rate",
            "Spiking Rate (%)",
            "Used Spiking Rate",
            "Used Spiking Rate (%)",
            "FLOPs per Patch",
            "Total FLOPs",
            "Energy Cost in CNP Single Operation (pJ)",
            "ANN Energy (J)",
            "CNP Energy (J)",
        ])

        for row in energy_rows:
            layer = row["Layer"]

            writer.writerow([
                layer,
                f"{spike_rates[layer]:.8f}",
                f"{spike_rates[layer] * 100:.4f}",
                f"{row['Used Spiking Rate']:.8f}",
                f"{row['Used Spiking Rate'] * 100:.4f}",
                f"{flops_per_patch[layer]}",
                f"{row['Total FLOPs']}",
                f"{row['Energy Cost in CNP Single Operation']:.1f}",
                f"{row['ANN Energy (J)']:.8e}",
                f"{row['CNP Energy (J)']:.8e}",
            ])

    print(f"\nCSV saved to: {save_path}")


# =========================================================
# 8. Main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "./weight/all_weights_4/epoch-527.pth"

    imageA_dir = "./Lytro/near"
    imageB_dir = "./Lytro/far"

    num_images = 20

    num_timesteps = 4
    window_size = 16
    stride = 4
    infer_batch_size = 256

    # True: 统计第 0 个时间步的 CNP 输出
    # False: 跳过第 0 个时间步，使统计更接近 spike_count / (T - 1)
    count_first_timestep = True

    net = SiameseCNPNetwork(batch_size=infer_batch_size).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        net.load_state_dict(state_dict["model_state_dict"])
    else:
        net.load_state_dict(state_dict)

    net.eval()

    meter = SpikeRateMeter(
        net=net,
        count_first_timestep=count_first_timestep
    )

    meter.register()

    total_patches = 0

    start_time = time.time()

    with torch.no_grad():
        for i in range(1, num_images + 1):
            imageA_path = os.path.join(imageA_dir, f"{i}.jpg")
            imageB_path = os.path.join(imageB_dir, f"{i}.jpg")

            if not os.path.exists(imageA_path):
                raise FileNotFoundError(imageA_path)

            if not os.path.exists(imageB_path):
                raise FileNotFoundError(imageB_path)

            imageA = load_gray_tensor(imageA_path, device)
            imageB = load_gray_tensor(imageB_path, device)

            if imageA.shape != imageB.shape:
                raise ValueError(f"Image shape mismatch: {imageA.shape} vs {imageB.shape}")

            n_patches = get_num_patches(
                image_tensor=imageA,
                window_size=window_size,
                stride=stride
            )

            total_patches += n_patches

            print(f"\n[{i}/{num_images}] {imageA_path} vs {imageB_path}, patches={n_patches}")

            _ = sliding_window_inference_unfold(
                net=net,
                imageA=imageA,
                imageB=imageB,
                num_timesteps=num_timesteps,
                window_size=window_size,
                stride=stride,
                infer_batch_size=infer_batch_size
            )

    meter.remove()

    elapsed = time.time() - start_time

    spike_rates = meter.rates()
    flops_per_patch = get_layer_flops()

    energy_rows = calculate_energy_rows(
        spike_rates=spike_rates,
        flops_per_patch=flops_per_patch,
        total_patches=total_patches,
        num_timesteps=num_timesteps
    )

    print_table(
        spike_rates=spike_rates,
        flops_per_patch=flops_per_patch,
        energy_rows=energy_rows
    )

    save_csv(
        spike_rates=spike_rates,
        flops_per_patch=flops_per_patch,
        energy_rows=energy_rows,
        save_path="./energy_results/spiking_rate_energy_table.csv"
    )

    print("\nFinished.")
    print(f"Total images: {num_images}")
    print(f"Total patches: {total_patches}")
    print(f"Time steps: {num_timesteps}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()