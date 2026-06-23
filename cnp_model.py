# %%
import torch
from torch import nn
import numpy as np

from spikingjelly.clock_driven import base, surrogate

from typing import Tuple


# import copy

# import surrogate
# import base

class Sigmoid(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.heaviside(input, torch.ones_like(input))  # t=1时刻都会发放脉冲

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (Sigmoid.scale * torch.abs(input) + 1.0) ** 2
        return grad


class SG(torch.autograd.Function):
    # Altered from code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


sigmoid = Sigmoid.apply


class CNPNode(base.MemoryModule):
    def __init__(self, tau_f: float, tau_l: float, tau_e: float, beta: float, ve: float,
                 in_channels: int, out_channels: int, kernel_size: Tuple[int, ...], stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 batch_size: int, shape: Tuple[int, ...],
                 local: bool = True, shared: bool = True, self_feedback: bool = True, multi_channel: bool = False,
                 learnable: bool = True,
                 linking: bool = True, encoding: bool = True):

        super().__init__()

        if local and shared and self_feedback and not multi_channel and learnable:
            self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        elif not local and not shared and self_feedback and not multi_channel and learnable:
            self.op = nn.Linear(*shape, *shape, bias=False)

        self.linking = linking
        self.encoding = encoding

        self.tau_f = tau_f
        self.tau_l = tau_l
        self.tau_e = tau_e
        self.ve = ve
        self.beta = beta

        self.surrogate_function = SG.apply

        self.compute_decays()

        self.register_memory('f', torch.zeros(batch_size, *shape))
        self.register_memory('l', torch.zeros(batch_size, *shape))
        self.register_memory('e', self.ve * torch.zeros(batch_size, *shape))
        self.register_memory('s', torch.zeros(batch_size, *shape))

    def forward(self, x: torch.Tensor):
        if self.encoding:
            self.f = x
        else:
            self.f = self.f * self.decay_f + x

        # linking input and modulation
        if self.linking:
            self.l = self.l * self.decay_l + self.op(self.s)
            self.u = self.f * (1 + self.beta * self.l)
            # self.u = self.f * (1 + self.l)
        else:
            self.u = self.f

        # dynamic threshold
        self.e = self.decay_e * self.e + self.s * self.ve

        # spiking generator
        self.s = self.surrogate_function(self.u - self.e, 1.0)
        return self.s

    def compute_decays(self):
        # computer
        self.decay_f = float(np.exp(-1 / self.tau_f))
        self.decay_l = float(np.exp(-1 / self.tau_l))
        self.decay_e = float(np.exp(-1 / self.tau_e))
