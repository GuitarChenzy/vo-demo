import torch
import torch.nn as nn


def conv(batch_norm, in_channel, out_channel, k_size=3, stride = 1):
    if batch_norm:
        net = nn.Sequential(
            nn.Conv2d
        )