import torch
import torch.nn as nn


class RIDNet(nn.Module):
    def __init__(self):
        super(RIDNet, self).__init__()

        self.head = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(
            *[self.residual_dense_block(64) for _ in range(3)]
        )
        self.tail = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

    def residual_dense_block(self, channels):
        
