import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=13, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, 5, stride=2, padding=11, bias=False)
        self.bn2_3 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, 4, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)

        self.out = nn.Linear(28800, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.bn3(x)
        x = F.dropout(x, p=0.4)

        x = self.conv2_1(x)
        x = F.leaky_relu(x)
        x = self.bn2_1(x)
        x = self.conv2_2(x)
        x = F.leaky_relu(x)
        x = self.bn2_2(x)
        x = self.conv2_3(x)
        x = F.leaky_relu(x)
        x = self.bn2_3(x)
        x = F.dropout(x, p=0.4)

        x = self.conv3_1(x)
        x = F.leaky_relu(x)
        x = self.bn3_1(x)

        x = x.view(-1, 28800)
        x = F.dropout(x, p=0.4)

        o = self.out(x)

        return o
