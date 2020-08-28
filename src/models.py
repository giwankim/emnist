import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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


Half_width = 128
layer_width = 128


class SpinalVGG(nn.Module):
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        return s

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(f2),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        return s

    def __init__(self, num_classes=10):
        super(SpinalVGG, self).__init__()

        # Convolutional layers
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        # Spinal fully-connected layers
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width, layer_width, bias=False),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width, bias=False),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width, bias=False),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width, bias=False),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(),
        )

        # Final output layer
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(layer_width * 4, num_classes),
        )

    def forward(self, x):
        # Extract features
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)

        # Get spinal fully-connected layer outputs
        x1 = x[:, 0:Half_width]
        x1 = self.fc_spinal_layer1(x1)

        x2 = torch.cat([x[:, Half_width : 2 * Half_width], x1], dim=1)
        x2 = self.fc_spinal_layer2(x2)

        x3 = torch.cat([x[:, 0:Half_width], x2], dim=1)
        x3 = self.fc_spinal_layer3(x3)

        x4 = torch.cat([x[:, Half_width : 2 * Half_width], x3], dim=1)
        x4 = self.fc_spinal_layer4(x4)

        # Get final output layer
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fc_out(x)

        # return F.log_softmax(x, dim=1)
        return x


## EfficentNetB0
def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    """Squeeze-and-Excitation block with Swish."""

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    """expansion + depthwise + pointwise + squeeze-excitation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio=1,
        se_ratio=0.0,
        drop_rate=0.0,
    ):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(1 if kernel_size == 3 else 2),
            groups=channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(
            channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg["out_channels"][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [
            self.cfg[k]
            for k in [
                "expansion",
                "out_channels",
                "num_blocks",
                "kernel_size",
                "stride",
            ]
        ]
        b = 0
        blocks = sum(self.cfg["num_blocks"])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg["drop_connect_rate"] * b / blocks
                layers.append(
                    Block(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        expansion,
                        se_ratio=0.25,
                        drop_rate=drop_rate,
                    )
                )
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.ZeroPad2d(4)(x)

        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg["dropout_rate"]
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def EfficientNetB0():
    cfg = {
        "num_blocks": [1, 2, 2, 3, 3, 4, 1],
        "expansion": [1, 6, 6, 6, 6, 6, 6],
        "out_channels": [16, 24, 40, 80, 112, 192, 320],
        "kernel_size": [3, 3, 5, 3, 5, 5, 3],
        "stride": [1, 2, 2, 2, 1, 2, 1],
        "dropout_rate": 0.2,
        "drop_connect_rate": 0.2,
    }
    return EfficientNet(cfg)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
