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
