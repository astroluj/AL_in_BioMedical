import torch
import torch.nn as nn
import torch.nn.functional as F


class LossNet(nn.Module):
    def __init__(self, num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.conv1 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_channels[1], num_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(num_channels[2], num_channels[2], kernel_size=3, stride=2, padding=1, bias=False)

        self.linear = nn.Linear(4 * interm_dim, 1)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        out1 = self.conv1(features[0])
        out1 = self.GAP(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.relu(self.FC1(out1))

        out2 = self.conv2(features[1])
        out2 = self.GAP(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.relu(self.FC2(out2))

        out3 = self.conv3(features[2])
        out3 = self.GAP(out3)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.relu(self.FC3(out3))

        out4 = self.GAP(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = self.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))

        return out