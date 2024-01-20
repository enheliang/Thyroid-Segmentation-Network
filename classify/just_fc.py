
import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  #
        attn_matrix = self.softmax(attn_matrix)  #
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #
        out = out.view(*input.shape)

        return self.gamma * out + input

class Just_FC(nn.Module):
    def __init__(self, num_classes=1000):
        super(Just_FC, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(6144, 2000)    # 2048*3   6144    5632   2816*3
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 2)
        self.conv_ratio = nn.Conv2d(in_channels=1, out_channels=512,
                                    kernel_size=1, stride=1, bias=False, padding=0)
        self.conv_w = nn.Conv2d(in_channels=1, out_channels=512,
                                    kernel_size=1, stride=1, bias=False, padding=0)
        self.attn = Attention(6144)

    def forward(self, x1, x2, x3):#, ratios, w  x3

        x1 = torch.cat([x1, x2], 1)

        x_t = torch.cat([x1, x3], 1)
        # x1 = x_t.transpose(1, 2)

        # a1 = self.attn(x_t)
        x5 = torch.flatten(x_t, 1)
        x_t = x5

        x_t = self.fc1(x_t)
        x_t = self.fc2(x_t)
        x_t = self.fc3(x_t)
        return x_t






