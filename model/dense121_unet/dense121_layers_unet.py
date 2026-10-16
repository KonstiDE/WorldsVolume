# Credit goes to the link below for providing the make_dense() method not described in Henry et al.:
# https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class PreDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, start=False):
        super(PreDenseBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.start = start

        self.start_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if start:
            self.feed_forward = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            )

    def forward(self, x):
        if self.start:
            x = s = self.start_conv(x)

            return self.feed_forward(x), s

        return self.feed_forward(x)


class DenseSingleLayer(nn.Module):
    def __init__(self, n_channels, growth_rate):
        super(DenseSingleLayer, self).__init__()
        self.bn = nn.BatchNorm2d(n_channels)
        self.conv = nn.Conv2d(n_channels, growth_rate, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat((x, out), 1)
        return out


class DenseTransition(nn.Module):
    def __init__(self, n_channels, n_out_channels, upsample):
        super(DenseTransition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1)

        self.upsample = upsample

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        if not self.upsample:
            out = F.avg_pool2d(out, 2)
        else:
            out = F.interpolate(out, scale_factor=2)

        return out


def make_dense(n_channels, growth_rate, n_dense_blocks):
    layers = []
    for i in range(int(n_dense_blocks)):
        layers.append(DenseSingleLayer(n_channels, growth_rate))
        n_channels += growth_rate
    return nn.Sequential(*layers)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_dense_blocks, k, transition=True, upsample=False):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.transition = transition

        self.dense = make_dense(n_channels=in_channels, growth_rate=k, n_dense_blocks=n_dense_blocks)

        if transition:
            self.trans = DenseTransition(in_channels + n_dense_blocks * k, out_channels, upsample)

    def forward(self, x, s=None):
        if self.transition:
            return self.trans(self.dense(x))

        return self.dense(x)


class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, double=False, intermezzo=False, attention=False):
        super(ConvTBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.attention = attention

        if double:
            if not intermezzo:
                self.convt = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(2, 2), stride=2)
                )
                if self.attention:
                    self.am = Attention(out_channels * 2, out_channels)
            else:
                self.convt = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=(2, 2), stride=2)
                )
                if self.attention:
                    self.am = Attention(out_channels * 4, out_channels)
        else:
            self.convt = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
            )
            if self.attention:
                self.am = Attention(out_channels, out_channels)

    def forward(self, x, s):
        if self.attention:
            # s is x in am, gating signal g is x here
            s = self.am(x, s)

        x = self.convt(x)

        return torch.cat((x, s), dim=1)


class Auxiliary(nn.Module):
    def __init__(self, in_channels=64, out_channels_deg=6, out_channels_dis=1):
        super(Auxiliary, self).__init__()

        self.pre = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.deg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_deg, kernel_size=(1, 1)),
            nn.Softmax2d()
        )

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_dis, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.pre(x)

        return torch.argmax(self.deg(x), dim=1, keepdim=True), self.dis(x)


class Attention(nn.Module):
    def __init__(self, g_in, channels):
        super(Attention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_in, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()

        self.aspp_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.aspp_conv(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()

        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=512):
        super(ASPP, self).__init__()
        # Standard dilation=1 (default)
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res = torch.cat(res, dim=1)

        return self.project(res)


if __name__ == '__main__':
    a = torch.randn([8, 64, 512, 512])

    m = Attention(channels=64)

    out = m(a)
    print(out.shape)


