import torch
import torch.nn as nn
import time

from model.unet.unet_layers import (
    DoubleConv, UpConv
)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET, self).__init__()

        self.dc1 = DoubleConv(in_channels=in_channels, out_channels=64)
        self.dc2 = DoubleConv(in_channels=64, out_channels=128)
        self.dc3 = DoubleConv(in_channels=128, out_channels=256)
        self.dc4 = DoubleConv(in_channels=256, out_channels=512)
        self.bc = DoubleConv(in_channels=512, out_channels=1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.uc4 = UpConv(in_channels=1024, out_channels=512)
        self.uc3 = UpConv(in_channels=512, out_channels=256)
        self.uc2 = UpConv(in_channels=256, out_channels=128)
        self.uc1 = UpConv(in_channels=128, out_channels=64)

        self.dc5 = DoubleConv(in_channels=1024, out_channels=512)
        self.dc6 = DoubleConv(in_channels=512, out_channels=256)
        self.dc7 = DoubleConv(in_channels=256, out_channels=128)
        self.dc8 = DoubleConv(in_channels=128, out_channels=64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=(1, 1))


    def forward(self, x):
        x = s1 = self.dc1(x)
        x = self.pool(x)

        x = s2 = self.dc2(x)
        x = self.pool(x)

        x = s3 = self.dc3(x)
        x = self.pool(x)

        x = s4 = self.dc4(x)
        x = self.pool(x)

        x = self.bc(x)

        x = self.uc4(x)
        x = self.dc5(torch.cat([x, s4], 1))

        x = self.uc3(x)
        x = self.dc6(torch.cat([x, s3], 1))

        x = self.uc2(x)
        x = self.dc7(torch.cat([x, s2], 1))

        x = self.uc1(x)
        x = self.dc8(torch.cat([x, s1], 1))

        return self.final(x), None, None


def test():
    a = torch.randn((8, 3, 512, 512)).cuda()
    unet = UNET(in_channels=3, out_channels=1).cuda()

    stamp = time.time()

    b, _, _ = unet(a)
    print("This took: {}".format(time.time() - stamp))

    print(b.shape)


if __name__ == "__main__":
    test()
