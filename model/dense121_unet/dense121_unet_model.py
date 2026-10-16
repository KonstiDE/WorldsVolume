import time

import torch
import torch.nn as nn

from model.dense121_unet.dense121_layers_unet import (
    PreDenseBlock, DenseBlock, ConvTBlock, Auxiliary, ASPP
)


class DenseNet121(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, k=32, attention=False, aspp=False, aux=False):
        super(DenseNet121, self).__init__()

        self.pd1 = PreDenseBlock(in_channels, 64, (7, 7), 2, 3, start=True)
        self.pd2 = PreDenseBlock(256, 128, (1, 1), 1, 0)
        self.pd3 = PreDenseBlock(512, 256, (1, 1), 1, 0)
        self.pd4 = PreDenseBlock(1024, 512, (1, 1), 1, 0)

        self.dn1 = DenseBlock(64, 256, 6, k)
        self.dn2 = DenseBlock(128, 512, 12, k)
        self.dn3 = DenseBlock(256, 1024, 24, k)
        self.dn4 = DenseBlock(512, 1024, 16, k, transition=False)

        self.dc1 = ConvTBlock(1024, 1024, attention=attention)
        self.dc2 = ConvTBlock(1024, 512, double=True, attention=attention)
        self.dc3 = ConvTBlock(512, 256, double=True, attention=attention)
        self.dc4 = ConvTBlock(256, 64, double=True, intermezzo=True, attention=attention)

        self.dn5 = DenseBlock(256, 1024, 24, k, upsample=True)
        self.dn6 = DenseBlock(128, 512, 12, k, upsample=True)
        self.dn7 = DenseBlock(64, 256, 6, k, upsample=True)

        self.decode_conv1 = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.decode_conv2 = nn.Conv2d(1024, 128, kernel_size=(1, 1))
        self.decode_conv3 = nn.Conv2d(512, 64, kernel_size=(1, 1))
        self.decode_conv4 = nn.Conv2d(128, 64, kernel_size=(7, 7), padding=3)

        self.final_block = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=(1, 1))
        )

        self.aux = aux
        self.aspp = aspp

        if aspp:
            self.aspp1 = ASPP(in_channels=256, out_channels=256, atrous_rates=[6, 12, 18])
            self.aspp2 = ASPP(in_channels=512, out_channels=512, atrous_rates=[6, 12, 18])
            self.aspp3 = ASPP(in_channels=1024, out_channels=1024, atrous_rates=[6, 12, 18])
            self.aspp4 = ASPP(in_channels=1024, out_channels=1024, atrous_rates=[6, 12, 18])

        if aux:
            self.pd1n = PreDenseBlock(64, 64, (7, 7), 2, 3, start=True)
            self.final_aux = Auxiliary()

            self.final_intermediate = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=(1, 1))
            )

            self.sn1_correct = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1))
            self.sn2_correct = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1))
            self.sn3_correct = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1))

            self.dn1n = DenseBlock(128, 256, 6, k)
            self.dn2n = DenseBlock(256, 512, 12, k)
            self.dn3n = DenseBlock(512, 1024, 24, k)
            self.dn4n = DenseBlock(512, 1024, 16, k, transition=False)

            self.pd2n = PreDenseBlock(256, 128, (1, 1), 1, 0)
            self.pd3n = PreDenseBlock(512, 256, (1, 1), 1, 0)
            self.pd4n = PreDenseBlock(1024, 512, (1, 1), 1, 0)

            self.dc1n = ConvTBlock(1024, 1024)
            self.dc2n = ConvTBlock(1024, 512, double=True)
            self.dc3n = ConvTBlock(512, 256, double=True)
            self.dc4n = ConvTBlock(256, 64, double=True, intermezzo=True)

            self.dn5n = DenseBlock(256, 1024, 24, k, upsample=True)
            self.dn6n = DenseBlock(128, 512, 12, k, upsample=True)
            self.dn7n = DenseBlock(64, 256, 6, k, upsample=True)

            self.decode_conv1n = nn.Conv2d(2048, 256, kernel_size=(1, 1))
            self.decode_conv2n = nn.Conv2d(1024, 128, kernel_size=(1, 1))
            self.decode_conv3n = nn.Conv2d(512, 64, kernel_size=(1, 1))
            self.decode_conv4n = nn.Conv2d(128, 64, kernel_size=(7, 7), padding=3)

    def forward(self, x):
        degree = distance = None
        sn1 = sn2 = sn3 = None

        # Encoder
        x, s1 = self.pd1(x)
        x = s2 = self.dn1(x)

        if self.aspp:
            x = self.aspp1(x)

        x = self.pd2(x)
        x = s3 = self.dn2(x)

        if self.aspp:
            x = self.aspp2(x)

        x = self.pd3(x)
        x = s4 = self.dn3(x)

        if self.aspp:
            x = self.aspp3(x)

        x = self.pd4(x)
        x = self.dn4(x)

        if self.aspp:
            x = self.aspp4(x)

        # Decoder
        x = self.dc1(x, s4)
        x = self.decode_conv1(x)
        x = self.dn5(x)

        if self.aux:
            sn3 = self.sn3_correct(x)

        x = self.dc2(x, s3)
        x = self.decode_conv2(x)
        x = self.dn6(x)

        if self.aux:
            sn2 = self.sn2_correct(x)

        x = self.dc3(x, s2)
        x = self.decode_conv3(x)
        x = self.dn7(x)

        if self.aux:
            sn1 = self.sn1_correct(x)

        x = self.dc4(x, s1)
        x = self.decode_conv4(x)

        if not self.aux:
            return self.final_block(x), degree, distance

        else:
            x = self.final_intermediate(x)

            degree, distance = self.final_aux(x)

            x, s1 = self.pd1n(x)
            x = s2 = self.dn1n(torch.cat([x, sn1], 1))

            x = self.pd2n(x)
            x = s3 = self.dn2n(torch.cat([x, sn2], 1))

            x = self.pd3n(x)
            x = s4 = self.dn3n(torch.cat([x, sn3], 1))

            x = self.pd4n(x)
            x = self.dn4n(x)

            x = self.dc1n(x, s4)
            x = self.decode_conv1n(x)
            x = self.dn5n(x)

            x = self.dc2n(x, s3)
            x = self.decode_conv2n(x)
            x = self.dn6n(x)

            x = self.dc3n(x, s2)
            x = self.decode_conv3n(x)
            x = self.dn7n(x)

            x = self.dc4n(x, s1)
            x = self.decode_conv4n(x)

            return self.final_block(x), degree, distance


if __name__ == '__main__':
    a = torch.randn((8, 3, 512, 512)).cuda()

    model = DenseNet121(in_channels=3, out_channels=1, attention=False, aspp=True, aux=False).cuda()

    stamp = time.time()

    b, _, _ = model(a)
    print("This took: {}".format(time.time() - stamp))

    print(b.shape)
