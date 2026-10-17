import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        def discrim_block(in_filter, out_filter, normalization=True):
            layers = [nn.Conv2d(in_filter, out_filter, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filter, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrim_block(in_channels + out_channels, 64, normalization=False),
            # Concatenates Sentinel input + DSM target/prediction
            *discrim_block(64, 128),
            *discrim_block(128, 256),
            *discrim_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, sentinel_img, dsm_img):
        # Concatenate conditioning image (Sentinel) and target/generated DSM along channel axis
        cat_input = torch.cat((sentinel_img, dsm_img), dim=1)
        return self.model(cat_input)