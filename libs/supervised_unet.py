import torch.nn as nn
import torch


class unet_EncodingStage(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unet_EncodingStage, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        s = self.layers(x)
        out = self.down(s)
        return out, s


class unet_CenterStage(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unet_CenterStage, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class unet_DecodingStage(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unet_DecodingStage, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, shortcut):
        x = self.up_conv(x)
        x = torch.cat((shortcut, x), dim=1)
        x = self.layers(x)
        return x


class unet(nn.Module):
    def __init__(self, img_channel, mask_channel, base_channel=64):
        super(unet, self).__init__()
        self.E1 = unet_EncodingStage(img_channel, base_channel)
        self.E2 = unet_EncodingStage(base_channel, base_channel * 2)
        self.E3 = unet_EncodingStage(base_channel * 2, base_channel * 4)
        self.E4 = unet_EncodingStage(base_channel * 4, base_channel * 8)

        self.Center = unet_CenterStage(base_channel * 8, base_channel * 16)

        self.D1 = unet_DecodingStage(base_channel * 16, base_channel * 8)
        self.D2 = unet_DecodingStage(base_channel * 8, base_channel * 4)
        self.D3 = unet_DecodingStage(base_channel * 4, base_channel * 2)
        self.D4 = unet_DecodingStage(base_channel * 2, base_channel)

        self.out = nn.Conv2d(base_channel, mask_channel, kernel_size=1, stride=1)
    
    def forward(self, x):
        x, s1 = self.E1(x)
        x, s2 = self.E2(x)
        x, s3 = self.E3(x)
        x, s4 = self.E4(x)

        x = self.Center(x)

        x = self.D1(x, s4)
        x = self.D2(x, s3)
        x = self.D3(x, s2)
        x = self.D4(x, s1)

        return self.out(x)