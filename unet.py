import torch 
import torch.nn as nn
import torch.nn.functional as F

class lReLU(nn.Module):
    def __init__(self):
        super(lReLU, self).__init__()

    def forward(self, x):
        return torch.max(x * 0.2, x)

class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Double_Conv2d, self).__init__()
        self.double_conv2d = torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            lReLU()
        )

    def forward(self, x):
        return self.double_conv2d(x)

class UNetSony(nn.Module):
    def __init__(self):
        super(UNetSony, self).__init__()
        self.conv1 = Double_Conv2d(4, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(64, 32)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 2)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

class UNetFuji(nn.Module):
    def __init__(self):
        super(UNetFuji, self).__init__()
        self.conv1 = Double_Conv2d(9, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(64, 32)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=27, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 3)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)