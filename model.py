import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.features = [64, 128, 256, 512, 1024]
        self.c1 = DoubleConv(in_channels, self.features[0])
        self.d1 = nn.MaxPool2d((2, 2))
        self.c2 = DoubleConv(self.features[0], self.features[1])
        self.d2 = nn.MaxPool2d((2, 2))
        self.c3 = DoubleConv(self.features[1], self.features[2])
        self.d3 = nn.MaxPool2d((2, 2))
        self.c4 = DoubleConv(self.features[2], self.features[3])
        self.d4 = nn.MaxPool2d((2, 2))
        self.c5 = DoubleConv(self.features[3], self.features[4])
        self.u1 = nn.ConvTranspose2d(in_channels=self.features[4], out_channels=self.features[3], kernel_size=(2, 2), stride=(2, 2))
        self.c6 = DoubleConv(self.features[4], self.features[3])
        self.u2 = nn.ConvTranspose2d(in_channels=self.features[3], out_channels=self.features[2], kernel_size=(2, 2), stride=(2, 2))
        self.c7 = DoubleConv(self.features[3], self.features[2])
        self.u3 = nn.ConvTranspose2d(in_channels=self.features[2], out_channels=self.features[1], kernel_size=(2, 2), stride=(2, 2))
        self.c8 = DoubleConv(self.features[2], self.features[1])
        self.u4 = nn.ConvTranspose2d(in_channels=self.features[1], out_channels=self.features[0], kernel_size=(2, 2), stride=(2, 2))
        self.c9 = DoubleConv(self.features[1], self.features[0])
        self.c10 = nn.Conv2d(self.features[0], out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        c1 = self.c1(x)
        d1 = self.d1(c1)
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        c4 = self.c4(d3)
        d4 = self.d4(c4)
        c5 = self.c5(d4)
        u1 = self.u1(c5)
        u1 = torch.cat((u1, c4), dim=1)
        c6 = self.c6(u1)
        u2 = self.u2(c6)
        u2 = torch.cat((u2, c3), dim=1)
        c7 = self.c7(u2)
        u3 = self.u3(c7)
        u3 = torch.cat((u3, c2), dim=1)
        c8 = self.c8(u3)
        u4 = self.u4(c8)
        u4 = torch.cat((u4, c1), dim=1)
        c9 = self.c9(u4)
        c10 = self.c10(c9)
        return c10
