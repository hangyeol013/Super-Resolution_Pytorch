import torch
import torch.nn as nn
import numpy as np


class FSRCNN(nn.Module):

    def __init__(self, sf):
        super().__init__()

        self.in_nc = 1  # RGB -> YCrCb, Only Y
        self.nc1 = 56
        self.nc2 = 12
        self.out_nc = 1  # Only Y component
        self.kernel_size1 = 5
        self.kernel_size2 = 1
        self.kernel_size3 = 3
        self.kernel_size4 = 9
        self.padding1 = 2
        self.padding3 = 1
        self.padding4 = 4
        self.sf = sf

        layers = []

        layers.append(nn.Conv2d(in_channels=self.in_nc, out_channels=self.nc1, kernel_size=self.kernel_size1,
                                padding=self.padding1, bias=True))
        layers.append(nn.PReLU())
        layers.append(nn.Conv2d(in_channels=self.nc1, out_channels=self.nc2, kernel_size=self.kernel_size2,
                                padding=0, bias=True))
        layers.append(nn.PReLU())
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels=self.nc2, out_channels=self.nc2, kernel_size=self.kernel_size3,
                                    padding=self.padding3, bias=True))
            layers.append(nn.PReLU())
        layers.append(nn.Conv2d(in_channels=self.nc2, out_channels=self.nc1, kernel_size=self.kernel_size2,
                                padding=0, bias=True))
        layers.append(nn.PReLU())

        self.main = nn.Sequential(*layers)

        self.m_up = nn.ConvTranspose2d(in_channels=self.nc1, out_channels=self.out_nc, kernel_size=self.kernel_size4,
                                       padding=self.padding4, stride=self.sf, bias=True)

    def forward(self, x):

        h, w = x.shape[-2:]
        paddingBottom = int(np.ceil(h/self.sf)*self.sf-h)
        paddingRight = int(np.ceil(w/self.sf)*self.sf-w)

        x = torch.nn.ReplicationPad2d(0, paddingRight, 0, paddingBottom)

        x = self.main(x)
        x = self.m_up(x)

        x = x[..., :h, :w]

        return x

if __name__ == '__main__':
    model = FSRCNN(sf=2)
    print(model)