import torch
import torch.nn as nn
import numpy as np

class SRCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.in_nc = 1          # RGB -> YCrCb, Only Y
        self.nc1 = 64
        self.nc2 = 32
        self.out_nc = 1         # Only Y component
        self.kernel_size1 = 9
        self.kernel_size2 = 5
        self.padding1 = 4
        self.padding2 = 2

        # Input image: Interpolated Image
        layers = []

        layers.append(nn.Conv2d(in_channels=self.in_nc, out_channels=self.nc1, kernel_size=self.kernel_size1,
                                padding=self.padding1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.nc1, out_channels=self.nc2, kernel_size=self.kernel_size2,
                                padding=self.padding2, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.nc2, out_channels=self.out_nc, kernel_size=self.kernel_size2,
                                padding=self.padding2, bias =True))

        self.main = nn.Sequential(*layers)


    def forward(self, x):

        # 어차피 interpolated image 만들 때 처리하지 않나? 확인해보고 지우자.
        h, w = x.shape[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)

        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))

        x = self.main(x)
        x = x[..., :h, :w]

        return x


if __name__ == '__main__':
    model = SRCNN()
    print(model)

