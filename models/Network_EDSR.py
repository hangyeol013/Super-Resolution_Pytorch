import torch.nn as nn
from .common import res_block, pixelShuffle


class EDSR(nn.Module):

    def __init__(self, scale):
        super().__init__()

        self.in_nc = 3  # RGB -> YCrCb, Only Y
        self.nc = 256
        self.out_nc = 3  # Only Y component
        self.kernel_size = 3
        self.padding = 1
        self.nblocks = 32
        self.scale = scale


        m_head = [nn.Conv2d(in_channels=self.in_nc, out_channels=self.nc, kernel_size=self.kernel_size,
                            padding=self.padding, bias=True)]

        m_body = [res_block(in_channels=self.nc, out_channels=self.nc, kernel_size=self.kernel_size,
                                   padding=self.padding) for _ in range(self.nblocks)]
        m_body.append(nn.Conv2d(in_channels=self.nc, out_channels=self.nc, kernel_size=self.kernel_size,
                                padding=self.padding, bias=True))

        m_tail = [pixelShuffle(scale=self.scale, nc=self.nc),
                  nn.Conv2d(in_channels=self.nc, out_channels=self.out_nc, kernel_size=self.kernel_size,
                            padding=self.padding, bias=True)]


        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x


if __name__ == '__main__':

    model = EDSR(scale=2)

    print(model)