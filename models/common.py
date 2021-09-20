import torch.nn as nn
import math



class res_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        L = []
        L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           padding=padding, bias=True))
        L.append(nn.ReLU())
        L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           padding=padding, bias=True))

        self.main = nn.Sequential(*L)
        self.res_scale = 0.1


    def forward(self, x):
        res = self.main(x).matmul(self.res_scale)
        res += x

        return res


def pixelShuffle(scale, nc):

    m = []
    if (scale & (scale-1)) == 0:
        for _ in range(int(math.log(scale,2))):
            m.append(nn.Conv2d(in_channels=nc, out_channels=nc*4, kernel_size=3,
                               padding=1, bias=True))
            m.append(nn.PixelShuffle(2))

    elif scale == 3:
        m.append(nn.Conv2d(in_channels=nc, out_channels=nc*9, kernel_size=3,
                           padding=1, bias=True))
        m.append(nn.PixelShuffle(3))

    x = nn.Sequential(*m)

    return x



def meanShift():

    rgb_mean = (0.4488, 0.4371, 0.4040)
    rgb_std = (1.0, 1.0, 1.0)

    pass



def embedding_block(nc, kernel_size, padding, layer_num):

    m = []

    for _ in range(layer_num):
        m.append(nn.LeakyReLU(negative_slope=0.2))
        m.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=kernel_size, padding=padding, bias=True))

    return nn.Sequential(*m)


class Embedding_module(nn.Module):

    def __init__(self, nc, kernel_size, padding, layer_num, recursive_num):
        super().__init__()
        self.embedding_block = embedding_block(nc=nc, kernel_size=kernel_size, padding=padding, layer_num=layer_num)
        self.R_num = recursive_num

        self.Transpose = nn.ConvTranspose2d(in_channels=nc, out_channels=nc, kernel_size=kernel_size, padding=padding,
                                            stride=2, bias=True)
        self.LReLU = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        output = x.clone()

        for _ in range(self.R_num):
            output = self.embedding_block(output) + x

        m_output = self.Transpose(output)
        m_output = self.LReLU(m_output)

        return m_output