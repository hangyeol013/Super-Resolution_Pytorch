import torch.nn as nn
import common




class MS_LapSRN(nn.Module):

    def __init__(self):
        super().__init__()

        self.in_nc = 3
        self.nc = 64
        self.out_nc = 3
        self.kernel_size = 3
        self.padding = 1
        self.D = 5
        self.R = 8


        self.m_init = nn.Conv2d(in_channels=self.in_nc, out_channels=self.nc, kernel_size=self.kernel_size,
                                padding=self.padding, bias=True)

        self.m_embedding = common.Embedding_module(nc=self.nc, kernel_size=self.kernel_size, padding=self.padding,
                                            layer_num=self.D, recursive_num=self.R)

        self.res_sub = nn.Conv2d(in_channels=self.nc, out_channels=self.out_nc, kernel_size=self.kernel_size,
                                 padding=self.padding, bias=True)

        self.m_upsampling = nn.ConvTranspose2d(in_channels=self.out_nc, out_channels=self.out_nc, kernel_size=self.kernel_size,
                                             padding=self.padding, stride=2, bias=True)


    def forward(self, x):

        initConv_x = self.m_init(x)

        upsample_1 = self.m_upsampling(x)
        embedding_1 = self.m_embedding(initConv_x)
        res_1 = self.res_sub(embedding_1)

        hr_x2 = upsample_1 + res_1

        upsample_2 = self.upsample(hr_x2)
        embedding_2 = self.m_embedding(embedding_1)
        res_2 = self.res_sub(embedding_2)

        hr_x4 = upsample_2 + res_2

        return hr_x2, hr_x4

if __name__ == '__main__':

    model = MS_LapSRN()
    print(model)