import torch.nn as nn



class VDSR(nn.Module):

    def __init__(self):
        super().__init__()

        # Input: Interpolated Y
        self.in_nc = 1  # RGB -> YCrCb, Only Y
        self.nc = 64
        self.out_nc = 1  # Only Y component
        self.kernel_size = 3
        self.padding = 1
        self.nlayers = 20

        layers = []

        layers.append(nn.Conv2d(in_channels=self.in_nc, out_channels=self.nc, kernel_size=self.kernel_size,
                                padding=self.padding, bias=True))
        layers.append(nn.ReLU())

        for _ in range(self.nlayers-2):
            layers.append(nn.Conv2d(in_channels=self.nc, out_channels=self.nc, kernel_size=self.kernel_size,
                                    padding=self.padding, bias=True))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=self.nc, out_channels=self.out_nc, kernel_size=self.kernel_size,
                                padding=self.padding, bias=True))

        self.main = nn.Sequential(*layers)


    def forward(self, x):

        input_img = x
        x = self.main(x)

        output_img = input_img + x

        return output_img

if __name__ =='__main__':

    model = VDSR()
    print(model)