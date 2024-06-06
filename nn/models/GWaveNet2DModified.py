import torch
from torch import nn


class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(CausalConv2d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(self.padding, 0),
            dilation=(dilation, 1),
        )

    def forward(self, x):
        x = nn.functional.pad(x, (0, 0, self.padding, 0), mode="replicate")
        x = self.conv(x)
        return x


class ModuleWavenet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ModuleWavenet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_dilated = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.gate_tanh = nn.ReLU6()
        self.gate_sigmoid = nn.Hardsigmoid()

        self.conv1x1 = nn.Conv2d(
            in_channels=out_channels, out_channels=32, kernel_size=(1, 1)
        )

    def forward(self, x):
        x_old = x

        x_a = self.conv_dilated(x)
        x_b = self.conv_dilated(x)

        x_a = self.batchnorm(x_a)
        x_b = self.batchnorm(x_b)

        # pixel gate
        x_skip = self.gate_tanh(x_a) * self.gate_sigmoid(x_b)
        x_skip = self.conv1x1(x_skip)

        # skip-connection
        if self.in_channels != self.out_channels:
            x_res = x_skip
        else:
            if x_old.size(2) < x_skip.size(2):
                x_res = x_old + x_skip[:, :, : x_old.size(2), :]
            else:
                x_res = x_skip + x_old[:, :, : x_skip.size(2), :]

        return x_res, x_skip


class GWaveNet2DModified(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super().__init__()

        self.Mod0 = ModuleWavenet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**0,
        )
        self.Mod1 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**1,
        )
        self.Mod2 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**2,
        )
        self.Mod3 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**3,
        )
        self.Mod4 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**4,
        )
        self.Mod5 = ModuleWavenet(
            in_channels=32,
            out_channels=32,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**5,
        )
        self.Mod6 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**6,
        )

        self.conv_1x1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1302, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # reshape input tensor to add dummy height dimension
        x = x.view(
            x.size(0),
            x.size(1),
            x.size(-1),
            1,
        )

        x, xs0 = self.Mod0(x)
        x, xs1 = self.Mod1(x)
        x, xs2 = self.Mod2(x)
        x, xs3 = self.Mod3(x)
        x, xs4 = self.Mod4(x)
        x, xs5 = self.Mod5(x)
        x, xs6 = self.Mod6(x)

        xs = xs6

        for i in [xs0, xs1, xs2, xs3, xs4, xs5]:
            if xs.size(2) < i.size(2):
                xs = xs + i[:, :, -xs.size(2) :, :]
            else:
                xs = i + xs[:, :, -i.size(2) :, :]

        xs = self.conv_1x1(xs)
        xs = self.relu(xs)

        xs = self.flatten(xs)
        xs = self.fc1(xs)
        xs = self.relu(xs)

        xs = self.fc2(xs)

        return xs


if __name__ == "__main__":
    from globals import DETECTORS, DILATION, FILTERS, STRIDE
    from torchsummary import summary

    Lin = 155648
    kernel_size_convolution = 2

    model = GWaveNet2DModified(
        in_channels=len(DETECTORS),
        out_channels=FILTERS,
        kernel_size=kernel_size_convolution,
        stride=STRIDE,
        dilation=DILATION,
    )

    summary(model, (len(DETECTORS), Lin))
