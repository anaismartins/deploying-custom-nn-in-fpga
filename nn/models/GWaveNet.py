import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(CausalConv1d, self).__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x


class ModuleWavenet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ModuleWavenet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.batch_norm = torch.nn.BatchNorm1d(out_channels)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

        # causal convolution
        self.conv_dilated = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        x_old = x

        x_a = self.conv_dilated(x)
        x_b = self.conv_dilated(x)

        # Pixel gate
        x_a = self.batch_norm(x_a)
        x_b = self.batch_norm(x_b)
        x_skip = self.gate_tanh(x_a) * self.gate_sigmoid(x_b)

        # Skip-connection
        if self.in_channels != self.out_channels:
            x_res = x_skip
        else:
            if x_old.size(2) < x_skip.size(2):
                x_res = x_old + x_skip[:, :, -x_old.size(2) :]
            else:
                x_res = x_skip + x_old[:, :, -x_skip.size(2) :]

        return x_res, x_skip


class GWaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(GWaveNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Define convolutions
        self.Mod0 = ModuleWavenet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**0,
        )
        self.Mod1 = ModuleWavenet(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**1,
        )
        self.Mod2 = ModuleWavenet(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**2,
        )
        self.Mod3 = ModuleWavenet(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**3,
        )
        self.Mod4 = ModuleWavenet(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**4,
        )
        self.Mod5 = ModuleWavenet(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**5,
        )
        self.Mod6 = ModuleWavenet(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**6,
        )

        self.conv_1x1 = torch.nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
        )

        torch.nn.init.kaiming_uniform_(self.conv_1x1.weight)

        self.linear = nn.Linear(
            2496,
            50,
        )
        self.linear2 = nn.Linear(50, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x, xs0 = self.Mod0(x)
        x, xs1 = self.Mod1(x)
        x, xs2 = self.Mod2(x)
        x, xs3 = self.Mod3(x)
        x, xs4 = self.Mod4(x)
        x, xs5 = self.Mod5(x)
        x, xs6 = self.Mod6(x)

        xs = xs6

        for i in [
            xs0,
            xs1,
            xs2,
            xs3,
            xs4,
            xs5,
        ]:
            if xs.size(2) < i.size(2):
                xs = xs + i[:, :, -xs.size(2) :]
            else:
                xs = i + xs[:, :, -i.size(2) :]

        xs = self.conv_1x1(xs)
        xs = self.relu(xs)

        xs = self.linear(xs)
        xs = self.relu(xs)
        xs = self.linear2(xs)

        xs = xs.view(xs.shape[0], 1)

        return xs


if __name__ == "__main__":
    import os

    from globals import DETECTORS, DILATION, FILTERS, KERNEL_SIZE_CONVOLUTION, STRIDE
    from torchsummary import summary
    from torchviz import make_dot

    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"
    Lin = 155648

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GWaveNet(
        in_channels=len(DETECTORS),
        out_channels=FILTERS,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
        dilation=DILATION,
    ).to(device)

    summary(model, (3, Lin))

    x = torch.randn(1, 3, Lin).to(device)
    flops = FlopCountAnalysis(model, x)
    print(f"Flops: {flops.total()/1e9} G")

    yhat = model(x)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(
        "gwavenet_torchviz", format="png"
    )
