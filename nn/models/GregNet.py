import torch
from fvcore.nn import FlopCountAnalysis
from torch import nn


class GregNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.batchnorm = nn.BatchNorm1d(3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=16)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16)

        self.fc1 = nn.Linear(37632, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    import os

    from globals import DETECTORS
    from torchsummary import summary
    from torchviz import make_dot

    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GregNet().to(device)
    Lin = 155648

    summary(model, (len(DETECTORS), Lin))

    x = torch.randn(1, len(DETECTORS), Lin).to(device)

    flops = FlopCountAnalysis(model, x)
    print(f"Flops: {flops.total()/1e9} G")

    yhat = model(x)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(
        "gregnet_torchviz", format="png"
    )
