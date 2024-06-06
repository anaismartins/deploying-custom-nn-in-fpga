import torch
from torch import nn


class GregNet2DModified(nn.Module):
    def __init__(self):
        super().__init__()

        self.batchnorm = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(16, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 1))
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(16, 1))

        self.fc1 = nn.Linear(18816, 128)
        self.fc2 = nn.Linear(128, 1)

        self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1))

    def forward(self, x):
        # reshape input tensor to add dummy height dimension
        x = x.view(
            x.size(0),
            x.size(1),
            x.size(-1),
            1,
        )

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

        x = self.conv1x1(x)
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

    Lin = 155648

    model = GregNet2DModified()
    summary(model, (len(DETECTORS), Lin))

    yhat = model(torch.randn(1, len(DETECTORS), Lin))

    make_dot(yhat, params=dict(list(model.named_parameters()))).render(
        "gregnet2dmodified_torchviz", format="png"
    )
