import torch

from helper import check_gen_schema


class CatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channel = 16
        self.conv1 = torch.nn.Conv2d(3, channel, 1)
        self.bn1 = torch.nn.BatchNorm2d(channel)
        self.conv2 = torch.nn.Conv2d(3, channel, 1)
        self.bn2 = torch.nn.BatchNorm2d(channel)

        self.conv3 = torch.nn.Conv2d(channel * 2, channel, 1)
        self.bn3 = torch.nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))

        x = torch.cat([x1, x2], dim=1)
        x = self.bn3(self.conv3(x))
        return x


class CatModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channel = 16
        self.conv1 = torch.nn.Conv2d(3, channel, 1)
        self.bn1 = torch.nn.BatchNorm2d(channel)
        self.conv2 = torch.nn.Conv2d(3, channel, 1)
        self.bn2 = torch.nn.BatchNorm2d(channel)

        self.conv3 = torch.nn.Conv2d(channel * 2, channel, 1)
        self.bn3 = torch.nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))

        x = torch.cat([x2, x1], dim=1)
        x = self.bn3(self.conv3(x))
        return x


def test_cat():
    check_gen_schema(CatModel())


def test_cat2():
    check_gen_schema(CatModel2())
